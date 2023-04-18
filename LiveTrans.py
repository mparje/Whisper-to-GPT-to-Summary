import streamlit as st
import config
import whisper
import os, glob
import sounddevice as sd
import wavio as wv
import datetime
import openai
import os

# set up OpenAI API credentials
openai.api_key = os.getenv("YOUR OPENAI_API_KEY")
openai.proxy = 'http://127.0.0.1:7890'

# list to keep track of conversation history
message_history = [{"role": "assistant", "content": f"OK"}]

# function to run GPT-3 model and generate response
def GPT(input):
    # tokenize the new input sentence
    message_history.append({"role": "user", "content": f"Please summarize this conversation in Chinese: {input}"}) 
    # It is up to you to ask the model to output bullet points or just a general summary
    prompt_history = [message_history[len(message_history)-2],message_history[len(message_history)-1]] 
    # I believe by putting the previous messages into the current context can improve the model's overall accuracy.
    completion = openai.Completion.create(
      engine="davinci", # use the davinci engine for better accuracy
      prompt="\n".join([f"{msg['role']}: {msg['content']}" for msg in prompt_history]), # format conversation history for input
      max_tokens=1024, # limit output to 1024 tokens
      n=1, # only generate one response
      stop=None, # don't stop generation until max_tokens is reached
    )
    st.write(f"{completion.total_characters} tokens consumed.")
    reply_content = completion.choices[0].text
    message_history.append({"role": "assistant", "content": f"{reply_content}"})
    return reply_content

# set up audio recording parameters
freq = 44100 
duration = 5 

# create recordings directory if it doesn't exist
if not os.path.exists('recordings'):
    os.mkdir('recordings')

# load pre-trained model for audio transcription
model = whisper.load_model("base")

# list to store which wav files have been transcribed
transcribed = []

# main Streamlit app
st.title("Whisper-Jarvis Demo")

st.write("Press the button below to start recording audio.")

# add a button to start recording audio
if st.button("Start Recording"):
    st.write('Recording...')

    # generate filename based on current timestamp
    ts = datetime.datetime.now()
    filename = ts.strftime("%Y_%m_%d_%H_%M_%S")

    # start audio recording
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
    sd.wait()

    # save audio recording to disk
    wv.write(f"./recordings/{filename}.wav", recording, freq, sampwidth=2)

    st.write(f"Saved audio recording to ./recordings/{filename}.wav")

    # get most recent wav recording in the recordings directory
    files = sorted(glob.iglob(os.path.join('recordings', '*')), key=os.path.getctime, reverse=True)

    if len(files) < 1:
        st.write("No audio files found.")
    else:
        latest_recording = files[0]
        latest_recording_filename = latest_recording.split('_')[1]

        # check if audio file has already been transcribed
        if os.path.exists(latest_recording) and not latest_recording in transcribed:
            # load audio file and preprocess for transcription
            audio = whisper.load_audio(latest_recording)
            audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)

        result = whisper.decode(model, mel, options)

        if result.no_speech_prob < 0.5:
            print(result.text)
            # append text to transcript file
            with open(config.TRANSCRIPT_FILE, 'a') as f:
                f.write(result.text)

            # save list of transcribed recordings so that we don't transcribe the same one again
            transcribed.append(latest_recording)
        
        #triggering phrase for GPT language model
        if 'Please summarize this conversation in sPANISH' in result.text:
            print("--------Deploying Jarvis--------")
            transcript = open('./transcriptions/transcript.txt', 'r').read()
           
            print("--------Deploying Jarvis--------")
            transcript = open('./transcriptions/transcript.txt', 'r').read()
            print(GPT(transcript))



