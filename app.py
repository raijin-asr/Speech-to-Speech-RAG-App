import os

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = r"C:\Program Files\eSpeak NG\libespeak-ng.dll"
os.environ["PHONEMIZER_ESPEAK_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

import streamlit as st
import pyaudio
import wave
import librosa
from transformers import pipeline
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
import torch
from Kokoro82M.kokoro import generate
from Kokoro82M.models import build_model

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize speech-to-text pipeline (Whisper)
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small")

# Function to record audio
def record_audio(filename, record_seconds=5, sample_rate=44100, chunk_size=1024):
    try:
        st.info(f"Recording for {record_seconds} seconds...")
        p = pyaudio.PyAudio()

        # Open the audio stream
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=sample_rate,
                       input=True,
                       frames_per_buffer=chunk_size)

        frames = []

        for _ in range(0, int(sample_rate / chunk_size * record_seconds)):
            data = stream.read(chunk_size)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

        # Save the recorded data to a .wav file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))

        st.success("Recording complete!")
        return filename

    except Exception as e:
        st.error(f"Error while recording audio: {e}")
        return None


# Function to transcribe audio using Whisper
def transcribe_audio(filename):
    try:
        # Load audio using librosa
        audio, sr = librosa.load(filename, sr=16000)

        # Transcribe the audio using Whisper
        transcription = asr_pipeline(audio)
        return transcription['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None


# Function to extract text from uploaded PDFs
def extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to create vector store using HuggingFace embeddings
def create_and_save_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Function to create the conversational chain using Gemini API
def create_prompt_template():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer contains any structured data like tables or lists, respond in the same format. 
    If the answer is not in the provided context, just say, "The answer is not available in the context." Do not provide a wrong answer.

    Context:
    {context}

    Question:
    {question}
    """
    return prompt_template


# Function to handle user input and provide a response
def handle_user_query(transcribed_text):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(transcribed_text)

    context = "\n\n".join([doc.page_content for doc in docs])
    prompt_template = create_prompt_template()
    formatted_prompt = prompt_template.format(context=context, question=transcribed_text)

    # Call Gemini API
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(formatted_prompt)
    return response.text if response.text else "No response generated."


# Function to initialize the TTS model
@st.cache_resource
def initialize_tts():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model('Kokoro82M/kokoro-v0_19.pth', device)
    voice_name = 'af'  # Default voice
    voicepack = torch.load(f'Kokoro82M/voices/{voice_name}.pt', weights_only=True).to(device)
    return model, voicepack, device


# Function to handle text-to-speech conversion
def text_to_speech(response_text, model, voicepack, device):
    try:
        audio, sample_rate = generate(model, response_text, voicepack, lang='a') 
        return audio, sample_rate
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None, None

def main():
    st.set_page_config("Speech-to-Speech RAG", layout="wide")
    st.title("Chat with PDF and Voice 🎙️")

    # Initialize session state variables
    if "transcribed_text" not in st.session_state:
        st.session_state.transcribed_text = ""

    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    if "response_text" not in st.session_state:
        st.session_state.response_text  = ""

    # File name for saving the recording
    filename = "recorded_audio.wav"

    # Sidebar for PDF upload
    with st.sidebar:
        st.title("Upload PDF 📂")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Process PDF"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = extract_pdf_text(pdf_docs)
                    text_chunks = split_text_into_chunks(raw_text)
                    create_and_save_vector_store(text_chunks)
                    st.session_state.pdf_processed = True
                    st.success("PDF processing complete!")
            else:
                st.error("Please upload PDF files before processing.")

    # Main section for audio recording and transcription
    record_seconds = st.slider("Select recording duration (seconds):", min_value=1, max_value=60, value=5)

    if st.button("Start Recording"):
        record_audio(filename, record_seconds=record_seconds)

    if st.button("Transcribe"):
        if filename:
            st.info("Transcribing the recorded audio...")
            transcribed_text = transcribe_audio(filename)
            if transcribed_text:
                st.session_state.transcribed_text = transcribed_text
                st.success("Transcription complete!")
                st.text_area("Transcribed Text", st.session_state.transcribed_text, height=150)
            else:
                st.error("Transcription failed.")
        else:
            st.error("Please record audio first.")

    if st.button("Reply"):
        if st.session_state.pdf_processed or st.session_state.transcribed_text:
            with st.spinner("Generating reply..."):
                query_text = st.session_state.transcribed_text
                response_text = handle_user_query(query_text)
                st.session_state.response_text = response_text  # Save the response to session state
                st.text_area("Reply", st.session_state.response_text, height=150)
        else:
            st.error("No PDF processed or transcribed text available. Please upload a PDF or transcribe audio first.")
    
    st.write("Debug: Response Text:", st.session_state.response_text)

    model, voicepack, device = initialize_tts()
    
    if st.button("Generate Speech"):
        st.write("Debug inside button: Response Text:", st.session_state.response_text)
        if st.session_state.response_text and st.session_state.response_text.strip():  
            with st.spinner("Generating audio..."):
                speech_text = st.session_state.response_text
                audio, sample_rate = text_to_speech(speech_text, model, voicepack, device)
                if audio is not None:
                    st.audio(audio, format="audio/wav", sample_rate=24000)
                else:
                    st.error("Error generating speech. Please try again.")
        else:
            st.error("Please get some reply before generating speech.")

if __name__ == "__main__":
    main()