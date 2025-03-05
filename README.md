# Speech-to-Speech-RAG-App
Both STT(Speech-to-text) and TTS(Text-to-speech) functionality App using RAG

This Streamlit application combines speech recognition, Retrieval Augmented Generation (RAG), and text-to-speech (TTS) to create a conversational AI that can interact with uploaded PDF documents.

## Features

* **PDF Upload and Processing:**
    * Upload multiple PDF documents.
    * Extract text from PDFs and split it into manageable chunks.
    * Create a vector store (FAISS) for efficient similarity search.
* **Speech Recognition:**
    * Record audio directly through the application.
    * Transcribe recorded audio using the Whisper model.
* **Retrieval Augmented Generation (RAG):**
    * Use the transcribed text as a query.
    * Retrieve relevant information from the vector store.
    * Generate a response using the Gemini API, incorporating retrieved context.
* **Text-to-Speech (TTS):**
    * Convert the generated response into spoken audio using the Kokoro82M model.
    * Play the generated speech in the browser.
* **User-Friendly Interface:**
    * Intuitive Streamlit interface for easy interaction.
 
## Screenshot
![speechRAG ss2](https://github.com/user-attachments/assets/9d3f29a8-2e1b-48cf-a86c-e60ea1d3eb77)


## Prerequisites

* Python 3.7+
* API keys for:
    * Google Gemini API
* Libraries:
    * `streamlit`
    * `pyaudio`
    * `wave`
    * `librosa`
    * `transformers`
    * `PyPDF2`
    * `langchain`
    * `faiss-cpu` (or `faiss-gpu`)
    * `sentence-transformers`
    * `google-generativeai`
    * `python-dotenv`
    * `torch`
    * `Kokoro82M`

## Installation

1.  Clone the repository:

    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```

2.  Install the required packages:

    ```bash
    pip install streamlit pyaudio librosa transformers PyPDF2 langchain faiss-cpu sentence-transformers google-generativeai python-dotenv torch Kokoro82M
    ```

3.  Set up environment variables:

    * Create a `.env` file in the project directory.
    * Add your Google Gemini API key:

        ```
        GEMINI_API_KEY=your_gemini_api_key
        ```
    * If using windows, add the following lines to your .env file, and point the path to your eSpeak NG installation.

        ```
        PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll
        PHONEMIZER_ESPEAK_PATH=C:\Program Files\eSpeak NG\espeak-ng.exe
        ```

4.  Run the Streamlit app:

    ```bash
    streamlit run app.py
    ```

## Usage

1.  Upload one or more PDF documents using the sidebar.
2.  Click "Process PDF" to prepare the documents for querying.
3.  Use the slider to select the recording duration.
4.  Click "Start Recording" to record your voice query.
5.  Click "Transcribe" to convert your recorded audio to text.
6.  Click "Reply" to generate a response based on the transcribed text and uploaded PDFs.
7.  Click "Generate Speech" to convert the generated response into spoken audio.

## Notes

* Ensure your microphone is properly configured.
* The quality of the speech recognition and generated responses depends on the clarity of the audio and the content of the PDFs.
* The Kokoro82M model requires a CUDA capable GPU to run efficiently. If a GPU is not available, it will run on the CPU, but performance will be significantly reduced.
* The path to the eSpeak NG installation must be provided correctly for the TTS model to work on windows.
