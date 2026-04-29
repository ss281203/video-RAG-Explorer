# Video RAG Explorer

## Overview

Video RAG Explorer is a Streamlit application that uses Retrieval-Augmented Generation (RAG) to allow users to query and extract information from YouTube videos. The application downloads videos, processes their content (including audio transcription and keyframe extraction), and creates a searchable knowledge graph that enables natural language queries about the video content.
Demo - https://drive.google.com/file/d/1pAsfIFLpO2ndYCHLqYuKywgswaGIkYOC/view?usp=sharing

## Features

- YouTube video download and processing
- Automatic transcription using Whisper and translation if needed
- Intelligent keyframe extraction and scene description using Gemini
- Creation of a semantic knowledge graph from video content
- Natural language querying of video content
- Visual exploration of video keyframes
- Time-stamped responses linked to video segments

## Prerequisites

- Python 3.8+
- FFmpeg installed on your system
- Google Cloud API key for Gemini
- Pinecone API key

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Sid-tyagi-ar/video-rag.git
cd video-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg

#### On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### On macOS (using Homebrew):
```bash
brew install ffmpeg
```

#### On Windows:
Install using Chocolatey:
```bash
choco install ffmpeg
```
You can also download it from the [official website](https://ffmpeg.org/download.html) and add it to PATH.

### 5. Set up API keys

 Update the API keys directly in the `Helper_function.py`  and 'streamlit_run.py' file.

## Running the Application

```bash
streamlit run streamlit_app.py
```

The application will be available at http://localhost:8501

## How to Use

1. Enter a YouTube URL in the input field and click "Process Video"
2. Wait for the processing to complete (this may take a few minutes depending on the video length)
3. Once processing is complete, you'll see keyframes extracted from the video
4. Enter your query about the video content in the "Ask About the Video" field
5. Review the AI-generated answer and explore the source timestamps and chunks

## Advanced Mode

Toggle "Advanced Mode" in the sidebar to access additional settings:
- Max Edge Depth: Controls how many related chunks to consider when retrieving content
- Bin Size: Adjusts the size of time segments for chunking the video content

## How It Works

### Video Processing Pipeline

1. **Download and Extraction**: The app downloads the YouTube video and its audio track
2. **Subtitle Extraction**: Attempts to extract existing subtitles from YouTube
3. **Transcription**: If subtitles aren't available, uses Whisper for audio transcription
4. **Translation**: Translates non-English content to English if needed
5. **Keyframe Extraction**: Extracts significant frames using computer vision techniques
6. **Scene Description**: Uses Gemini to describe the content of keyframes

### RAG System

1. **Chunking**: Divides the transcript into meaningful segments
2. **Integration**: Combines transcript chunks with keyframe descriptions
3. **Embedding**: Creates vector embeddings of each chunk
4. **Graph Building**: Creates a knowledge graph connecting related chunks
5. **Indexing**: Stores chunks and relationships in Pinecone
6. **Retrieval**: Finds the most relevant chunks for a query
7. **Generation**: Uses Gemini to generate answers based on retrieved chunks

## Troubleshooting

- **FFmpeg errors**: Ensure FFmpeg is properly installed and in your PATH.
- **API key errors**: Verify your Pinecone and Gemini API keys are correct.
- **Memory issues**: For long videos, you may need to increase your system's memory or reduce the processing quality.
- **Language support**: For non-English videos and non-spring-asr coverage videos, check if the language is supported by Whisper.
- **Gen-ai** - It might be the case that you can get an error using 'from google import genai' on the local machine, so one can use 'import google.generativeai as genai' that works too.

## Notes
- Usage of pikachu running gif to make the loading appealing and also keyframe detection would take major time so while watching demo prefer 2x during that portion, thanks 
