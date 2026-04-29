#%%
import cv2
import numpy as np
from pathlib import Path
import yt_dlp
import os
import shutil
import webvtt  
import whisper
import google.generativeai as genai
import PIL.Image
import time
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
import math
import streamlit as st
import re
import logging
import json
load_dotenv()

#%%

INDEX_NAME = "video-rag"
Pinecone_api_key = os.getenv("PINECONE_API_KEY")
gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)
pc = Pinecone(api_key=Pinecone_api_key)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

class Video_processor:
    def __init__(self, output_dir="downloads", keyframe_dir="keyframes", ffmpeg_path=None):
        self.output_dir = output_dir
        self.keyframe_dir = keyframe_dir
        self.ffmpeg_path = ffmpeg_path or shutil.which("ffmpeg") or r"C:\ProgramData\chocolatey\bin\ffmpeg.exe"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(keyframe_dir, exist_ok=True)

    def extract_video_id(self, url):
        """Extract video ID from YouTube URL."""
        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        return match.group(1) if match else "unknown"

    def download_video(self, url):
        video_id = self.extract_video_id(url)
        video_file = f"{self.output_dir}/{video_id}.mp4"
        ydl_opts = {
            'format':'best[ext=mp4]/best',
            'outtmpl': f'{self.output_dir}/%(id)s.%(ext)s',
            'merge_output_format': 'mp4',
            'ffmpeg_location': self.ffmpeg_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        return video_file

    def download_audio(self, url, audio_format="mp3"):
        video_id = self.extract_video_id(url)
        audio_file = f"{self.output_dir}/{video_id}.{audio_format}"
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': f'{self.output_dir}/%(id)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': audio_format,
            }],
            'ffmpeg_location': self.ffmpeg_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return audio_file

    def check_existing_subtitles(self, url, sub_format="vtt", preferred_lang="en"):
        video_id = self.extract_video_id(url)
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
            'subtitlesformat': sub_format,
            'subtitleslangs': [preferred_lang, 'en'] + [lang for lang in ['hi', 'ta', 'te'] if lang != preferred_lang],
            'outtmpl': f'{self.output_dir}/{video_id}.%(ext)s',
            'ffmpeg_location': self.ffmpeg_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            available_subs = info.get('subtitles', {}) or info.get('automatic_captions', {})
            if not available_subs:
                logging.info("No subtitles available.")
                return None, None, None
            lang = next((l for l in [preferred_lang, 'en'] + list(available_subs.keys()) if l in available_subs), None)
            if not lang:
                logging.info("No usable subtitle language found.")
                return None, None, None
            ydl_opts['subtitleslangs'] = [lang]
            ydl.download([url])
            subtitle_file = f"{self.output_dir}/{video_id}.{lang}.{sub_format}"
            if os.path.exists(subtitle_file):
                title = info.get('title', video_id)
                logging.info(f"Subtitles downloaded: {subtitle_file} (lang: {lang})")
                return subtitle_file, title, lang
            return None, None, None

    def detect_language(self, audio_file):
        model = whisper.load_model("base")
        audio = whisper.load_audio(audio_file)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        _, probs = model.detect_language(mel)
        return max(probs, key=probs.get)

    # def translate_vtt(self, vtt_file, src_lang):
    #     logging.info(f"Translating VTT from {src_lang} to English...")
    #     vtt = webvtt.read(vtt_file)
    #     translated_vtt = webvtt.WebVTT()
    #     for caption in vtt:
    #         translated = self.translator.translate(caption.text, src=src_lang, dest='en')
    #         translated_vtt.captions.append(webvtt.Caption(
    #             start=caption.start,
    #             end=caption.end,
    #             text=translated.text if translated else caption.text
    #         ))
    #     video_id = os.path.basename(vtt_file).split('.')[0]
    #     translated_file = f"{self.output_dir}/{video_id}.vtt"  # Overwrite as URI.vtt
    #     translated_vtt.save(translated_file)
    #     with open(translated_file, 'r', encoding='utf-8') as f:
    #         return translated_file, f.read()
    def translate_vtt(self, vtt_file, src_lang):
        logging.info(f"Skipping translation. Using original VTT for language: {src_lang}")
        with open(vtt_file, "r", encoding="utf-8") as f:
            return vtt_file, f.read()

    def generate_transcript_spring_api(self, audio_file, language):
        video_id = os.path.basename(audio_file).split('.')[0]
        files = {'file': open(audio_file, 'rb'), 'language': (None, language), 'vtt': (None, 'true')}
        response = requests.post('https://asr.iitm.ac.in/internal/asr/decode', files=files)
        if response.status_code != 200 or not response.json().get("status") == "success":
            raise Exception(f"Spring API failed: {response.text}")
        vtt_content = response.json().get("vtt")
        output_file = f"{self.output_dir}/{video_id}.vtt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(vtt_content)
        return response.json().get("transcript"), output_file

    def generate_transcript_whisper(self, audio_file):
        video_id = os.path.basename(audio_file).split('.')[0]
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language='en')
        output_file = f"{self.output_dir}/{video_id}.vtt"
        vtt = webvtt.WebVTT()
        for segment in result["segments"]:
            vtt.captions.append(webvtt.Caption(
                start=self.seconds_to_timestamp(segment["start"]),
                end=self.seconds_to_timestamp(segment["end"]),
                text=segment["text"]
            ))
        vtt.save(output_file)
        return result["text"], output_file

    def seconds_to_timestamp(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    def extract_keyframes(self, video_path, sift_threshold=0.7, lap_threshold=0.5, frame_skip=15, 
                     w_sift=0.6, w_lap=0.4, total_threshold=0.5, resize_height=360):
        video_id = os.path.basename(video_path).split('.')[0]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("Could not open video file.")
        keyframes = []
        frame_count = 0
        keyframe_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sift = cv2.SIFT_create()
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        prev_kp, prev_des, prev_edges = None, None, None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            h, w = frame.shape[:2]
            scale = resize_height / h
            frame = cv2.resize(frame, (int(w * scale), resize_height), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
            kp, des = sift.detectAndCompute(smoothed, None)

            # SIFT score
            sift_score = 0.0
            if prev_des is not None and des is not None and len(kp) > 0 and len(prev_kp) > 0:
                matches = bf.match(prev_des, des)
                match_ratio = len(matches) / min(len(prev_kp), len(kp))
                sift_score = 1.0 - match_ratio

            # Laplacian score
            edges = cv2.convertScaleAbs(cv2.Laplacian(smoothed, cv2.CV_64F))
            lap_score = 0.0
            if prev_edges is not None:
                diff = np.mean(np.abs(edges - prev_edges))
                lap_score = min(diff / 255.0, 1.0)

            # Keyframe decision
            if prev_des is not None and prev_edges is not None:
                total_score = w_sift * sift_score + w_lap * lap_score
                if total_score > total_threshold:
                    timestamp = frame_count / fps
                    keyframe_path = f"{self.keyframe_dir}/{video_id}_frame_{keyframe_idx:03d}.jpg"
                    cv2.imwrite(keyframe_path, frame)
                    keyframes.append((frame_count, timestamp, keyframe_path))
                    keyframe_idx += 1

            prev_kp, prev_des, prev_edges = kp, des, edges
            frame_count += 1

        cap.release()
        logging.info(f"Extracted {len(keyframes)} keyframes from {video_path}")
        return keyframes
        
    def describe_keyframes(self, keyframes, gemini_api_key):
        model = genai.GenerativeModel("gemini-2.0-flash")  
        
        prompt = (
            "You are supposed to give a short descriptor for the image provided."
            "Example: Input: Image showcasing a dog in park "
            "Output The image showcases a brown colored dog running in park and is seemingly happy. "
            "Now do the same kind of reasoning for the input given."
        )

        descriptions = {}
        for i,(frame_num, timestamp, path) in enumerate(keyframes):
            print(f"Describing {path} at {timestamp:.1f}s")
            try:
                # Use Pillow to open image
                image = PIL.Image.open(path)
                
                # Generate content with prompt + image
                response = model.generate_content([prompt, image])
                print(f"Gemini Response: {response.text}")
                
                if hasattr(response, 'text') and response.text:
                    descriptions[path] = {"desc": response.text.strip(), "timestamp": timestamp}
                else:
                    print(f"No text in response for {path}")
                    descriptions[path] = {"desc": "No description generated", "timestamp": timestamp}
                if (i + 1) % 5 == 0:
                    print("Pausing 2 seconds to respect rate limits...")
                    time.sleep(10)
            except Exception as e:
                print(f"Error for {path}: {str(e)}")
                descriptions[path] = {"desc": f"Error: {str(e)}", "timestamp": timestamp}

        
        return descriptions

    def process_video(self, url, audio_format="mp3", preferred_lang="en", 
                     sift_threshold=0.7, lap_threshold=0.5, frame_skip=15, gemini_api_key=None):
        try:
            video_id = self.extract_video_id(url)
            video_file = self.download_video(url)
            audio_file = self.download_audio(url, audio_format)
            keyframes = self.extract_keyframes(video_file, sift_threshold, lap_threshold, frame_skip)
            keyframe_desc = self.describe_keyframes(keyframes, gemini_api_key) if gemini_api_key and keyframes else None

            # Check YouTube subtitles first—stop if successful
            subtitle_file, title, subtitle_lang = self.check_existing_subtitles(url, sub_format="vtt", preferred_lang=preferred_lang)
            if subtitle_file:
                transcript_file = f"{self.output_dir}/{video_id}.vtt"
                if subtitle_lang != 'en':
                    logging.info(f"Translating YouTube subs from {subtitle_lang} to English...")
                    transcript_file, transcript = self.translate_vtt(subtitle_file, subtitle_lang)
                    os.remove(subtitle_file)  # Clean up original
                else:
                    os.rename(subtitle_file, transcript_file)  # Rename to video_id.vtt
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        transcript = f.read()
                return {
                    "status": "success",
                    "video_file": video_file,
                    "audio_file": audio_file,
                    "keyframes": keyframes,
                    "keyframe_descriptions": keyframe_desc,
                    "transcript_file": transcript_file,
                    "language": "en",
                    "transcript": transcript,
                    "source": "youtube"
                }

            # Fallback to Spring/Whisper only if no subs
            detected_lang = self.detect_language(audio_file)
            indian_languages = {"bn": "bengali", "en": "english", "gu": "gujarati", "hi": "hindi", 
                                "kn": "kannada", "ml": "malayalam", "mr": "marathi", "or": "odia", 
                                "pa": "punjabi", "sa": "sanskrit", "ta": "tamil", "te": "telugu", "ur": "urdu"}
            if detected_lang in indian_languages and detected_lang != 'en':
                spring_lang = indian_languages[detected_lang]
                transcript, vtt_file = self.generate_transcript_whisper(audio_file)
                transcript_file, transcript = self.translate_vtt(vtt_file, detected_lang)
                source = "spring_lab"
            else:
                transcript, transcript_file = self.generate_transcript_whisper(audio_file)
                if detected_lang != 'en':
                    transcript_file, transcript = self.translate_vtt(transcript_file, detected_lang)
                source = "whisper"
            return {
                "status": "success",
                "video_file": video_file,
                "audio_file": audio_file,
                "keyframes": keyframes,
                "keyframe_descriptions": keyframe_desc,
                "transcript_file": transcript_file,
                "language": "en",
                "transcript": transcript,
                "source": source
            }
        except Exception as e:
            logging.error(f"Process video failed: {e}")
            return {"status": "error", "message": str(e)}

class VideoRAGChunker:
    def __init__(self, output_dir="downloads", bin_size=10.0):
        self.output_dir = output_dir
        self.bin_size = bin_size
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def ts_to_seconds(self, ts):
        try:
            if len(ts.split(':')) == 3:
                h, m, s = ts.split(':')
                return int(h) * 3600 + int(m) * 60 + float(s)
            elif len(ts.split(':')) == 2:
                m, s = ts.split(':')
                return int(m) * 60 + float(s)
            raise ValueError(f"Unknown timestamp format: {ts}")
        except Exception as e:
            logging.error(f"Error parsing timestamp '{ts}': {e}")
            return 0.0

    def preprocess_transcript(self, vtt_file):
        """Preprocess with sentence-level deduplication per bin."""
        vtt = webvtt.read(vtt_file) if os.path.exists(vtt_file) else []
        segments = [(self.ts_to_seconds(c.start), self.ts_to_seconds(c.end), c.text.strip()) 
                    for c in vtt if c.text.strip()]
        if not segments:
            logging.error("No segments found in .vtt")
            return []

        total_duration = max(end for _, end, _ in segments)
        num_bins = math.ceil(total_duration / self.bin_size)
        bins = [[] for _ in range(num_bins)]

        # Assign segments to bins by midpoint
        for start, end, text in segments:
            midpoint = (start + end) / 2
            bin_idx = int(midpoint // self.bin_size)
            if bin_idx < num_bins:
                bins[bin_idx].append((start, end, text))

        # Deduplicate within each bin
        bin_descriptions = []
        for i, bin_segments in enumerate(bins):
            if not bin_segments:
                continue
            
            # Sort by start time
            bin_segments.sort(key=lambda x: x[0])
            seen_sentences = set()
            ordered_sentences = []
            
            for start, end, text in bin_segments:
                # Clean text
                cleaned = re.sub(r'<[\d:.]+>|<c>|</c>', '', text).replace('\n', ' ').strip()
                if not cleaned:
                    continue
                
                # Split into sentences
                sentences = re.split(r'(?<=[.!?])\s+', cleaned)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if sentence and sentence not in seen_sentences:
                        seen_sentences.add(sentence)
                        ordered_sentences.append(sentence)

            if ordered_sentences:
                bin_start = i * self.bin_size
                bin_end = min((i + 1) * self.bin_size, total_duration)
                combined_text = " ".join(ordered_sentences)
                bin_descriptions.append({"start": bin_start, "end": bin_end, "text": combined_text})

        logging.info(f"Created {len(bin_descriptions)} bins from transcript")
        return bin_descriptions

    def integrate_keyframes(self, bins, keyframe_descs):
        if not keyframe_descs:
            logging.warning("No keyframes provided. Returning bins with empty text if not set.")
            # Ensure all bins have a 'text' key, even if empty
            for b in bins:
                if "text" not in b:
                    b["text"] = ""
            return bins

        try:
            # Sort keyframes by timestamp
            keyframes = sorted(keyframe_descs.items(), key=lambda x: x[1]["timestamp"])
            max_keyframe_ts = max(info["timestamp"] for _, info in keyframes) if keyframes else 0
            max_bin_end = max(b["end"] for b in bins) if bins else 0
            total_duration = max(max_bin_end, max_keyframe_ts)
            num_bins = math.ceil(total_duration / self.bin_size)

            # Extend bins if necessary
            if num_bins > len(bins):
                for i in range(len(bins), num_bins):
                    bins.append({
                        "start": i * self.bin_size,
                        "end": min((i + 1) * self.bin_size, total_duration),
                        "text": ""
                    })

            # Integrate keyframes into bins
            for _, info in keyframes:
                ts = info["timestamp"]
                desc = f"The video has scenes showcasing: {info['desc']}"
                bin_idx = int(ts // self.bin_size)
                if bin_idx < len(bins):
                    bins[bin_idx]["text"] = bins[bin_idx]["text"] + " " + desc if bins[bin_idx]["text"] else desc

            # Ensure all bins have a 'text' key, even if empty, and don’t filter out empty bins
            for b in bins:
                if "text" not in b:
                    b["text"] = ""

            logging.info(f"Integrated keyframes, final bins: {len(bins)}")
            return bins

        except Exception as e:
            logging.error(f"Error integrating keyframes: {str(e)}")
            raise
    
    def build_graph(self, bins, video_id):
        nodes = [{"id": f"{video_id}_{i}", "chunk": {"text": bin["text"], "timestamp_coverage": [{"start": bin["start"], "end": bin["end"]}]}} 
                 for i, bin in enumerate(bins)]
        embeddings = [self.model.encode(node["chunk"]["text"]) for node in nodes]
        edges = []
        # Fix: Ensure no duplicate edges, strict i < j
        for i in range(len(embeddings)):
            emb_i = embeddings[i].reshape(1, -1)
            for j in range(i + 1, len(embeddings)):
                emb_j = embeddings[j].reshape(1, -1)
                sim = cosine_similarity(emb_i, emb_j)[0][0]
                if sim > 0.65:  # Threshold from your original
                    edges.append({"source": nodes[i]["id"], "target": nodes[j]["id"], "similarity": float(sim)})
        logging.info(f"Built graph: {len(nodes)} nodes, {len(edges)} edges")
        return nodes, edges

    def save_to_pinecone(self, nodes, edges, video_metadata, namespace="temp"):
        video_id = self.extract_video_id(video_metadata["uri"])  # Use full URL parsing
        if INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=INDEX_NAME,
                dimension=384,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        index = pc.Index(INDEX_NAME)

        edge_map = {node["id"]: [] for node in nodes}
        for edge in edges:
            edge_map[edge["source"]].append(edge)
            edge_map[edge["target"]].append(edge)

        to_upsert = []
        for node in nodes:
            embedding = self.model.encode(node["chunk"]["text"]).tolist()
            top_edges = sorted(edge_map[node["id"]], key=lambda x: x["similarity"], reverse=True)[:3]
            metadata = {
                "text": node["chunk"]["text"],
                "timestamp_coverage": json.dumps(node["chunk"]["timestamp_coverage"]),
                "top_edges": json.dumps(top_edges),
                "video_id": video_id  # Consistent with retrieve_chunks
            }
            to_upsert.append((node["id"], embedding, metadata))
        index.upsert(vectors=to_upsert, namespace=namespace)
        logging.info(f"Saved {len(nodes)} nodes to Pinecone namespace {namespace}")

    def extract_video_id(self, uri):  # Borrowed from Video_processor
        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", uri)
        return match.group(1) if match else "unknown"

def process_video_rag(transcript_file, keyframe_descs=None, video_metadata=None):
    chunker = VideoRAGChunker()
    bins = chunker.preprocess_transcript(transcript_file)
    bins_with_keyframes = chunker.integrate_keyframes(bins, keyframe_descs)
    nodes, edges = chunker.build_graph(bins_with_keyframes, video_metadata["uri"].split("=")[-1])
    chunker.save_to_pinecone(nodes, edges, video_metadata)
    return nodes, edges

class VideoRAGRetriever:
    def __init__(self):
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.index = pc.Index(INDEX_NAME)

    def retrieve_chunks(self, query, video_id, top_k=1, max_edge_depth=3, namespace="temp"):
        query_emb = self.embed_model.encode([query]).tolist()
        results = self.index.query(
            vector=query_emb,
            top_k=top_k,
            filter={"video_id": video_id},
            include_metadata=True,
            namespace=namespace
        )
        logging.info(f"Pinecone query results for '{query}': {results}")
        if not results["matches"]:
            return []
        top_match = results["matches"][0]
        chunks = [{
            "node_id": top_match["id"],
            "text": top_match["metadata"]["text"],
            "timestamp_coverage": json.loads(top_match["metadata"]["timestamp_coverage"]),
            "similarity": top_match["score"]
        }]
        top_edges = json.loads(top_match["metadata"]["top_edges"])
        seen = {top_match["id"]}
        for edge in top_edges[:max_edge_depth]:
            next_id = edge["target"] if edge["source"] == top_match["id"] else edge["source"]
            if next_id not in seen:
                fetch_response = self.index.fetch(ids=[next_id], namespace=namespace)
                next_node = fetch_response.vectors.get(next_id)
                if next_node:
                    chunks.append({
                        "node_id": next_id,
                        "text": next_node["metadata"]["text"],
                        "timestamp_coverage": json.loads(next_node["metadata"]["timestamp_coverage"]),
                        "similarity": edge["similarity"]
                    })
                    seen.add(next_id)
        return chunks

    def answer_query(self, query, video_id, top_k=1, max_edge_depth=3, namespace="temp"):
        """Generate answer using Gemini from retrieved chunks."""
        chunks = self.retrieve_chunks(query, video_id, top_k, max_edge_depth, namespace)
        if not chunks:
            return "Sorry, couldn’t find anything relevant for that query."

        # Prepare context for Gemini
        context = "\n".join([f"Chunk {c['node_id']} (t={c['timestamp_coverage'][0]['start']}-{c['timestamp_coverage'][0]['end']}s): {c['text']}" 
                             for c in chunks])
        prompt = (
            f"Answer the query '{query}' using only the following context from video. Be comprehensive and also the provided context may have sentence repetition ignore repeated sentences and answer correctly taking in context, the retrieved context:\n\n{context}\n\nAnswer:"
        )

        try:
            response = self.gemini_model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            logging.error(f"Gemini generation failed: {e}")
            answer = "Error generating answer, but here’s the raw data:\n" + context

        return {
            "answer": answer,
            "retrieved_chunks": chunks,
            "uri": f"https://youtube.com/watch?v={video_id}"
        }
    def cleanup_namespace(self, namespace="temp"):
        self.index.delete(delete_all=True, namespace=namespace)
        logging.info(f"Cleaned up namespace {namespace}")


