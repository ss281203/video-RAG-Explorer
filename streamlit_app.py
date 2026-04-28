import streamlit as st
import os
import shutil
import atexit
import yt_dlp
import subprocess
from Helper_function import Video_processor, VideoRAGChunker, VideoRAGRetriever
import logging
import time
import uuid

st.set_page_config(
    page_title="Video RAG Explorer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: visible;}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.stButton > button {
    background: linear-gradient(90deg, #14B8A6, #06B6D4);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 0.7rem 1.2rem;
    font-weight: 600;
}

.stTextInput > div > div > input,
.stTextArea textarea {
    border-radius: 10px;
}

.card {
    background-color: #111827;
    padding: 1.2rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)



# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------



# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Cache video processing
@st.cache_data
def cached_process_video(url, frame_skip, gemini_api_key):
    processor = Video_processor(output_dir="downloads", keyframe_dir="keyframes")
    return processor.process_video(url, frame_skip=frame_skip, gemini_api_key=gemini_api_key)

# Cleanup function
def cleanup():
    retriever = VideoRAGRetriever()
    retriever.cleanup_namespace("test_session")
    for dir in ["downloads", "keyframes", "temp_clips"]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
    logging.info("Cleaned up files, temp clips, and Pinecone namespace.")

atexit.register(cleanup)

# Get original video path by URI
def get_original_video_path(url, output_dir="downloads"):
    video_id = Video_processor().extract_video_id(url)
    # Assume original video is saved as {video_id}.mp4 or .mkv
    possible_extensions = [".mp4", ".mkv"]
    for ext in possible_extensions:
        original_path = os.path.join(output_dir, f"{video_id}{ext}")
        if os.path.exists(original_path):
            return original_path
    return None

# Streamlit app
st.title("Video RAG Explorer")

# Center input bar with form
st.markdown("<h3 style='text-align: center;'>Enter YouTube URL</h3>", unsafe_allow_html=True)
with st.form(key="url_form"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        url = st.text_input("", "https://www.youtube.com/watch?v=ftDsSB3F5kg", label_visibility="collapsed")
        submit_button = st.form_submit_button(label="Process Video")

# Advanced mode toggle and stop session button (sidebar)
with st.sidebar:
    advanced_mode = st.toggle("Advanced Mode", value=False)
    if advanced_mode:
        edge_depth = st.slider("Max Edge Depth", 1, 5, 3)
        bin_size = st.slider("Bin Size (seconds)", 5, 20, 10)
    
    if st.button("Stop Session"):
        cleanup()
        st.session_state.clear()
        st.success("Session stopped and files cleaned up.")
        st.rerun()

# Process video on submit
if submit_button:
    GEMINI_API_KEY = "your key"  
    video_id = Video_processor().extract_video_id(url)
    
    # Check if video is already downloaded
    original_video = get_original_video_path(url)
    if original_video:
        st.info(f"Video already downloaded: {original_video}. Using cached data.")
        result = cached_process_video(url, frame_skip=15, gemini_api_key=GEMINI_API_KEY)
    else:
        pikachu_gif = "https://64.media.tumblr.com/tumblr_m4azcqZ9uV1qge5e6o1_500.gif"
        with st.spinner("Processing..."):
            st.markdown(f"<div style='text-align: center;'><img src='{pikachu_gif}' width='200'></div>", unsafe_allow_html=True)
            result = cached_process_video(url, frame_skip=15, gemini_api_key=GEMINI_API_KEY)
    
    if result["status"] == "success":
        st.session_state["result"] = result
        st.session_state["video_id"] = video_id
        st.session_state["original_video_path"] = get_original_video_path(url)  # Store for later use
        
        # Build RAG graph
        chunker = VideoRAGChunker(output_dir="downloads", bin_size=bin_size if advanced_mode else 10.0)
        bins = chunker.preprocess_transcript(result["transcript_file"])
        bins_with_keyframes = chunker.integrate_keyframes(bins, result["keyframe_descriptions"])
        nodes, edges = chunker.build_graph(bins_with_keyframes, video_id)
        video_metadata = {
            "title": "Title of the Video",
            "description": "Description of the Video",
            "uri": url
        }
        chunker.save_to_pinecone(nodes, edges, video_metadata, namespace="test_session")
        
        # Show keyframes
        st.write("### Keyframes")
        if result["keyframe_descriptions"] is None or not result["keyframe_descriptions"]:
            st.info("No keyframes detected in this video.")
        else:
            cols = st.columns(4)
            for i, (path, info) in enumerate(result["keyframe_descriptions"].items()):
                with cols[i % 4]:
                    st.image(path, caption=f"{os.path.basename(path)} at {info['timestamp']:.1f}s")
        
        st.session_state["processed"] = True
        st.success("Video processed! Ready for your query.")
    else:
        st.error(f"Failed: {result['message']}")

# Query input after processing
if "processed" in st.session_state and st.session_state["processed"]:
    st.markdown("<h3 style='text-align: center;'>Ask About the Video</h3>", unsafe_allow_html=True)
    with st.form(key="query_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            query = st.text_input("", placeholder="e.g., Explain director responsibilities", label_visibility="collapsed")
            query_button = st.form_submit_button(label="Submit Query")
    
    if query_button and query:
        retriever = VideoRAGRetriever()
        max_edge_depth = edge_depth if advanced_mode else 3
        try:
            result = retriever.answer_query(query, st.session_state["video_id"], top_k=1, max_edge_depth=max_edge_depth, namespace="test_session")
            
            # Robust handling of query result
            if isinstance(result, dict) and "answer" in result:
                st.success(result["answer"])
            else:
                st.warning("Query returned an unexpected result. No answer available.")
                logging.warning(f"Unexpected result from answer_query: {result}")
            
            # Right-side blocks with short clips
            col1, col2 = st.columns([3, 1])
            with col2:
                if isinstance(result, dict) and "retrieved_chunks" in result:
                    with st.expander("Timestamps"):
                        for chunk in result["retrieved_chunks"]:
                            ts = chunk["timestamp_coverage"][0]
                            st.write(f"Node {chunk['node_id']}: {ts['start']}–{ts['end']}s")
                    with st.expander("Chunks Used"):
                        for chunk in result["retrieved_chunks"]:
                            st.write(f"Node {chunk['node_id']}: {chunk['text'][:50]}... (Sim: {chunk['similarity']:.2f})")
                else:
                    st.info("No chunk data available for display.")
            
            # Temporary clips
            with col1:
                st.write("### Relevant Clips")
                video_path = st.session_state.get("original_video_path")
                if video_path and os.path.exists(video_path) and isinstance(result, dict) and "retrieved_chunks" in result:
                    # Generate a unique session ID for this query
                    query_session_id = str(uuid.uuid4())[:8]
                    
                    # Clear previous temp clips
                    temp_dir = "temp_clips"
                    if os.path.exists(temp_dir):
                        try:
                            shutil.rmtree(temp_dir)
                            time.sleep(0.5)  # Give OS time to release file handles
                        except Exception as e:
                            logging.warning(f"Could not clear temp directory: {e}")
                            # Use a new directory instead
                            temp_dir = f"temp_clips_{query_session_id}"
                    
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Store clips in session state to track what's been shown
                    if "shown_clips" not in st.session_state:
                        st.session_state["shown_clips"] = []
                    
                    # Clear previous clips list for new query
                    st.session_state["shown_clips"] = []
                    
                    for i, chunk in enumerate(result["retrieved_chunks"]):
                        ts = chunk["timestamp_coverage"][0]
                        start, end = ts["start"], ts["end"]
                        
                        # Create unique clip name based on query and timestamps
                        clip_name = f"{query_session_id}_{i}_{start:.1f}-{end:.1f}"
                        clip_path = os.path.join(temp_dir, f"{clip_name}.mp4")
                        
                        try:
                            subprocess.run([
                                "ffmpeg", "-i", video_path, "-ss", str(start), "-t", str(end - start),
                                "-c:v", "copy", "-c:a", "copy", clip_path, "-y"
                            ], check=True, capture_output=True, text=True)
                            
                            # Only show clip if it was successfully created
                            if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
                                st.session_state["shown_clips"].append({
                                    "path": clip_path,
                                    "start": start,
                                    "end": end,
                                    "node_id": chunk["node_id"]
                                })
                                st.write(f"**Clip {i+1}:** Node {chunk['node_id']} ({start:.1f}s - {end:.1f}s)")
                                st.video(clip_path)
                            else:
                                st.warning(f"Clip extraction failed for {start:.1f}s-{end:.1f}s: File not created")
                        except subprocess.CalledProcessError as e:
                            st.warning(f"Failed to extract clip for {start:.1f}s-{end:.1f}s: {e.stderr}")
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logging.error(f"Query processing failed: {str(e)}")


# --- Chatbot section (integrated but same heading text) ---

if "processed" in st.session_state and st.session_state["processed"]:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I can answer questions about your uploaded video."}
        ]

    st.subheader("Video Chat Assistant")

    # Show chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    prompt = st.chat_input("Ask something about the video...")

    if prompt:
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Use your existing retriever for answers
        try:
            retriever = VideoRAGRetriever()
            max_edge_depth = edge_depth if "edge_depth" in locals() else 3

            result = retriever.answer_query(
                prompt,
                st.session_state["video_id"],
                top_k=1,
                max_edge_depth=max_edge_depth,
                namespace="test_session"
            )

            if isinstance(result, dict) and "answer" in result:
                answer = result["answer"]
            else:
                answer = "I could not retrieve a confident answer for that question."

        except Exception as e:
            answer = f"Error while answering: {e}"
            logging.error(f"Chatbot error: {e}")

        # Assistant message
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Process a video first, then the Video Chat Assistant will be available.")

# Cleanup reminder
st.sidebar.write("Note: Files and Pinecone data will be cleaned up when you exit or stop the session.")
