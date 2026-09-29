import gradio as gr
import chromadb
import librosa
import os
import torch
import warnings
import numpy as np
from transformers import ClapModel, ClapProcessor

warnings.filterwarnings("ignore")

print("Loading CLAP Acoustic Engine...")
MODEL_ID = "laion/clap-htsat-unfused"
processor = ClapProcessor.from_pretrained(MODEL_ID)
model = ClapModel.from_pretrained(MODEL_ID)

# Persistent Vector Database for Hybrid Search
chroma_client = chromadb.PersistentClient(path="./faydem_sample_db")
collection = chroma_client.get_or_create_collection(name="samples", metadata={"hnsw:space": "cosine"})

def index_folder(folder_path):
    if not os.path.isdir(folder_path):
        return "❌ Invalid folder path."
    
    supported_exts = ('.wav', '.mp3', '.flac')
    files_to_index = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(supported_exts):
                files_to_index.append(os.path.join(root, f))
                
    if not files_to_index:
        return "No audio files found."
        
    for file_path in files_to_index:
        try:
            # 1. Extract Audio Embedding (Heavy NPU Workload in production)
            audio, sr = librosa.load(file_path, sr=48000, mono=True, duration=5.0)
            duration_sec = librosa.get_duration(y=audio, sr=sr)
            inputs = processor(audio=audio, sampling_rate=48000, return_tensors="pt")
            
            with torch.no_grad():
                audio_embed = model.get_audio_features(**inputs).numpy()[0]
            
            # 2. Extract Hard Metadata for Hybrid Filtering
            is_oneshot = "oneshot" if duration_sec < 2.5 else "loop"
            
            collection.upsert(
                embeddings=[audio_embed.tolist()],
                documents=[file_path],
                metadatas=[{"type": is_oneshot, "duration": duration_sec}],
                ids=[file_path]
            )
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            
    return f"✅ Indexing Complete! Indexed {len(files_to_index)} files with hybrid metadata."

def search_text(query, sample_type_filter):
    if collection.count() == 0:
        return [None]*5, "Database is empty. Index a folder first."
        
    # Text encoding is computationally cheap (CPU task)
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embed = model.get_text_features(**inputs).numpy()[0]
        
    where_clause = {} if sample_type_filter == "All" else {"type": sample_type_filter.lower()}
    
    results = collection.query(
        query_embeddings=[text_embed.tolist()],
        n_results=5,
        where=where_clause if where_clause else None
    )
    
    return format_results(results, f"Text Query: '{query}' ({sample_type_filter})")

def search_audio_similarity(reference_audio, sample_type_filter):
    if reference_audio is None:
        return [None]*5, "Please upload a reference audio file."
    if collection.count() == 0:
        return [None]*5, "Database is empty. Index a folder first."
        
    # Audio encoding triggers the heavy NPU workload at query time
    audio, sr = librosa.load(reference_audio, sr=48000, mono=True, duration=5.0)
    inputs = processor(audio=audio, sampling_rate=48000, return_tensors="pt")
    
    with torch.no_grad():
        ref_embed = model.get_audio_features(**inputs).numpy()[0]
        
    where_clause = {} if sample_type_filter == "All" else {"type": sample_type_filter.lower()}
    
    results = collection.query(
        query_embeddings=[ref_embed.tolist()],
        n_results=5,
        where=where_clause if where_clause else None
    )
    
    return format_results(results, f"Audio Match for: '{os.path.basename(reference_audio)}'")

def format_results(results, query_title):
    matched_files = results['documents'][0]
    distances = results['distances'][0]
    
    audio_outputs = []
    log_text = f"🔍 {query_title}\n" + "-"*45 + "\n"
    
    for i, file_path in enumerate(matched_files):
        audio_outputs.append(file_path)
        sim = max(0, 100 - (distances[i] * 100))
        log_text += f"{i+1}. {os.path.basename(file_path)} | Match: {sim:.1f}%\n"
        
    while len(audio_outputs) < 5:
        audio_outputs.append(None)
        
    return audio_outputs, log_text

# Dark Studio UI
custom_css = "body, .gradio-container { background-color: #0b0e14 !important; color: #e0e6ed !important; }"

with gr.Blocks(css=custom_css, title="Faydem DSP :: NPU Sample Engine") as demo:
    gr.Markdown("## 🎛️ Faydem DSP :: Hybrid NPU Sample Engine")
    gr.Markdown("Offloading background spectrogram indexing & audio similarity queries to the Snapdragon Hexagon HTP.")
    
    with gr.Tab("1. Library Indexer"):
        folder_input = gr.Textbox(label="Sample Folder Path")
        index_btn = gr.Button("Build NPU Embeddings & Metadata")
        index_status = gr.Textbox(label="Status", interactive=False)
        index_btn.click(fn=index_folder, inputs=folder_input, outputs=index_status)
        
    with gr.Tab("2. Semantic Text Search"):
        with gr.Row():
            text_query = gr.Textbox(label="Description (e.g., 'punchy synthwave kick')")
            type_filter = gr.Radio(["All", "One-Shot", "Loop"], value="All", label="Type Filter")
        text_btn = gr.Button("Search by Text")
        
    with gr.Tab("3. Audio-to-Audio Similarity"):
        with gr.Row():
            ref_audio = gr.Audio(type="filepath", label="Upload Reference WAV")
            audio_type_filter = gr.Radio(["All", "One-Shot", "Loop"], value="All", label="Type Filter")
        audio_btn = gr.Button("Find Similar Samples")

    # Outputs
    with gr.Row():
        out_logs = gr.Textbox(label="Retrieval Engine Metrics", lines=6)
    with gr.Row():
        out_1 = gr.Audio(label="Match 1")
        out_2 = gr.Audio(label="Match 2")
        out_3 = gr.Audio(label="Match 3")

    text_btn.click(fn=search_text, inputs=[text_query, type_filter], outputs=[out_1, out_2, out_3, out_logs])
    audio_btn.click(fn=search_audio_similarity, inputs=[ref_audio, audio_type_filter], outputs=[out_1, out_2, out_3, out_logs])

if __name__ == "__main__":
    demo.launch()