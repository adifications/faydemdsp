# Faydem DSP : Hybrid NPU Semantic Sample Indexer
**Snapdragon® AI Lab Build & Present Challenge Submission**  
*Made by: A Aditya Nair*

---

## The Studio Engineering Problem
Modern music producers manage massive local sample libraries often exceeding 200,000 individual audio files. Traditional folder browsing relying on file names (`kick_03_dirty.wav`) breaks creative momentum, while standard local AI indexing tools place heavy computational loads on the host CPU.

During active Digital Audio Workstation (DAW) sessions (e.g., FL Studio, Ableton Live, Cubase), background audio indexing or audio-to-audio similarity searches compete directly with active VST plugins. This causes host CPU thread starvation, audio buffer underruns (XRUNs), and audio dropouts. Furthermore, base text-to-audio models (like LAION CLAP) struggle with percussion accuracy, often returning full loops when a producer specifically needs a one-shot sample.

---

## The Qualcomm Hardware Solution
**Faydem DSP** operates as an open-source, hybrid semantic sample manager engineered specifically as a reference implementation for Windows-on-Snapdragon Copilot+ PCs. It allows producers to search the library they already own by acoustic sound and metadata—while their DAW is actively playing—without impacting host CPU performance.

### Key Architectural Pillars
1. **NPU-Accelerated Spectrogram Ingestion:** The computationally expensive Audio Encoder from the LAION CLAP model was isolated, traced via PyTorch (`torch.jit.trace`), and compiled through Qualcomm AI Hub for Hexagon HTP architecture. It executes on-device via the ONNX Runtime `QNNExecutionProvider` / QAIRT framework.
2. **Audio-to-Audio Similarity Querying:** When a producer inputs a reference `.wav` file to find similar samples, the query spectrogram is encoded directly on the NPU in real time, bypassing host CPU audio threads.
3. **Hybrid Retrieval Engine:** Lightweight text prompt encoding and ChromaDB vector retrieval run locally on the CPU. To fix pure semantic accuracy limitations on percussive audio, the indexer extracts hard acoustic metadata (duration and transient classification for One-Shots vs. Loops) and pairs it with vector embeddings.

---

## Snapdragon® X Elite CRD Benchmarks

Physical device profiling on the Qualcomm AI Hub Workbench confirms the following performance metrics:

| Metric | Measured Value / Specification |
| :--- | :--- |
| **Target Hardware** | Snapdragon® X Elite CRD (SC8380XP, Windows 11) |
| **Execution Provider** | ONNX Runtime 1.27.1 / QAIRT |
| **Minimum Inference Latency** | **105.6 ms** (per 5-second 48kHz audio spectrogram batch) |
| **Estimated Peak Memory** | **58 MB** RAM footprint |
| **Hardware Node Mapping** | **538 Compute Units** natively mapped to Hexagon HTP |
| **Qualcomm AI Hub Compile Job** | `jp0mmoveg` |
| **Qualcomm AI Hub Profile Job** | `jgolld8dg` |

> **DAW Headroom Protection:** Offloading dense tensor graph execution to the Hexagon NPU keeps host CPU utilization during search queries under 3%. With a peak RAM footprint of just 58 MB, Faydem DSP eliminates audio driver glitches while keeping system memory free for heavy synthesizers and sample libraries.

---

## Repository Architecture

* `app.py` — The primary desktop interface (Gradio) implementing Hybrid Metadata search, Text-to-Audio retrieval, and Audio-to-Audio similarity queries.
* `compile_clap.py` — PyTorch tracing and Qualcomm AI Hub submission pipeline for the CLAP Audio Encoder.
* `profile_clap.py` — Hardware benchmark profiling script for capturing NPU latency and memory footprint metrics on physical Snapdragon hardware.
* `requirements.txt` — Python library dependencies (`gradio`, `transformers`, `torch`, `chromadb`, `qai-hub`, `librosa`).

---

## Local Installation & Quickstart

### Prerequisites
* Python 3.10+
* Virtual Environment (`venv` or `conda`)

### Setup Instructions
1. Clone the repository:
   ```bash
   git clone [https://github.com/adifications/faydemdsp.git](https://github.com/adifications/faydemdsp.git)
   cd faydemdsp
