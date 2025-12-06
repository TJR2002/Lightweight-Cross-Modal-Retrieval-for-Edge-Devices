📘 Lightweight Cross-Modal Retrieval for Edge Devices
⚡ A Privacy-First Multimedia Search System Using Tri-Compression

This project implements a fully offline, lightweight cross-modal retrieval system capable of searching images, videos, and audio using text queries.
The model is optimized through a Tri-Compression Pipeline consisting of Knowledge Distillation, Structured Pruning, and INT8 Quantization, achieving 7.6× model compression and 10× faster search while running efficiently on consumer edge hardware.

🚀 Features
🔍 Cross-Modal Retrieval

Search images, videos, and audio using natural language prompts.

🔐 Fully Offline, Privacy-First

No cloud processing. All inference, indexing, and retrieval happens offline.

⚙️ Tri-Compression Pipeline

Knowledge Distillation → 151M → 20M parameters

Structured Pruning → removes 35% redundant weights

INT8 Quantization → 4× smaller, faster execution

⚡ High Performance

10× faster search

4× faster embedding generation

Works on dual-core CPUs with 1.5GB RAM

🗄 Efficient Storage

FAISS-based vector index + SQLite metadata database.

🏗 Architecture Overview
            ┌─────────────────────┐
            │     User Query      │
            └─────────┬───────────┘
                      │ Text Encoder
            ┌─────────▼───────────┐
            │   Compressed Model   │
            │ (Tri-Compression CLIP) 
            └─────────┬───────────┘
                      │ 256-D Embedding
            ┌─────────▼───────────┐
            │     FAISS Index      │
            └─────────┬───────────┘
                      │ Nearest Neighbor Search
            ┌─────────▼───────────┐
            │   SQLite Metadata    │
            └─────────┬───────────┘
                      ▼
                Search Results

📁 Project Structure
MIR_compression/
│
├── cmrs_tricompression.py        # Main model pipeline (distill → prune → quantize)
├── train_tricompression.py       # Training script
├── compare_models.py             # Compare model sizes & accuracy
├── fix_database.py               # Utilities for index/db
├── finish_stage3.py              # Final quantization tools
│
├── config.ini                    # Configurations
├── requirements.txt              # Dependencies
│
├── benchmark_results.json        # Performance results
│
├── Daigram.drawio                # Architecture diagram
│
└── logs/                         # Training & evaluation logs
├── benchmark_plots/              # Visual performance analysis

📦 Installation
1️⃣ Clone the repository
git clone https://github.com/TJR2002/Lightweight-Cross-Modal-Retrieval-for-Edge-Devices.git
cd Lightweight-Cross-Modal-Retrieval-for-Edge-Devices

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ How to Use
1. Run the quick demo
python quick_start.py

2. Train the Tri-Compression Pipeline
python train_tricompression.py --data-dir <path_to_media_folder>

3. Compare different compressed models
python compare_models.py

📊 Benchmark Results
Metric	Baseline CLIP	Final Model	Improvement
Encoding Time	72.57 ms	16.79 ms	4.3×
Search Time	48.17 ms	4.80 ms	10×
Model Size	577 MB	75.7 MB	7.6×
Accuracy	High	High	Minimal Loss
🏆 Key Achievements

✔ 7.6× compression (577 MB → 75.7 MB)
✔ 10× faster multimedia search
✔ Works offline on edge devices
✔ Supports text → image/video/audio retrieval
✔ Scaled successfully to 22,000+ files

🔮 Future Work

Android/iOS deployment using NNAPI / CoreML

Quantization-Aware Training (QAT) for recovering accuracy

FAISS IVF indexing for million-scale datasets

Real-time video frame embedding

📜 License

MIT License (recommended — I can generate it if you want)

🙌 Authors

Thejas Rao

Sk Abdur Razzaq

Guide: Dr. Janani T (NITK Surathkal)
