# ECHOTALES – GENERATIVE AI STORYTELLING PLATFORM

![EchoTales Banner](https://img.shields.io/badge/Generative%20AI-Storytelling%20Platform-8A2BE2?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=PyTorch&logoColor=white) ![Azure](https://img.shields.io/badge/azure-%230072C6.svg?style=for-the-badge&logo=microsoftazure&logoColor=white)

EchoTales is a cutting-edge, end-to-end Generative AI platform that transforms simple text prompts or audio recordings into fully narrated, illustrated, and animated story videos. It orchestrates multiple state-of-the-art AI models—from Large Language Models (LLMs) to Latent Diffusion Models and Vector Databases—into a seamless, unified creative pipeline.

---

## ✨ Core Capabilities

* **Multi-Modal Input Architecture:** Supply a creative prompt via text, or speak directly into a microphone using our highly optimized, locally-run `faster-whisper` transcription engine.
* **LLM-Driven Narrative Generation:** Powered by Groq (Llama-3), the platform instantly crafts structured 10-sentence stories and intelligently extracts the narrative's main characters for interaction.
* **Cinematic Visual Synthesis:** Leverages Stable Diffusion (`DreamShaper-8`) to autonomously generate vibrant, context-aware 512x512 illustrations for each sentence of the generated story.
* **Vector Memory & Retrieval:** Sentences are mapped into high-dimensional vector embeddings using `SentenceTransformer` and stored in a local **Qdrant** database, laying the foundation for semantic search and character memory.
* **Automated Audio-Visual Assembly:** Synthesizes Text-to-Speech (TTS) narration using `gTTS` and dynamically stitches the generated audio and imagery into a finalized `story_video.mp4` using `MoviePy`.
* **Interactive Character AI:** Engage in real-time, in-character conversations with the protagonists of your generated stories via a dedicated chat interface.
* **Enterprise Cloud Integration:** Automatically uploads the final video, audio assets, images, and JSON metadata manifests to **Azure Blob Storage**, automatically generating 7-day Secure Access Signature (SAS) URLs for secure distribution.

---

## 🛠️ Technical Architecture & Optimizations

EchoTales was engineered to run effectively on consumer-grade hardware (specifically optimized for 4GB VRAM NVIDIA GPUs and systems with strict RAM/Disk constraints).

### Hardware-Level Optimizations
1. **Aggressive VRAM Management (`diffusers`)**:
   * Implemented `enable_sequential_cpu_offload()` to stream Stable Diffusion layers to the GPU one at a time, keeping peak VRAM usage under 2GB.
   * Utilized `enable_attention_slicing()` to halve memory requirements during the attention step.
2. **RAM & Disk Conservation**:
   * Replaced the standard `openai-whisper` library with `faster-whisper` (CTranslate2) utilizing `int8` quantization and CPU thread limiting to drastically reduce memory fragmentation.
   * Explicitly mapped HuggingFace and PyTorch caches (`HF_HOME`, `TORCH_HOME`) to a secondary drive to prevent OS-level crashes due to Pagefile exhaustion.
3. **Resilient CUDA Backend**:
   * Built-in bypass for broken/corrupted Windows cuDNN drivers (`cudnnGetLibConfig Error 127`) by forcing native PyTorch ATen kernels when necessary.

---

## 🚀 Quick Start Guide

### Prerequisites
* Python 3.10+
* An active **Azure Storage Account** with Blob access.
* A **Groq API Key** for LLM access.
* **Qdrant** running locally (or via Docker) on port `6333`.

### Installation

1. **Clone the repository and activate your environment**
   ```powershell
   git clone https://github.com/yourusername/echotales.git
   cd echotales
   .\venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file in the root directory and add your credentials:
   ```env
   GROQ_API_KEY=your_groq_api_key
   AZURE_STORAGE_CONNECTION_STRING=your_azure_connection_string
   ```

4. **Launch the Platform**
   We have provided helper scripts for easy launching. Simply run:
   ```powershell
   .\run.ps1
   ```
   *Alternatively: `streamlit run main.py`*

5. Open your browser to `http://localhost:8501` to start generating stories!

---

## 📁 Repository Structure

```text
ECHOTALES/
├── main.py                # Core Streamlit application & AI orchestration logic
├── requirements.txt       # Python dependencies
├── run.ps1 / run.bat      # Helper scripts for launching the app
├── .env                   # API Keys and Environment Variables (Not committed)
├── .gitignore             # Version control exclusions
└── README.md              # Project documentation
```

*(Note: Cache folders like `venv/`, `generated_images/`, and `.cache/` are automatically generated upon first run and excluded from version control).*
