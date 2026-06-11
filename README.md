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

## 🎬 Sample Output

### 📁 Included in This Repository (`samples/`)

| File | Story Prompt |
|---|---|
| [`Demo 1`](./samples/echotales%20demo1(a%20mother%20and%20a%20son%20on%20their%20vacation).mp4) | *"A mother and a son on their vacation"* |
| [`Demo 2`](./samples/echotales%20demo2(Tell%20me%20an%20emotional%20dark%20fantasy%20story%20set%20in%20a%20ruined%20kingdom%20where%20dragons%20once%20protected%20humanity%20but%20suddenly%20disappeared%20100%20years%20ago.).mp4) | *"Emotional dark fantasy — ruined kingdom where dragons disappeared 100 years ago"* |
| [`Demo 3`](./samples/echotales%20demo3(Captain%20Arjun%20and%20his%20AI%20companion%20NOVA%20must%20sail%20through%20a%20storm%20of%20magnetic%20lightning).mp4) | *"Captain Arjun and his AI companion NOVA sailing through magnetic lightning"* |
| [`narration.mp3`](./samples/narration.mp3) | TTS audio narration sample |

---

### 🌐 Full Demo on Google Drive

> 🎥 **[View Full Demo → Google Drive](https://drive.google.com/drive/folders/1_QTTj2v9ESarEr-uqim0nFR_alm4zfse?usp=drive_link)**

The Google Drive folder contains the **complete end-to-end demo**, including:

| Demo | Description |
|---|---|
| 🖥️ **Streamlit UI Demo** | Full walkthrough of story generation, image synthesis, and character chat |
| ☁️ **Azure Integration Demo** | Uploading assets to Azure Blob Storage & generating SAS URLs |
| 🖼️ **Generated Scene Images** | All 10 AI-generated illustrations from a sample story run |
| 🎬 **Final Story Video** | Full HD narrated video assembled by MoviePy |
| 🔊 **Audio Narration** | gTTS-generated audio from the story |

---

## 📁 Repository Structure

```text
ECHOTALES/
├── main.py                # Core Streamlit application & AI orchestration logic
├── requirements.txt       # Python dependencies
├── run.ps1 / run.bat      # Helper scripts for launching the app
├── .env                   # API Keys and Environment Variables (Not committed)
├── .gitignore             # Version control exclusions
├── samples/               # Sample outputs from a real EchoTales run
│   ├── screenshot.png     # App UI screenshot
│   ├── story_video.mp4    # Generated story video
│   └── *.png              # Sample scene illustrations
└── README.md              # Project documentation
```

*(Note: Cache folders like `venv/`, `generated_images/`, and `.cache/` are automatically generated upon first run and excluded from version control).*
