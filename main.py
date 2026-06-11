import os
import sys
import json
import tempfile
import time
import re
import logging
from uuid import uuid4
from datetime import datetime, timedelta, timezone

# ── Load .env before anything reads environment variables ─────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on system environment variables


# ══════════════════════════════════════════════════════════════════════════════
# REDIRECT MODEL CACHES TO E: DRIVE
# C: drive is nearly full (~10 MB free). All HuggingFace, Torch, and
# faster-whisper model downloads must go to E:\story\.cache instead.
# This MUST be set before any model library is imported.
# ══════════════════════════════════════════════════════════════════════════════
_CACHE_ROOT = r"E:\story\.cache"
os.makedirs(_CACHE_ROOT, exist_ok=True)

os.environ["HF_HOME"]                  = _CACHE_ROOT
os.environ["HUGGINGFACE_HUB_CACHE"]    = _CACHE_ROOT
os.environ["TRANSFORMERS_CACHE"]       = _CACHE_ROOT
os.environ["TORCH_HOME"]               = _CACHE_ROOT
os.environ["XDG_CACHE_HOME"]           = _CACHE_ROOT
os.environ["FASTER_WHISPER_CACHE_DIR"] = _CACHE_ROOT

# Limit OpenBLAS / OMP threads to avoid memory fragmentation on low-RAM systems
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS",      "1")
os.environ.setdefault("MKL_NUM_THREADS",      "1")


# ── Whisper: prefer faster-whisper (CTranslate2 — 4× less RAM) over openai-whisper ──
_FASTER_WHISPER_AVAILABLE = False
_WHISPER_AVAILABLE = False

try:
    from faster_whisper import WhisperModel as _FasterWhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except Exception as e:
    import traceback
    print("FASTER WHISPER IMPORT ERROR:", type(e), e)
    traceback.print_exc()

import streamlit as st
import torch

# Fix for "Could not load symbol cudnnGetLibConfig. Error code 127"
# This bypasses the broken cuDNN DLLs and uses native PyTorch CUDA kernels instead.
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False

from PIL import Image
from azure.storage.blob import (
    BlobServiceClient,
    generate_blob_sas,
    BlobSasPermissions,
)
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from gtts import gTTS
from moviepy.editor import (
    ImageClip, AudioFileClip,
    concatenate_videoclips, concatenate_audioclips,
)

# ── Diffusers import with graceful fallback ───────────────────────────────────
try:
    from diffusers import AutoPipelineForText2Image, DEISMultistepScheduler
    _USE_AUTO_PIPELINE = True
except ImportError:
    from diffusers import StableDiffusionPipeline, DEISMultistepScheduler  # type: ignore
    _USE_AUTO_PIPELINE = False

# ==========================
# CONFIG
# ==========================

st.set_page_config(page_title="AI Story Generator", page_icon="📖", layout="wide")

GROQ_API_KEY       = os.getenv("GROQ_API_KEY", "")
AZURE_ACCOUNT_NAME = "moulistorage2026"
AZURE_ACCOUNT_KEY  = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
CONNECTION_STRING  = (
    f"DefaultEndpointsProtocol=https;AccountName={AZURE_ACCOUNT_NAME};"
    f"AccountKey={AZURE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"
)
CONTAINER_NAME   = "story-assets"
IMAGE_DIR        = os.path.join(os.getcwd(), "generated_images")
SAS_EXPIRY_HOURS = 168   # 7 days

# Root logger — won't be reconfigured per-run; handlers are added per-run instead
_root_logger = logging.getLogger("story_app")
_root_logger.setLevel(logging.INFO)
if not _root_logger.handlers:
    _root_logger.addHandler(logging.StreamHandler(sys.stdout))

def _get_run_logger(log_path: str) -> logging.Logger:
    """Return a run-scoped logger that writes to log_path."""
    logger = logging.getLogger(f"story_run_{os.path.basename(log_path)}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(fh)
    return logger


# ==========================
# AZURE BLOB  (SAS URLs)
# ==========================

@st.cache_resource(show_spinner=False)
def get_container_client():
    service   = BlobServiceClient.from_connection_string(CONNECTION_STRING)
    container = service.get_container_client(CONTAINER_NAME)
    if not container.exists():
        container.create_container()
    # Try to set public access — silently ignore if the account policy blocks it
    try:
        container.set_container_access_policy(public_access="blob")
    except Exception as ex:
        _root_logger.warning(f"Unable to set public container access: {ex}")
    return container


def _make_sas_url(blob_name: str) -> str:
    """Return a 7-day SAS URL for any blob — no public access required."""
    sas_token = generate_blob_sas(
        account_name=AZURE_ACCOUNT_NAME,
        container_name=CONTAINER_NAME,
        blob_name=blob_name,
        account_key=AZURE_ACCOUNT_KEY,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(timezone.utc) + timedelta(hours=SAS_EXPIRY_HOURS),
    )
    return (
        f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net"
        f"/{CONTAINER_NAME}/{blob_name}?{sas_token}"
    )


def upload_to_blob(local_path: str, blob_name: str) -> str:
    """Upload file → Azure Blob and return a SAS URL (fallback: local path)."""
    try:
        container = get_container_client()
        with open(local_path, "rb") as data:
            container.get_blob_client(blob_name).upload_blob(data, overwrite=True)
        return _make_sas_url(blob_name)
    except Exception as ex:
        _root_logger.error(f"Blob upload failed for {blob_name}: {ex}")
        return local_path  # graceful fallback


# ==========================
# LLM
# ==========================

def get_llm(primary: bool = True):
    model = "llama-3.3-70b-versatile" if primary else "llama-3.1-8b-instant"
    return ChatGroq(api_key=GROQ_API_KEY, model=model, temperature=0.7)


# ==========================
# EMBEDDINGS & VECTOR DB
# ==========================

@st.cache_resource(show_spinner=False)
def get_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def get_qdrant_client():
    return QdrantClient(host="localhost", port=6333)


# ==========================
# IMAGE PIPELINE
# ==========================
# Why DreamShaper-8 fails:
#   The model is 3.44 GB in fp16.  On machines without a GPU (or with a small
#   VRAM budget) the fp16 allocation fails with "memory allocation … bytes
#   failed".  We fix this by:
#     1. Detecting CUDA availability and choosing dtype accordingly.
#     2. Falling back progressively: DreamShaper-8 → SD v1-5 → CPU fp32.
#     3. Skipping enable_model_cpu_offload when no CUDA is present (it calls
#        .to("cuda") internally and crashes on CPU-only machines).


@st.cache_resource(show_spinner=False)
def get_image_pipe():
    """Load the best available SD pipeline for the current hardware.

    DreamShaper-8 space requirements
    ---------------------------------
    • Model weights (fp16): ~3.44 GB download / ~2.1 GB VRAM peak when
      enable_sequential_cpu_offload() is used (layers streamed one-by-one).
    • With enable_model_cpu_offload() the whole UNet must fit in VRAM at
      once (≈ 3.1 GB), which overflows a 4 GB card.
    • With enable_sequential_cpu_offload() peak VRAM stays under 2 GB,
      so it runs on your RTX 3050 4 GB.
    • Hard disk: ~7 GB total (fp32 cache + safetensors).

    Fallback chain: DreamShaper-8 → SD v1-5 → error.
    """
    has_cuda = torch.cuda.is_available()
    dtype    = torch.float16 if has_cuda else torch.float32
    device   = "cuda" if has_cuda else "cpu"

    # Free any leftover VRAM before loading
    if has_cuda:
        torch.cuda.empty_cache()

    pipe = None

    # ── 1. Try DreamShaper-8 ──────────────────────────────────────────────
    if _USE_AUTO_PIPELINE:
        try:
            pipe = AutoPipelineForText2Image.from_pretrained(
                "Lykon/dreamshaper-8",
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                # low_cpu_mem_usage keeps peak RAM manageable during load
                low_cpu_mem_usage=True,
            )
            _root_logger.info("Loaded DreamShaper-8.")
        except Exception as e:
            _root_logger.warning(f"DreamShaper-8 failed: {e}. Falling back to SD v1-5…")
            pipe = None
            if has_cuda:
                torch.cuda.empty_cache()

    # ── 2. Fallback: SD v1-5 (≈ 2.2 GB download, fits easily) ───────────
    if pipe is None:
        try:
            ctor = AutoPipelineForText2Image if _USE_AUTO_PIPELINE else StableDiffusionPipeline  # type: ignore[name-defined]
            pipe = ctor.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
                low_cpu_mem_usage=True,
            )
            _root_logger.info("Loaded SD v1-5 as fallback.")
        except Exception as e:
            _root_logger.error(f"SD v1-5 also failed: {e}")
            raise RuntimeError(
                "Could not load any image generation model. "
                "Check GPU VRAM / disk space."
            ) from e

    # ── Memory optimisations ──────────────────────────────────────────────
    pipe.scheduler = DEISMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()   # halves attention memory
    pipe.enable_vae_slicing()         # decode images one-at-a-time

    if has_cuda:
        # enable_sequential_cpu_offload: streams individual model layers to GPU
        # one at a time → peak VRAM ~1.5–2 GB (vs ~3.1 GB for model_cpu_offload)
        # This is the key fix for 4 GB cards like the RTX 3050.
        pipe.enable_sequential_cpu_offload()
    else:
        pipe = pipe.to(device)

    return pipe



# ==========================
# WHISPER
# ==========================

class _FasterWrapper:
    """Wrap faster-whisper so it exposes the same API as openai-whisper."""
    def __init__(self, model):
        self._model = model

    def transcribe(self, audio_path: str) -> dict:
        segments, _ = self._model.transcribe(audio_path)
        text = " ".join(seg.text.strip() for seg in segments)
        return {"text": text}


@st.cache_resource(show_spinner=False)
def load_whisper():
    """Load Whisper model safely. Returns None if unavailable or OOM."""
    if _FASTER_WHISPER_AVAILABLE:
        try:
            model = _FasterWhisperModel(
                "small",
                device="cpu",
                compute_type="int8",
                cpu_threads=1,
                num_workers=1,
            )
            return _FasterWrapper(model)
        except Exception as fw_err:
            _root_logger.warning(f"faster-whisper load failed: {fw_err}. Audio disabled.")
            return None
    elif _WHISPER_AVAILABLE:
        try:
            return _whisper_lib.load_model("tiny")
        except Exception as w_err:
            _root_logger.warning(f"openai-whisper load failed: {w_err}. Audio disabled.")
            return None
    return None   # no whisper installed


# NOTE: whisper_model is loaded lazily inside the audio upload handler,
# NOT at module startup — avoids MemoryError when RAM is low.
_whisper_model_cache = None


def get_whisper_model():
    """Return cached whisper model, loading on first call."""
    global _whisper_model_cache
    if _whisper_model_cache is None:
        _whisper_model_cache = load_whisper()
    return _whisper_model_cache


# ==========================
# TTS
# ==========================

def generate_tts(sentence: str, out_path: str, retries: int = 3):
    for attempt in range(retries):
        try:
            gTTS(text=sentence, lang="en", slow=False).save(out_path)
            return out_path
        except Exception:
            if attempt < retries - 1:
                time.sleep(1.5)
    return None


def build_narration(tts_paths: list, narration_path: str):
    valid = [p for p in tts_paths if p and os.path.exists(p)]
    if not valid:
        return None
    clips = [AudioFileClip(p) for p in valid]
    full  = concatenate_audioclips(clips)
    full.write_audiofile(narration_path, logger=None)
    for c in clips:
        c.close()
    full.close()
    return narration_path


# ==========================
# VIDEO ASSEMBLY
# ==========================

def build_story_video(sentences, image_path_map, tts_paths, video_path):
    clips          = []
    last_valid_img = None

    for i, _sent in enumerate(sentences):
        audio_path = tts_paths[i] if i < len(tts_paths) else None
        if not audio_path or not os.path.exists(audio_path):
            continue

        audio_clip = AudioFileClip(audio_path)
        duration   = audio_clip.duration

        img_path = image_path_map.get(i)
        if img_path and os.path.exists(img_path):
            last_valid_img = img_path
        elif last_valid_img:
            img_path = last_valid_img
        else:
            black    = Image.new("RGB", (512, 512), color=(20, 20, 30))
            img_path = os.path.join(IMAGE_DIR, f"_black_{i}.png")
            black.save(img_path)

        img_clip = (
            ImageClip(img_path)
            .set_duration(duration)
            .set_audio(audio_clip)
        )
        clips.append(img_clip)

    if not clips:
        return None

    video = concatenate_videoclips(clips, method="compose")
    video.write_videofile(video_path, fps=1, codec="libx264", audio_codec="aac", logger=None)
    video.close()
    for c in clips:
        c.close()
    return video_path


# ==========================
# LLM RESPONSE PARSER
# ==========================

def _parse_llm_json(raw: str) -> dict:
    """Extract and parse the first valid JSON object from an LLM response."""
    text = raw.strip()

    # Strip markdown code fences
    if "```" in text:
        try:
            text = text.split("```json")[1].split("```")[0].strip()
        except IndexError:
            text = text.replace("```", "").strip()

    # If still not a bare JSON object, extract via regex
    if not (text.startswith("{") and text.endswith("}")):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)

    return json.loads(text)


# ==========================
# SESSION STATE
# ==========================

_defaults: dict = {
    "story":            None,
    "characters":       [],
    "chat_history":     {},
    "raw_response":     None,
    "image_sas_urls":   [],
    "narration_sas_url": None,
    "video_sas_url":    None,
    "json_sas_url":     None,
    "log_sas_url":      None,
    "story_id":         str(uuid4()),
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ==========================
# UI — HEADER & INPUT
# ==========================

st.title("📖 AI Story Generator")

is_whisper_installed = _WHISPER_AVAILABLE or _FASTER_WHISPER_AVAILABLE

if not is_whisper_installed:
    st.warning(
        "⚠️ No Whisper implementation found. Audio transcription is disabled. "
        "Install `openai-whisper` or `faster-whisper` to enable it."
    )

input_mode = st.radio(
    "Choose input method",
    options=["Prompt only", "Audio only", "Both"],
    index=0,
    horizontal=True,
    key="input_mode_selector",
    disabled=not is_whisper_installed,  # disable audio modes when no whisper
)

text_prompt    = ""
uploaded_audio = None

if input_mode in ("Prompt only", "Both"):
    text_prompt = st.text_area("Enter Story Prompt", height=120)
if input_mode in ("Audio only", "Both") and is_whisper_installed:
    uploaded_audio = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a", "ogg"])

transcribed_text = ""

if uploaded_audio is not None and is_whisper_installed:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_audio.read())
        temp_path = tmp.name
    with st.spinner("Transcribing audio…"):
        model = get_whisper_model()
        if model:
            result = model.transcribe(temp_path)
            transcribed_text = result["text"]
        else:
            st.error("Audio transcription failed to load due to low memory.")
    if transcribed_text:
        st.subheader("🎤 Transcribed Text")
        st.write(transcribed_text)

# ==========================
# GENERATE STORY
# ==========================

has_text  = bool(text_prompt.strip())
has_audio = bool(transcribed_text.strip())

if has_text or has_audio:
    if has_text and has_audio:
        st.info("Both inputs detected — AI will merge them.")
    else:
        st.subheader("📝 Final Prompt")
        st.write(text_prompt.strip() if has_text else transcribed_text.strip())

    if st.button("🚀 Generate Story", type="primary"):
        with st.spinner("Generating story…"):
            llm = get_llm()

            # Merge if both inputs provided
            if has_text and has_audio:
                merge_resp = llm.invoke([HumanMessage(content=(
                    "Merge the following two story ideas into one cohesive story prompt.\n\n"
                    f"Text Prompt: {text_prompt.strip()}\n"
                    f"Audio Transcription: {transcribed_text.strip()}\n\n"
                    "Return ONLY the merged prompt. No explanations."
                ))])
                resolved_prompt = merge_resp.content.strip()
            else:
                resolved_prompt = text_prompt.strip() if has_text else transcribed_text.strip()

            story_prompt = f"""Generate a creative story from this idea:

{resolved_prompt}

Requirements:
- Exactly 10 sentences.
- Each sentence must end with a period, exclamation mark, or question mark.
- Coherent and engaging with dialogues where appropriate.
- Give proper names to important characters.
- Make it cinematic.

IMPORTANT: Return ONLY a single valid JSON object with NO extra text:
{{
    "story": "The full story as one string.",
    "characters": ["Name1", "Name2"]
}}"""

            try:
                response = llm.invoke([HumanMessage(content=story_prompt)])
                raw_content = response.content

                try:
                    data = _parse_llm_json(raw_content)
                except json.JSONDecodeError as je:
                    st.error(f"LLM returned invalid JSON: {je}")
                    st.code(raw_content, language="text")
                    raise

                story      = data.get("story", "").strip()
                characters = data.get("characters", [])

                if not story:
                    raise ValueError("LLM returned an empty story field.")

                # ── Reset session state ────────────────────────────────────
                st.session_state.story            = story
                st.session_state.characters       = characters
                st.session_state.chat_history     = {}
                st.session_state.raw_response     = None
                st.session_state.image_sas_urls   = []
                st.session_state.narration_sas_url = None
                st.session_state.video_sas_url    = None
                st.session_state.json_sas_url     = None
                st.session_state.log_sas_url      = None
                st.session_state.story_id         = str(uuid4())

                # ── Set up per-run log file ────────────────────────────────
                log_dir  = os.path.join(IMAGE_DIR, "logs")
                os.makedirs(log_dir, exist_ok=True)
                log_path = os.path.join(log_dir, f"run_{st.session_state.story_id}.log")
                run_log  = _get_run_logger(log_path)
                run_log.info("Story generation started.")

                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", story) if s.strip()]
                os.makedirs(IMAGE_DIR, exist_ok=True)

                # ── Qdrant embeddings ──────────────────────────────────────
                try:
                    with st.spinner("Storing embeddings…"):
                        transformer = get_sentence_transformer()
                        qdrant      = get_qdrant_client()
                        embeddings  = transformer.encode(sentences, normalize_embeddings=True).tolist()
                        col_name    = f"story_{st.session_state.story_id}"
                        if not qdrant.collection_exists(col_name):
                            qdrant.recreate_collection(
                                collection_name=col_name,
                                vectors_config=models.VectorParams(
                                    size=len(embeddings[0]),
                                    distance=models.Distance.COSINE,
                                ),
                            )
                        qdrant.upsert(
                            collection_name=col_name,
                            points=[
                                models.PointStruct(id=i, vector=emb, payload={"sentence": s})
                                for i, (emb, s) in enumerate(zip(embeddings, sentences))
                            ],
                        )
                        run_log.info("Embeddings stored in Qdrant.")
                except Exception as qe:
                    st.warning(f"Qdrant (non-fatal): {qe}")
                    run_log.warning(f"Qdrant failed: {qe}")

                # ── Image generation ───────────────────────────────────────
                pipe           = get_image_pipe()
                image_path_map = {}
                img_errors     = []

                with st.spinner("Generating images…"):
                    prog = st.progress(0, text="Generating images…")
                    for i, sent in enumerate(sentences):
                        prog.progress(i / len(sentences), text=f"Image {i+1}/{len(sentences)}…")
                        try:
                            with torch.inference_mode():
                                out = pipe(
                                    f"Cinematic illustration, detailed, vibrant: {sent}",
                                    num_inference_steps=8,
                                    width=512,
                                    height=512,
                                    guidance_scale=7.5,
                                )
                            img_path            = os.path.join(IMAGE_DIR, f"img_{i:03d}.png")
                            out.images[0].save(img_path)
                            image_path_map[i]   = img_path
                            run_log.info(f"Image {i} saved: {img_path}")
                        except Exception as ie:
                            img_errors.append(f"Image {i}: {ie}")
                            run_log.error(f"Image {i} failed: {ie}")
                    prog.progress(1.0, text="Images done!")

                if img_errors:
                    st.warning("Some images failed:\n" + "\n".join(img_errors))

                # Upload images → get SAS URLs
                image_sas_urls = []
                for idx in sorted(image_path_map):
                    blob_name = f"{st.session_state.story_id}/images/img_{idx:03d}.png"
                    sas_url   = upload_to_blob(image_path_map[idx], blob_name)
                    image_sas_urls.append(sas_url)
                    run_log.info(f"Image {idx} uploaded: {sas_url[:80]}…")
                st.session_state.image_sas_urls = image_sas_urls

                # ── TTS per sentence ───────────────────────────────────────
                tts_paths = []
                with st.spinner("Generating narration…"):
                    tts_prog = st.progress(0, text="Generating TTS…")
                    for i, sent in enumerate(sentences):
                        tts_prog.progress(i / len(sentences), text=f"TTS {i+1}/{len(sentences)}…")
                        out_mp3 = os.path.join(IMAGE_DIR, f"audio_{i:03d}.mp3")
                        tts_paths.append(generate_tts(sent, out_mp3))
                        time.sleep(0.3)
                    tts_prog.progress(1.0, text="TTS done!")

                narration_path = os.path.join(IMAGE_DIR, "narration.mp3")
                if build_narration([p for p in tts_paths if p], narration_path):
                    blob_name = f"{st.session_state.story_id}/narration.mp3"
                    st.session_state.narration_sas_url = upload_to_blob(narration_path, blob_name)
                    run_log.info("Narration uploaded.")

                # ── Video assembly ─────────────────────────────────────────
                video_path = os.path.join(IMAGE_DIR, "story_video.mp4")
                with st.spinner("Assembling video…"):
                    result_video = build_story_video(sentences, image_path_map, tts_paths, video_path)
                    if result_video and os.path.exists(result_video):
                        blob_name = f"{st.session_state.story_id}/story_video.mp4"
                        st.session_state.video_sas_url = upload_to_blob(result_video, blob_name)
                        run_log.info("Video uploaded.")

                # ── JSON manifest ──────────────────────────────────────────
                story_data = {
                    "story_id":         st.session_state.story_id,
                    "story":            story,
                    "characters":       characters,
                    "image_sas_urls":   image_sas_urls,
                    "narration_sas_url": st.session_state.narration_sas_url,
                    "video_sas_url":    st.session_state.video_sas_url,
                }
                json_path      = os.path.join(IMAGE_DIR, "story.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(story_data, f, ensure_ascii=False, indent=2)
                json_blob_name = f"{st.session_state.story_id}/story.json"
                st.session_state.json_sas_url = upload_to_blob(json_path, json_blob_name)
                run_log.info("JSON manifest uploaded.")

                # ── Upload log file ────────────────────────────────────────
                # Flush handlers so all entries are written before upload
                for h in run_log.handlers:
                    h.flush()
                log_blob_name = f"{st.session_state.story_id}/logs/run_{st.session_state.story_id}.log"
                st.session_state.log_sas_url = upload_to_blob(log_path, log_blob_name)

                st.rerun()

            except Exception as err:
                import traceback
                tb = traceback.format_exc()
                st.session_state.raw_response = tb
                st.error(f"Error: {err}")


# ==========================
# DISPLAY — TWO-COLUMN LAYOUT
# ==========================

if st.session_state.story:

    left, right = st.columns([3, 2])

    # ── LEFT COLUMN ────────────────────────────────────────────────────────
    with left:
        st.subheader("📖 Generated Story")
        st.write(st.session_state.story)

        st.subheader("🎭 Characters")
        if st.session_state.characters:
            char_cols = st.columns(min(len(st.session_state.characters), 5))
            for idx, char in enumerate(st.session_state.characters):
                with char_cols[idx % len(char_cols)]:
                    st.info(f"👤 {char}")
        else:
            st.write("No characters found.")

        st.markdown("---")
        st.subheader("🖼️ Story Images")
        imgs = st.session_state.image_sas_urls
        if imgs:
            for i in range(0, len(imgs), 4):
                row = st.columns(4)
                for col, url in zip(row, imgs[i : i + 4]):
                    with col:
                        st.image(url, use_container_width=True)
        else:
            st.write("No images generated yet.")

        st.markdown("---")
        st.subheader("🔊 Full Narration Audio")
        narr = st.session_state.narration_sas_url
        if narr:
            st.audio(narr)
            st.markdown(f"[⬇️ Download MP3]({narr})")
        else:
            st.write("Narration not available.")

        st.markdown("---")
        st.subheader("🎬 Story Video")
        vid = st.session_state.video_sas_url
        if vid:
            st.video(vid)
        else:
            st.write("Video not available.")

        # Downloads row
        dl_cols = st.columns(2)
        with dl_cols[0]:
            if st.session_state.get("json_sas_url"):
                st.markdown(f"[⬇️ Download JSON Manifest]({st.session_state.json_sas_url})")
        with dl_cols[1]:
            if st.session_state.get("log_sas_url"):
                st.markdown(f"[⬇️ Download Run Log]({st.session_state.log_sas_url})")

        st.markdown("---")
        st.subheader("🖼️ Generate Additional Image")
        custom_prompt = st.text_input("Enter image description", key="custom_img_input")
        if st.button("Generate Custom Image", key="gen_custom_img"):
            if custom_prompt.strip():
                with st.spinner("Generating…"):
                    try:
                        pipe = get_image_pipe()
                        with torch.inference_mode():
                            out = pipe(
                                custom_prompt,
                                num_inference_steps=8,
                                width=512,
                                height=512,
                                guidance_scale=7.5,
                            )
                        os.makedirs(IMAGE_DIR, exist_ok=True)
                        img_path  = os.path.join(IMAGE_DIR, f"custom_{uuid4().hex[:8]}.png")
                        out.images[0].save(img_path)
                        blob_name = f"{st.session_state.story_id}/images/{os.path.basename(img_path)}"
                        sas_url   = upload_to_blob(img_path, blob_name)
                        st.session_state.image_sas_urls.append(sas_url)
                        st.success("Custom image generated!")
                        st.image(sas_url, width=400)
                    except Exception as ce:
                        st.error(f"Failed: {ce}")

    # ── RIGHT COLUMN — CHARACTER CHAT ──────────────────────────────────────
    with right:
        st.subheader("💬 Chat with Characters")

        if not st.session_state.characters:
            st.write("No characters found in the story.")
        else:
            col_sel, col_clr = st.columns([3, 1])
            with col_sel:
                selected_char = st.selectbox(
                    "Choose character:",
                    options=st.session_state.characters,
                    key="char_selector",
                )
            with col_clr:
                st.write("")  # vertical spacing
                if st.button("🧹 Clear", key="clear_chat"):
                    st.session_state.chat_history[selected_char] = []
                    st.rerun()

            if selected_char not in st.session_state.chat_history:
                st.session_state.chat_history[selected_char] = []

            # Chat history display
            chat_container = st.container(height=450)
            with chat_container:
                for msg in st.session_state.chat_history[selected_char]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            # Chat input
            user_msg = st.chat_input(f"Message {selected_char}…", key="char_chat_input")
            if user_msg:
                st.session_state.chat_history[selected_char].append(
                    {"role": "user", "content": user_msg}
                )
                with st.spinner(f"{selected_char} is thinking…"):
                    chat_llm    = get_llm(primary=False)
                    system_prompt = (
                        f"You are {selected_char} from the story below.\n"
                        "Stay completely in character. Respond naturally and concisely.\n\n"
                        f"STORY:\n{st.session_state.story}"
                    )
                    messages = [SystemMessage(content=system_prompt)]
                    for m in st.session_state.chat_history[selected_char]:
                        cls = HumanMessage if m["role"] == "user" else AIMessage
                        messages.append(cls(content=m["content"]))

                    reply = chat_llm.invoke(messages).content.strip()
                    st.session_state.chat_history[selected_char].append(
                        {"role": "assistant", "content": reply}
                    )
                st.rerun()

elif st.session_state.raw_response:
    st.subheader("⚠️ Debug — Raw Error")
    st.code(st.session_state.raw_response)