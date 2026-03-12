import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import faiss
import gradio as gr
import open_clip
from huggingface_hub import hf_hub_download

# ==========================================
# 1. KONFIGURASI PATH & PENGUNDUHAN MODEL
# ==========================================
# PENTING: Ganti dengan username dan nama repo tempat Anda menyimpan file .pt 1GB
REPO_ID = "bgaspra/eva-generative-recommender" 
FILENAME = "eva_stage2_heavy_text_best.pt"

print("Downloading/Loading model weights from Hub...")
try:
    EVA_FINETUNE_CKPT = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
except Exception as e:
    print(f"Error downloading model: {e}")
    EVA_FINETUNE_CKPT = None

# Karena di Hugging Face Space semua file di-upload di folder utama,
# kita cukup memanggil nama filenya secara langsung.
CENTROIDS_PATH = "stage2_centroids.npy"
META_PATH_CSV  = "stage2_centroids_meta.csv"
BALANCED_CSV   = "balanced_min6_max15.csv"

# ==========================================
# 2. UTILITIES & DATA LOADING
# ==========================================
def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)

def build_model_url_map(df: pd.DataFrame, max_imgs_per_model: int = 3) -> Dict[str, List[str]]:
    if "Model" not in df.columns or "url" not in df.columns:
        raise ValueError("CSV must contain columns: Model, url")
    tmp = df.copy()
    tmp["Model"] = tmp["Model"].astype(str)
    tmp["url"] = tmp["url"].astype(str)
    tmp = tmp[(tmp["Model"].str.len() > 0) & (tmp["url"].str.len() > 0)]
    out: Dict[str, List[str]] = {}
    for m, g in tmp.groupby("Model"):
        urls = g["url"].dropna().astype(str).tolist()
        out[m] = urls[:max_imgs_per_model]
    return out

def load_model_names_from_meta_csv(path: str) -> List[str]:
    if not os.path.exists(path): raise FileNotFoundError(f"Missing meta: {path}")
    dfm = pd.read_csv(path, low_memory=False)
    for col in ["model_names", "Model", "model", "model_name", "name", "0"]:
        if col in dfm.columns: return dfm[col].astype(str).tolist()
    if dfm.shape[1] == 1: return dfm.iloc[:, 0].astype(str).tolist()
    raise ValueError("Meta CSV has no recognizable model-name column")

def make_faiss_index(centroids_norm: np.ndarray) -> faiss.Index:
    d = centroids_norm.shape[1]
    idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efSearch = 256
    idx.hnsw.efConstruction = 200
    idx.add(centroids_norm.astype(np.float32))
    return idx

def _clean_state_dict(state_dict: dict) -> dict:
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."): k = k[len("module."):]
        if k.startswith("model."): k = k[len("model."):]
        new_sd[k] = v
    return new_sd

def _extract_state_dict(ckpt_obj: object) -> dict:
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj: return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj: return ckpt_obj["model_state_dict"]
        if "model" in ckpt_obj: return ckpt_obj["model"]
        return ckpt_obj
    raise ValueError("Unknown checkpoint format")

# ==========================================
# 3. KELAS ENCODER & ARTIFAK
# ==========================================
@dataclass
class EVAEncoder:
    model: torch.nn.Module
    preprocess: callable
    tokenizer: callable
    device: str

    @torch.no_grad()
    def encode_image(self, pil_img: Image.Image) -> np.ndarray:
        x = self.preprocess(pil_img.convert("RGB")).unsqueeze(0).to(self.device)
        feat = self.model.encode_image(x)
        feat = F.normalize(feat, dim=-1)
        return feat.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text: raise ValueError("Empty text")
        toks = self.tokenizer([text]).to(self.device)
        feat = self.model.encode_text(toks)
        feat = F.normalize(feat, dim=-1)
        return feat.detach().cpu().numpy().astype(np.float32)

@dataclass
class Artifacts:
    centroids: np.ndarray
    model_names: List[str]
    index: faiss.Index
    model_to_urls: Dict[str, List[str]]

# --- Inisialisasi Model & Data ---
device = "cpu" # Wajib CPU untuk HF Space gratis
print(f"🔹 Loading EVA02-L-14 on {device}...")
model_clip, _, preprocess_clip = open_clip.create_model_and_transforms(
    "EVA02-L-14", pretrained="merged2b_s4b_b131k", device=device
)
tokenizer_clip = open_clip.get_tokenizer("EVA02-L-14")

if EVA_FINETUNE_CKPT and os.path.exists(EVA_FINETUNE_CKPT):
    ckpt = torch.load(EVA_FINETUNE_CKPT, map_location=device)
    sd = _clean_state_dict(_extract_state_dict(ckpt))
    model_clip.load_state_dict(sd, strict=False)
    print("✅ Loaded Stage-2 weights from Hub!")
else:
    print("⚠️ Stage-2 weights not found. Using pretrained base.")

model_clip.eval()
eva = EVAEncoder(model_clip, preprocess_clip, tokenizer_clip, device)

print("🔹 Loading artifacts...")
centroids_raw = np.load(CENTROIDS_PATH).astype(np.float32)
model_names_list = load_model_names_from_meta_csv(META_PATH_CSV)

if len(model_names_list) != centroids_raw.shape[0]:
    raise ValueError(f"Mismatch: {centroids_raw.shape[0]} centroids vs {len(model_names_list)} names")

centroids_norm = l2_normalize(centroids_raw)
index_faiss = make_faiss_index(centroids_norm)

df_balanced = pd.read_csv(BALANCED_CSV, low_memory=False)
model_to_urls_map = build_model_url_map(df_balanced, max_imgs_per_model=3)

art = Artifacts(centroids_norm, model_names_list, index_faiss, model_to_urls_map)
print(f"✅ Ready. Indexing {len(model_names_list)} models.")

# ==========================================
# 4. LOGIKA PENCARIAN & GRADIO UI
# ==========================================
def search_and_format(query_vec: np.ndarray, k: int, imgs_per_model: int):
    # 1. FAISS Search
    fetch = min(100, len(art.model_names))
    D, I = art.index.search(query_vec, fetch)
    
    # 2. Extract Results
    recs = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0: continue
        recs.append((art.model_names[idx], float(score)))
        if len(recs) >= k: break
    
    # 3. Build Gallery Items
    items = []
    text_log = ["### Top Matches:"]
    
    for i, (name, score) in enumerate(recs, 1):
        urls = art.model_to_urls.get(name, [])
        text_log.append(f"{i}. **{name}**")
        
        if not urls:
            items.append((None, f"{name}"))
        else:
            for u in urls[:imgs_per_model]:
                items.append((u, f"{name}"))
                
    return "\n".join(text_log), items

# CSS Custom
CUSTOM_CSS = """
.container { max-width: 1200px !important; margin: auto; padding-top: 20px; }
#gallery_img, #gallery_txt {
    height: auto !important;
    max-height: none !important;
    overflow: visible !important;
    min-height: 500px;
}
"""

# Membangun UI Gradio
with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="Multimodal Recommender") as demo:
    
    # --- Header ---
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# ✨ Multimodal Model Recommender")
            gr.Markdown("Search for Generative AI models using **Images** or **Text** description.")

    # --- Settings ---
    with gr.Accordion("⚙️ Search Settings", open=False):
        with gr.Row():
            k_slider = gr.Slider(1, 10, value=5, step=1, label="Top-K Models")
            img_count_slider = gr.Slider(1, 3, value=2, step=1, label="Images per Model (Display)")

    # --- Tabs for Interaction ---
    with gr.Tabs():
        
        # === TAB 1: IMAGE SEARCH ===
        with gr.Tab("📷 Image Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(type="pil", label="Upload Reference Image", height=300)
                    btn_img = gr.Button("🔍 Find Similar Models", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    md_res_img = gr.Markdown()
                    gal_img = gr.Gallery(
                        label="Recommendations", 
                        columns=4, 
                        elem_id="gallery_img",
                        object_fit="contain"
                    )

            def run_img_search(img, k, ipm):
                if img is None: return "⚠️ Please upload an image.", []
                q = eva.encode_image(img)
                return search_and_format(q, int(k), int(ipm))

            btn_img.click(run_img_search, inputs=[img_input, k_slider, img_count_slider], outputs=[md_res_img, gal_img])

        # === TAB 2: TEXT SEARCH ===
        with gr.Tab("📝 Text Search"):
            with gr.Row():
                with gr.Column(scale=1):
                    txt_input = gr.Textbox(
                        lines=4, 
                        label="Describe Style / Concept", 
                        placeholder="e.g., cyberpunk city with neon lights, anime style, highly detailed..."
                    )
                    btn_txt = gr.Button("🔍 Find Models by Text", variant="primary", size="lg")
                
                with gr.Column(scale=3):
                    md_res_txt = gr.Markdown()
                    gal_txt = gr.Gallery(
                        label="Recommendations", 
                        columns=4, 
                        elem_id="gallery_txt",
                        object_fit="contain"
                    )

            def run_txt_search(text, k, ipm):
                if not text: return "⚠️ Please enter text.", []
                q = eva.encode_text(text)
                return search_and_format(q, int(k), int(ipm))

            btn_txt.click(run_txt_search, inputs=[txt_input, k_slider, img_count_slider], outputs=[md_res_txt, gal_txt])

# Launch the App
if __name__ == "__main__":
    demo.queue().launch()