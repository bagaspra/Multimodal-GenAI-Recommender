import gradio as gr
import torch
import open_clip
import faiss
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

# 1. Download model dari Hugging Face Models milikmu
model_path = hf_hub_download(repo_id="bgaspra/eva-generative-recommender", filename="eva_stage2_heavy_text_best.pt")

# 2. Load Model EVA-CLIP
device = "cpu" # Gunakan CPU karena HF Spaces gratis tidak pakai GPU
model, _, preprocess = open_clip.create_model_and_transforms("EVA02-L-14", pretrained="merged2b_s4b_b131k")
tokenizer = open_clip.get_tokenizer("EVA02-L-14")

# Load weights yang baru saja di-download
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

# 3. Load Data & FAISS Index
centroids = np.load("stage2_centroids.npy")
meta_df = pd.read_csv("stage2_centroids_meta.csv")
dataset_df = pd.read_csv("balanced_min6_max15.csv")

# Buat FAISS Index
res = faiss.StandardGpuResources() # Hapus ini jika pakai CPU
index = faiss.IndexFlatIP(768)
index.add(centroids)

# 4. Fungsi Rekomendasi (Sama seperti di notebook Anda)
def recommend(image, text):
    # Logika ekstraksi embedding image/text pakai model
    # Logika index.search(vektor, k=5)
    # Return output gambar ke Gradio
    return list_of_images 

# 5. Gradio UI
with gr.Blocks() as ui:
    gr.Markdown("# Multimodal Generative AI Recommender")
    # ... (Tambahkan UI komponen Anda di sini)

ui.launch()