---
title: Multimodal GenAI Recommender
emoji: 🎨
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🎨 Multimodal Generative AI Recommender

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/bgaspra/multimodal-genai-recommender)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)](https://pytorch.org/)

Finding the right Generative AI model (Checkpoints/LoRA) usually requires tedious trial-and-error. **Multimodal Generative AI Recommender** is an end-to-end machine learning system that allows users to discover the perfect AI model for their desired visual style using either **image references** or **text prompts**.

This project was developed as part of a Master's Thesis research, transitioning from a standard classification approach to a highly scalable **retrieval-based architecture**.

👉 **[Try the Live Interactive Demo Here!](https://huggingface.co/spaces/bgaspra/multimodal-genai-recommender)**

---

## 📸 Sneak Peek
![App Screenshot](https://ibb.co/fzmwtBQ9)

---

## ✨ Key Features
* **Multimodal Search:** Seamlessly search for models using an uploaded Image or a Text description.
* **Model-Level Representation (Centroid Aggregation):** Synthesized 27,000+ individual image samples into stable, high-dimensional vector representations for 1,900+ unique AI models.
* **Ultra-Fast Retrieval:** Powered by **FAISS** (Facebook AI Similarity Search) to perform L2-normalized nearest-neighbor lookups in milliseconds.
* **Fine-Tuned Embeddings:** Utilizes a fine-tuned **EVA-CLIP (EVA02-L-14)** vision-language model to accurately capture complex visual styles and semantics.
* **Interactive Web Interface:** Built with Gradio and deployed on Hugging Face Spaces for real-time user interaction.

---

## 🛠️ Tech Stack & Architecture
* **Core ML / Deep Learning:** PyTorch, EVA-CLIP (OpenCLIP)
* **Vector Database & Search:** FAISS (IndexHNSWFlat / IndexFlatIP)
* **Data Processing:** Pandas, NumPy
* **Frontend & Deployment:** Gradio, Hugging Face Spaces, Git LFS

### System Workflow
1. **Input:** User provides an image or a text prompt.
2. **Embedding Extraction:** The input is passed through the fine-tuned EVA-CLIP encoder to generate a 768-dimensional normalized feature vector.
3. **Similarity Search:** The vector is queried against a FAISS index containing pre-computed centroids of 1,900+ Generative AI models.
4. **Recommendation:** The system returns the Top-K models that best match the requested visual style, complete with preview images.

---

## 📊 Performance Metrics
The system was rigorously evaluated using ranking metrics, achieving strong recommendation performance on unseen validation data:
* **nDCG@5:** 0.79
* **Recall@10:** 0.89

---

## 💻 How to Run Locally

If you want to run this recommendation system on your local machine:

**1. Clone the repository:**
```bash
git clone [https://github.com/bgaspra/multimodal-genai-recommender.git](https://github.com/bgaspra/multimodal-genai-recommender.git)
cd multimodal-genai-recommender

2. Install dependencies:

Bash
pip install -r requirements.txt
3. Run the application:

Bash
python app.py
Note: The script is configured to automatically download the 1.7GB fine-tuned model weights from the Hugging Face Model Hub on the first run.

👨‍💻 About The Author
Bagas Prasetyo | AI Engineer | Master's Graduate (2026)

Passionate about building scalable AI systems, multimodal architectures, and intelligent web applications.