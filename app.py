import streamlit as st
import torch
import clip
import pickle
import faiss
from PIL import Image
import numpy as np

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load features and image paths
with open("features.pkl", "rb") as f:
    data = pickle.load(f)

image_features = data["features"]
image_paths = data["paths"]

# Create FAISS index
index = faiss.IndexFlatIP(image_features.shape[1])
index.add(image_features)

# UI
import streamlit as st

# Streamlit app UI
st.title("AI-Powered Interior Design Image Search")

text_query = st.text_input("Enter a description of the style or room you're looking for:")

if text_query:
    # Encode the text query
    text_input = clip.tokenize([text_query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_input)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Similarity search for images
    similarities = np.dot(image_features, text_features.cpu().numpy().T)
    top_k = 5
    top_k_indices = similarities.squeeze().argsort()[-top_k:][::-1]

    st.subheader(f"Top {top_k} matching images:")
    for idx in top_k_indices:
        img_path = image_paths[idx]
        st.image(img_path, caption=f"Match: {img_path}", use_column_width=True)

    st.markdown("---")
  
   
