# app.py
import os
import io
from PIL import Image, ImageOps
import numpy as np
from sklearn.cluster import KMeans
import streamlit as st
import openai
from sklearn.metrics.pairwise import cosine_similarity

# ----- Configure OpenAI -----
openai.api_key = os.getenv("OPENAI_API_KEY")
TEXT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# ----- Utilities: image processing -----
def load_image(file) -> Image.Image:
    img = Image.open(file).convert("RGB")
    return img

def get_dominant_colors(image: Image.Image, k=3):
    small = image.resize((150, 150))
    arr = np.array(small).reshape(-1, 3).astype(float)
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(arr)
    centers = kmeans.cluster_centers_.astype(int)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    total = counts.sum()
    order = np.argsort(-counts)
    colors = []
    for idx in order:
        c = centers[idx]
        pct = counts[idx] / total
        hexc = '#%02x%02x%02x' % tuple(c)
        colors.append({"hex": hexc, "pct": float(pct)})
    return colors

def edge_density(image: Image.Image):
    gray = ImageOps.grayscale(image).resize((150, 150))
    arr = np.array(gray).astype(float) / 255.0
    gy, gx = np.gradient(arr)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.mean(mag))

def build_feature_summary(colors, edge_d):
    main = colors[0]
    color_desc = f"dominant color {main['hex']} ({int(main['pct']*100)}%)"
    other = ""
    if len(colors) > 1:
        other = ", plus " + ", ".join([c["hex"] for c in colors[1:]])
    texture = "textured" if edge_d > 0.06 else "smooth"
    return f"{color_desc}{other}; overall vibe: {texture} (edge={edge_d:.3f})"

def generate_descriptor_text(feature_summary: str):
    prompt = (
        "You are a playful but kind stylist and dating-app copywriter for socks. "
        "Turn this technical feature summary into a one-sentence, human-friendly descriptor "
        "that mentions color and vibe in a fun, short way. "
        "Example: 'Bright blue stripes, playful and adventurous.'\n\n"
        f"Feature summary: {feature_summary}\n\nDescriptor:"
    )
    resp = openai.ChatCompletion.create(
        model=TEXT_MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=60,
        temperature=0.8,
    )
    text = resp.choices[0].message.content.strip()
    return text

def generate_match_story(desc_a: str, desc_b: str, score_pct: float):
    prompt = (
        "You are a whimsical matchmaker who writes one short, witty 'love story' line for two socks "
        "based on their descriptors. Keep it 1-2 sentences, playful, and family-friendly. "
        f"Sock A: {desc_a}\nSock B: {desc_b}\nMatch score: {score_pct:.0f}%\nStory:"
    )
    resp = openai.ChatCompletion.create(
        model=TEXT_MODEL,
        messages=[{"role":"user","content":prompt}],
        max_tokens=80,
        temperature=0.9,
    )
    return resp.choices[0].message.content.strip()

def get_embedding(text: str):
    resp = openai.Embedding.create(model=EMBED_MODEL, input=text)
    return np.array(resp.data[0].embedding, dtype=float)

st.set_page_config(page_title="SockMatch 🧦", layout="centered")
st.title("SockMatch — find your sock's sole mate")

col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload sock photo A", type=["png","jpg","jpeg"], key="a")
with col2:
    file_b = st.file_uploader("Upload sock photo B", type=["png","jpg","jpeg"], key="b")

if st.button("Find My Match"):
    if not file_a or not file_b:
        st.error("Please upload two sock images.")
    else:
        img_a = load_image(file_a)
        img_b = load_image(file_b)
        st.image([img_a, img_b], caption=["Sock A","Sock B"], width=180)

        colors_a = get_dominant_colors(img_a, k=3)
        colors_b = get_dominant_colors(img_b, k=3)
        edge_a = edge_density(img_a)
        edge_b = edge_density(img_b)
        feat_a = build_feature_summary(colors_a, edge_a)
        feat_b = build_feature_summary(colors_b, edge_b)

        st.markdown("**Visual Analysis**")
        st.write("Sock A:", feat_a)
        st.write("Sock B:", feat_b)

        with st.spinner("Describing socks..."):
            desc_a = generate_descriptor_text(feat_a)
            desc_b = generate_descriptor_text(feat_b)

        st.markdown("**Descriptors**")
        st.write("A →", desc_a)
        st.write("B →", desc_b)

        emb_a = get_embedding(desc_a)
        emb_b = get_embedding(desc_b)
        from sklearn.metrics.pairwise import cosine_similarity
        score = float(cosine_similarity([emb_a],[emb_b])[0][0])
        pct = max(0, min(100, (score + 1) / 2 * 100))

        with st.spinner("Writing the matchmaking story..."):
            story = generate_match_story(desc_a, desc_b, pct)

        st.markdown("---")
        st.header(f"Match Score: {pct:.0f}%")
        st.subheader(story)
