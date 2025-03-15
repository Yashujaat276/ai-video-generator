# app.py

import streamlit as st
import torch
import tempfile
import shutil
from diffusers import DiffusionPipeline

# Page Title
st.title("🎬 AI Image-to-Video Generator (Runway ML Alternative)")

# GPU Check
def check_gpu():
    if torch.cuda.is_available():
        return "✅ GPU Available"
    else:
        return "❌ GPU Not Available (Slower on CPU)"

st.sidebar.write(check_gpu())

# Load Model (using ModelScope)
@st.cache_resource()
def load_model():
    model_id = "damo-vilab/modelscope-diffusion"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# Input: Prompt
prompt = st.text_input("Enter your prompt (e.g., 'A futuristic city at sunset')", "A tiger running in the jungle")

# Video Generation Settings
num_frames = st.slider("Number of Frames", 16, 48, 24)
fps = st.slider("Frames per Second (FPS)", 8, 30, 12)

# Generate Video Function
def generate_video(prompt, num_frames, fps):
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = f"{temp_dir}/output_video.mp4"

        st.info("Generating video... This may take a few minutes ⏳")
        
        video = pipe(prompt=prompt, num_inference_steps=num_frames).frames
        video[0].save(video_path)

        shutil.copy(video_path, "output_video.mp4")
        st.success("✅ Video generation complete!")
        return "output_video.mp4"

# Button to Generate Video
if st.button("Generate Video"):
    output_file = generate_video(prompt, num_frames, fps)

    # Show Video
    st.video(output_file)

    # Download Button
    with open(output_file, "rb") as file:
        st.download_button("📥 Download Your AI Video", file, "output_video.mp4")
