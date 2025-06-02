import streamlit as st
from PIL import Image
import os
import shutil
import subprocess

STYLE_DIR = "input/style"
CONTENT_DIR = "input/images"
OUTPUT_DIR = "output/final_outputs"

os.makedirs(STYLE_DIR, exist_ok=True)
os.makedirs(CONTENT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

st.title("Neural Style Transfer")

# Image upload
style_image = st.file_uploader("Upload Style Image", type=["png"])
content_image = st.file_uploader("Upload Content Image", type=["jpg"])

if st.button("Run Style Transfer"):
    if style_image and content_image:
        style_path = os.path.join(STYLE_DIR, "style.png")
        content_path = os.path.join(CONTENT_DIR, "inference.jpg")

        with open(style_path, "wb") as f:
            f.write(style_image.read())
        with open(content_path, "wb") as f:
            f.write(content_image.read())

        st.success("Images saved. Running preprocessing...")

        try:
            subprocess.run(["python", "preprocess.py"], check=True)
            st.success("Preprocessing done.")

            subprocess.run(["python", "final.py"], check=True)
            st.success("Style transfer complete.")

            # Show result
            output_path = os.path.join(OUTPUT_DIR, "final_inference.jpg")
            if os.path.exists(output_path):
                st.image(Image.open(output_path), caption="Stylized Output")
            else:
                st.warning("Output image not found.")

        except subprocess.CalledProcessError as e:
            st.error(f"Processing failed: {e}")

        finally:
            if os.path.exists(style_path):
                os.remove(style_path)
            if os.path.exists(content_path):
                os.remove(content_path)
            st.info("Temporary files cleaned up.")

    else:
        st.warning("Please upload both images and provide a name.")
