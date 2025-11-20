# streamlit_app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os
import base64
from io import BytesIO
import json
import random
import requests
import urllib.request
import pandas as pd
import torch
import torch.nn as nn
from gtts import gTTS
import warnings
import subprocess
import sys
import shutil
from pathlib import Path

warnings.filterwarnings('ignore')

# ======================== SAFE IMPORT OF CV2 ========================
def ensure_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        with st.spinner("Installing OpenCV (cv2)... This takes ~20 seconds only once"):
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless", "--quiet"])
        import cv2
        st.success("OpenCV installed!")
        return cv2

cv2 = ensure_cv2()

# ======================== SAFE MOVIEPY INSTALL ========================
def install_moviepy():
    try:
        from moviepy.editor import (AudioFileClip, ImageClip, concatenate_videoclips,
                                   VideoFileClip, concatenate_audioclips)
        return True
    except ImportError:
        with st.spinner("Installing MoviePy + FFmpeg tools..."):
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "moviepy", "imageio", "imageio-ffmpeg", "--quiet"
            ])
        from moviepy.editor import (AudioFileClip, ImageClip, concatenate_videoclips,
                                   VideoFileClip, concatenate_audioclips)
        st.success("MoviePy ready!")
        return True

MOVIEPY_AVAILABLE = install_moviepy()

if MOVIEPY_AVAILABLE:
    from moviepy.editor import (AudioFileClip, ImageClip, concatenate_videoclips,
                               VideoFileClip, concatenate_audioclips)
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout
    from moviepy.video.fx.all import resize
    from moviepy.config import change_settings

    # Try to configure ImageMagick (for text, optional)
    try:
        result = subprocess.run(['which', 'convert'], capture_output=True, text=True)
        if result.returncode == 0:
            path = result.stdout.strip()
            os.environ["IMAGEMAGICK_BINARY"] = path
            change_settings({"IMAGEMAGICK_BINARY": path})
    except:
        pass

# ======================== PAGE CONFIG & CSS ========================
st.set_page_config(page_title="Uganda Bird Spotter", page_icon="Bird", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 3.5rem; color: #2E86AB; text-align: center; margin: 2rem 0;
                  font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.1);}
    .glass-card {background: rgba(255,255,255,0.25); backdrop-filter: blur(15px);
                 border-radius: 20px; padding: 25px; margin: 15px 0; box-shadow: 0 8px 32px rgba(0,0,0,0.1);}
    .story-box {background: rgba(255,248,225,0.9); border-left: 6px solid #FFD700;
                padding: 20px; border-radius: 12px; font-size: 1.1rem; line-height: 1.7;}
    .stButton>button {background: #2E86AB; color: white; border-radius: 12px; font-weight: 600;}
</style>
""", unsafe_allow_html=True)

# ======================== STORY TEMPLATES ========================
TEMPLATES = [
    "Deep in Uganda's lush forests, the magnificent {name} flashes its {color_phrase} feathers as it moves gracefully through the canopy. {desc}",
    "Along the majestic Nile's banks, the elegant {name} stands tall with its stunning {color_phrase} plumage shimmering in the morning light. {desc}",
    "In Queen Elizabeth National Park, the {name} soars above ancient acacia trees, its {color_phrase} wings creating beautiful patterns against the golden sky. {desc}",
    "Near the tranquil shores of Lake Victoria, the {name} perches quietly. {desc} Children say seeing its {color_phrase} colors means a good fishing day.",
    "High in the mystical Rwenzori Mountains, the {name} sings through the mist with its {color_phrase} feathers glowing in rare light. {desc}"
]

class BirdStoryGenerator:
    def __init__(self, templates): 
        self.templates = templates
    def __call__(self, name, description="", colors=None):
        if colors is None: colors = []
        color_phrase = ", ".join([c.strip() for c in colors]) if colors else "vibrant"
        desc = description.strip().capitalize() if description else "This remarkable bird is cherished across Uganda."
        return random.choice(self.templates).format(name=name, color_phrase=color_phrase, desc=desc)

story_generator = BirdStoryGenerator(TEMPLATES)

# ======================== VIDEO GENERATOR ========================
class AdvancedVideoGenerator:
    def __init__(self):
        self.story_generator = story_generator
        self.moviepy_available = MOVIEPY_AVAILABLE

    def natural_tts(self, text, filename):
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filename)
            return filename
        except:
            return None

    def create_placeholder_image(self, species_name, output_path, variation=0):
        try:
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            colors = [[70,130,180],[60,179,113],[186,85,211],[255,165,0],[106,90,205]]
            img[:,:] = colors[variation % len(colors)]

            if cv2 is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, species_name, (50, 200), font, 1.2, (255,255,255), 3)
                cv2.putText(img, "Uganda Bird Spotter", (140, 300), font, 0.8, (220,220,220), 2)
                cv2.imwrite(output_path, img)
            else:
                pil_img = Image.fromarray(img)
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 48)
                except:
                    font = ImageFont.load_default()
                draw.text((50, 180), species_name, fill=(255,255,255), font=font)
                draw.text((140, 280), "Uganda Bird Spotter", fill=(220,220,220), font=font)
                pil_img.save(output_path)
            return True
        except:
            return False

    def get_bird_images(self, species_name, max_images=8):
        # Try to get real images from bird_db or model data
        # Fallback to placeholders
        images = []
        for i in range(max_images):
            path = f"temp_placeholder_{i}.jpg"
            if self.create_placeholder_image(species_name, path, i):
                images.append(path)
        return images

    def ken_burns_effect(self, img_path, duration=5.0):
        if not MOVIEPY_AVAILABLE: return None
        try:
            clip = ImageClip(img_path).set_duration(duration)
            w, h = clip.size
            zoom = 1 + 0.2 * (np.sin(np.linspace(0, np.pi, int(duration*24))) ** 2)
            clip = clip.resize(lambda t: 1 + 0.15 * (t/duration))
            clip = clip.set_position("center")
            return clip.fadein(0.7).fadeout(0.7)
        except:
            return ImageClip(img_path).set_duration(duration)

    def generate_story_video(self, species_name):
        try:
            story = self.story_generator(species_name)
            st.markdown(f'<div class="story-box">{story}</div>', unsafe_allow_html=True)

            audio_file = f"temp_{species_name}.mp3"
            if not self.natural_tts(story, audio_file):
                st.error("Failed to generate audio")
                return None

            images = self.get_bird_images(species_name, 8)
            clips = [self.ken_burns_effect(img, 5.0) for img in images]

            if MOVIEPY_AVAILABLE and clips:
                video = concatenate_videoclips(clips, method="compose")
                audio = AudioFileClip(audio_file)
                video = video.set_audio(audio_fadein(audio, 1).set_duration(video.duration))
                output = f"uganda_bird_{species_name.replace(' ', '_')}.mp4"
                video.write_videofile(output, fps=24, codec="libx264", audio_codec="aac", verbose=False, logger=None)
                return output, story
            else:
                st.error("Video creation unavailable")
                return None, story
        except Exception as e:
            st.error(f"Video error: {e}")
            return None, None

video_gen = AdvancedVideoGenerator()

# ======================== MAIN APP ========================
def main():
    st.markdown('<div class="main-header">Uganda Bird Spotter</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">Upload a bird photo to identify it and generate a beautiful narrated story video!</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload Photo", type="primary", use_container_width=True):
            st.session_state.mode = "upload"
    with col2:
        if st.button("Take Photo", use_container_width=True):
            st.session_state.mode = "camera"

    if 'mode' not in st.session_state:
        st.session_state.mode = "upload"

    img = None
    if st.session_state.mode == "upload":
        uploaded = st.file_uploader("Choose bird image...", type=['png','jpg','jpeg'])
        if uploaded: img = Image.open(uploaded)
    else:
        captured = st.camera_input("Take a photo")
        if captured: img = Image.open(captured)

    if img:
        st.image(img, "Your Bird")

        species = st.text_input("Enter species name (or leave blank for placeholder)", 
                                placeholder="e.g. Grey Crowned Crane").strip()
        if not species:
            species = "Unknown Bird"

        if st.button("Generate Story Video", type="primary"):
            with st.spinner("Creating your beautiful video..."):
                video_path, story = video_gen.generate_story_video(species)
                if video_path and os.path.exists(video_path):
                    with open(video_path, "rb") as f:
                        st.video(f.read())
                    st.download_button("Download Video", f.read(), 
                                       file_name=os.path.basename(video_path), mime="video/mp4")
                    # Cleanup
                    for f in os.listdir('.'):
                        if f.startswith('temp_') or f.startswith('uganda_bird_'):
                            try: os.remove(f)
                            except: pass

if __name__ == "__main__":
    main()