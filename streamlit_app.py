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
import pandas as pd
import torch
import torch.nn as nn
from gtts import gTTS
import warnings
import subprocess
import sys
import platform
warnings.filterwarnings('ignore')

# ========== ROBUST MOVIEPY INSTALLATION ==========
def install_moviepy():
    """Install MoviePy with comprehensive error handling"""
    st.info("🔧 Setting up MoviePy for video generation...")
    
    # List of packages to install
    packages = [
        "moviepy==1.0.3",
        "decorator==5.1.1", 
        "proglog==0.1.10",
        "imageio==2.31.1",
        "imageio-ffmpeg==0.4.8",
        "Pillow==10.0.1",
        "numpy==1.24.3"
    ]
    
    success_count = 0
    for package in packages:
        try:
            st.write(f"📦 Installing {package}...")
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                package, "--quiet", "--no-warn-script-location"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                success_count += 1
                st.success(f"✅ {package} installed successfully")
            else:
                st.warning(f"⚠️ {package} installation had issues: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            st.warning(f"⏰ Timeout installing {package}, skipping...")
        except Exception as e:
            st.warning(f"⚠️ Error installing {package}: {str(e)}")
    
    return success_count >= 4  # At least core packages installed

def initialize_moviepy():
    """Initialize MoviePy with proper configuration"""
    # Try to install first
    install_success = install_moviepy()
    
    if not install_success:
        st.error("❌ MoviePy installation failed. Using fallback mode.")
        return False
    
    # Now try to import
    try:
        # Clear any cached imports
        import importlib
        if 'moviepy' in sys.modules:
            del sys.modules['moviepy']
        if 'moviepy.editor' in sys.modules:
            del sys.modules['moviepy.editor']
        
        # Import MoviePy components
        from moviepy.editor import (
            AudioFileClip, ImageClip, concatenate_videoclips,
            VideoFileClip, concatenate_audioclips
        )
        from moviepy.audio.fx.all import audio_fadein, audio_fadeout
        from moviepy.video.fx.all import resize
        import moviepy.config as moviepy_config
        
        # Configure MoviePy
        try:
            moviepy_config.change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
        except:
            pass
            
        st.success("✅ MoviePy loaded successfully!")
        return True
        
    except ImportError as e:
        st.error(f"❌ MoviePy import failed: {e}")
        return False
    except Exception as e:
        st.error(f"❌ MoviePy initialization error: {e}")
        return False

# Initialize MoviePy
MOVIEPY_AVAILABLE = initialize_moviepy()

# Set page configuration
st.set_page_config(
    page_title="Uganda Bird Spotter",
    page_icon="🦅",
    layout="wide"
)

# Custom CSS with Glass Morphism
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
    }
    .title-image {
        width: 80px;
        height: 80px;
        border-radius: 16px;
        object-fit: cover;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .glass-upload {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 30px;
        text-align: center;
        margin: 20px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
    }
    .info-box {
        background: rgba(255, 248, 225, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #FFD700;
    }
    .warning-box {
        background: rgba(255, 193, 7, 0.2);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STORY TEMPLATES AND GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATES = [
    "Deep in Uganda's lush forests, the {name} flashes its {color_phrase} feathers. {desc} It dances on branches at dawn, a true jewel of the Pearl of Africa.",
    "Along the Nile's banks, the {name} stands tall with {color_phrase} plumage. {desc} Fishermen smile when they hear its melodic call at sunrise.",
    "In Queen Elizabeth National Park, the {name} soars above acacia trees. {desc} Its {color_phrase} wings catch the golden light of the savanna.",
    "Near Lake Victoria, the {name} perches quietly. {desc} Children in fishing villages know its {color_phrase} colors mean good luck for the day.",
    "High in the Rwenzori Mountains, the {name} sings through mist. {desc} Its {color_phrase} feathers shine like emeralds in the cloud forest.",
]

class BirdStoryGenerator:
    def __init__(self, templates): 
        self.templates = templates
    
    def __call__(self, name, description="", colors=None):
        if colors is None: 
            colors = []
        color_phrase = ", ".join([c.strip() for c in colors]) if colors else "vibrant"
        desc = description.strip().capitalize() if description else "A fascinating bird with unique habits."
        tmpl = random.choice(self.templates)
        return tmpl.format(name=name, color_phrase=color_phrase, desc=desc)

# ========== FALLBACK VIDEO GENERATOR (No MoviePy) ==========
class SimpleVideoGenerator:
    def __init__(self):
        self.story_generator = BirdStoryGenerator(TEMPLATES)
        
    def natural_tts(self, text, filename):
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            return filename
        except Exception as e:
            st.error(f"❌ Error generating speech: {e}")
            return None

    def create_placeholder_image(self, species_name, output_path, variation=0):
        """Create a placeholder image using PIL"""
        try:
            width, height = 600, 400
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            colors = [
                (70, 130, 180), (60, 179, 113), (186, 85, 211),
                (255, 165, 0), (106, 90, 205)
            ]
            
            bg_color = colors[variation % len(colors)]
            draw.rectangle([0, 0, width, height], fill=bg_color)
            
            # Add text
            text = species_name
            try:
                font = ImageFont.truetype("Arial", size=24)
                small_font = ImageFont.truetype("Arial", size=16)
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = (width - text_width) // 2
                text_y = (height - 50) // 2
                
                draw.text((text_x, text_y), text, fill='white', font=font)
                draw.text((200, 300), f"Image {variation + 1}", fill=(200, 200, 200), font=small_font)
            except:
                draw.text((width//2 - 100, height//2 - 10), text, fill='white')
                draw.text((200, 300), f"Image {variation + 1}", fill=(200, 200, 200))
            
            # Add bird shape
            center_x, center_y = 300, 150
            draw.ellipse([center_x-40, center_y-25, center_x+40, center_y+25], fill='white')
            draw.ellipse([center_x-20, center_y-45, center_x+20, center_y-5], fill='white')
            
            img.save(output_path, "JPEG")
            return True
        except Exception as e:
            st.error(f"❌ Error creating image: {e}")
            return False

    def generate_story_content(self, species_name):
        """Generate story and audio without video"""
        try:
            # Get bird info
            bird_info = bird_db.get(species_name, {})
            description = bird_info.get("desc", "A beautiful bird native to Uganda.")
            colors = bird_info.get("colors", ["colorful"])
            
            # Generate story
            story_text = self.story_generator(species_name, description, colors)
            
            # Generate audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                audio_path = temp_audio.name
            
            audio_path = self.natural_tts(story_text, audio_path)
            
            if audio_path and os.path.exists(audio_path):
                # Create placeholder images
                image_paths = []
                for i in range(3):
                    placeholder_path = f"./temp_img_{species_name.replace(' ', '_')}_{i}.jpg"
                    if self.create_placeholder_image(species_name, placeholder_path, i):
                        image_paths.append(placeholder_path)
                
                return story_text, audio_path, image_paths
            else:
                return story_text, None, []
                
        except Exception as e:
            st.error(f"❌ Story generation error: {e}")
            return None, None, []

# ========== ADVANCED VIDEO GENERATOR (With MoviePy) ==========
class AdvancedVideoGenerator:
    def __init__(self):
        self.moviepy_available = MOVIEPY_AVAILABLE
        self.story_generator = BirdStoryGenerator(TEMPLATES)
        
    def natural_tts(self, text, filename):
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            return filename
        except Exception as e:
            st.error(f"❌ Error generating speech: {e}")
            return None

    def create_video_with_moviepy(self, images, audio_path, output_path):
        """Create video using MoviePy"""
        try:
            from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips, concatenate_audioclips
            from moviepy.audio.fx.all import audio_fadein, audio_fadeout
            
            # Load audio
            raw_audio = AudioFileClip(audio_path)
            narration = audio_fadein(raw_audio, 0.6)
            narration = audio_fadeout(narration, 1.2)

            # Create clips
            img_duration = 4.0
            total_duration = img_duration * len(images)

            # Adjust audio
            if narration.duration < total_duration:
                loops = int(total_duration / narration.duration) + 1
                narration = concatenate_audioclips([narration] * loops).subclip(0, total_duration)
            else:
                narration = narration.subclip(0, total_duration)

            # Create image clips with simple effects
            clips = []
            for img in images:
                clip = ImageClip(img).set_duration(img_duration)
                clip = clip.fadein(0.5).fadeout(0.5)
                clips.append(clip)

            # Combine everything
            video = concatenate_videoclips(clips, method="compose").set_audio(narration)
            video = video.resize(height=480)  # Smaller for faster processing
            
            # Write video
            video.write_videofile(
                output_path, 
                fps=15,  # Lower fps for faster processing
                codec="libx264", 
                audio_codec="aac", 
                preset="fast",  # Faster encoding
                verbose=False,
                logger=None
            )
            
            # Clean up
            for clip in clips:
                clip.close()
            video.close()
            raw_audio.close()
            narration.close()
            
            return output_path
            
        except Exception as e:
            st.error(f"❌ MoviePy video creation failed: {e}")
            return None

    def generate_story_video(self, species_name):
        """Generate complete story video"""
        try:
            # Get bird info
            bird_info = bird_db.get(species_name, {})
            description = bird_info.get("desc", "A beautiful bird native to Uganda.")
            colors = bird_info.get("colors", ["colorful"])
            
            # Generate story
            story_text = self.story_generator(species_name, description, colors)
            
            # Generate audio
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                audio_path = temp_audio.name
            
            audio_path = self.natural_tts(story_text, audio_path)
            
            if not audio_path:
                return None, None, []

            # Create placeholder images
            image_paths = []
            for i in range(3):
                placeholder_path = f"./temp_img_{species_name.replace(' ', '_')}_{i}.jpg"
                if self.create_placeholder_image(species_name, placeholder_path, i):
                    image_paths.append(placeholder_path)

            if not image_paths:
                return story_text, audio_path, []

            # Create video if MoviePy is available
            video_path = None
            if self.moviepy_available:
                with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                    video_path = temp_video.name
                
                video_path = self.create_video_with_moviepy(image_paths, audio_path, video_path)

            return story_text, audio_path, image_paths, video_path
            
        except Exception as e:
            st.error(f"❌ Video generation error: {e}")
            return None, None, [], None

    def create_placeholder_image(self, species_name, output_path, variation=0):
        """Create placeholder image (same as simple generator)"""
        try:
            width, height = 600, 400
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            colors = [
                (70, 130, 180), (60, 179, 113), (186, 85, 211),
                (255, 165, 0), (106, 90, 205)
            ]
            
            bg_color = colors[variation % len(colors)]
            draw.rectangle([0, 0, width, height], fill=bg_color)
            
            text = species_name
            try:
                font = ImageFont.truetype("Arial", size=24)
                small_font = ImageFont.truetype("Arial", size=16)
                
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_x = (width - text_width) // 2
                text_y = (height - 50) // 2
                
                draw.text((text_x, text_y), text, fill='white', font=font)
                draw.text((200, 300), f"Image {variation + 1}", fill=(200, 200, 200), font=small_font)
            except:
                draw.text((width//2 - 100, height//2 - 10), text, fill='white')
                draw.text((200, 300), f"Image {variation + 1}", fill=(200, 200, 200))
            
            center_x, center_y = 300, 150
            draw.ellipse([center_x-40, center_y-25, center_x+40, center_y+25], fill='white')
            draw.ellipse([center_x-20, center_y-45, center_x+20, center_y-5], fill='white')
            
            img.save(output_path, "JPEG")
            return True
        except Exception as e:
            return False

# ========== BIRD DATABASE ==========
bird_db = {
    "African Fish Eagle": {
        "desc": "A majestic bird of prey found near water bodies. Known for its distinctive cry and excellent fishing skills.",
        "colors": ["white", "brown", "black"]
    },
    "Grey Crowned Crane": {
        "desc": "National bird of Uganda with golden crown. Performs beautiful mating dances in wetlands.",
        "colors": ["grey", "white", "gold"]
    },
    "Shoebill Stork": {
        "desc": "Large stork-like bird with shoe-shaped bill. Known for its prehistoric appearance and patient hunting.",
        "colors": ["blue-grey", "white"]
    },
    "Lilac-breasted Roller": {
        "desc": "Colorful bird with vibrant plumage. Famous for its acrobatic flight displays during mating season.",
        "colors": ["lilac", "blue", "green", "brown"]
    },
    "African Darter": {
        "desc": "Also known as snakebird for its long slender neck. Excellent swimmer and diver.",
        "colors": ["black", "brown", "white"]
    }
}

# ========== BIRD DETECTION MODEL ==========
class BirdDetectionModel:
    def __init__(self):
        self.bird_species = list(bird_db.keys())
    
    def predict_bird_species(self, image):
        """Simple prediction returning random species for demo"""
        random_species = random.choice(self.bird_species)
        return [((100, 100, 200, 200), 0.85)], [(random_species, 0.92)], np.array(image)

# ========== MAIN APP ==========
def main():
    st.title("🦅 Uganda Bird Spotter")
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.bird_model = BirdDetectionModel()
        st.session_state.video_generator = AdvancedVideoGenerator() if MOVIEPY_AVAILABLE else SimpleVideoGenerator()
        st.session_state.simple_generator = SimpleVideoGenerator()
        st.session_state.detection_complete = False
        st.session_state.selected_species = None

    # Status display
    col1, col2 = st.columns(2)
    with col1:
        if MOVIEPY_AVAILABLE:
            st.success("🎬 Video Generation: ✅ Available")
        else:
            st.warning("🎬 Video Generation: ⚠️ Basic Mode (Audio + Images)")
    
    with col2:
        st.info(f"📚 Bird Species: {len(bird_db)} available")

    # Image upload
    st.subheader("📸 Upload Bird Image")
    uploaded_file = st.file_uploader("Choose a bird image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Identify Bird Species"):
            with st.spinner("Analyzing..."):
                detections, classifications, _ = st.session_state.bird_model.predict_bird_species(image)
                st.session_state.detection_complete = True
                if classifications:
                    st.session_state.selected_species = classifications[0][0]
                    st.success(f"✅ Identified: **{st.session_state.selected_species}**")

    # Manual species selection
    st.subheader("🎬 Generate Bird Story")
    selected_species = st.selectbox(
        "Select bird species:",
        options=list(bird_db.keys()),
        index=list(bird_db.keys()).index(st.session_state.selected_species) if st.session_state.selected_species else 0
    )

    if st.button("📖 Generate Story & Content"):
        with st.spinner("Creating amazing bird content..."):
            if MOVIEPY_AVAILABLE:
                # Try advanced generator first
                story_text, audio_path, image_paths, video_path = st.session_state.video_generator.generate_story_video(selected_species)
            else:
                # Use simple generator
                story_text, audio_path, image_paths = st.session_state.simple_generator.generate_story_content(selected_species)
                video_path = None

            # Display results
            if story_text:
                st.markdown(f'<div class="info-box"><strong>📖 Story:</strong><br>{story_text}</div>', unsafe_allow_html=True)
                
                # Display images
                if image_paths:
                    st.subheader("🖼️ Generated Images")
                    cols = st.columns(len(image_paths))
                    for idx, img_path in enumerate(image_paths):
                        with cols[idx]:
                            st.image(img_path, use_column_width=True)
                
                # Audio player
                if audio_path and os.path.exists(audio_path):
                    st.subheader("🔊 Audio Story")
                    with open(audio_path, "rb") as audio_file:
                        audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/mp3")
                    
                    # Download audio
                    st.download_button(
                        label="📥 Download Audio",
                        data=audio_bytes,
                        file_name=f"{selected_species.replace(' ', '_')}_story.mp3",
                        mime="audio/mp3"
                    )
                
                # Video display
                if video_path and os.path.exists(video_path):
                    st.subheader("🎬 Story Video")
                    with open(video_path, "rb") as video_file:
                        video_bytes = video_file.read()
                    st.video(video_bytes)
                    
                    # Download video
                    st.download_button(
                        label="📥 Download Video",
                        data=video_bytes,
                        file_name=f"{selected_species.replace(' ', '_')}_story.mp4",
                        mime="video/mp4"
                    )
                elif MOVIEPY_AVAILABLE:
                    st.warning("🎬 Video generation failed, but audio and images are available!")
                
                # Download story text
                st.download_button(
                    label="📝 Download Story Text",
                    data=story_text.encode('utf-8'),
                    file_name=f"{selected_species.replace(' ', '_')}_story.txt",
                    mime="text/plain"
                )
            else:
                st.error("❌ Failed to generate content")

    # Instructions for MoviePy issues
    if not MOVIEPY_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
        <strong>💡 Video Generation Tips:</strong><br>
        • The app is running in basic mode with audio and images<br>
        • For full video features, ensure MoviePy dependencies are installed<br>
        • Manual install: <code>pip install moviepy imageio-ffmpeg</code><br>
        • Refresh the page after installation
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()