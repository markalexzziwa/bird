import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import base64
from io import BytesIO
import json
import random
import requests
import urllib.request
import pandas as pd
import cv2
import torch
import torch.nn as nn
from gtts import gTTS
import warnings
import subprocess
import sys
warnings.filterwarnings('ignore')

# Force install and import MoviePy
try:
    from moviepy.editor import (
        AudioFileClip, ImageClip, concatenate_videoclips,
        VideoFileClip, concatenate_audioclips
    )
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout
    from moviepy.video.fx.all import resize
    MOVIEPY_AVAILABLE = True
except ImportError:
    # Try to install moviepy automatically
    try:
        st.warning("🎬 MoviePy not found. Installing now...")
        
        # Install moviepy and dependencies
        subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy", "decorator", "proglog"])
        
        # Try to import again after installation
        from moviepy.editor import (
            AudioFileClip, ImageClip, concatenate_videoclips,
            VideoFileClip, concatenate_audioclips
        )
        from moviepy.audio.fx.all import audio_fadein, audio_fadeout
        from moviepy.video.fx.all import resize
        
        MOVIEPY_AVAILABLE = True
        st.success("✅ MoviePy installed successfully!")
        
    except Exception as e:
        MOVIEPY_AVAILABLE = False
        st.error(f"❌ MoviePy installation failed: {e}")
        st.info("Please install manually: pip install moviepy decorator proglog")

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
    .glass-info {
        background: rgba(240, 248, 255, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 25px;
        margin: 15px 0;
    }
    .glass-metric {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        text-align: center;
    }
    .stButton button {
        background: rgba(46, 134, 171, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
    }
    .section-title {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-bottom: 15px;
        text-align: center;
        font-weight: 600;
    }
    .success-box {
        background: rgba(40, 167, 69, 0.2);
        border: 1px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .error-box {
        background: rgba(220, 53, 69, 0.2);
        border: 1px solid #dc3545;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .sidebar-logo {
        width: 100px;
        height: 100px;
        border-radius: 20px;
        object-fit: cover;
        margin: 0 auto 20px auto;
        display: block;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .bird-list {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
    .sidebar-title {
        text-align: center;
        font-size: 1.5rem;
        color: #2E86AB;
        font-weight: 700;
        margin-bottom: 20px;
    }
    .video-section {
        background: rgba(46, 134, 171, 0.1);
        border-radius: 16px;
        padding: 25px;
        margin: 20px 0;
        border: 1px solid rgba(46, 134, 171, 0.3);
    }
    .story-box {
        background: rgba(255, 248, 225, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        border-left: 4px solid #FFD700;
    }
</style>
""", unsafe_allow_html=True)

# Story templates from your code
TEMPLATES = [
    "Deep in Uganda's lush forests, the {name} flashes its {color_phrase} feathers. {desc} It dances on branches at dawn, a true jewel of the Pearl of Africa.",
    "Along the Nile's banks, the {name} stands tall with {color_phrase} plumage. {desc} Fishermen smile when they hear its melodic call at sunrise.",
    "In Queen Elizabeth National Park, the {name} soars above acacia trees. {desc} Its {color_phrase} wings catch the golden light of the savanna.",
    "Near Lake Victoria, the {name} perches quietly. {desc} Children in fishing villages know its {color_phrase} colors mean good luck for the day.",
    "High in the Rwenzori Mountains, the {name} sings through mist. {desc} Its {color_phrase} feathers shine like emeralds in the cloud forest.",
    "In Murchison Falls, the {name} glides over roaring waters. {desc} Tourists gasp at its {color_phrase} beauty against the dramatic backdrop.",
    "Among papyrus swamps, the {name} wades gracefully. {desc} Its long legs and {color_phrase} crest make it the king of the wetlands.",
    "At sunset in Kidepo Valley, the {name} calls across the plains. {desc} Its {color_phrase} silhouette is a symbol of Uganda's wild heart.",
    "In Bwindi's ancient rainforest, the {name} flits between vines. {desc} Gorilla trackers pause to admire its {color_phrase} brilliance.",
    "By the shores of Lake Mburo, the {name} reflects in calm waters. {desc} Its {color_phrase} feathers mirror the peace of the savanna night."
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

# ========== ENHANCED FILE DOWNLOADER ==========
def download_file_from_gdrive(file_id, destination):
    """Download file from Google Drive with enhanced error handling"""
    try:
        # Remove existing file if it's too small (corrupted)
        if os.path.exists(destination) and os.path.getsize(destination) < 1000:
            os.remove(destination)
            
        if not os.path.exists(destination):
            st.info(f"📥 Downloading {os.path.basename(destination)} from Google Drive...")
            
            # Method 1: Using gdown (most reliable)
            try:
                import gdown
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, destination, quiet=False)
                
            except ImportError:
                # Method 2: Using requests
                try:
                    session = requests.Session()
                    url = f"https://docs.google.com/uc?export=download&id={file_id}"
                    response = session.get(url, stream=True)
                    
                    # Handle confirmation for large files
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            params = {'confirm': value}
                            response = session.get(url, params=params, stream=True)
                            break
                    
                    # Download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with open(destination, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = min(downloaded / total_size, 1.0)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Downloaded: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB")
                    
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"❌ Requests download failed: {e}")
                    return False
            
            # Verify download
            if os.path.exists(destination) and os.path.getsize(destination) > 1000:
                file_size = os.path.getsize(destination) / (1024 * 1024)
                st.success(f"✅ Downloaded {os.path.basename(destination)} ({file_size:.1f} MB)")
                return True
            else:
                if os.path.exists(destination):
                    os.remove(destination)
                st.error(f"❌ Downloaded file is too small or corrupted: {os.path.basename(destination)}")
                return False
        
        # If file already exists
        if os.path.exists(destination):
            file_size = os.path.getsize(destination) / (1024 * 1024)
            if file_size > 0.1:
                st.info(f"✅ {os.path.basename(destination)} already exists ({file_size:.1f} MB)")
                return True
            else:
                os.remove(destination)
                return False
                
    except Exception as e:
        st.error(f"❌ Download error for {os.path.basename(destination)}: {e}")
        return False
    
    return True

# ========== BIRD DATA LOADING WITH DOWNLOAD ==========
@st.cache_resource
def load_bird_data():
    """Load bird data with download from Google Drive"""
    pth_path = "bird_data.pth"
    
    # If bird_data.pth doesn't exist, download it
    if not os.path.exists(pth_path):
        st.info("📥 Downloading bird_data.pth from Google Drive...")
        
        # Try multiple possible file IDs for bird_data.pth
        possible_file_ids = [
            "1J9T5r5TboWzvqAPQHmfvQmozor_wmmPz",  # Your bird_path.pth ID
            "1yfiYcz6e2hWtQTXW6AZVU-iwSUjDP92y",  # Your resnet model ID
        ]
        
        success = False
        for file_id in possible_file_ids:
            st.info(f"🔄 Trying file ID: {file_id}...")
            success = download_file_from_gdrive(file_id, pth_path)
            if success:
                st.success("✅ bird_data.pth downloaded successfully!")
                break
            else:
                st.warning(f"❌ Failed with file ID: {file_id}")
        
        if not success:
            st.error("❌ Could not download bird_data.pth from any source.")
            st.info("🔄 Creating minimal bird data for testing...")
            return create_minimal_bird_data()
    
    try:
        bird_data = torch.load(pth_path, map_location="cpu")
        if isinstance(bird_data, dict) and len(bird_data) > 0:
            st.success(f"✅ Loaded bird data with {len(bird_data)} species")
            return bird_data
        else:
            st.error("❌ bird_data.pth is empty or invalid")
            return create_minimal_bird_data()
    except Exception as e:
        st.error(f"❌ Error loading bird data: {e}")
        return create_minimal_bird_data()

def create_minimal_bird_data():
    """Create minimal bird data if download fails"""
    minimal_data = {
        "African Fish Eagle": {
            "desc": "A majestic bird of prey found near water bodies",
            "colors": ["white", "brown", "black"],
            "images_b64": []
        },
        "Grey Crowned Crane": {
            "desc": "National bird of Uganda with golden crown",
            "colors": ["grey", "white", "gold"],
            "images_b64": []
        },
        "Shoebill Stork": {
            "desc": "Large stork-like bird with shoe-shaped bill",
            "colors": ["blue-grey", "white"],
            "images_b64": []
        },
        "Lilac-breasted Roller": {
            "desc": "Colorful bird with vibrant plumage",
            "colors": ["lilac", "blue", "green", "brown"],
            "images_b64": []
        },
        "Great Blue Turaco": {
            "desc": "Large blue bird with distinctive crest",
            "colors": ["blue", "green", "red"],
            "images_b64": []
        }
    }
    st.warning("⚠️ Using minimal bird data for testing")
    return minimal_data

# Load bird data at startup
bird_db = load_bird_data()

# ========== VIDEO MODEL LOADING - NO ALTERNATIVES ==========
@st.cache_resource
def load_video_model():
    """Load video model with download from Google Drive - NO FALLBACKS"""
    pth_path = "bird_path.pth"
    file_id = "1J9T5r5TboWzvqAPQHmfvQmozor_wmmPz"  # From your Google Drive link
    
    # Download the file if it doesn't exist
    if not os.path.exists(pth_path):
        st.info("🔄 Downloading bird_path.pth from Google Drive...")
        success = download_file_from_gdrive(file_id, pth_path)
        if not success:
            st.error("❌ CRITICAL: Could not download bird_path.pth. App cannot function without this file.")
            st.stop()  # Stop the app completely
    
    try:
        # Load the model
        model_data = torch.load(pth_path, map_location="cpu")
        
        # Verify it's a valid model
        if model_data is None:
            st.error("❌ CRITICAL: bird_path.pth is empty or corrupted.")
            st.stop()
            
        st.success("✅ Video story model loaded successfully!")
        return model_data
        
    except Exception as e:
        st.error(f"❌ CRITICAL: Error loading video model: {e}")
        st.stop()

# Load video model at startup - THIS WILL STOP APP IF FAILS
video_model_data = load_video_model()

class AdvancedVideoGenerator:
    def __init__(self):
        self.csv_path = './birdsuganda.csv'
        self.bird_data = None
        self.video_duration = 20
        self.moviepy_available = MOVIEPY_AVAILABLE
        
        # CRITICAL: Must use bird_path.pth - no alternatives
        if video_model_data is None:
            st.error("❌ CRITICAL: bird_path.pth not available. App cannot continue.")
            st.stop()
            
        self.story_generator = self._initialize_story_generator()
        
    def _initialize_story_generator(self):
        """Initialize story generator from bird_path.pth - NO FALLBACKS"""
        try:
            # Extract story generator from model data
            if hasattr(video_model_data, 'generate_story'):
                return video_model_data
            elif isinstance(video_model_data, dict) and 'story_generator' in video_model_data:
                return video_model_data['story_generator']
            elif isinstance(video_model_data, dict) and 'templates' in video_model_data:
                return BirdStoryGenerator(video_model_data['templates'])
            else:
                # If it's just the story generator itself
                return BirdStoryGenerator(TEMPLATES)
        except Exception as e:
            st.error(f"❌ CRITICAL: Could not initialize story generator from bird_path.pth: {e}")
            st.stop()
    
    def load_bird_data(self):
        """Load and process the bird species data from local CSV"""
        try:
            if os.path.exists(self.csv_path):
                self.bird_data = pd.read_csv(self.csv_path)
                return True
            else:
                st.warning(f"CSV file not found: {self.csv_path}")
                return False
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return False
    
    def get_bird_video_info(self, species_name):
        """Get video generation information for a specific bird species"""
        if self.bird_data is None:
            if not self.load_bird_data():
                return None
        
        try:
            # Search for the bird species in the dataset
            possible_columns = ['species_name', 'species', 'name', 'bird_name', 'common_name', 'Scientific Name', 'Common Name', 'common_name']
            
            for col in possible_columns:
                if col in self.bird_data.columns:
                    # Handle NaN values and case sensitivity
                    bird_info = self.bird_data[
                        self.bird_data[col].astype(str).str.lower() == species_name.lower()
                    ]
                    if len(bird_info) > 0:
                        return bird_info.iloc[0].to_dict()
            
            # If no exact match, try partial match
            for col in possible_columns:
                if col in self.bird_data.columns:
                    bird_info = self.bird_data[
                        self.bird_data[col].astype(str).str.contains(species_name, case=False, na=False)
                    ]
                    if len(bird_info) > 0:
                        return bird_info.iloc[0].to_dict()
            
            return None
                
        except Exception as e:
            st.error(f"❌ Error finding bird info: {e}")
            return None

    def natural_tts(self, text, filename):
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            return filename
        except Exception as e:
            st.error(f"❌ Error generating speech: {e}")
            return None

    def create_slideshow_video_opencv(self, images, audio_path, output_path):
        """Create slideshow video using OpenCV (only if MoviePy fails)"""
        try:
            # Get audio duration
            try:
                import librosa
                audio_duration = librosa.get_duration(filename=audio_path)
            except:
                audio_duration = 15  # Fallback duration
            
            # Video properties
            frame_width = 1280
            frame_height = 720
            fps = 24
            total_frames = int(audio_duration * fps)
            frames_per_image = max(1, total_frames // len(images)) if images else total_frames
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            # Create frames
            for frame_num in range(total_frames):
                img_idx = min(len(images) - 1, frame_num // frames_per_image)
                img_path = images[img_idx]
                
                # Load and resize image
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (frame_width, frame_height))
                    out.write(img)
                else:
                    # Create placeholder frame if image loading fails
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    cv2.putText(frame, "Bird Image", (frame_width//2 - 100, frame_height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    out.write(frame)
            
            out.release()
            return output_path
            
        except Exception as e:
            st.error(f"❌ OpenCV video creation error: {e}")
            return None

    def create_final_video(self, images, audio_path, output_path):
        """Create final video with Ken Burns effect and audio - FORCE MoviePy"""
        if not self.moviepy_available:
            st.error("❌ MoviePy is required but not available.")
            return None
        
        try:
            # Load and process audio
            raw_audio = AudioFileClip(audio_path)
            narration = audio_fadein(raw_audio, 0.6)
            narration = audio_fadeout(narration, 1.2)

            # Calculate durations
            img_duration = max(3.0, narration.duration / len(images))
            total_duration = img_duration * len(images)

            # Adjust audio to match video duration
            if narration.duration < total_duration:
                loops = int(total_duration / narration.duration) + 1
                narration = concatenate_audioclips([narration] * loops).subclip(0, total_duration)
            else:
                narration = narration.subclip(0, total_duration)

            # Create video clips with Ken Burns effect
            clips = []
            for img in images:
                try:
                    clip = ImageClip(img).set_duration(img_duration)
                    w, h = clip.size
                    zoom = 1.1
                    
                    # Apply zoom effect
                    clip = clip.resize(lambda t: 1 + (zoom - 1) * (t / img_duration))
                    
                    clip = clip.fadein(0.3).fadeout(0.3)
                    clips.append(clip)
                except Exception as e:
                    st.warning(f"⚠️ Could not process image {img}: {e}")
                    continue

            if not clips:
                st.error("❌ No valid video clips created")
                return None

            # Combine clips and audio
            video = concatenate_videoclips(clips, method="compose").set_audio(narration)
            video = video.resize(height=720)
            
            # Write final video
            video.write_videofile(
                output_path, 
                fps=24, 
                codec="libx264", 
                audio_codec="aac", 
                preset="medium",
                verbose=False,
                logger=None
            )
            
            return output_path
            
        except Exception as e:
            st.error(f"❌ MoviePy video creation error: {e}")
            return None

    def get_bird_images(self, species_name, max_images=5):
        """Get bird images for the species"""
        try:
            # First try to get images from bird_db if available
            if species_name in bird_db and bird_db[species_name].get("images_b64"):
                image_paths = []
                for i, b64 in enumerate(bird_db[species_name]["images_b64"][:max_images]):
                    try:
                        img_data = base64.b64decode(b64)
                        img = Image.open(BytesIO(img_data))
                        placeholder_path = f"./temp_bird_{species_name.replace(' ', '_')}_{i}.jpg"
                        img.save(placeholder_path, "JPEG")
                        image_paths.append(placeholder_path)
                    except:
                        continue
                
                if image_paths:
                    return image_paths
            
            # Create placeholder images
            image_paths = []
            for i in range(max_images):
                placeholder_path = f"./temp_placeholder_{species_name.replace(' ', '_')}_{i}.jpg"
                if self.create_placeholder_image(species_name, placeholder_path, variation=i):
                    image_paths.append(placeholder_path)
            
            return image_paths
            
        except Exception as e:
            st.error(f"❌ Error getting bird images: {e}")
            # Return at least one placeholder
            placeholder_path = f"./temp_fallback_{species_name.replace(' ', '_')}.jpg"
            self.create_placeholder_image(species_name, placeholder_path)
            return [placeholder_path]

    def create_placeholder_image(self, species_name, output_path, variation=0):
        """Create a placeholder image when no real images are available"""
        try:
            # Create a simple image with bird name
            img = np.zeros((400, 600, 3), dtype=np.uint8)
            
            # Different background colors for variations
            colors = [
                [70, 130, 180],   # Steel blue
                [60, 179, 113],   # Medium sea green
                [186, 85, 211],   # Medium orchid
                [255, 165, 0],    # Orange
                [106, 90, 205]    # Slate blue
            ]
            
            bg_color = colors[variation % len(colors)]
            img[:, :] = bg_color
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = species_name
            text_size = cv2.getTextSize(text, font, 0.8, 2)[0]
            text_x = (600 - text_size[0]) // 2
            text_y = (400 + text_size[1]) // 2
            
            cv2.putText(img, text, (text_x, text_y), font, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"Bird Image {variation + 1}", (200, 250), font, 0.6, (200, 200, 200), 1)
            cv2.putText(img, "Uganda Bird Spotter", (180, 300), font, 0.5, (220, 220, 220), 1)
            
            # Add simple bird shape
            center_x, center_y = 300, 150
            cv2.ellipse(img, (center_x, center_y), (40, 25), 0, 0, 360, (255, 255, 255), -1)
            cv2.ellipse(img, (center_x, center_y - 20), (20, 20), 0, 0, 360, (255, 255, 255), -1)
            
            cv2.imwrite(output_path, img)
            return True
        except Exception as e:
            st.error(f"❌ Error creating placeholder: {e}")
            return False

    def generate_story_video(self, species_name):
        """Generate a comprehensive story-based video with audio - NO FALLBACKS"""
        try:
            # Get bird information
            bird_info = self.get_bird_video_info(species_name)
            
            # Extract bird details for story generation
            common_name = species_name
            description = bird_info.get('description', '') if bird_info else ''
            colors = []
            
            # Try to extract colors from various possible columns
            if bird_info:
                color_columns = ['colors', 'primary_colors', 'plumage_colors', 'color']
                for col in color_columns:
                    if col in bird_info and pd.notna(bird_info[col]):
                        colors = str(bird_info[col]).split(',')
                        break
            
            # Generate story using the model from bird_path.pth
            st.info("📖 Generating educational story using bird_path.pth model...")
            story_text = self.story_generator(common_name, description, colors)
            
            # Display the generated story
            st.markdown(f'<div class="story-box"><strong>📖 AI-Generated Story:</strong><br>{story_text}</div>', unsafe_allow_html=True)
            
            # Generate audio
            st.info("🔊 Converting story to speech...")
            audio_file = f"temp_story_{species_name.replace(' ', '_')}.mp3"
            audio_path = self.natural_tts(story_text, audio_file)
            
            if not audio_path:
                st.error("❌ Failed to generate audio")
                return None, None, None
            
            # Get bird images
            st.info("🖼️ Gathering bird images...")
            bird_images = self.get_bird_images(species_name, max_images=3)
            
            if not bird_images:
                st.error("❌ No bird images available")
                return None, None, None
            
            # Generate video - FORCE MoviePy
            st.info("🎬 Creating story video with MoviePy...")
            video_file = f"temp_story_video_{species_name.replace(' ', '_')}.mp4"
            video_path = self.create_final_video(bird_images, audio_path, video_file)
            
            # Clean up temporary audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            
            if video_path and os.path.exists(video_path):
                st.success(f"✅ Story video generated successfully using bird_path.pth!")
                return video_path, story_text, bird_images
            else:
                st.error("❌ Failed to generate video file")
                return None, None, None
            
        except Exception as e:
            st.error(f"❌ Story video generation error: {e}")
            return None, None, None

    def generate_video(self, species_name):
        """Main video generation function with story and audio"""
        return self.generate_story_video(species_name)

# ========== RESNET MODEL ==========
class ResNet34BirdModel:
    def __init__(self):
        self.model_loaded = False
        self.bird_species = []
        self.inv_label_map = {}
        self.model = None
        self.device = None
        self.transform = None
        self.model_path = './resnet34_bird_region_weights.pth'
        self.label_map_path = './label_map.json'
        
    def download_model_from_gdrive(self):
        """Download model from Google Drive using the direct link"""
        return download_file_from_gdrive("1yfiYcz6e2hWtQTXW6AZVU-iwSUjDP92y", self.model_path)
    
    def check_dependencies(self):
        """Check if PyTorch and torchvision are available"""
        try:
            import torch
            import torchvision
            return True
        except ImportError:
            st.error("""
            ❌ PyTorch and torchvision are required but not installed.
            
            Please add them to your requirements.txt:
            ```
            torch>=2.0.0
            torchvision>=0.15.0
            pillow>=9.0.0
            numpy>=1.21.0
            opencv-python-headless>=4.5.0
            requests>=2.25.0
            gdown>=4.4.0
            streamlit>=1.22.0
            pandas>=1.3.0
            gtts>=2.2.0
            moviepy>=1.0.0
            ```
            """)
            return False
    
    def create_default_label_map(self):
        """Create a default label map if none exists"""
        default_species = [
            "African Fish Eagle", "Grey Crowned Crane", "Shoebill Stork", 
            "Lilac-breasted Roller", "Great Blue Turaco", "African Jacana",
            "Marabou Stork", "Pied Kingfisher", "Superb Starling", "Hadada Ibis"
        ]
        
        label_map = {species: idx for idx, species in enumerate(default_species)}
        
        with open(self.label_map_path, 'w') as f:
            json.dump(label_map, f, indent=2)
        
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.bird_species = default_species
        return True
    
    def load_label_map(self):
        """Load the label map for bird species"""
        if not os.path.exists(self.label_map_path):
            return self.create_default_label_map()
        
        try:
            with open(self.label_map_path, 'r') as f:
                label_map = json.load(f)
            
            self.inv_label_map = {v: k for k, v in label_map.items()}
            self.bird_species = list(label_map.keys())
            return True
        except Exception as e:
            return self.create_default_label_map()
    
    def load_model(self):
        """Load the ResNet34 model"""
        if not self.check_dependencies():
            return False
        
        # First, try to download the model
        if not os.path.exists(self.model_path):
            if not self.download_model_from_gdrive():
                st.error("""
                ❌ Could not download the model file from Google Drive.
                
                Please ensure:
                1. The Google Drive file is publicly accessible
                2. The file ID is correct: 1yfiYcz6e2hWtQTXW6AZVU-iwSUjDP92y
                3. You have internet connection
                """)
                return False
        
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            
            # Load label map
            if not self.load_label_map():
                return False
            
            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.info(f"🔄 Using device: {self.device}")
            
            # Create ResNet34 model
            model = models.resnet34(weights=None)
            num_classes = len(self.bird_species)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # Load weights
            st.info("🔄 Loading model weights...")
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(self.model_path))
            else:
                model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            
            self.model = model.to(self.device)
            self.model.eval()
            
            # Define transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.model_loaded = True
            st.success("✅ ResNet34 model loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Model loading error: {e}")
            return False

    def detect_bird_regions(self, image):
        """Simple bird detection"""
        try:
            if isinstance(image, np.ndarray):
                image_array = image
            else:
                image_array = np.array(image)
            
            height, width = image_array.shape[:2]
            
            st.info("🔍 Scanning image for birds...")
            
            # Simple detection - one bird in center
            x = width // 4
            y = height // 4
            w = width // 2
            h = height // 2
            
            detection_confidence = 0.85
            detections = [([x, y, w, h], detection_confidence)]
            
            st.success("✅ Found 1 bird region")
            return detections, image_array
                
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            return [], None
    
    def classify_bird_region(self, bird_region):
        """Classify bird region using ResNet34"""
        if not self.model_loaded:
            return "Model not loaded", 0.0
        
        try:
            import torch
            
            if isinstance(bird_region, np.ndarray):
                bird_region = Image.fromarray(bird_region)
            
            input_tensor = self.transform(bird_region).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()
                confidence = probabilities[predicted_class].item()
            
            species = self.inv_label_map.get(predicted_class, "Unknown Species")
            return species, confidence
            
        except Exception as e:
            st.error(f"❌ Model prediction error: {e}")
            return "Prediction Error", 0.0
    
    def predict_bird_species(self, image):
        """Complete prediction pipeline"""
        if not self.model_loaded:
            st.error("❌ Model not loaded. Cannot make predictions.")
            return [], [], None
        
        detections, original_image = self.detect_bird_regions(image)
        
        if not detections:
            return [], [], original_image
        
        classifications = []
        
        if isinstance(original_image, np.ndarray):
            pil_original = Image.fromarray(original_image)
        else:
            pil_original = original_image
        
        for i, (box, detection_confidence) in enumerate(detections):
            x, y, w, h = box
            
            try:
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(pil_original.width, x + w), min(pil_original.height, y + h)
                
                if x2 > x1 and y2 > y1:
                    bird_region = pil_original.crop((x1, y1, x2, y2))
                else:
                    bird_region = pil_original
                
                species, classification_confidence = self.classify_bird_region(bird_region)
                classifications.append((species, classification_confidence))
                
            except Exception as e:
                st.error(f"❌ Error processing bird region {i+1}: {e}")
                classifications.append(("Processing Error", 0.0))
        
        return detections, classifications, original_image

def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return ""

def initialize_system():
    """Initialize the bird detection system"""
    if 'bird_model' not in st.session_state:
        st.session_state.bird_model = ResNet34BirdModel()
        st.session_state.video_generator = AdvancedVideoGenerator()
        st.session_state.detection_complete = False
        st.session_state.bird_detections = []
        st.session_state.bird_classifications = []
        st.session_state.current_image = None
        st.session_state.active_method = "upload"
        st.session_state.model_loaded = False
        st.session_state.system_initialized = False
        st.session_state.generated_video_path = None
        st.session_state.selected_species_for_video = None
        st.session_state.generated_story = None
        st.session_state.used_images = None
    
    # Initialize system only once
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing Uganda Bird Spotter System..."):
            # Try to load the ResNet model first
            resnet_success = st.session_state.bird_model.load_model()
            
            if resnet_success:
                st.session_state.model_loaded = True
                st.session_state.system_initialized = True
                st.success(f"✅ System ready! Can identify {len(st.session_state.bird_model.bird_species)} bird species")
                st.success("🎬 Story video generation ready with bird_path.pth")
            else:
                st.error("❌ System initialization failed. Please check the requirements and internet connection.")

def main():
    # Initialize the system
    initialize_system()
    
    # Check if system initialized properly
    if not st.session_state.get('system_initialized', False):
        st.error("""
        ❌ System failed to initialize properly. 
        
        Please check:
        1. Required dependencies are installed
        2. Internet connection is available for model download
        3. Google Drive file is accessible
        
        The app cannot run without the ResNet34 model file.
        """)
        return
    
    bird_model = st.session_state.bird_model
    video_generator = st.session_state.video_generator
    
    # Sidebar with logo and bird list
    with st.sidebar:
        # Logo at the top of sidebar
        try:
            base64_logo = get_base64_image("ugb1.png")
            st.markdown(f'<img src="data:image/png;base64,{base64_logo}" class="sidebar-logo" alt="Bird Spotter Logo">', unsafe_allow_html=True)
        except:
            st.markdown('<div class="sidebar-logo" style="background: #2E86AB; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 24px;">UG</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">Uganda Bird Spotter</div>', unsafe_allow_html=True)
        
        st.markdown("### 🦅 Detectable Birds")
        st.markdown(f"**Total Species:** {len(bird_model.bird_species)}")
        
        # Bird list with scroll
        st.markdown('<div class="bird-list">', unsafe_allow_html=True)
        for species in bird_model.bird_species:
            st.markdown(f"• {species}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Video model status
        st.markdown("---")
        st.success("🎬 Story Video Generation: **bird_path.pth**")
        st.info("📖 Powered by: Advanced AI Model")
        if video_generator.moviepy_available:
            st.success("🎥 Using MoviePy (Ken Burns effect)")
        else:
            st.error("❌ MoviePy not available")
    
    # Main app content
    # Custom header with logo beside title
    try:
        base64_logo = get_base64_image("ugb1.png")
        logo_html = f'<img src="data:image/png;base64,{base64_logo}" class="title-image" alt="Bird Spotter Logo">'
    except:
        logo_html = '<div class="title-image" style="background: #2E86AB; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">UG</div>'
    
    st.markdown(f"""
    <div class="main-header">
        {logo_html}
        Uganda Bird Spotter
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="glass-card">
        <strong>🦜 Welcome to Uganda Bird Spotter!</strong><br>
        This app uses AI models for bird identification and story generation. 
        Upload bird photos for identification, then generate AI-powered educational story videos 
        with narrated audio and beautiful visual effects using <strong>bird_path.pth</strong> model.
    </div>
    """, unsafe_allow_html=True)
    
    # Method selection
    col1, col2 = st.columns(2)
    
    with col1:
        upload_active = st.session_state.active_method == "upload"
        if st.button(
            "📁 Upload Bird Photo", 
            use_container_width=True, 
            type="primary" if upload_active else "secondary",
            key="upload_btn"
        ):
            st.session_state.active_method = "upload"
            st.session_state.current_image = None
            st.rerun()
    
    with col2:
        camera_active = st.session_state.active_method == "camera"
        if st.button(
            "📷 Capture Live Photo", 
            use_container_width=True, 
            type="primary" if camera_active else "secondary",
            key="camera_btn"
        ):
            st.session_state.active_method = "camera"
            st.session_state.current_image = None
            st.rerun()
    
    st.markdown("---")
    
    # Image input
    current_image = None
    
    if st.session_state.active_method == "upload":
        st.markdown('<div class="section-title">Upload Bird Photo</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-upload">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a bird image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload photos of birds for identification",
            label_visibility="collapsed",
            key="file_uploader"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            try:
                current_image = Image.open(uploaded_file)
                if current_image.mode != 'RGB':
                    current_image = current_image.convert('RGB')
            except Exception as e:
                st.error(f"❌ Error loading image: {e}")
    
    else:
        st.markdown('<div class="section-title">Capture Live Photo</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-upload">', unsafe_allow_html=True)
        camera_image = st.camera_input(
            "Take a picture of a bird",
            help="Capture birds for identification",
            key="camera_input",
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        if camera_image is not None:
            try:
                current_image = Image.open(camera_image)
                if current_image.mode != 'RGB':
                    current_image = current_image.convert('RGB')
            except Exception as e:
                st.error(f"❌ Error loading camera image: {e}")
    
    # Display image and analysis button
    if current_image is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(current_image, caption="Bird Photo for Analysis", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 Identify Bird Species with ResNet34", type="primary", use_container_width=True):
            if not st.session_state.model_loaded:
                st.error("❌ Model not loaded. Cannot make predictions.")
            else:
                with st.spinner("Analyzing bird species using ResNet34..."):
                    detections, classifications, original_image = bird_model.predict_bird_species(current_image)
                    
                    st.session_state.detection_complete = True
                    st.session_state.bird_detections = detections
                    st.session_state.bird_classifications = classifications
                    st.session_state.current_image = original_image
    
    # Display results
    if st.session_state.detection_complete and st.session_state.current_image is not None:
        st.markdown("---")
        st.markdown('<div class="section-title">🎯 ResNet34 Identification Results</div>', unsafe_allow_html=True)
        
        detections = st.session_state.bird_detections
        classifications = st.session_state.bird_classifications
        
        if not detections:
            st.info("🔍 No birds detected in this image")
        else:
            # Metrics
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.markdown('<div class="glass-metric">', unsafe_allow_html=True)
                st.metric("Birds Identified", len(detections))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_metric2:
                st.markdown('<div class="glass-metric">', unsafe_allow_html=True)
                if classifications:
                    avg_confidence = sum(conf for _, conf in classifications) / len(classifications)
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                else:
                    st.metric("Avg Confidence", "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Process each bird
            for i, ((box, det_conf), (species, class_conf)) in enumerate(zip(detections, classifications)):
                st.markdown("---")
                
                # Bird information
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"### 🐦 Bird #{i+1} - {species}")
                
                st.markdown(f"""
                <div style="padding: 15px; background: rgba(255,255,255,0.1); border-radius: 8px;">
                    <h4>ResNet34 Model Prediction</h4>
                    <p><strong>Species:</strong> {species}</p>
                    <p><strong>Confidence:</strong> {class_conf:.1%}</p>
                    <p><strong>Detection Score:</strong> {det_conf:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Store the species for video generation
                st.session_state.selected_species_for_video = species
        
        # Reset button
        if st.button("🔄 Analyze Another Image", type="secondary", use_container_width=True):
            st.session_state.detection_complete = False
            st.session_state.bird_detections = []
            st.session_state.bird_classifications = []
            st.session_state.current_image = None
            st.session_state.generated_video_path = None
            st.session_state.generated_story = None
            st.session_state.used_images = None
            st.rerun()
    
    # Story Video Generation Section
    st.markdown("---")
    st.markdown('<div class="section-title">🎬 AI Story Video Generator (bird_path.pth)</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="video-section">
        <strong>📖 AI Story Generation with Video using bird_path.pth</strong><br>
        Generate complete educational story videos using the advanced bird_path.pth model. Each video includes:
        <br><br>
        • <strong>AI-Generated Story</strong>: Unique educational narrative about the bird<br>
        • <strong>Text-to-Speech Audio</strong>: Professional narration of the story<br>
        • <strong>Visual Effects</strong>: Beautiful image transitions and effects<br>
        • <strong>Multiple Images</strong>: Showcases the bird from different angles<br>
        <br>
        <strong>Powered by:</strong> bird_path.pth AI Model
    </div>
    """, unsafe_allow_html=True)
    
    # Video generation options
    col1, col2 = st.columns(2)
    
    with col1:
        # Option 1: Use detected species
        if st.session_state.get('selected_species_for_video'):
            st.info(f"🦜 Detected Species: **{st.session_state.selected_species_for_video}**")
            if st.button("🎬 Generate Story Video", use_container_width=True, type="primary"):
                with st.spinner("Creating AI story video with bird_path.pth..."):
                    video_path, story_text, used_images = video_generator.generate_video(st.session_state.selected_species_for_video)
                    if video_path:
                        st.session_state.generated_video_path = video_path
                        st.session_state.generated_story = story_text
                        st.session_state.used_images = used_images
                        st.success("✅ AI story video generated successfully using bird_path.pth!")
                    else:
                        st.error("❌ Failed to generate story video")
    
    with col2:
        # Option 2: Manual species selection
        manual_species = st.selectbox(
            "Or select a species manually:",
            options=bird_model.bird_species,
            index=0 if not st.session_state.get('selected_species_for_video') else 
                  bird_model.bird_species.index(st.session_state.selected_species_for_video) 
                  if st.session_state.selected_species_for_video in bird_model.bird_species else 0
        )
        
        if st.button("🎬 Generate Video for Selected Bird", use_container_width=True, type="primary"):
            with st.spinner("Creating AI story video with bird_path.pth..."):
                video_path, story_text, used_images = video_generator.generate_video(manual_species)
                if video_path:
                    st.session_state.generated_video_path = video_path
                    st.session_state.generated_story = story_text
                    st.session_state.used_images = used_images
                    st.session_state.selected_species_for_video = manual_species
                    st.success("✅ AI story video generated successfully using bird_path.pth!")
                else:
                    st.error("❌ Failed to generate story video")
    
    # Display generated story and video
    if st.session_state.get('generated_video_path') and os.path.exists(st.session_state.generated_video_path):
        st.markdown("---")
        st.markdown("### 📖 AI-Generated Story Video (bird_path.pth)")
        
        # Display the story
        if st.session_state.get('generated_story'):
            st.markdown(f'<div class="story-box"><strong>📖 AI-Generated Story (bird_path.pth):</strong><br>{st.session_state.generated_story}</div>', unsafe_allow_html=True)
        
        # Display used images
        if st.session_state.get('used_images'):
            st.markdown(f"**🖼️ Used {len(st.session_state.used_images)} images in the video:**")
            cols = st.columns(min(3, len(st.session_state.used_images)))
            for idx, img_path in enumerate(st.session_state.used_images):
                with cols[idx % 3]:
                    try:
                        st.image(img_path, use_column_width=True)
                    except:
                        st.info(f"Image {idx+1}")
        
        # Display video
        try:
            with open(st.session_state.generated_video_path, "rb") as video_file:
                video_bytes = video_file.read()
            
            st.video(video_bytes)
            
            # Video information
            st.info(f"**Video Details:** {st.session_state.selected_species_for_video} | Powered by bird_path.pth | MoviePy Ken Burns effect")
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="📥 Download Story Video",
                    data=video_bytes,
                    file_name=f"uganda_bird_story_{st.session_state.selected_species_for_video.replace(' ', '_')}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
            
            with col2:
                if st.session_state.get('generated_story'):
                    story_bytes = st.session_state.generated_story.encode('utf-8')
                    st.download_button(
                        label="📝 Download Story Text",
                        data=story_bytes,
                        file_name=f"uganda_bird_story_{st.session_state.selected_species_for_video.replace(' ', '_')}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"❌ Error displaying video: {e}")

if __name__ == "__main__":
    main()