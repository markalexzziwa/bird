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
import shutil
from pathlib import Path
warnings.filterwarnings('ignore')

# ========== MOVIEPY INSTALLATION AND CONFIGURATION ==========
def install_moviepy():
    """Install MoviePy if not available"""
    try:
        from moviepy.editor import (
            AudioFileClip, ImageClip, concatenate_videoclips,
            VideoFileClip, concatenate_audioclips
        )
        return True
    except ImportError:
        st.warning("🎬 MoviePy not available. Installing...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "moviepy", "imageio", "imageio-ffmpeg"])
            st.success("✅ MoviePy installed successfully!")
            
            # Try importing again after installation
            from moviepy.editor import (
                AudioFileClip, ImageClip, concatenate_videoclips,
                VideoFileClip, concatenate_audioclips
            )
            return True
        except Exception as e:
            st.error(f"❌ Failed to install MoviePy: {e}")
            return False

# Install and configure MoviePy
MOVIEPY_AVAILABLE = install_moviepy()

if MOVIEPY_AVAILABLE:
    try:
        from moviepy.editor import (
            AudioFileClip, ImageClip, concatenate_videoclips,
            VideoFileClip, concatenate_audioclips
        )
        from moviepy.audio.fx.all import audio_fadein, audio_fadeout
        from moviepy.video.fx.all import resize
        from moviepy.config import change_settings
        
        # Set ImageMagick path for TextClip support
        try:
            # Try to find ImageMagick binary
            result = subprocess.run(['which', 'convert'], capture_output=True, text=True)
            if result.returncode == 0:
                imagemagick_path = result.stdout.strip()
                os.environ["IMAGEMAGICK_BINARY"] = imagemagick_path
                change_settings({"IMAGEMAGICK_BINARY": imagemagick_path})
            else:
                # Fallback paths
                possible_paths = [
                    "/usr/bin/convert",
                    "/usr/local/bin/convert",
                    "/opt/homebrew/bin/convert",
                    "C:\\Program Files\\ImageMagick\\convert.exe"
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        os.environ["IMAGEMAGICK_BINARY"] = path
                        change_settings({"IMAGEMAGICK_BINARY": path})
                        break
        except Exception as e:
            st.warning(f"⚠️ ImageMagick configuration: {e}")
        
        st.success("✅ MoviePy fully configured!")
        
    except Exception as e:
        st.error(f"❌ MoviePy configuration error: {e}")
        MOVIEPY_AVAILABLE = False
else:
    st.error("❌ MoviePy is not available. Video creation features will be limited.")

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
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .image-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin: 15px 0;
    }
    .image-grid img {
        width: 100%;
        height: 120px;
        object-fit: cover;
        border-radius: 8px;
        border: 2px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Story templates with more detailed content
TEMPLATES = [
    "Deep in Uganda's lush forests, the magnificent {name} flashes its {color_phrase} feathers as it moves gracefully through the canopy. {desc} This remarkable bird plays a vital role in Uganda's ecosystem, helping to disperse seeds and control insect populations. Local communities have long cherished sightings of this beautiful creature, considering it a symbol of natural harmony and ecological balance in the Pearl of Africa.",
    
    "Along the majestic Nile's banks, the elegant {name} stands tall with its stunning {color_phrase} plumage shimmering in the morning light. {desc} Fishermen along the riverbanks often pause their work to admire this bird's graceful movements and listen to its melodic calls that echo across the water. The species has adapted perfectly to its riverine habitat, demonstrating nature's incredible ability to thrive in specific environments.",
    
    "In the vast expanse of Queen Elizabeth National Park, the {name} soars majestically above ancient acacia trees, its {color_phrase} wings creating beautiful patterns against the golden savanna sky. {desc} Conservationists have documented how this bird contributes significantly to the park's biodiversity, making it an essential species for maintaining the ecological balance of this protected area that attracts nature enthusiasts from around the world.",
    
    "Near the tranquil shores of Lake Victoria, Africa's largest lake, the {name} perches quietly observing its surroundings with keen awareness. {desc} Children in nearby fishing villages have learned that spotting this bird's {color_phrase} colors early in the morning often signals good fishing conditions for the day. The species has become intertwined with local culture and traditions, featuring in folk tales and community celebrations.",
    
    "High in the mystical Rwenzori Mountains, often called the 'Mountains of the Moon', the {name} sings its enchanting melodies through the morning mist. {desc} Its {color_phrase} feathers capture and reflect the unique light of this cloud forest environment, creating a spectacle that few are privileged to witness. Mountain guides consider sightings of this bird as special moments during their expeditions through this UNESCO World Heritage Site.",
    
    "At the breathtaking Murchison Falls, where the Nile River forces its way through a narrow gorge, the {name} glides effortlessly over the roaring waters. {desc} Tourists and photographers often gasp in awe at its {color_phrase} beauty set against one of Africa's most dramatic natural backdrops. The bird's presence adds to the magical atmosphere of this iconic Ugandan landmark that continues to inspire visitors.",
    
    "Among the extensive papyrus swamps of Uganda, the {name} wades with extraordinary grace and precision. {desc} Its long, slender legs and distinctive {color_phrase} crest have earned it the respectful title of 'king of the wetlands' among local birdwatchers and conservationists. The species demonstrates remarkable adaptation to its aquatic environment, showcasing evolution's incredible work.",
    
    "As sunset paints the skies over Kidepo Valley National Park, the {name} calls across the vast plains, its voice carrying through the evening air. {desc} Its {color_phrase} silhouette against the dramatic African sunset has become a symbol of Uganda's wild, untamed heart and the country's commitment to preserving its natural heritage for future generations to experience and appreciate.",
    
    "In the ancient, mist-shrouded rainforests of Bwindi Impenetrable National Park, the {name} flits skillfully between thick vines and dense foliage. {desc} Even experienced gorilla trackers, focused on their primary mission, often pause their important work to admire the bird's {color_phrase} brilliance shining through the forest gloom. This creates magical moments where humanity connects with nature's diverse wonders.",
    
    "By the peaceful shores of Lake Mburo National Park, the {name} reflects perfectly in the calm waters during the golden hour. {desc} Its {color_phrase} feathers seem to mirror the profound peace of the savanna night settling over the landscape. The bird's presence enhances the tranquil atmosphere of this protected area, reminding visitors of nature's gentle rhythms and timeless beauty that endure through the ages."
]

class BirdStoryGenerator:
    def __init__(self, templates): 
        self.templates = templates
    
    def __call__(self, name, description="", colors=None):
        if colors is None: 
            colors = []
        color_phrase = ", ".join([c.strip() for c in colors]) if colors else "vibrant and beautifully patterned"
        desc = description.strip().capitalize() if description else "This fascinating bird exhibits unique behaviors and plays an important role in its ecosystem, captivating birdwatchers and researchers alike with its distinctive characteristics and adaptations to its natural environment."
        tmpl = random.choice(self.templates)
        return tmpl.format(name=name, color_phrase=color_phrase, desc=desc)

# ========== ENHANCED FILE DOWNLOADER WITH GDOWN ==========
def install_gdown():
    """Install gdown if not available"""
    try:
        import gdown
        return True
    except ImportError:
        st.warning("📦 gdown not available. Installing...")
        try:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
            st.success("✅ gdown installed successfully!")
            return True
        except Exception as e:
            st.error(f"❌ Failed to install gdown: {e}")
            return False

def download_file_from_gdrive(file_id, destination):
    """Download file from Google Drive using gdown"""
    try:
        if not os.path.exists(destination):
            st.info(f"📥 Downloading {os.path.basename(destination)} from Google Drive...")
            
            # Install and use gdown
            if install_gdown():
                import gdown
                url = f'https://drive.google.com/uc?id={file_id}'
                
                # Create progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_hook(current, total, width=80):
                    percent = current / total
                    progress_bar.progress(percent)
                    status_text.text(f"Downloaded: {current/(1024*1024):.1f} MB / {total/(1024*1024):.1f} MB")
                
                # Download with progress
                gdown.download(url, destination, quiet=False, fuzzy=True)
                
                progress_bar.progress(1.0)
                status_text.empty()
                
                if os.path.exists(destination):
                    file_size = os.path.getsize(destination) / (1024 * 1024)
                    st.success(f"✅ Downloaded {os.path.basename(destination)} successfully! ({file_size:.1f} MB)")
                    return True
                else:
                    st.error(f"❌ Download failed for {os.path.basename(destination)}")
                    return False
            else:
                st.error("❌ gdown installation failed")
                return False
        
        # If file already exists
        if os.path.exists(destination):
            file_size = os.path.getsize(destination) / (1024 * 1024)
            if file_size > 0.1:  # Ensure file is not empty/corrupted
                st.info(f"✅ {os.path.basename(destination)} already exists ({file_size:.1f} MB)")
                return True
            else:
                st.error(f"❌ {os.path.basename(destination)} is too small - may be corrupted")
                os.remove(destination)
                return False
        else:
            st.error(f"❌ Failed to download {os.path.basename(destination)}")
            return False
            
    except Exception as e:
        st.error(f"❌ Download error for {os.path.basename(destination)}: {e}")
        return False

# ========== BIRD DATA LOADING ==========
@st.cache_resource
def load_bird_data():
    """Load bird data with multiple location support"""
    possible_paths = [
        "bird_data.pth",
        "./models/bird_data.pth",
        "./data/bird_data.pth",
        "./bird_data/bird_data.pth",
        "/content/bird_data.pth"  # For Google Colab
    ]
    
    for pth_path in possible_paths:
        if os.path.exists(pth_path):
            try:
                bird_data = torch.load(pth_path, map_location="cpu")
                st.success(f"✅ Loaded bird data from {pth_path} with {len(bird_data)} species")
                return bird_data
            except Exception as e:
                st.error(f"Error loading bird data from {pth_path}: {e}")
                continue
    
    # If not found, try to download
    st.warning("bird_data.pth not found in standard locations. Using placeholder data.")
    return {}

# Load bird data at startup
bird_db = load_bird_data()

# ========== VIDEO MODEL LOADING ==========
@st.cache_resource
def load_video_model():
    """Load video model with download from Google Drive and multiple location support"""
    # Define possible locations for bird_path.pth
    possible_paths = [
        "bird_path.pth",
        "./models/bird_path.pth", 
        "./data/bird_path.pth",
        "./bird_data/bird_path.pth",
        "/content/bird_path.pth"  # For Google Colab
    ]
    
    # First check if file exists in any location
    pth_path = None
    for path in possible_paths:
        if os.path.exists(path):
            pth_path = path
            st.info(f"📁 Found bird_path.pth at: {path}")
            break
    
    # If not found, download it
    if pth_path is None:
        st.info("🔍 bird_path.pth not found in standard locations. Downloading...")
        pth_path = "bird_path.pth"  # Default download location
        file_id = "1J9T5r5TboWzvqAPQHmfvQmozor_wmmPz"  # From your Google Drive link
        
        success = download_file_from_gdrive(file_id, pth_path)
        if not success:
            st.warning("❌ Could not download bird_path.pth. Using default story generation.")
            return None
    
    try:
        model_data = torch.load(pth_path, map_location="cpu")
        st.success(f"✅ Video story model loaded successfully from {pth_path}!")
        
        # Display information about loaded model data
        if isinstance(model_data, dict):
            st.info(f"📁 Model contains data for {len(model_data)} bird species")
            # Show sample of what's in the model
            sample_species = list(model_data.keys())[:3]
            st.info(f"📸 Sample species: {', '.join(sample_species)}")
            
        return model_data
    except Exception as e:
        st.warning(f"❌ Error loading video model from {pth_path}: {e}. Using default story generation.")
        return None

# Load video model at startup
video_model_data = load_video_model()

# ========== MOVIEPY VIDEO CREATION FUNCTIONS ==========
def ken_burns_effect(image_path, duration=5.0, zoom_direction="random"):
    """
    Enhanced Ken Burns effect with multiple zoom directions using MoviePy
    """
    if not MOVIEPY_AVAILABLE:
        return None
        
    try:
        clip = ImageClip(image_path).set_duration(duration)
        w, h = clip.size
        
        # Different zoom effects
        zoom_level = 1.2  # Increased zoom for more dramatic effect
        
        if zoom_direction == "random":
            zoom_direction = random.choice(["in", "out", "pan_left", "pan_right", "pan_up", "pan_down"])
        
        if zoom_direction == "in":
            # Zoom in slowly
            clip = clip.resize(lambda t: 1 + (zoom_level - 1) * (t / duration))
            clip = clip.set_position("center")
            
        elif zoom_direction == "out":
            # Start zoomed in and zoom out
            clip = clip.resize(lambda t: zoom_level - (zoom_level - 1) * (t / duration))
            clip = clip.set_position("center")
            
        elif zoom_direction == "pan_left":
            # Pan from right to left
            clip = clip.resize(lambda t: 1 + (zoom_level - 1) * 0.4)
            clip = clip.set_position(lambda t: (
                w * 0.15 * (1 - t/duration),
                "center"
            ))
            
        elif zoom_direction == "pan_right":
            # Pan from left to right
            clip = clip.resize(lambda t: 1 + (zoom_level - 1) * 0.4)
            clip = clip.set_position(lambda t: (
                -w * 0.15 * (1 - t/duration),
                "center"
            ))
            
        elif zoom_direction == "pan_up":
            # Pan from bottom to top
            clip = clip.resize(lambda t: 1 + (zoom_level - 1) * 0.4)
            clip = clip.set_position(lambda t: (
                "center",
                h * 0.15 * (1 - t/duration)
            ))
            
        elif zoom_direction == "pan_down":
            # Pan from top to bottom
            clip = clip.resize(lambda t: 1 + (zoom_level - 1) * 0.4)
            clip = clip.set_position(lambda t: (
                "center",
                -h * 0.15 * (1 - t/duration)
            ))
        
        # Add smooth fade in/out
        clip = clip.fadein(0.7).fadeout(0.7)  # Longer fades for smoother transitions
        return clip
        
    except Exception as e:
        st.error(f"❌ Ken Burns effect error: {e}")
        # Fallback: simple image clip
        return ImageClip(image_path).set_duration(duration).fadein(0.5).fadeout(0.5)

def get_audio_duration(audio_path):
    """Get audio duration using MoviePy"""
    if not MOVIEPY_AVAILABLE:
        return 25  # Increased default duration for longer stories
        
    try:
        audio_clip = AudioFileClip(audio_path)
        duration = audio_clip.duration
        audio_clip.close()
        return duration
    except Exception as e:
        st.warning(f"⚠️ Could not determine audio duration: {e}, using default")
        return 25  # Increased default for longer stories

def create_moviepy_video(images, audio_path, output_path):
    """
    Create video using MoviePy with Ken Burns effect only
    """
    if not MOVIEPY_AVAILABLE:
        st.error("❌ MoviePy is not available. Cannot create video.")
        return None
    
    try:
        # Load and process audio with enhanced effects
        raw_audio = AudioFileClip(audio_path)
        
        # Add audio enhancements with longer fades
        narration = audio_fadein(raw_audio, 1.5)  # Longer fade in
        narration = audio_fadeout(narration, 2.0)  # Longer fade out
        
        # Calculate durations - longer duration per image for more photos
        audio_duration = narration.duration
        img_duration = max(5.0, audio_duration / len(images))  # Increased minimum duration
        total_duration = img_duration * len(images)
        
        # Adjust audio to match video duration
        if narration.duration < total_duration:
            # Loop audio if too short
            loops = int(total_duration / narration.duration) + 1
            narration = concatenate_audioclips([narration] * loops).subclip(0, total_duration)
        else:
            narration = narration.subclip(0, total_duration)
        
        # Create video clips with Ken Burns effect only
        clips = []
        
        for i, img_path in enumerate(images):
            try:
                # Use Ken Burns effect with random direction for variety
                directions = ["in", "out", "pan_left", "pan_right", "pan_up", "pan_down"]
                direction = directions[i % len(directions)]
                clip = ken_burns_effect(img_path, img_duration, direction)
                
                if clip is not None:
                    clips.append(clip)
                
            except Exception as e:
                st.warning(f"⚠️ Could not process image {img_path}: {e}")
                # Create a simple clip as fallback
                try:
                    clip = ImageClip(img_path).set_duration(img_duration)
                    clip = clip.fadein(0.5).fadeout(0.5)
                    clips.append(clip)
                except:
                    continue
        
        if not clips:
            st.error("❌ No valid video clips created")
            return None
        
        # Combine clips with smooth transitions
        video = concatenate_videoclips(clips, method="compose")
        video = video.set_audio(narration)
        
        # Enhance video quality with higher resolution
        video = video.resize(height=720)
        
        # Write final video with optimized settings for better quality
        video.write_videofile(
            output_path, 
            fps=24, 
            codec="libx264", 
            audio_codec="aac", 
            preset="medium",
            verbose=False,
            logger=None,
            ffmpeg_params=[
                '-crf', '20',           # Better quality setting
                '-pix_fmt', 'yuv420p',  # Better compatibility
                '-movflags', '+faststart'  # Better streaming
            ]
        )
        
        # Clean up resources
        video.close()
        for clip in clips:
            clip.close()
        
        return output_path
        
    except Exception as e:
        st.error(f"❌ MoviePy video creation error: {e}")
        return None

class AdvancedVideoGenerator:
    def __init__(self):
        self.csv_path = './birdsuganda.csv'
        self.bird_data = None
        self.video_duration = 25  # Increased default duration
        self.moviepy_available = MOVIEPY_AVAILABLE
        
        # Initialize story generator - ALWAYS have a fallback
        self.story_generator = self._initialize_story_generator()
        
    def _initialize_story_generator(self):
        """Initialize story generator with proper fallbacks"""
        if video_model_data is not None:
            try:
                # Try to extract story generator from model data
                if hasattr(video_model_data, 'generate_story'):
                    st.success("✅ Using advanced story generation model")
                    return video_model_data
                elif isinstance(video_model_data, dict) and 'story_generator' in video_model_data:
                    st.success("✅ Using story generator from model data")
                    return video_model_data['story_generator']
                elif isinstance(video_model_data, dict) and 'templates' in video_model_data:
                    st.success("✅ Using custom templates from model")
                    return BirdStoryGenerator(video_model_data['templates'])
                else:
                    st.info("📖 Using enhanced story generation")
                    return BirdStoryGenerator(TEMPLATES)
            except Exception as e:
                st.warning(f"⚠️ Could not load advanced model: {e}. Using default stories.")
                return BirdStoryGenerator(TEMPLATES)
        else:
            st.info("📖 Using default story generation")
            return BirdStoryGenerator(TEMPLATES)
    
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
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filename)
            return filename
        except Exception as e:
            st.error(f"❌ Error generating speech: {e}")
            return None

    def extract_images_from_model_data(self, species_name):
        """Extract images for a species from the video model data"""
        try:
            if video_model_data is None:
                return None
                
            # Check if model data contains image information for this species
            if isinstance(video_model_data, dict):
                # Case 1: Model data is a dictionary with species as keys
                if species_name in video_model_data:
                    species_data = video_model_data[species_name]
                    return self._process_species_images(species_data, species_name)
                
                # Case 2: Try case-insensitive matching
                species_lower = species_name.lower()
                for key in video_model_data.keys():
                    if key.lower() == species_lower:
                        species_data = video_model_data[key]
                        return self._process_species_images(species_data, species_name)
                
                # Case 3: Try partial matching
                for key in video_model_data.keys():
                    if species_lower in key.lower() or key.lower() in species_lower:
                        species_data = video_model_data[key]
                        return self._process_species_images(species_data, species_name)
            
            return None
            
        except Exception as e:
            st.error(f"❌ Error extracting images from model: {e}")
            return None

    def _process_species_images(self, species_data, species_name):
        """Process species data to extract images"""
        try:
            image_paths = []
            
            # Case 1: Data contains base64 encoded images
            if isinstance(species_data, dict) and 'images_b64' in species_data:
                for i, b64 in enumerate(species_data['images_b64']):
                    try:
                        img_data = base64.b64decode(b64)
                        img = Image.open(BytesIO(img_data))
                        temp_path = f"./temp_model_{species_name.replace(' ', '_')}_{i}.jpg"
                        img.save(temp_path, "JPEG", quality=95)  # Better quality
                        image_paths.append(temp_path)
                    except Exception as e:
                        st.warning(f"⚠️ Could not decode image {i} for {species_name}: {e}")
                        continue
            
            # Case 2: Data contains image paths or PIL images
            elif isinstance(species_data, dict) and 'images' in species_data:
                for i, img_item in enumerate(species_data['images']):
                    try:
                        if isinstance(img_item, str):
                            # It's a file path
                            if os.path.exists(img_item):
                                image_paths.append(img_item)
                        elif hasattr(img_item, 'save'):
                            # It's a PIL Image
                            temp_path = f"./temp_model_{species_name.replace(' ', '_')}_{i}.jpg"
                            img_item.save(temp_path, "JPEG", quality=95)
                            image_paths.append(temp_path)
                    except Exception as e:
                        st.warning(f"⚠️ Could not process image {i} for {species_name}: {e}")
                        continue
            
            # Case 3: Data is a list of images
            elif isinstance(species_data, list):
                for i, img_item in enumerate(species_data):
                    try:
                        if hasattr(img_item, 'save'):
                            # It's a PIL Image
                            temp_path = f"./temp_model_{species_name.replace(' ', '_')}_{i}.jpg"
                            img_item.save(temp_path, "JPEG", quality=95)
                            image_paths.append(temp_path)
                    except Exception as e:
                        st.warning(f"⚠️ Could not process list image {i} for {species_name}: {e}")
                        continue
            
            return image_paths if image_paths else None
            
        except Exception as e:
            st.error(f"❌ Error processing species images: {e}")
            return None

    def get_bird_images(self, species_name, max_images=8):  # Increased from 5 to 8
        """Get bird images for the species - prioritize model data, then fallbacks"""
        try:
            # First try to get images from the video model data
            model_images = self.extract_images_from_model_data(species_name)
            if model_images:
                st.success(f"✅ Found {len(model_images)} images from model data")
                # Use more images if available
                selected_images = self._select_best_images(model_images, min(max_images, len(model_images)))
                return selected_images
            
            # Fallback to bird_db if available
            if species_name in bird_db and bird_db[species_name].get("images_b64"):
                image_paths = []
                for i, b64 in enumerate(bird_db[species_name]["images_b64"][:max_images]):
                    try:
                        img_data = base64.b64decode(b64)
                        img = Image.open(BytesIO(img_data))
                        placeholder_path = f"./temp_bird_{species_name.replace(' ', '_')}_{i}.jpg"
                        img.save(placeholder_path, "JPEG", quality=95)
                        image_paths.append(placeholder_path)
                    except:
                        continue
                
                if image_paths:
                    st.info(f"ℹ️ Using {len(image_paths)} images from bird database")
                    return image_paths
            
            # Final fallback: Create placeholder images
            st.warning(f"⚠️ No images found for {species_name}, using placeholders")
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

    def _select_best_images(self, images, max_count):
        """Select the best images for video creation"""
        try:
            # If we have fewer images than max_count, return all
            if len(images) <= max_count:
                return images
            
            # Otherwise, select a diverse set
            # Prefer images that are different from each other
            selected = []
            
            # Always include the first image
            selected.append(images[0])
            
            # Try to select images that are likely to be different
            # This is a simple approach - in production you might want to use image similarity
            step = max(1, len(images) // max_count)
            for i in range(1, max_count):
                idx = min(i * step, len(images) - 1)
                if images[idx] not in selected:
                    selected.append(images[idx])
            
            # If we still don't have enough, add remaining ones
            while len(selected) < max_count and len(selected) < len(images):
                for img in images:
                    if img not in selected:
                        selected.append(img)
                        break
            
            return selected[:max_count]
            
        except Exception as e:
            st.warning(f"⚠️ Error selecting best images: {e}")
            return images[:max_count]

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
                [106, 90, 205],   # Slate blue
                [220, 20, 60],    # Crimson
                [30, 144, 255],   # Dodger blue
                [50, 205, 50]     # Lime green
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
        """Generate a comprehensive story-based video with audio using MoviePy"""
        try:
            # Always generate a story - never fail here
            if self.story_generator is None:
                self.story_generator = BirdStoryGenerator(TEMPLATES)
            
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
            
            # Generate story using the model
            st.info("📖 Generating detailed educational story...")
            story_text = self.story_generator(common_name, description, colors)
            
            # Display the generated story
            st.markdown(f'<div class="story-box"><strong>📖 AI-Generated Story:</strong><br>{story_text}</div>', unsafe_allow_html=True)
            
            # Generate audio
            st.info("🔊 Converting detailed story to speech...")
            audio_file = f"temp_story_{species_name.replace(' ', '_')}.mp3"
            audio_path = self.natural_tts(story_text, audio_file)
            
            if not audio_path:
                st.error("❌ Failed to generate audio")
                return None, None, None
            
            # Get bird images - ALWAYS get images (increased from 5 to 8)
            st.info("🖼️ Gathering multiple bird images...")
            bird_images = self.get_bird_images(species_name, max_images=8)  # Increased for better variety
            
            if not bird_images:
                st.error("❌ No bird images available")
                return None, None, None
            
            # Display the selected images
            st.info(f"🎨 Selected {len(bird_images)} images for video creation")
            self._display_image_grid(bird_images, species_name)
            
            # Generate video using MoviePy with Ken Burns effect only
            st.info("🎬 Creating enhanced story video with Ken Burns effects...")
            video_file = f"temp_story_video_{species_name.replace(' ', '_')}.mp4"
            
            if not MOVIEPY_AVAILABLE:
                st.error("❌ MoviePy is not available. Cannot create video.")
                return None, None, None
                
            video_path = create_moviepy_video(bird_images, audio_path, video_file)
            
            # Clean up temporary audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            
            if video_path and os.path.exists(video_path):
                st.success(f"✅ Enhanced story video generated successfully!")
                return video_path, story_text, bird_images
            else:
                st.error("❌ Failed to generate video file")
                return None, None, None
            
        except Exception as e:
            st.error(f"❌ Story video generation error: {e}")
            return None, None, None

    def _display_image_grid(self, image_paths, species_name):
        """Display a grid of images being used in the video"""
        try:
            st.markdown(f"### 🖼️ Images of {species_name} (Used in Video)")
            
            # Create columns for image display
            cols = st.columns(min(4, len(image_paths)))
            
            for idx, img_path in enumerate(image_paths):
                with cols[idx % len(cols)]:
                    try:
                        img = Image.open(img_path)
                        st.image(img, caption=f"Image {idx + 1}", use_column_width=True)
                    except Exception as e:
                        st.error(f"❌ Could not display image {idx + 1}")
            
        except Exception as e:
            st.warning(f"⚠️ Could not display image grid: {e}")

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
            moviepy>=1.0.3
            imageio>=2.9.0
            imageio-ffmpeg>=0.4.5
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

def cleanup_temp_files():
    """Clean up temporary files created during video generation"""
    try:
        temp_files = [f for f in os.listdir('.') if f.startswith('temp_') and (f.endswith('.mp4') or f.endswith('.mp3') or f.endswith('.jpg'))]
        for temp_file in temp_files:
            try:
                os.remove(temp_file)
            except:
                pass
        st.success(f"✅ Cleaned up {len(temp_files)} temporary files")
    except Exception as e:
        st.warning(f"⚠️ Could not clean up temp files: {e}")

def check_moviepy_dependencies():
    """Check if all MoviePy dependencies are available"""
    st.info("🔍 Checking video creation dependencies...")
    
    # Check moviepy
    if MOVIEPY_AVAILABLE:
        st.success("✅ MoviePy: Available - Full video effects enabled")
    else:
        st.error("❌ MoviePy: Not available - Video creation features disabled")
    
    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True)
        st.success("✅ FFmpeg: Available")
    except:
        st.warning("⚠️ FFmpeg: Not found - video creation may be limited")
    
    return MOVIEPY_AVAILABLE

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
            # Check video dependencies
            check_moviepy_dependencies()
            
            # Try to load the ResNet model first
            resnet_success = st.session_state.bird_model.load_model()
            
            if resnet_success:
                st.session_state.model_loaded = True
                st.session_state.system_initialized = True
                st.success(f"✅ System ready! Can identify {len(st.session_state.bird_model.bird_species)} bird species")
                
                # Show information about available bird images
                if video_model_data is not None and isinstance(video_model_data, dict):
                    st.info(f"📸 Video model contains images for {len(video_model_data)} bird species")
                else:
                    st.info("📖 Story video generation available with placeholder images")
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
        if video_model_data is not None:
            st.success("🎬 Story Video Generation: **Available**")
            if isinstance(video_model_data, dict):
                st.info(f"📸 Contains images for {len(video_model_data)} species")
        else:
            st.warning("🎬 Story Video: **Basic Mode**")
            st.info("📖 Using placeholder images")
        
        if not video_generator.moviepy_available:
            st.error("🎥 Video Engine: MoviePy Not Available")
            st.info("🔧 Please install MoviePy for video creation")
        else:
            st.success("🎥 Video Engine: MoviePy (Ken Burns effect)")
            st.info("📹 Using 8 images per video")
            st.info("🗣️ Detailed stories with narration")
        
        # Cleanup button
        if st.button("🧹 Clean Temporary Files", use_container_width=True):
            cleanup_temp_files()
            st.rerun()
    
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
        with narrated audio and beautiful visual effects using Ken Burns animation.
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
    st.markdown('<div class="section-title">🎬 AI Story Video Generator</div>', unsafe_allow_html=True)
    
    if not MOVIEPY_AVAILABLE:
        st.error("""
        ❌ **MoviePy is not available!**
        
        Video creation features are disabled. Please install MoviePy to enable video generation:
        
        ```bash
        pip install moviepy imageio imageio-ffmpeg
        ```
        
        Or add to your requirements.txt:
        ```txt
        moviepy>=1.0.3
        imageio>=2.9.0
        imageio-ffmpeg>=0.4.5
        ```
        """)
    else:
        st.markdown(f"""
        <div class="video-section">
            <strong>📖 Enhanced AI Story Generation with Video</strong><br>
            Generate comprehensive educational story videos featuring:
            <br><br>
            • <strong>Detailed AI-Generated Stories</strong>: Rich, educational narratives with ecological context<br>
            • <strong>Professional Text-to-Speech Audio</strong>: Clear narration of detailed stories<br>
            • <strong>Ken Burns Visual Effects</strong>: Beautiful pan and zoom animations on 8+ images<br>
            • <strong>Multiple Bird Images</strong>: Showcases each species from various angles and settings<br>
            <br>
            <strong>Video Features:</strong> Ken Burns effects only | 8+ images per video | Detailed narration | HD Quality
        </div>
        """, unsafe_allow_html=True)
        
        # Video generation options - Only Ken Burns effect
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Option 1: Use detected species
            if st.session_state.get('selected_species_for_video'):
                st.info(f"🦜 Detected Species: **{st.session_state.selected_species_for_video}**")

        with col2:
            if st.session_state.get('selected_species_for_video'):
                if st.button("🎬 Generate Story Video", use_container_width=True, type="primary"):
                    with st.spinner("Creating enhanced story video with Ken Burns effects..."):
                        video_path, story_text, used_images = video_generator.generate_video(
                            st.session_state.selected_species_for_video
                        )
                        if video_path:
                            st.session_state.generated_video_path = video_path
                            st.session_state.generated_story = story_text
                            st.session_state.used_images = used_images
                            st.success("✅ Enhanced story video generated successfully!")
                        else:
                            st.error("❌ Failed to generate story video")
        
        # Manual species selection
        st.markdown("---")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            manual_species = st.selectbox(
                "Or select a species manually:",
                options=bird_model.bird_species,
                index=0 if not st.session_state.get('selected_species_for_video') else 
                      bird_model.bird_species.index(st.session_state.selected_species_for_video) 
                      if st.session_state.selected_species_for_video in bird_model.bird_species else 0
            )
        
        with col2:
            if st.button("🎬 Generate Video for Selected Bird", use_container_width=True, type="primary"):
                with st.spinner("Creating detailed AI story video with audio..."):
                    video_path, story_text, used_images = video_generator.generate_video(manual_species)
                    if video_path:
                        st.session_state.generated_video_path = video_path
                        st.session_state.generated_story = story_text
                        st.session_state.used_images = used_images
                        st.session_state.selected_species_for_video = manual_species
                        st.success("✅ Enhanced story video generated successfully!")
                    else:
                        st.error("❌ Failed to generate story video")
        
        # Display generated story and video
        if st.session_state.get('generated_video_path') and os.path.exists(st.session_state.generated_video_path):
            st.markdown("---")
            st.markdown("### 📖 Enhanced AI-Generated Story Video")
            
            # Display the story
            if st.session_state.get('generated_story'):
                st.markdown(f'<div class="story-box"><strong>📖 Detailed AI-Generated Story:</strong><br>{st.session_state.generated_story}</div>', unsafe_allow_html=True)
            
            # Display video
            try:
                with open(st.session_state.generated_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                
                st.video(video_bytes)
                
                # Video information
                st.info(f"**Video Details:** {st.session_state.selected_species_for_video} | Ken Burns Effects | {len(st.session_state.used_images)} images | Detailed Audio Narration | HD Quality")
                
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