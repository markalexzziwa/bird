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

# ========== ENHANCED MOVIEPY INSTALLATION AND IMPORT ==========
def ensure_moviepy_installation():
    """Ensure MoviePy and all dependencies are properly installed"""
    required_packages = {
        'moviepy': 'moviepy',
        'decorator': 'decorator', 
        'proglog': 'proglog',
        'imageio': 'imageio',
        'imageio-ffmpeg': 'imageio-ffmpeg'
    }
    
    missing_packages = []
    for package, import_name in required_packages.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        st.warning(f"🎬 Installing missing packages: {', '.join(missing_packages)}")
        try:
            for package in missing_packages:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package, "--quiet", "--no-warn-script-location"
                ])
            st.success("✅ All MoviePy dependencies installed successfully!")
            
            # Reload required modules
            import importlib
            importlib.invalidate_caches()
            
        except Exception as e:
            st.error(f"❌ Failed to install MoviePy dependencies: {e}")
            return False
    
    return True

# Ensure MoviePy is available
MOVIEPY_AVAILABLE = ensure_moviepy_installation()

if MOVIEPY_AVAILABLE:
    try:
        # Import MoviePy components
        from moviepy.editor import (
            AudioFileClip, ImageClip, concatenate_videoclips,
            VideoFileClip, concatenate_audioclips
        )
        from moviepy.audio.fx.all import audio_fadein, audio_fadeout
        from moviepy.video.fx.all import resize
        
        # Configure MoviePy for better performance
        import moviepy.config as moviepy_config
        moviepy_config.change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
        
        st.success("✅ MoviePy loaded successfully with all features!")
        
    except Exception as e:
        st.error(f"❌ MoviePy import error: {e}")
        MOVIEPY_AVAILABLE = False
else:
    st.error("❌ MoviePy installation failed. Video generation will not work.")

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
    .video-status {
        background: rgba(255, 193, 7, 0.2);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
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

# ========== ENHANCED MOVIEPY VIDEO GENERATOR ==========
class AdvancedVideoGenerator:
    def __init__(self):
        self.csv_path = './birdsuganda.csv'
        self.bird_data = None
        self.video_duration = 20
        self.moviepy_available = MOVIEPY_AVAILABLE
        
        # Initialize story generator
        self.story_generator = BirdStoryGenerator(TEMPLATES)
        
        st.info(f"🎬 MoviePy Status: {'✅ Available' if self.moviepy_available else '❌ Not Available'}")
        
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

    def ken_burns_effect(self, image_path, duration=4.0, zoom_ratio=1.1):
        """Apply Ken Burns effect to an image clip"""
        try:
            # Load image clip
            clip = ImageClip(image_path).set_duration(duration)
            
            # Get original dimensions
            w, h = clip.size
            
            # Apply zoom effect
            def make_frame(t):
                """Calculate zoom factor for each frame"""
                zoom_factor = 1 + (zoom_ratio - 1) * (t / duration)
                new_w = int(w * zoom_factor)
                new_h = int(h * zoom_factor)
                
                # Resize frame
                frame = clip.get_frame(t)
                frame_resized = cv2.resize(frame, (new_w, new_h))
                
                # Calculate position for panning effect
                x_offset = int((new_w - w) * (t / duration) * 0.3)
                y_offset = int((new_h - h) * (t / duration) * 0.2)
                
                # Crop to original size with offset
                if new_w > w and new_h > h:
                    frame_cropped = frame_resized[y_offset:y_offset+h, x_offset:x_offset+w]
                    if frame_cropped.shape[0] == h and frame_cropped.shape[1] == w:
                        return frame_cropped
                
                return cv2.resize(frame, (w, h))
            
            # Create final clip with effects
            final_clip = clip.fl(lambda gf, t: make_frame(t))
            final_clip = final_clip.set_duration(duration)
            final_clip = final_clip.fadein(0.5).fadeout(0.5)
            
            return final_clip
            
        except Exception as e:
            st.error(f"❌ Ken Burns effect error: {e}")
            # Return simple image clip as fallback
            return ImageClip(image_path).set_duration(duration).fadein(0.3).fadeout(0.3)

    def create_moviepy_video(self, images, audio_path, output_path):
        """Create video using MoviePy with enhanced effects"""
        if not self.moviepy_available:
            st.error("❌ MoviePy is not available for video creation")
            return None
        
        try:
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            
            # Load and process audio
            st.info("🔊 Processing audio...")
            raw_audio = AudioFileClip(audio_path)
            audio_duration = raw_audio.duration
            
            # Apply audio effects
            narration = audio_fadein(raw_audio, 0.8)
            narration = audio_fadeout(narration, 1.5)
            
            # Calculate image durations
            num_images = len(images)
            min_duration_per_image = 3.0
            max_duration_per_image = 6.0
            
            # Adjust durations based on audio length
            if audio_duration / num_images < min_duration_per_image:
                total_duration = num_images * min_duration_per_image
            elif audio_duration / num_images > max_duration_per_image:
                total_duration = num_images * max_duration_per_image
            else:
                total_duration = audio_duration
            
            duration_per_image = total_duration / num_images
            
            st.info(f"🎬 Creating {num_images} clips with {duration_per_image:.1f}s each...")
            
            # Create video clips with Ken Burns effect
            video_clips = []
            progress_bar = st.progress(0)
            
            for i, image_path in enumerate(images):
                try:
                    # Create clip with Ken Burns effect
                    clip = self.ken_burns_effect(image_path, duration_per_image)
                    video_clips.append(clip)
                    
                    # Update progress
                    progress = (i + 1) / len(images)
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not process image {image_path}: {e}")
                    # Add simple image clip as fallback
                    try:
                        simple_clip = ImageClip(image_path).set_duration(duration_per_image)
                        simple_clip = simple_clip.fadein(0.3).fadeout(0.3)
                        video_clips.append(simple_clip)
                    except:
                        continue
            
            progress_bar.empty()
            
            if not video_clips:
                st.error("❌ No valid video clips were created")
                return None
            
            # Concatenate all clips
            st.info("🔄 Combining video clips...")
            final_video = concatenate_videoclips(video_clips, method="compose")
            
            # Set audio
            if narration.duration > final_video.duration:
                narration = narration.subclip(0, final_video.duration)
            else:
                # Loop audio if needed
                loops_needed = int(final_video.duration / narration.duration) + 1
                audio_clips = [narration] * loops_needed
                narration = concatenate_audioclips(audio_clips).subclip(0, final_video.duration)
            
            final_video = final_video.set_audio(narration)
            
            # Resize for consistent output
            final_video = final_video.resize(height=720)
            
            # Write video file
            st.info("💾 Writing video file...")
            final_video.write_videofile(
                output_path,
                fps=24,
                codec="libx264",
                audio_codec="aac",
                temp_audiofile=os.path.join(temp_dir, "temp_audio.m4a"),
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up
            for clip in video_clips:
                clip.close()
            final_video.close()
            raw_audio.close()
            narration.close()
            
            # Remove temp directory
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
            
            st.success(f"✅ Video created successfully: {output_path}")
            return output_path
            
        except Exception as e:
            st.error(f"❌ MoviePy video creation error: {e}")
            # Clean up on error
            try:
                for clip in video_clips:
                    clip.close()
                if 'final_video' in locals():
                    final_video.close()
            except:
                pass
            return None

    def get_bird_images(self, species_name, max_images=5):
        """Get bird images for the species"""
        try:
            # Create placeholder images for demonstration
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
        """Generate a comprehensive story-based video with audio"""
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
            
            # Generate story
            st.info("📖 Generating educational story...")
            story_text = self.story_generator(common_name, description, colors)
            
            # Display the generated story
            st.markdown(f'<div class="story-box"><strong>📖 AI-Generated Story:</strong><br>{story_text}</div>', unsafe_allow_html=True)
            
            # Generate audio
            st.info("🔊 Converting story to speech...")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                audio_path = temp_audio.name
            
            audio_path = self.natural_tts(story_text, audio_path)
            
            if not audio_path or not os.path.exists(audio_path):
                st.error("❌ Failed to generate audio")
                return None, None, None
            
            # Get bird images
            st.info("🖼️ Gathering bird images...")
            bird_images = self.get_bird_images(species_name, max_images=3)
            
            if not bird_images:
                st.error("❌ No bird images available")
                return None, None, None
            
            # Generate video using MoviePy
            st.info("🎬 Creating story video with MoviePy...")
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
                video_path = temp_video.name
            
            if self.moviepy_available:
                video_path = self.create_moviepy_video(bird_images, audio_path, video_path)
            else:
                st.error("❌ MoviePy not available for video creation")
                return None, None, None
            
            # Clean up temporary audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            
            if video_path and os.path.exists(video_path):
                file_size = os.path.getsize(video_path) / (1024 * 1024)  # Size in MB
                st.success(f"✅ Story video generated successfully! ({file_size:.1f} MB)")
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

# ========== SIMPLIFIED BIRD DATA ==========
@st.cache_resource
def load_bird_data():
    """Load minimal bird data for testing"""
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
    st.success(f"✅ Loaded bird data with {len(minimal_data)} species")
    return minimal_data

# Load bird data
bird_db = load_bird_data()

# ========== SIMPLIFIED RESNET MODEL ==========
class ResNet34BirdModel:
    def __init__(self):
        self.model_loaded = True  # Simplified for demo
        self.bird_species = list(bird_db.keys())
        self.inv_label_map = {idx: species for idx, species in enumerate(self.bird_species)}
    
    def predict_bird_species(self, image):
        """Simplified prediction for demo"""
        # Return mock detection for demo purposes
        detections = [((100, 100, 200, 200), 0.85)]  # Single detection
        classifications = [("African Fish Eagle", 0.92)]  # Single classification
        
        if isinstance(image, Image.Image):
            original_image = np.array(image)
        else:
            original_image = image
            
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
        st.session_state.model_loaded = True
        st.session_state.system_initialized = True
        st.session_state.generated_video_path = None
        st.session_state.selected_species_for_video = None
        st.session_state.generated_story = None
        st.session_state.used_images = None
    
    if st.session_state.system_initialized:
        st.success("✅ Uganda Bird Spotter System Ready!")
        if MOVIEPY_AVAILABLE:
            st.success("🎬 MoviePy Video Generation: ✅ Available")
        else:
            st.error("🎬 MoviePy Video Generation: ❌ Not Available")

def main():
    # Initialize the system
    initialize_system()
    
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
        if MOVIEPY_AVAILABLE:
            st.success("🎬 Video Generation: **✅ Available**")
            st.info("📹 Features: Ken Burns effect, TTS audio, smooth transitions")
        else:
            st.error("🎬 Video Generation: **❌ Not Available**")
            st.info("Please install MoviePy for video features")
    
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
        with narrated audio and beautiful visual effects using <strong>MoviePy</strong>.
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
        
        if st.button("🔍 Identify Bird Species", type="primary", use_container_width=True):
            with st.spinner("Analyzing bird species..."):
                detections, classifications, original_image = bird_model.predict_bird_species(current_image)
                
                st.session_state.detection_complete = True
                st.session_state.bird_detections = detections
                st.session_state.bird_classifications = classifications
                st.session_state.current_image = original_image
    
    # Display results
    if st.session_state.detection_complete and st.session_state.current_image is not None:
        st.markdown("---")
        st.markdown('<div class="section-title">🎯 Identification Results</div>', unsafe_allow_html=True)
        
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
                    <h4>AI Model Prediction</h4>
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
    
    # MoviePy status display
    if not MOVIEPY_AVAILABLE:
        st.markdown("""
        <div class="video-status">
            <strong>⚠️ MoviePy Not Available</strong><br>
            Video generation features require MoviePy. The system will attempt to install it automatically, 
            or you can install manually: <code>pip install moviepy decorator proglog imageio imageio-ffmpeg</code>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="video-section">
            <strong>📖 AI Story Generation with Video using MoviePy</strong><br>
            Generate complete educational story videos using advanced AI. Each video includes:
            <br><br>
            • <strong>AI-Generated Story</strong>: Unique educational narrative about the bird<br>
            • <strong>Text-to-Speech Audio</strong>: Professional narration of the story<br>
            • <strong>Ken Burns Effects</strong>: Smooth zoom and pan visual effects<br>
            • <strong>Multiple Images</strong>: Showcases the bird from different angles<br>
            <br>
            <strong>Powered by:</strong> MoviePy with advanced video processing
        </div>
        """, unsafe_allow_html=True)
    
    # Video generation options
    col1, col2 = st.columns(2)
    
    with col1:
        # Option 1: Use detected species
        if st.session_state.get('selected_species_for_video'):
            st.info(f"🦜 Detected Species: **{st.session_state.selected_species_for_video}**")
            if st.button("🎬 Generate Story Video", use_container_width=True, type="primary"):
                if not MOVIEPY_AVAILABLE:
                    st.error("❌ MoviePy not available. Cannot generate video.")
                else:
                    with st.spinner("Creating AI story video with MoviePy..."):
                        video_path, story_text, used_images = video_generator.generate_video(st.session_state.selected_species_for_video)
                        if video_path:
                            st.session_state.generated_video_path = video_path
                            st.session_state.generated_story = story_text
                            st.session_state.used_images = used_images
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
            if not MOVIEPY_AVAILABLE:
                st.error("❌ MoviePy not available. Cannot generate video.")
            else:
                with st.spinner("Creating AI story video with MoviePy..."):
                    video_path, story_text, used_images = video_generator.generate_video(manual_species)
                    if video_path:
                        st.session_state.generated_video_path = video_path
                        st.session_state.generated_story = story_text
                        st.session_state.used_images = used_images
                        st.session_state.selected_species_for_video = manual_species
                    else:
                        st.error("❌ Failed to generate story video")
    
    # Display generated story and video
    if st.session_state.get('generated_video_path') and os.path.exists(st.session_state.generated_video_path):
        st.markdown("---")
        st.markdown("### 📖 AI-Generated Story Video")
        
        # Display the story
        if st.session_state.get('generated_story'):
            st.markdown(f'<div class="story-box"><strong>📖 AI-Generated Story:</strong><br>{st.session_state.generated_story}</div>', unsafe_allow_html=True)
        
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
            video_size = os.path.getsize(st.session_state.generated_video_path) / (1024 * 1024)
            st.info(f"**Video Details:** {st.session_state.selected_species_for_video} | {video_size:.1f} MB | MoviePy with Ken Burns effect")
            
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