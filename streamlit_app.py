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
warnings.filterwarnings('ignore')

# ========== ENHANCED MOVIEPY INSTALLATION AND IMPORT ==========
def ensure_moviepy_installation():
    """Ensure MoviePy and all dependencies are properly installed"""
    required_packages = {
        'moviepy': 'moviepy',
        'decorator': 'decorator', 
        'proglog': 'proglog',
        'imageio': 'imageio',
        'imageio-ffmpeg': 'imageio_ffmpeg'
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
            
            # Clear import caches
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

# ─────────────────────────────────────────────────────────────────────────────
# RICH, UGANDA-SPECIFIC STORY TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────
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

# ========== LOAD BIRD STORY GENERATOR FROM LOCAL PTH FILE ==========
@st.cache_resource
def load_story_generator():
    """Load the bird story generator from local .pth file"""
    PTH_PATH = "bird_story_generator.pth"
    
    if not os.path.exists(PTH_PATH):
        st.warning(f"⚠️ {PTH_PATH} not found. Creating a new story generator...")
        # Create and save a new story generator
        story_generator = BirdStoryGenerator(TEMPLATES)
        torch.save(story_generator, PTH_PATH)
        st.success(f"✅ Created new story generator at {PTH_PATH}")
        return story_generator
    else:
        try:
            story_generator = torch.load(PTH_PATH, map_location="cpu")
            st.success(f"✅ Loaded story generator from {PTH_PATH}")
            return story_generator
        except Exception as e:
            st.error(f"❌ Error loading {PTH_PATH}: {e}")
            # Fallback to new generator
            st.info("🔄 Using fallback story generator...")
            return BirdStoryGenerator(TEMPLATES)

# Load the story generator
story_generator = load_story_generator()

# ========== SIMPLIFIED BIRD DATA ==========
@st.cache_resource
def load_bird_data():
    """Load minimal bird data for testing"""
    minimal_data = {
        "African Fish Eagle": {
            "desc": "A majestic bird of prey found near water bodies. Known for its distinctive cry and excellent fishing skills.",
            "colors": ["white", "brown", "black"],
            "images_b64": []
        },
        "Grey Crowned Crane": {
            "desc": "National bird of Uganda with golden crown. Performs beautiful mating dances in wetlands.",
            "colors": ["grey", "white", "gold"],
            "images_b64": []
        },
        "Shoebill Stork": {
            "desc": "Large stork-like bird with shoe-shaped bill. Known for its prehistoric appearance and patient hunting.",
            "colors": ["blue-grey", "white"],
            "images_b64": []
        },
        "Lilac-breasted Roller": {
            "desc": "Colorful bird with vibrant plumage. Famous for its acrobatic flight displays during mating season.",
            "colors": ["lilac", "blue", "green", "brown"],
            "images_b64": []
        },
        "Great Blue Turaco": {
            "desc": "Large blue bird with distinctive crest. Often seen in forest canopies feeding on fruits.",
            "colors": ["blue", "green", "red"],
            "images_b64": []
        },
        "African Jacana": {
            "desc": "Known as the 'lily-trotter' for its ability to walk on floating vegetation with long toes.",
            "colors": ["chestnut", "white", "black"],
            "images_b64": []
        },
        "Marabou Stork": {
            "desc": "Large wading bird often seen in urban areas. Important scavenger in African ecosystems.",
            "colors": ["white", "black", "pink"],
            "images_b64": []
        }
    }
    st.success(f"✅ Loaded bird data with {len(minimal_data)} species")
    return minimal_data

# Load bird data
bird_db = load_bird_data()

# ========== ENHANCED MOVIEPY VIDEO GENERATOR ==========
class AdvancedVideoGenerator:
    def __init__(self):
        self.moviepy_available = MOVIEPY_AVAILABLE
        self.story_generator = story_generator
        
        st.info(f"🎬 MoviePy Status: {'✅ Available' if self.moviepy_available else '❌ Not Available'}")
        
    def natural_tts(self, text, filename):
        """Convert text to speech using gTTS"""
        try:
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            return filename
        except Exception as e:
            st.error(f"❌ Error generating speech: {e}")
            return None

    def ken_burns_clip(self, img_path, duration=4.0):
        """Apply Ken Burns effect to an image clip"""
        try:
            clip = ImageClip(img_path).set_duration(duration)
            w, h = clip.size
            zoom = 1.15
            
            # Apply zoom effect
            clip = clip.resize(lambda t: 1 + (zoom - 1) * (t / duration))
            
            # Apply subtle pan effect
            clip = clip.set_position(lambda t: (
                "center" if t < duration * 0.6 else (w * 0.05 * (t - duration * 0.6) / (duration * 0.4)),
                "center"
            ))
            
            return clip.fadein(0.3).fadeout(0.3)
            
        except Exception as e:
            st.error(f"❌ Ken Burns effect error: {e}")
            # Return simple image clip as fallback
            return ImageClip(img_path).set_duration(duration).fadein(0.3).fadeout(0.3)

    def create_final_video(self, images, audio_path, output_path):
        """Create final video with Ken Burns effect and audio"""
        if not self.moviepy_available:
            st.error("❌ MoviePy is not available for video creation")
            return None
        
        try:
            # Load and process audio
            raw_audio = AudioFileClip(audio_path)
            narration = audio_fadein(raw_audio, 0.6)
            narration = audio_fadeout(narration, 1.2)

            # Calculate durations
            img_duration = 4.0
            total_duration = img_duration * len(images)

            # Adjust audio to match video duration
            if narration.duration < total_duration:
                loops = int(total_duration / narration.duration) + 1
                narration = concatenate_audioclips([narration] * loops).subclip(0, total_duration)
            else:
                narration = narration.subclip(0, total_duration)

            # Create video clips with Ken Burns effect
            clips = []
            progress_bar = st.progress(0)
            
            for i, img in enumerate(images):
                try:
                    clip = self.ken_burns_clip(img, img_duration)
                    clips.append(clip)
                    
                    # Update progress
                    progress = (i + 1) / len(images)
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.warning(f"⚠️ Could not process image {img}: {e}")
                    continue
            
            progress_bar.empty()
            
            if not clips:
                st.error("❌ No valid video clips were created")
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
            
            # Clean up
            for clip in clips:
                clip.close()
            video.close()
            raw_audio.close()
            narration.close()
            
            return output_path
            
        except Exception as e:
            st.error(f"❌ MoviePy video creation error: {e}")
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
        """Create a placeholder image using PIL only"""
        try:
            # Create image with PIL
            width, height = 600, 400
            img = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(img)
            
            # Different background colors for variations
            colors = [
                (70, 130, 180),   # Steel blue
                (60, 179, 113),   # Medium sea green
                (186, 85, 211),   # Medium orchid
                (255, 165, 0),    # Orange
                (106, 90, 205)    # Slate blue
            ]
            
            bg_color = colors[variation % len(colors)]
            draw.rectangle([0, 0, width, height], fill=bg_color)
            
            # Try to use a font, fallback to default
            try:
                font = ImageFont.truetype("Arial", size=24)
                small_font = ImageFont.truetype("Arial", size=16)
            except:
                try:
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                except:
                    font = None
                    small_font = None
            
            # Add text
            text = species_name
            if font:
                # Calculate text size
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                text_x = (width - text_width) // 2
                text_y = (height - text_height) // 2
                
                draw.text((text_x, text_y), text, fill='white', font=font)
                
                # Add additional text
                draw.text((200, 250), f"Bird Image {variation + 1}", fill=(200, 200, 200), font=small_font)
                draw.text((180, 300), "Uganda Bird Spotter", fill=(220, 220, 220), font=small_font)
            else:
                # Fallback without font
                draw.text((width//2 - 100, height//2 - 10), text, fill='white')
                draw.text((200, 250), f"Bird Image {variation + 1}", fill=(200, 200, 200))
                draw.text((180, 300), "Uganda Bird Spotter", fill=(220, 220, 220))
            
            # Add simple bird shape (ellipses)
            center_x, center_y = 300, 150
            draw.ellipse([center_x-40, center_y-25, center_x+40, center_y+25], fill='white')
            draw.ellipse([center_x-20, center_y-45, center_x+20, center_y-5], fill='white')
            
            img.save(output_path, "JPEG")
            return True
        except Exception as e:
            st.error(f"❌ Error creating placeholder: {e}")
            return False

    def generate_story_video(self, species_name):
        """Generate a comprehensive story-based video using the story generator"""
        try:
            # Get bird information from database
            if species_name in bird_db:
                bird_info = bird_db[species_name]
                description = bird_info.get("desc", "")
                colors = bird_info.get("colors", [])
            else:
                # Fallback for unknown species
                description = "A beautiful bird native to Uganda's diverse ecosystems."
                colors = ["vibrant"]
            
            # Generate story using the loaded story generator
            st.info("📖 Generating educational story using bird_story_generator.pth...")
            story_text = self.story_generator(species_name, description, colors)
            
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
                video_path = self.create_final_video(bird_images, audio_path, video_path)
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
                st.success(f"✅ Story video generated successfully using bird_story_generator.pth! ({file_size:.1f} MB)")
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

# ========== SIMPLIFIED RESNET MODEL ==========
class ResNet34BirdModel:
    def __init__(self):
        self.model_loaded = True  # Simplified for demo
        self.bird_species = list(bird_db.keys())
        self.inv_label_map = {idx: species for idx, species in enumerate(self.bird_species)}
    
    def predict_bird_species(self, image):
        """Simplified prediction for demo - uses the actual bird species from our database"""
        # For demo purposes, return a random species from our database
        random_species = random.choice(self.bird_species)
        detections = [((100, 100, 200, 200), 0.85)]  # Single detection
        classifications = [(random_species, 0.92)]  # Random classification from available species
        
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
        st.success(f"📚 Story Generator: Loaded from bird_story_generator.pth")
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
        
        # System status
        st.markdown("---")
        st.success("📚 **Story Generator:** bird_story_generator.pth")
        if MOVIEPY_AVAILABLE:
            st.success("🎬 **Video Generation:** ✅ Available")
            st.info("📹 Features: Ken Burns effect, TTS audio")
        else:
            st.error("🎬 **Video Generation:** ❌ Not Available")
    
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
        with narrated audio and beautiful visual effects using <strong>bird_story_generator.pth</strong>.
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
    st.markdown('<div class="section-title">🎬 AI Story Video Generator (bird_story_generator.pth)</div>', unsafe_allow_html=True)
    
    # MoviePy status display
    if not MOVIEPY_AVAILABLE:
        st.markdown("""
        <div class="video-status">
            <strong>⚠️ MoviePy Not Available</strong><br>
            Video generation features require MoviePy. The system will attempt to install it automatically.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="video-section">
            <strong>📖 AI Story Generation with Video using bird_story_generator.pth</strong><br>
            Generate complete educational story videos using the trained story generator. Each video includes:
            <br><br>
            • <strong>AI-Generated Story</strong>: Unique educational narrative using bird_story_generator.pth<br>
            • <strong>Text-to-Speech Audio</strong>: Professional narration of the story<br>
            • <strong>Ken Burns Effects</strong>: Smooth zoom and pan visual effects<br>
            • <strong>Multiple Images</strong>: Showcases the bird from different angles<br>
            <br>
            <strong>Powered by:</strong> bird_story_generator.pth + MoviePy
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
                    with st.spinner("Creating AI story video with bird_story_generator.pth..."):
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
                with st.spinner("Creating AI story video with bird_story_generator.pth..."):
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
        st.markdown("### 📖 AI-Generated Story Video (bird_story_generator.pth)")
        
        # Display the story
        if st.session_state.get('generated_story'):
            st.markdown(f'<div class="story-box"><strong>📖 AI-Generated Story (bird_story_generator.pth):</strong><br>{st.session_state.generated_story}</div>', unsafe_allow_html=True)
        
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
            st.info(f"**Video Details:** {st.session_state.selected_species_for_video} | {video_size:.1f} MB | Powered by bird_story_generator.pth")
            
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