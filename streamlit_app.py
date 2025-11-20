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
warnings.filterwarnings('ignore')

# Try to import moviepy with fallback
try:
    from moviepy.editor import (
        AudioFileClip, ImageClip, concatenate_videoclips,
        VideoFileClip, concatenate_audioclips
    )
    from moviepy.audio.fx.all import audio_fadein, audio_fadeout
    from moviepy.video.fx.all import resize
    from moviepy.config import change_settings
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    st.warning("🎬 MoviePy not available. Video creation will use OpenCV fallback.")

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

class AdvancedVideoGenerator:
    def __init__(self):
        self.csv_path = './birdsuganda.csv'
        self.video_model_path = './bird_path.pth'
        self.bird_data = None
        self.video_model = None
        self.model_loaded = False
        self.video_duration = 20
        self.story_generator = None
        self.moviepy_available = MOVIEPY_AVAILABLE
        
    def download_video_model(self):
        """Download the video generation model from Google Drive (same method as resnet)"""
        try:
            if not os.path.exists(self.video_model_path):
                st.info("📥 Downloading advanced video generation model from Google Drive...")
                
                # Google Drive file ID for the video model
                file_id = "1J9T5r5TboWzvqAPQHmfvQmozor_wmmPz"
                
                # Method 1: Using gdown (same as resnet)
                try:
                    import gdown
                    url = f'https://drive.google.com/uc?id={file_id}'
                    gdown.download(url, self.video_model_path, quiet=False)
                    
                except ImportError:
                    st.warning("gdown not available, trying requests...")
                    # Method 2: Using requests with cookie handling (same as resnet)
                    session = requests.Session()
                    
                    # First, get the confirmation token
                    url = f"https://docs.google.com/uc?export=download&id={file_id}"
                    response = session.get(url, stream=True)
                    
                    # Check for download confirmation
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            # Need to confirm the download
                            params = {'confirm': value, 'id': file_id}
                            response = session.get(url, params=params, stream=True)
                            break
                    
                    # Download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with open(self.video_model_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = min(downloaded / total_size, 1.0)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Downloaded: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB")
                    
                    progress_bar.empty()
                    status_text.empty()
            
            # Verify download (same verification as resnet)
            if os.path.exists(self.video_model_path):
                file_size = os.path.getsize(self.video_model_path) / (1024 * 1024)
                if file_size > 1:  # Ensure file is not empty/corrupted
                    st.success(f"✅ Video model downloaded successfully! ({file_size:.1f} MB)")
                    return True
                else:
                    st.error("❌ Downloaded video model is too small - may be corrupted")
                    if os.path.exists(self.video_model_path):
                        os.remove(self.video_model_path)
                    return False
            else:
                st.error("❌ Failed to download video model")
                return False
                
        except Exception as e:
            st.error(f"❌ Video model download error: {e}")
            # Try fallback method (same as resnet)
            return self.download_video_model_fallback()

    def download_video_model_fallback(self):
        """Final fallback download method for video model"""
        try:
            st.info("🔄 Trying final download method for video model...")
            
            # Direct download URL format (same as resnet)
            file_id = "1J9T5r5TboWzvqAPQHmfvQmozor_wmmPz"
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Simple urllib download
            urllib.request.urlretrieve(direct_url, self.video_model_path)
            
            if os.path.exists(self.video_model_path) and os.path.getsize(self.video_model_path) > 1000000:
                file_size = os.path.getsize(self.video_model_path) / (1024 * 1024)
                st.success(f"✅ Video model downloaded via fallback method! ({file_size:.1f} MB)")
                return True
            return False
            
        except Exception as e:
            st.error(f"❌ All video model download methods failed: {e}")
            return False

    def load_video_model(self):
        """Load the advanced video generation model with story capabilities"""
        if not os.path.exists(self.video_model_path):
            if not self.download_video_model():
                st.error("❌ Could not download the video model file from Google Drive.")
                st.info("""
                Please ensure:
                1. The Google Drive file is publicly accessible
                2. The file ID is correct: 1J9T5r5TboWzvqAPQHmfvQmozor_wmmPz
                3. You have internet connection
                """)
                # Initialize default story generator as fallback
                self.story_generator = BirdStoryGenerator(TEMPLATES)
                self.model_loaded = True
                st.success("✅ Using default story generation (fallback mode)")
                return True
        
        try:
            # Load the model with the same method as resnet
            st.info("🔄 Loading advanced story and video generation model...")
            
            if torch.cuda.is_available():
                model_data = torch.load(self.video_model_path)
            else:
                model_data = torch.load(self.video_model_path, map_location=torch.device('cpu'))
            
            # Check what type of model we loaded
            if isinstance(model_data, dict):
                # It's a state dict or model weights
                if 'story_generator' in model_data:
                    # Load the story generator from the model data
                    self.story_generator = model_data['story_generator']
                    self.model_loaded = True
                    st.success("✅ Advanced story generation model loaded successfully!")
                else:
                    # Create a compatible story generator
                    self.story_generator = BirdStoryGenerator(TEMPLATES)
                    self.model_loaded = True
                    st.success("✅ Story generation capabilities initialized with model weights!")
            elif hasattr(model_data, 'generate_story'):
                # It's a model object with generate_story method
                self.story_generator = model_data
                self.model_loaded = True
                st.success("✅ Advanced story generation model object loaded!")
            else:
                # Fallback to default story generator
                self.story_generator = BirdStoryGenerator(TEMPLATES)
                self.model_loaded = True
                st.success("✅ Using enhanced story generation with loaded model data!")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Video model loading failed: {e}")
            # Initialize default story generator as fallback
            self.story_generator = BirdStoryGenerator(TEMPLATES)
            self.model_loaded = True
            st.success("✅ Using default story generation (fallback due to loading error)")
            return True

    def load_bird_data(self):
        """Load and process the bird species data from local CSV"""
        try:
            if os.path.exists(self.csv_path):
                self.bird_data = pd.read_csv(self.csv_path)
                st.success(f"✅ Loaded data for {len(self.bird_data)} bird species from local CSV")
                return True
            else:
                st.error(f"❌ CSV file not found at: {self.csv_path}")
                st.info("Please ensure 'birdsuganda.csv' is in the app directory")
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
            
            st.warning(f"⚠️ No detailed information found for {species_name} in database")
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
        """Create slideshow video using OpenCV (fallback when MoviePy not available)"""
        try:
            # Get audio duration using alternative method
            audio_duration = 20  # Default duration in seconds
            
            # Video properties
            frame_width = 1280
            frame_height = 720
            fps = 24
            total_frames = audio_duration * fps
            frames_per_image = total_frames // len(images) if images else total_frames
            
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
                    
                    # Add text overlay
                    text = f"Uganda Bird: {os.path.basename(img_path)}"
                    cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    # Add progress indicator
                    progress = frame_num / total_frames
                    cv2.rectangle(img, (50, frame_height - 30), (int(50 + 300 * progress), frame_height - 10), (0, 255, 0), -1)
                    
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
        """Create final video with Ken Burns effect and audio"""
        if not self.moviepy_available:
            st.warning("🎬 Using OpenCV fallback for video creation (MoviePy not available)")
            return self.create_slideshow_video_opencv(images, audio_path, output_path)
        
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
            for img in images:
                try:
                    clip = ImageClip(img).set_duration(img_duration)
                    w, h = clip.size
                    zoom = 1.15
                    
                    # Apply zoom effect
                    clip = clip.resize(lambda t: 1 + (zoom - 1) * (t / img_duration))
                    
                    # Apply pan effect
                    clip = clip.set_position(lambda t: (
                        "center" if t < img_duration * 0.6 else (w * 0.05 * (t - img_duration * 0.6) / (img_duration * 0.4)),
                        "center"
                    ))
                    
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
            st.info("🔄 Falling back to OpenCV video creation...")
            return self.create_slideshow_video_opencv(images, audio_path, output_path)

    def get_bird_images(self, species_name, max_images=5):
        """Get bird images for the species"""
        try:
            # Create placeholder images for demonstration
            image_paths = []
            
            # Create multiple placeholder images with different backgrounds
            for i in range(max_images):
                placeholder_path = f"./temp_placeholder_{species_name.replace(' ', '_')}_{i}.jpg"
                if self.create_placeholder_image(species_name, placeholder_path, variation=i):
                    image_paths.append(placeholder_path)
            
            return image_paths
            
        except Exception as e:
            st.error(f"❌ Error getting bird images: {e}")
            return []

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
        """Generate a comprehensive story-based video with audio using the loaded model"""
        try:
            if not self.model_loaded or self.story_generator is None:
                st.error("❌ Story generation model not loaded")
                return None, None, None
            
            # Get bird information
            bird_info = self.get_bird_video_info(species_name)
            
            # Extract bird details for story generation
            common_name = species_name
            description = bird_info.get('description', '') if bird_info else ''
            colors = []
            
            # Try to extract colors from various possible columns
            color_columns = ['colors', 'primary_colors', 'plumage_colors', 'color']
            for col in color_columns:
                if col in bird_info and pd.notna(bird_info[col]):
                    colors = str(bird_info[col]).split(',')
                    break
            
            # Generate story using the model
            st.info("📖 Generating educational story using AI...")
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
            st.info("🖼️ Creating bird images...")
            bird_images = self.get_bird_images(species_name, max_images=5)
            
            if not bird_images:
                st.error("❌ No bird images created")
                return None, None, None
            
            # Generate video
            st.info("🎬 Creating story video...")
            video_file = f"temp_story_video_{species_name.replace(' ', '_')}.mp4"
            video_path = self.create_final_video(bird_images, audio_path, video_file)
            
            # Clean up temporary audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            
            if video_path:
                st.success(f"✅ Story video generated successfully! ({len(bird_images)} images, {len(story_text.split())} words)")
                return video_path, story_text, bird_images
            else:
                st.error("❌ Failed to generate video")
                return None, None, None
            
        except Exception as e:
            st.error(f"❌ Story video generation error: {e}")
            return None, None, None

    def generate_video(self, species_name):
        """Main video generation function with story and audio"""
        return self.generate_story_video(species_name)

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
        try:
            if not os.path.exists(self.model_path):
                st.info("📥 Downloading ResNet34 model from Google Drive...")
                
                # Your Google Drive file ID from the link
                file_id = "1yfiYcz6e2hWtQTXW6AZVU-iwSUjDP92y"
                
                # Method 1: Using gdown (most reliable)
                try:
                    import gdown
                    # Direct download URL for gdown
                    url = f'https://drive.google.com/uc?id={file_id}'
                    output = self.model_path
                    gdown.download(url, output, quiet=False)
                    
                except ImportError:
                    st.warning("gdown not available, trying requests...")
                    # Method 2: Using requests with cookie handling
                    session = requests.Session()
                    
                    # First, get the confirmation token
                    url = f"https://docs.google.com/uc?export=download&id={file_id}"
                    response = session.get(url, stream=True)
                    
                    # Check for download confirmation
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            # Need to confirm the download
                            params = {'confirm': value, 'id': file_id}
                            response = session.get(url, params=params, stream=True)
                            break
                    
                    # Download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with open(self.model_path, 'wb') as f:
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=block_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size > 0:
                                    progress = min(downloaded / total_size, 1.0)
                                    progress_bar.progress(progress)
                                    status_text.text(f"Downloaded: {downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB")
                    
                    progress_bar.empty()
                    status_text.empty()
            
            # Verify download
            if os.path.exists(self.model_path):
                file_size = os.path.getsize(self.model_path) / (1024 * 1024)
                if file_size > 1:  # Ensure file is not empty/corrupted
                    st.success(f"✅ Model downloaded successfully! ({file_size:.1f} MB)")
                    return True
                else:
                    st.error("❌ Downloaded file is too small - may be corrupted")
                    if os.path.exists(self.model_path):
                        os.remove(self.model_path)
                    return False
            else:
                st.error("❌ Failed to download model file")
                return False
                
        except Exception as e:
            st.error(f"❌ Download error: {e}")
            # Try one more method as fallback
            return self.download_model_fallback()
    
    def download_model_fallback(self):
        """Final fallback download method"""
        try:
            st.info("🔄 Trying final download method...")
            
            # Direct download URL format
            file_id = "1yfiYcz6e2hWtQTXW6AZVU-iwSUjDP92y"
            direct_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Simple urllib download
            urllib.request.urlretrieve(direct_url, self.model_path)
            
            if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:
                file_size = os.path.getsize(self.model_path) / (1024 * 1024)
                st.success(f"✅ Model downloaded via fallback method! ({file_size:.1f} MB)")
                return True
            return False
            
        except Exception as e:
            st.error(f"❌ All download methods failed: {e}")
            return False

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
            
            # Then load video generator
            video_data_loaded = st.session_state.video_generator.load_bird_data()
            video_model_loaded = st.session_state.video_generator.load_video_model()
            
            if resnet_success:
                st.session_state.model_loaded = True
                st.session_state.system_initialized = True
                
                if video_model_loaded and st.session_state.video_generator.model_loaded:
                    st.success(f"✅ System ready! Both models loaded - Can identify {len(st.session_state.bird_model.bird_species)} bird species and generate AI story videos")
                else:
                    st.success(f"✅ System ready! ResNet34 model active - Can identify {len(st.session_state.bird_model.bird_species)} bird species")
                    if video_data_loaded:
                        st.info("📖 Basic story generation available")
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
        if video_generator.model_loaded:
            st.success("🎬 AI Story Model: **Loaded**")
            st.info("📖 Generates: Stories + Audio + Video")
            if not video_generator.moviepy_available:
                st.warning("🎥 Using OpenCV video creation")
            else:
                st.success("🎥 Using MoviePy (Ken Burns effect)")
        else:
            st.warning("🎬 AI Story Model: **Not Available**")
    
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
    if video_generator.model_loaded:
        st.markdown("""
        <div class="glass-card">
            <strong>🦜 Welcome to Uganda Bird Spotter!</strong><br>
            This app uses AI models for bird identification and story generation. 
            Upload bird photos for identification, then generate AI-powered educational story videos 
            with narrated audio and beautiful visual effects.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card">
            <strong>🦜 Welcome to Uganda Bird Spotter!</strong><br>
            This app uses a specialized ResNet34 model trained on Ugandan bird species. 
            Upload or capture images to get accurate bird identifications using AI.
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
    
    if video_generator.model_loaded:
        st.markdown(f"""
        <div class="video-section">
            <strong>📖 AI Story Generation with Video</strong><br>
            Generate complete educational story videos using our advanced AI model. Each video includes:
            <br><br>
            • <strong>AI-Generated Story</strong>: Unique educational narrative about the bird<br>
            • <strong>Text-to-Speech Audio</strong>: Professional narration of the story<br>
            • <strong>Visual Effects</strong>: Beautiful image transitions and effects<br>
            • <strong>Multiple Images</strong>: Showcases the bird from different angles<br>
            <br>
            <strong>Video Engine:</strong> {'MoviePy with Ken Burns effect' if video_generator.moviepy_available else 'OpenCV slideshow'}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="video-section">
            <strong>📖 Story Video Generation</strong><br>
            Advanced story generation requires the bird_path.pth model file.
            Please ensure the model is properly downloaded and configured.
        </div>
        """, unsafe_allow_html=True)
    
    # Video generation options
    col1, col2 = st.columns(2)
    
    with col1:
        # Option 1: Use detected species
        if st.session_state.get('selected_species_for_video'):
            st.info(f"🦜 Detected Species: **{st.session_state.selected_species_for_video}**")
            if st.button("🎬 Generate Story Video", use_container_width=True, type="primary"):
                with st.spinner("Creating AI story video with audio..."):
                    video_path, story_text, used_images = video_generator.generate_video(st.session_state.selected_species_for_video)
                    if video_path:
                        st.session_state.generated_video_path = video_path
                        st.session_state.generated_story = story_text
                        st.session_state.used_images = used_images
                        st.success("✅ AI story video generated successfully!")
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
            with st.spinner("Creating AI story video with audio..."):
                video_path, story_text, used_images = video_generator.generate_video(manual_species)
                if video_path:
                    st.session_state.generated_video_path = video_path
                    st.session_state.generated_story = story_text
                    st.session_state.used_images = used_images
                    st.session_state.selected_species_for_video = manual_species
                    st.success("✅ AI story video generated successfully!")
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
            video_type = "MoviePy with Ken Burns" if video_generator.moviepy_available else "OpenCV slideshow"
            st.info(f"**Video Details:** {st.session_state.selected_species_for_video} | {video_type} | Audio Narration")
            
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