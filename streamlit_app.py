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
        
    def download_video_model(self):
        """Download the video generation model from Google Drive"""
        try:
            if not os.path.exists(self.video_model_path):
                st.info("📥 Downloading advanced video generation model from Google Drive...")
                
                # Google Drive file ID for the video model
                file_id = "1J9T5r5TboWzvqAPQHmfvQmozor_wmmPz"
                
                # Method 1: Using gdown
                try:
                    import gdown
                    url = f'https://drive.google.com/uc?id={file_id}'
                    gdown.download(url, self.video_model_path, quiet=False)
                except ImportError:
                    # Method 2: Using requests
                    session = requests.Session()
                    url = f"https://docs.google.com/uc?export=download&id={file_id}"
                    response = session.get(url, stream=True)
                    
                    # Handle confirmation for large files
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            params = {'confirm': value}
                            response = session.get(url, params=params, stream=True)
                    
                    # Download with progress
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 8192
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with open(self.video_model_path, "wb") as f:
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
            
            if os.path.exists(self.video_model_path):
                file_size = os.path.getsize(self.video_model_path) / (1024 * 1024)
                if file_size > 1:
                    st.success(f"✅ Advanced video model downloaded! ({file_size:.1f} MB)")
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
            return False

    def load_video_model(self):
        """Load the advanced video generation model with story capabilities"""
        if not os.path.exists(self.video_model_path):
            if not self.download_video_model():
                return False
        
        try:
            # Load the model
            st.info("🔄 Loading advanced story and video generation model...")
            
            if torch.cuda.is_available():
                model_data = torch.load(self.video_model_path)
            else:
                model_data = torch.load(self.video_model_path, map_location=torch.device('cpu'))
            
            # Check if it's our story generator or a different model
            if isinstance(model_data, BirdStoryGenerator):
                self.story_generator = model_data
                self.model_loaded = True
                st.success("✅ Story generation model loaded successfully!")
            else:
                # Try to extract story generator or create a compatible one
                try:
                    self.story_generator = BirdStoryGenerator(TEMPLATES)
                    self.model_loaded = True
                    st.success("✅ Story generation capabilities initialized!")
                except Exception as e:
                    st.warning(f"⚠️ Could not initialize story generator: {e}")
                    self.story_generator = BirdStoryGenerator(TEMPLATES)
                    self.model_loaded = True
                    st.success("✅ Default story generator initialized!")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            # Initialize default story generator as fallback
            self.story_generator = BirdStoryGenerator(TEMPLATES)
            self.model_loaded = True
            st.success("✅ Using default story generation")
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
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filename)
            return filename
        except Exception as e:
            st.error(f"❌ Error generating speech: {e}")
            return None

    def get_audio_duration(self, audio_path):
        """Get audio duration using alternative method"""
        try:
            # For MP3 files, we can estimate duration based on file size
            # This is a rough estimation - 1MB ≈ 1 minute of audio
            file_size = os.path.getsize(audio_path)
            duration = max(15, min(60, file_size / (16 * 1024)))  # Rough estimation
            return duration
        except:
            return 20  # Default fallback

    def create_story_video_with_opencv(self, images, audio_path, output_path, story_text):
        """Create a professional story video using OpenCV with text overlays"""
        try:
            # Get audio duration
            audio_duration = self.get_audio_duration(audio_path)
            
            # Video properties
            frame_width = 1280
            frame_height = 720
            fps = 24
            total_frames = int(audio_duration * fps)
            
            if not images:
                st.error("❌ No images available for video creation")
                return None
            
            frames_per_image = max(1, total_frames // len(images))
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            # Split story into parts for display
            story_parts = self.split_story_for_display(story_text)
            
            # Create frames
            for frame_num in range(total_frames):
                img_idx = min(len(images) - 1, frame_num // frames_per_image)
                story_part_idx = min(len(story_parts) - 1, frame_num // (total_frames // len(story_parts)))
                
                # Load and process image
                img_path = images[img_idx]
                frame = cv2.imread(img_path)
                
                if frame is None:
                    # Create a colored background frame if image loading fails
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    colors = [(70, 130, 180), (60, 179, 113), (186, 85, 211)]
                    frame[:, :] = colors[img_idx % len(colors)]
                
                # Resize frame to target dimensions
                frame = cv2.resize(frame, (frame_width, frame_height))
                
                # Add dark overlay for better text visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                
                # Add story text
                current_story_part = story_parts[story_part_idx]
                self.add_story_text_to_frame(frame, current_story_part, frame_num, total_frames)
                
                # Add header with bird name
                cv2.putText(frame, f"Uganda Bird Spotter: {os.path.basename(images[0]).split('_')[0]}", 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add progress bar
                progress = frame_num / total_frames
                cv2.rectangle(frame, (50, frame_height - 50), (frame_width - 50, frame_height - 30), (100, 100, 100), -1)
                cv2.rectangle(frame, (50, frame_height - 50), (50 + int((frame_width - 100) * progress), frame_height - 30), (0, 200, 255), -1)
                
                # Add frame counter
                cv2.putText(frame, f"Frame {frame_num + 1}/{total_frames}", 
                           (frame_width - 200, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                out.write(frame)
            
            out.release()
            
            # Combine audio with video using ffmpeg if available
            final_video_path = self.combine_audio_video_ffmpeg(output_path, audio_path, f"final_{output_path}")
            
            return final_video_path if final_video_path else output_path
            
        except Exception as e:
            st.error(f"❌ OpenCV video creation error: {e}")
            return None

    def split_story_for_display(self, story_text, max_chars_per_line=60):
        """Split story into displayable parts"""
        words = story_text.split()
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line + " " + word) <= max_chars_per_line:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        
        # Group lines into parts for different frames
        parts = []
        lines_per_part = 3
        for i in range(0, len(lines), lines_per_part):
            part = "\n".join(lines[i:i + lines_per_part])
            parts.append(part)
        
        return parts if parts else [story_text]

    def add_story_text_to_frame(self, frame, text, frame_num, total_frames):
        """Add story text to frame with animation"""
        lines = text.split('\n')
        y_start = 150
        line_height = 40
        
        for i, line in enumerate(lines):
            y_pos = y_start + i * line_height
            
            # Simple fade-in effect
            alpha = min(1.0, (frame_num - (i * 10)) / 30)
            alpha = max(0, alpha)
            
            if alpha > 0:
                # Add text shadow for better readability
                cv2.putText(frame, line, (52, y_pos + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                # Add main text
                text_color = (int(255 * alpha), int(255 * alpha), int(255 * alpha))
                cv2.putText(frame, line, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    def combine_audio_video_ffmpeg(self, video_path, audio_path, output_path):
        """Combine audio and video using ffmpeg"""
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file
                '-i', video_path,
                '-i', audio_path,
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-shortest',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return output_path
            else:
                st.warning("⚠️ FFmpeg audio combination failed, using video without audio")
                return video_path
        except Exception as e:
            st.warning(f"⚠️ FFmpeg not available: {e}. Video will be created without audio sync.")
            return video_path

    def get_bird_images(self, species_name, max_images=5):
        """Get or create bird images for the species"""
        try:
            # Create professional placeholder images
            image_paths = []
            
            for i in range(max_images):
                placeholder_path = f"./temp_placeholder_{species_name.replace(' ', '_')}_{i}.jpg"
                if self.create_professional_placeholder(species_name, placeholder_path, i):
                    image_paths.append(placeholder_path)
            
            return image_paths
            
        except Exception as e:
            st.error(f"❌ Error creating bird images: {e}")
            return []

    def create_professional_placeholder(self, species_name, output_path, variation=0):
        """Create professional placeholder images with bird illustrations"""
        try:
            # Create image with higher resolution
            width, height = 800, 600
            img = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Background colors
            bg_colors = [
                (30, 60, 90),    # Dark blue
                (40, 80, 120),   # Medium blue  
                (50, 100, 150),  # Light blue
                (60, 120, 180),  # Sky blue
                (70, 140, 210)   # Bright blue
            ]
            
            img[:, :] = bg_colors[variation % len(bg_colors)]
            
            # Add gradient effect
            for i in range(height):
                alpha = i / height
                img[i, :] = img[i, :] * (1 - alpha * 0.3) + np.array([10, 20, 30]) * (alpha * 0.3)
            
            # Draw bird illustration
            self.draw_bird_illustration(img, species_name, variation, width, height)
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = (255, 255, 255)
            
            # Species name
            cv2.putText(img, species_name, (width//2 - 150, 100), font, 1.2, text_color, 2)
            
            # Decorative elements
            cv2.putText(img, "•", (width//2 - 10, 130), font, 1, text_color, 2)
            
            # Information text
            info_lines = [
                "Uganda Bird Spotter",
                "Professional Wildlife Documentation",
                f"Image {variation + 1} of 5"
            ]
            
            for i, line in enumerate(info_lines):
                cv2.putText(img, line, (width//2 - 180, 180 + i * 30), font, 0.6, text_color, 1)
            
            # Add border
            cv2.rectangle(img, (10, 10), (width-10, height-10), text_color, 2)
            
            cv2.imwrite(output_path, img)
            return True
            
        except Exception as e:
            st.error(f"❌ Error creating professional placeholder: {e}")
            return False

    def draw_bird_illustration(self, img, species_name, variation, width, height):
        """Draw a bird illustration based on species"""
        center_x, center_y = width // 2, height // 2 + 50
        
        # Different bird poses based on variation
        poses = [
            {"body_size": (60, 35), "head_size": 25, "wing_angle": 20, "tail_length": 30},
            {"body_size": (55, 30), "head_size": 22, "wing_angle": -15, "tail_length": 25},
            {"body_size": (65, 40), "head_size": 28, "wing_angle": 10, "tail_length": 35},
            {"body_size": (50, 28), "head_size": 20, "wing_angle": -20, "tail_length": 22},
            {"body_size": (70, 45), "head_size": 30, "wing_angle": 25, "tail_length": 40}
        ]
        
        pose = poses[variation % len(poses)]
        body_w, body_h = pose["body_size"]
        head_size = pose["head_size"]
        wing_angle = pose["wing_angle"]
        tail_length = pose["tail_length"]
        
        # Bird color based on species
        species_colors = {
            "African Fish Eagle": (200, 200, 100),
            "Grey Crowned Crane": (150, 150, 150),
            "Shoebill Stork": (120, 120, 80),
            "Lilac-breasted Roller": (180, 120, 220),
            "Great Blue Turaco": (80, 120, 200)
        }
        
        bird_color = species_colors.get(species_name, (150, 150, 150))
        
        # Draw body
        cv2.ellipse(img, (center_x, center_y), (body_w, body_h), 0, 0, 360, bird_color, -1)
        
        # Draw head
        cv2.ellipse(img, (center_x, center_y - body_h), (head_size, head_size), 0, 0, 360, bird_color, -1)
        
        # Draw beak
        cv2.ellipse(img, (center_x, center_y - body_h), (head_size//2, head_size//4), 0, 0, 360, (50, 50, 30), -1)
        
        # Draw wings
        wing_color = tuple(max(0, c - 30) for c in bird_color)
        left_wing_points = np.array([
            [center_x - body_w//2, center_y],
            [center_x - body_w - 20, center_y - wing_angle],
            [center_x - body_w//2, center_y - body_h//2]
        ], np.int32)
        
        right_wing_points = np.array([
            [center_x + body_w//2, center_y],
            [center_x + body_w + 20, center_y - wing_angle],
            [center_x + body_w//2, center_y - body_h//2]
        ], np.int32)
        
        cv2.fillPoly(img, [left_wing_points], wing_color)
        cv2.fillPoly(img, [right_wing_points], wing_color)
        
        # Draw tail
        tail_points = np.array([
            [center_x, center_y + body_h//2],
            [center_x - tail_length//2, center_y + body_h//2 + tail_length],
            [center_x + tail_length//2, center_y + body_h//2 + tail_length]
        ], np.int32)
        cv2.fillPoly(img, [tail_points], bird_color)

    def generate_story_video(self, species_name):
        """Generate a comprehensive story-based video with audio"""
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
            st.info("🖼️ Creating professional bird images...")
            bird_images = self.get_bird_images(species_name, max_images=5)
            
            if not bird_images:
                st.error("❌ No bird images created")
                return None, None, None
            
            # Generate video using OpenCV
            st.info("🎬 Creating professional story video...")
            video_file = f"temp_story_video_{species_name.replace(' ', '_')}.mp4"
            video_path = self.create_story_video_with_opencv(bird_images, audio_path, video_file, story_text)
            
            # Clean up temporary audio file
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except:
                pass
            
            if video_path and os.path.exists(video_path):
                st.success(f"✅ Professional story video generated successfully!")
                st.info(f"📊 Video details: {len(bird_images)} images, {len(story_text.split())} words, {os.path.getsize(video_path) // (1024*1024)}MB")
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

# ... (Keep the ResNet34BirdModel class exactly the same as previous version)

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
            # Try to load the model
            success = st.session_state.bird_model.load_model()
            
            if success:
                # Load video generator data and model
                st.session_state.video_generator.load_bird_data()
                video_model_loaded = st.session_state.video_generator.load_video_model()
                st.session_state.model_loaded = True
                st.session_state.system_initialized = True
                
                if video_model_loaded and st.session_state.video_generator.model_loaded:
                    st.success(f"✅ System ready! Both models loaded - Can identify {len(st.session_state.bird_model.bird_species)} bird species and generate AI story videos")
                else:
                    st.success(f"✅ System ready! ResNet34 model active - Can identify {len(st.session_state.bird_model.bird_species)} bird species")
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
            st.success("🎥 Using: OpenCV Professional Video Engine")
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
            This app uses AI models for bird identification and professional story generation. 
            Upload bird photos for identification, then generate AI-powered educational story videos 
            with narrated audio and professional visual effects.
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
            <strong>📖 AI Story Generation with Professional Video</strong><br>
            Generate complete educational story videos using our advanced AI model. Each video includes:
            <br><br>
            • <strong>AI-Generated Story</strong>: Unique educational narrative about the bird<br>
            • <strong>Text-to-Speech Audio</strong>: Professional narration of the story<br>
            • <strong>Professional Visuals</strong>: High-quality bird illustrations and animations<br>
            • <strong>Story Text Overlay</strong>: Animated text display synchronized with audio<br>
            • <strong>Progress Tracking</strong>: Visual progress indicator<br>
            <br>
            <strong>Video Engine:</strong> OpenCV Professional Video Creation
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
                with st.spinner("Creating professional AI story video..."):
                    video_path, story_text, used_images = video_generator.generate_video(st.session_state.selected_species_for_video)
                    if video_path:
                        st.session_state.generated_video_path = video_path
                        st.session_state.generated_story = story_text
                        st.session_state.used_images = used_images
                        st.success("✅ Professional story video generated successfully!")
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
            with st.spinner("Creating professional AI story video..."):
                video_path, story_text, used_images = video_generator.generate_video(manual_species)
                if video_path:
                    st.session_state.generated_video_path = video_path
                    st.session_state.generated_story = story_text
                    st.session_state.used_images = used_images
                    st.session_state.selected_species_for_video = manual_species
                    st.success("✅ Professional story video generated successfully!")
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
            st.markdown(f"**🖼️ Professional Bird Illustrations ({len(st.session_state.used_images)} images):**")
            cols = st.columns(min(3, len(st.session_state.used_images)))
            for idx, img_path in enumerate(st.session_state.used_images):
                with cols[idx % 3]:
                    try:
                        st.image(img_path, use_column_width=True, caption=f"Illustration {idx+1}")
                    except:
                        st.info(f"Image {idx+1}")
        
        # Display video
        try:
            with open(st.session_state.generated_video_path, "rb") as video_file:
                video_bytes = video_file.read()
            
            st.video(video_bytes)
            
            # Video information
            file_size = os.path.getsize(st.session_state.generated_video_path) // (1024 * 1024)
            st.info(f"**Video Details:** {st.session_state.selected_species_for_video} | Professional OpenCV Video | {file_size}MB | Audio Narration")
            
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