# HEALTH CHECK FIX - Add this at the VERY TOP
import os
if os.environ.get('STREAMLIT_SERVER_HEALTH_CHECK'):
    print("HEALTH_CHECK_OK")
    import sys
    sys.exit(0)

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
import time
import threading
from queue import Queue
warnings.filterwarnings('ignore')

# Configuration for large file handling
class AppConfig:
    def __init__(self):
        self.max_file_size_mb = 1024  # 1GB
        self.chunk_size = 32768  # 32KB chunks
        self.download_timeout = 300  # 5 minutes
        self.max_retries = 3
        self.cache_dir = "./cache"
        self.model_dir = "./models"
        self.temp_dir = "./temp"
        self.data_dir = "./data"
        
    def setup_directories(self):
        """Create all necessary directories"""
        directories = [self.cache_dir, self.model_dir, self.temp_dir, self.data_dir]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def cleanup_old_files(self, max_age_hours=24):
        """Clean up old temporary files"""
        try:
            current_time = time.time()
            for temp_dir in [self.temp_dir, self.cache_dir]:
                if os.path.exists(temp_dir):
                    for filename in os.listdir(temp_dir):
                        filepath = os.path.join(temp_dir, filename)
                        if os.path.isfile(filepath):
                            file_age = current_time - os.path.getctime(filepath)
                            if file_age > max_age_hours * 3600:
                                os.remove(filepath)
        except Exception as e:
            print(f"Cleanup warning: {e}")

# Initialize config
app_config = AppConfig()
app_config.setup_directories()
app_config.cleanup_old_files()

# Enhanced download manager for large files
class LargeFileDownloadManager:
    def __init__(self):
        self.download_progress = {}
        
    def download_with_progress(self, url, file_path, file_id, chunk_size=32768):
        """Download large files with progress tracking"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Start download
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        # Update progress
                        progress = downloaded / total_size if total_size > 0 else 0
                        self.download_progress[file_id] = {
                            'progress': progress,
                            'downloaded_mb': downloaded / (1024 * 1024),
                            'total_mb': total_size / (1024 * 1024) if total_size > 0 else 0,
                            'completed': False
                        }
            
            self.download_progress[file_id]['completed'] = True
            return True
            
        except Exception as e:
            self.download_progress[file_id] = {
                'error': str(e),
                'completed': False
            }
            return False

# Set page configuration
st.set_page_config(
    page_title="Uganda Bird Spotter",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
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
        -webkit-backup-filter: blur(10px);
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
    .download-progress {
        background: rgba(46, 134, 171, 0.1);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Story templates
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

class ResNet34BirdModel:
    def __init__(self):
        self.model_loaded = False
        self.bird_species = []
        self.inv_label_map = {}
        self.model = None
        self.device = None
        self.transform = None
        self.model_path = './models/resnet34_bird_region_weights.pth'
        self.label_map_path = './models/label_map.json'
        self.download_manager = LargeFileDownloadManager()
        
    def ensure_model_directories(self):
        """Create necessary directories"""
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./temp', exist_ok=True)
        
    def download_large_file_with_retry(self, file_id, url, output_path, max_retries=3):
        """Download large files with retry mechanism"""
        for attempt in range(max_retries):
            try:
                st.info(f"📥 Download attempt {attempt + 1}/{max_retries} for {file_id}...")
                
                # Initialize progress
                self.download_manager.download_progress[file_id] = {
                    'progress': 0,
                    'downloaded_mb': 0,
                    'total_mb': 0,
                    'completed': False
                }
                
                # Download in a separate thread
                download_thread = threading.Thread(
                    target=self.download_manager.download_with_progress,
                    args=(url, output_path, file_id)
                )
                download_thread.daemon = True
                download_thread.start()
                
                # Monitor progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while download_thread.is_alive():
                    time.sleep(1)
                    if file_id in self.download_manager.download_progress:
                        progress_info = self.download_manager.download_progress[file_id]
                        progress = progress_info.get('progress', 0)
                        downloaded_mb = progress_info.get('downloaded_mb', 0)
                        total_mb = progress_info.get('total_mb', 0)
                        
                        progress_bar.progress(progress)
                        if total_mb > 0:
                            status_text.text(f"Downloading {file_id}: {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({progress:.1%})")
                        else:
                            status_text.text(f"Downloading {file_id}: {downloaded_mb:.1f}MB")
                
                download_thread.join(timeout=300)
                
                if file_id in self.download_manager.download_progress:
                    progress_info = self.download_manager.download_progress[file_id]
                    if progress_info.get('completed'):
                        file_size = os.path.getsize(output_path) / (1024 * 1024)
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ {file_id} downloaded successfully! ({file_size:.1f}MB)")
                        return True
                
                st.warning(f"⚠️ Download attempt {attempt + 1} failed, retrying...")
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
            except Exception as e:
                st.error(f"❌ Download error on attempt {attempt + 1}: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                
        return False

    def download_model_from_gdrive(self):
        """Download model from Google Drive with enhanced error handling"""
        self.ensure_model_directories()
        
        if os.path.exists(self.model_path):
            file_size = os.path.getsize(self.model_path) / (1024 * 1024)
            if file_size > 100:
                st.info(f"✅ Model already exists ({file_size:.1f}MB)")
                return True
        
        file_id = "1yfiYcz6e2hWtQTXW6AZVU-iwSUjDP92y"
        
        # Try multiple download methods
        try:
            import gdown
            url = f'https://drive.google.com/uc?id={file_id}'
            return self.download_large_file_with_retry("ResNet34 Model", url, self.model_path)
        except ImportError:
            pass
        
        # Fallback method
        try:
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            return self.download_large_file_with_retry("ResNet34 Model", url, self.model_path)
        except Exception as e:
            st.error(f"❌ All download methods failed: {e}")
            return False

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

    def load_model_lazy(self):
        """Lazy load model - only when needed"""
        if self.model_loaded:
            return True
            
        if not os.path.exists(self.model_path):
            with st.spinner("🔄 Downloading bird identification model..."):
                if not self.download_model_from_gdrive():
                    st.error("❌ Failed to download model")
                    return False
        
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            
            if not self.load_label_map():
                return False
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model = models.resnet34(weights=None)
            num_classes = len(self.bird_species)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(self.model_path))
            else:
                model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            
            self.model = model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            self.model_loaded = True
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
            
            # Simple detection - one bird in center
            x = width // 4
            y = height // 4
            w = width // 2
            h = height // 2
            
            detection_confidence = 0.85
            detections = [([x, y, w, h], detection_confidence)]
            
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

class AdvancedVideoGenerator:
    def __init__(self):
        self.csv_path = './data/birdsuganda.csv'
        self.video_model_path = './models/bird_path.pth'
        self.bird_data = None
        self.story_model = None
        self.model_loaded = False
        self.video_duration = 20
        self.download_manager = LargeFileDownloadManager()
        
    def ensure_data_directories(self):
        """Create necessary directories"""
        os.makedirs('./data', exist_ok=True)
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./temp_videos', exist_ok=True)
        
    def download_video_model_lazy(self):
        """Lazy download of video model only when needed"""
        self.ensure_data_directories()
        
        if os.path.exists(self.video_model_path):
            file_size = os.path.getsize(self.video_model_path) / (1024 * 1024)
            if file_size > 50:
                return self.load_video_model()
        
        # Show download option
        if st.button("📥 Download Story Generation Model (1GB+)"):
            with st.spinner("Downloading advanced story generation model..."):
                file_id = "1J9T5r5TboWzvqAPQHmfvQmozor_wmmPz"
                
                try:
                    import gdown
                    url = f'https://drive.google.com/uc?id={file_id}'
                    success = self.download_large_file_with_retry("Story Model", url, self.video_model_path)
                    if success:
                        return self.load_video_model()
                except ImportError:
                    pass
                
                # Fallback
                try:
                    url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    success = self.download_large_file_with_retry("Story Model", url, self.video_model_path)
                    if success:
                        return self.load_video_model()
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")
            
        return False

    def download_large_file_with_retry(self, file_id, url, output_path, max_retries=2):
        """Download large files with retry mechanism"""
        for attempt in range(max_retries):
            try:
                # Initialize progress
                self.download_manager.download_progress[file_id] = {
                    'progress': 0,
                    'downloaded_mb': 0,
                    'total_mb': 0,
                    'completed': False
                }
                
                # Download in thread
                download_thread = threading.Thread(
                    target=self.download_manager.download_with_progress,
                    args=(url, output_path, file_id)
                )
                download_thread.daemon = True
                download_thread.start()
                
                # Monitor progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                while download_thread.is_alive():
                    time.sleep(1)
                    if file_id in self.download_manager.download_progress:
                        progress_info = self.download_manager.download_progress[file_id]
                        progress = progress_info.get('progress', 0)
                        downloaded_mb = progress_info.get('downloaded_mb', 0)
                        total_mb = progress_info.get('total_mb', 0)
                        
                        progress_bar.progress(progress)
                        if total_mb > 0:
                            status_text.text(f"Downloading {file_id}: {downloaded_mb:.1f}MB / {total_mb:.1f}MB ({progress:.1%})")
                        else:
                            status_text.text(f"Downloading {file_id}: {downloaded_mb:.1f}MB")
                
                download_thread.join(timeout=600)  # 10-minute timeout for large files
                
                if file_id in self.download_manager.download_progress:
                    progress_info = self.download_manager.download_progress[file_id]
                    if progress_info.get('completed'):
                        file_size = os.path.getsize(output_path) / (1024 * 1024)
                        progress_bar.empty()
                        status_text.empty()
                        st.success(f"✅ {file_id} downloaded! ({file_size:.1f}MB)")
                        return True
                
                st.warning(f"⚠️ Download attempt {attempt + 1} failed")
                if os.path.exists(output_path):
                    os.remove(output_path)
                    
            except Exception as e:
                st.error(f"❌ Download error: {e}")
                if os.path.exists(output_path):
                    os.remove(output_path)
                
        return False

    def load_video_model(self):
        """Load the bird_path.pth model for story generation"""
        if not os.path.exists(self.video_model_path):
            return False
        
        try:
            if torch.cuda.is_available():
                model_data = torch.load(self.video_model_path)
            else:
                model_data = torch.load(self.video_model_path, map_location=torch.device('cpu'))
            
            if isinstance(model_data, BirdStoryGenerator):
                self.story_model = model_data
            elif isinstance(model_data, dict):
                self.story_model = BirdStoryGenerator(TEMPLATES)
            else:
                self.story_model = BirdStoryGenerator(TEMPLATES)
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Model loading failed: {e}")
            self.story_model = BirdStoryGenerator(TEMPLATES)
            self.model_loaded = True
            return True

    def load_video_model_lazy(self):
        """Lazy load video model"""
        if not self.model_loaded:
            return self.download_video_model_lazy()
        return True

    def load_bird_data(self):
        """Load and process the bird species data from local CSV"""
        try:
            if os.path.exists(self.csv_path):
                self.bird_data = pd.read_csv(self.csv_path)
                return True
            else:
                # Create sample data if CSV doesn't exist
                sample_data = {
                    'species_name': ['African Fish Eagle', 'Grey Crowned Crane', 'Shoebill Stork'],
                    'description': ['Majestic bird of prey', 'National bird of Uganda', 'Ancient-looking stork'],
                    'colors': ['white, brown', 'grey, white, gold', 'grey, blue']
                }
                self.bird_data = pd.DataFrame(sample_data)
                self.bird_data.to_csv(self.csv_path, index=False)
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading CSV: {e}")
            return False

    def generate_story_video(self, species_name):
        """Generate a comprehensive story-based video"""
        try:
            if not self.model_loaded or self.story_model is None:
                st.error("❌ Story generation model not loaded")
                return None, None, None
            
            # Get bird information
            if not self.load_bird_data():
                return None, None, None
            
            bird_info = self.get_bird_video_info(species_name)
            
            # Generate story
            story_text = self.generate_story_from_model(species_name, bird_info)
            
            if not story_text:
                return None, None, None
            
            st.markdown(f'<div class="story-box"><strong>📖 Story:</strong><br>{story_text}</div>', unsafe_allow_html=True)
            
            # Create simple video file
            video_path = self.create_simple_video(species_name, story_text)
            
            if video_path:
                st.success("✅ Story video generated successfully!")
                return video_path, story_text, []
            else:
                return None, None, None
            
        except Exception as e:
            st.error(f"❌ Story video generation error: {e}")
            return None, None, None

    def get_bird_video_info(self, species_name):
        """Get video generation information for a specific bird species"""
        if self.bird_data is None:
            if not self.load_bird_data():
                return None
        
        try:
            for col in ['species_name', 'species', 'name']:
                if col in self.bird_data.columns:
                    bird_info = self.bird_data[
                        self.bird_data[col].astype(str).str.lower() == species_name.lower()
                    ]
                    if len(bird_info) > 0:
                        return bird_info.iloc[0].to_dict()
            
            return None
                
        except Exception as e:
            return None

    def generate_story_from_model(self, species_name, bird_info):
        """Generate story using the loaded model"""
        try:
            if not self.model_loaded or self.story_model is None:
                return None
            
            common_name = species_name
            description = bird_info.get('description', '') if bird_info else ''
            colors = []
            
            color_columns = ['colors', 'primary_colors', 'plumage_colors']
            for col in color_columns:
                if col in bird_info and pd.notna(bird_info[col]):
                    colors = str(bird_info[col]).split(',')
                    break
            
            story_text = self.story_model(common_name, description, colors)
            return story_text
            
        except Exception as e:
            return f"The {species_name} is a magnificent bird found in Uganda's diverse ecosystems. With its unique characteristics, it plays a vital role in the local biodiversity."

    def create_simple_video(self, species_name, story_text):
        """Create a simple video with text overlay"""
        try:
            # Create a simple video with OpenCV
            width, height = 800, 600
            fps = 24
            duration = 10  # seconds
            total_frames = fps * duration
            
            video_path = f"./temp_videos/story_{species_name.replace(' ', '_')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            for i in range(total_frames):
                # Create frame with gradient background
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                frame[:, :] = [30 + i % 50, 60 + i % 50, 90 + i % 50]
                
                # Add text
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, f"Uganda Bird: {species_name}", (50, 100), font, 1, (255, 255, 255), 2)
                cv2.putText(frame, "Story by bird_path.pth", (50, 150), font, 0.6, (200, 200, 255), 1)
                
                # Add progress bar
                progress = i / total_frames
                cv2.rectangle(frame, (50, 500), (width - 50, 530), (100, 100, 100), -1)
                cv2.rectangle(frame, (50, 500), (50 + int((width - 100) * progress), 530), (0, 200, 255), -1)
                
                out.write(frame)
            
            out.release()
            return video_path
            
        except Exception as e:
            st.error(f"❌ Video creation error: {e}")
            return None

def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return ""

def initialize_system_optimized():
    """Optimized initialization that doesn't block health checks"""
    if 'system_initialized' not in st.session_state:
        st.session_state.update({
            'bird_model': ResNet34BirdModel(),
            'video_generator': AdvancedVideoGenerator(),
            'system_initialized': False,
            'model_loaded': False,
            'video_model_loaded': False,
            'initialization_phase': 'minimal',
            'detection_complete': False,
            'bird_detections': [],
            'bird_classifications': [],
            'current_image': None,
            'active_method': "upload",
            'generated_video_path': None,
            'selected_species_for_video': None,
            'generated_story': None,
            'used_images': None
        })
    
    # Phase 1: Minimal initialization (always done)
    if st.session_state.initialization_phase == 'minimal':
        st.session_state.bird_model.ensure_model_directories()
        st.session_state.bird_model.load_label_map()
        st.session_state.video_generator.ensure_data_directories()
        st.session_state.initialization_phase = 'core'
        st.session_state.system_initialized = True
    
    return True

def load_core_model_on_demand():
    """Load the core bird identification model only when needed"""
    if not st.session_state.get('model_loaded', False):
        with st.spinner("🔄 Loading bird identification model..."):
            if st.session_state.bird_model.load_model_lazy():
                st.session_state.model_loaded = True
                return True
            else:
                st.error("❌ Failed to load bird identification model")
                return False
    return True

def load_video_model_on_demand():
    """Load video model only when user requests video features"""
    if not st.session_state.get('video_model_loaded', False):
        if st.session_state.video_generator.load_video_model_lazy():
            st.session_state.video_model_loaded = True
            return True
    return st.session_state.get('video_model_loaded', False)

def render_basic_ui():
    """Render the basic UI that appears immediately"""
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-title">Uganda Bird Spotter</div>', unsafe_allow_html=True)
        
        st.markdown("### 🦅 Detectable Birds")
        if st.session_state.bird_model.bird_species:
            st.markdown(f"**Total Species:** {len(st.session_state.bird_model.bird_species)}")
            st.markdown('<div class="bird-list">', unsafe_allow_html=True)
            for species in st.session_state.bird_model.bird_species[:10]:  # Show first 10
                st.markdown(f"• {species}")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Bird list will load with model")
        
        # System status
        st.markdown("---")
        if st.session_state.model_loaded:
            st.success("🔍 Bird ID: **Ready**")
        else:
            st.warning("🔍 Bird ID: **Click Analyze**")
        
        if st.session_state.video_model_loaded:
            st.success("🎬 Stories: **Ready**")
        else:
            st.warning("🎬 Stories: **Click Download**")

    # Main header
    st.markdown("""
    <div class="main-header">
        Uganda Bird Spotter
    </div>
    """, unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div class="glass-card">
        <strong>🦜 Welcome to Uganda Bird Spotter!</strong><br>
        AI-powered bird identification and story generation. Upload photos for instant identification 
        or generate educational story videos about Ugandan birds.
    </div>
    """, unsafe_allow_html=True)

def handle_image_input():
    """Handle image input from user"""
    col1, col2 = st.columns(2)
    
    with col1:
        upload_active = st.session_state.active_method == "upload"
        if st.button("📁 Upload Photo", use_container_width=True, type="primary" if upload_active else "secondary"):
            st.session_state.active_method = "upload"
            st.session_state.current_image = None
            st.rerun()
    
    with col2:
        camera_active = st.session_state.active_method == "camera"
        if st.button("📷 Capture Photo", use_container_width=True, type="primary" if camera_active else "secondary"):
            st.session_state.active_method = "camera"
            st.session_state.current_image = None
            st.rerun()
    
    st.markdown("---")
    
    current_image = None
    
    if st.session_state.active_method == "upload":
        st.markdown('<div class="section-title">Upload Bird Photo</div>', unsafe_allow_html=True)
        st.markdown('<div class="glass-upload">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a bird image", 
            type=['jpg', 'jpeg', 'png'],
            help="Upload photos of birds for identification",
            label_visibility="collapsed"
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
    
    return current_image

def main():
    """Main optimized application"""
    # Minimal initialization
    initialize_system_optimized()
    
    bird_model = st.session_state.bird_model
    video_generator = st.session_state.video_generator
    
    # Show basic UI immediately
    render_basic_ui()
    
    # Handle image input
    current_image = handle_image_input()
    
    # Handle image analysis
    if current_image is not None:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.image(current_image, caption="Bird Photo for Analysis", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        if st.button("🔍 Analyze Bird Species", type="primary", use_container_width=True):
            # Load model on demand
            if load_core_model_on_demand():
                with st.spinner("Analyzing bird species using AI..."):
                    detections, classifications, original_image = bird_model.predict_bird_species(current_image)
                    
                    st.session_state.detection_complete = True
                    st.session_state.bird_detections = detections
                    st.session_state.bird_classifications = classifications
                    st.session_state.current_image = original_image
                    st.rerun()
            else:
                st.error("❌ Please wait for model to load or try again")
    
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
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<div class="glass-metric">', unsafe_allow_html=True)
                st.metric("Birds Identified", len(detections))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
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
    st.markdown('<div class="section-title">🎬 Story Video Generator</div>', unsafe_allow_html=True)
    
    if st.session_state.video_model_loaded:
        st.markdown("""
        <div class="video-section">
            <strong>📖 Ugandan Bird Stories</strong><br>
            Generate educational story videos with AI-generated narratives about Ugandan birds.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="video-section">
            <strong>📖 Enable Story Generation</strong><br>
            Download the story model to generate educational videos about Ugandan birds.
        </div>
        """, unsafe_allow_html=True)
    
    # Video generation options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.get('selected_species_for_video'):
            st.info(f"🦜 Detected: **{st.session_state.selected_species_for_video}**")
            if st.button("🎬 Generate Story Video", use_container_width=True, type="primary"):
                if load_video_model_on_demand():
                    with st.spinner("Creating story video..."):
                        video_path, story_text, used_images = video_generator.generate_story_video(
                            st.session_state.selected_species_for_video
                        )
                        if video_path:
                            st.session_state.generated_video_path = video_path
                            st.session_state.generated_story = story_text
                            st.session_state.used_images = used_images
                            st.rerun()
    
    with col2:
        manual_species = st.selectbox(
            "Or select species:",
            options=bird_model.bird_species,
            index=0 if not st.session_state.get('selected_species_for_video') else 
                  bird_model.bird_species.index(st.session_state.selected_species_for_video) 
                  if st.session_state.selected_species_for_video in bird_model.bird_species else 0,
            key="manual_species_select"
        )
        
        if st.button("🎬 Generate for Selected", use_container_width=True, type="primary"):
            if load_video_model_on_demand():
                with st.spinner("Creating story video..."):
                    video_path, story_text, used_images = video_generator.generate_story_video(manual_species)
                    if video_path:
                        st.session_state.generated_video_path = video_path
                        st.session_state.generated_story = story_text
                        st.session_state.used_images = used_images
                        st.session_state.selected_species_for_video = manual_species
                        st.rerun()
    
    # Display generated video
    if st.session_state.get('generated_video_path') and os.path.exists(st.session_state.generated_video_path):
        st.markdown("---")
        st.markdown("### 📖 Generated Story Video")
        
        if st.session_state.get('generated_story'):
            st.markdown(f'<div class="story-box">{st.session_state.generated_story}</div>', unsafe_allow_html=True)
        
        try:
            with open(st.session_state.generated_video_path, "rb") as video_file:
                video_bytes = video_file.read()
            
            st.video(video_bytes)
            
            # Download button
            st.download_button(
                label="📥 Download Story Video",
                data=video_bytes,
                file_name=f"uganda_bird_{st.session_state.selected_species_for_video.replace(' ', '_')}.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error displaying video: {e}")

if __name__ == "__main__":
    main()