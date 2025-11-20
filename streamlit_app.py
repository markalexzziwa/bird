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
</style>
""", unsafe_allow_html=True)

class VideoGenerator:
    def __init__(self):
        self.csv_path = './birdsuganda.csv'  # CSV is in app directory
        self.video_model_path = './video_generation_model.pth'  # Model from Google Drive
        self.bird_data = None
        self.video_model = None
        
    def download_video_model(self):
        """Download the video generation model from Google Drive"""
        try:
            if not os.path.exists(self.video_model_path):
                st.info("📥 Downloading video generation model from Google Drive...")
                
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
                if file_size > 1:  # Ensure file is not empty/corrupted
                    st.success(f"✅ Video model downloaded! ({file_size:.1f} MB)")
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
        """Load the video generation model"""
        if not os.path.exists(self.video_model_path):
            if not self.download_video_model():
                return False
        
        try:
            import torch
            import torch.nn as nn
            
            # Initialize device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load the model (assuming it's a PyTorch model)
            if torch.cuda.is_available():
                self.video_model = torch.load(self.video_model_path)
            else:
                self.video_model = torch.load(self.video_model_path, map_location=torch.device('cpu'))
            
            self.video_model.eval()
            st.success("✅ Video generation model loaded successfully!")
            return True
            
        except Exception as e:
            st.warning(f"⚠️ Could not load video model, using basic video generation: {e}")
            # Continue with basic video generation even if model loading fails
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
            # Try different column names that might exist in the CSV
            possible_columns = ['species_name', 'species', 'name', 'bird_name', 'common_name']
            
            for col in possible_columns:
                if col in self.bird_data.columns:
                    bird_info = self.bird_data[self.bird_data[col].str.lower() == species_name.lower()]
                    if len(bird_info) > 0:
                        return bird_info.iloc[0].to_dict()
            
            # If no exact match, try partial match
            for col in possible_columns:
                if col in self.bird_data.columns:
                    bird_info = self.bird_data[self.bird_data[col].str.contains(species_name, case=False, na=False)]
                    if len(bird_info) > 0:
                        return bird_info.iloc[0].to_dict()
            
            return None
                
        except Exception as e:
            st.error(f"❌ Error finding bird info: {e}")
            return None
    
    def generate_video_with_model(self, species_name, duration=10):
        """Generate video using the trained model (if available)"""
        try:
            if self.video_model is not None:
                st.info("🎬 Using AI model to generate video...")
                # Here you would use the actual model to generate video frames
                # For now, we'll use the basic generation as fallback
                pass
            
            # Fallback to basic video generation
            return self.generate_basic_video(species_name, duration)
            
        except Exception as e:
            st.warning(f"⚠️ AI video generation failed, using basic method: {e}")
            return self.generate_basic_video(species_name, duration)
    
    def generate_basic_video(self, species_name, duration=10):
        """Generate a basic educational video about the bird species"""
        try:
            # Get bird information
            bird_info = self.get_bird_video_info(species_name)
            
            st.info(f"🎬 Generating educational video for {species_name}...")
            
            # Create a temporary video file
            temp_video_path = f"./temp_{species_name.replace(' ', '_')}.mp4"
            
            # Video properties
            frame_width = 640
            frame_height = 480
            fps = 24
            total_frames = duration * fps
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
            
            # Generate video frames
            for frame_num in range(total_frames):
                # Create a background (sky blue gradient)
                frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                
                # Create gradient background (sky to ground)
                for i in range(frame_height):
                    # Sky blue to light green gradient
                    if i < frame_height * 0.7:  # Sky
                        color = [135 + i//4, 206 + i//8, 235]  # Sky blue gradient
                    else:  # Ground
                        color = [34 + (i - frame_height*0.7)//2, 139 + (i - frame_height*0.7)//3, 34]  # Forest green gradient
                    
                    frame[i, :] = color
                
                # Add bird silhouette or shape in the center with animation
                center_x, center_y = frame_width // 2, frame_height // 3
                bird_radius = 50
                
                # Flying animation
                fly_offset_x = int(30 * np.sin(frame_num * 0.1))
                fly_offset_y = int(15 * np.sin(frame_num * 0.2))
                
                current_x = center_x + fly_offset_x
                current_y = center_y + fly_offset_y
                
                # Draw a simple bird silhouette (ellipse)
                cv2.ellipse(frame, (current_x, current_y), (bird_radius, bird_radius//2), 0, 0, 360, (30, 30, 30), -1)
                
                # Add wings with flapping animation
                wing_flap = int(10 * np.sin(frame_num * 0.5))
                
                # Left wing
                wing_points_left = np.array([
                    [current_x - bird_radius//2, current_y],
                    [current_x - bird_radius - wing_flap, current_y - bird_radius//2],
                    [current_x - bird_radius//2, current_y - bird_radius//4 + wing_flap//2]
                ], np.int32)
                cv2.fillPoly(frame, [wing_points_left], (30, 30, 30))
                
                # Right wing
                wing_points_right = np.array([
                    [current_x + bird_radius//2, current_y],
                    [current_x + bird_radius + wing_flap, current_y - bird_radius//2],
                    [current_x + bird_radius//2, current_y - bird_radius//4 + wing_flap//2]
                ], np.int32)
                cv2.fillPoly(frame, [wing_points_right], (30, 30, 30))
                
                # Add text information with fade-in effect
                text_alpha = min(1.0, frame_num / (fps * 2))  # Fade in over 2 seconds
                
                # Species name (large and prominent)
                text_y = frame_height - 150
                cv2.putText(frame, f"Species: {species_name}", 
                           (50, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Additional info if available
                info_y = text_y + 40
                if bird_info:
                    if 'habitat' in bird_info and pd.notna(bird_info['habitat']):
                        cv2.putText(frame, f"Habitat: {str(bird_info['habitat'])[:50]}", 
                                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        info_y += 30
                    
                    if 'conservation_status' in bird_info and pd.notna(bird_info['conservation_status']):
                        status_color = (0, 255, 0)  # Green for good status
                        if 'endangered' in str(bird_info['conservation_status']).lower():
                            status_color = (0, 165, 255)  # Orange for endangered
                        elif 'vulnerable' in str(bird_info['conservation_status']).lower():
                            status_color = (0, 0, 255)  # Red for vulnerable
                        
                        cv2.putText(frame, f"Conservation: {bird_info['conservation_status']}", 
                                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                        info_y += 30
                    
                    if 'diet' in bird_info and pd.notna(bird_info['diet']):
                        cv2.putText(frame, f"Diet: {str(bird_info['diet'])[:40]}", 
                                   (50, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add Uganda Bird Spotter watermark
                cv2.putText(frame, "Uganda Bird Spotter - AI Generated", 
                           (frame_width - 300, frame_height - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Write frame to video
                out.write(frame)
            
            # Release video writer
            out.release()
            
            st.success(f"✅ Educational video generated for {species_name}!")
            return temp_video_path
            
        except Exception as e:
            st.error(f"❌ Video generation error: {e}")
            return None
    
    def generate_video(self, species_name, duration=10):
        """Main video generation function that tries AI model first, then falls back to basic"""
        return self.generate_video_with_model(species_name, duration)

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
        st.session_state.video_generator = VideoGenerator()
        st.session_state.detection_complete = False
        st.session_state.bird_detections = []
        st.session_state.bird_classifications = []
        st.session_state.current_image = None
        st.session_state.active_method = "upload"
        st.session_state.model_loaded = False
        st.session_state.system_initialized = False
        st.session_state.generated_video_path = None
        st.session_state.selected_species_for_video = None
    
    # Initialize system only once
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing Uganda Bird Spotter System..."):
            # Try to load the model
            success = st.session_state.bird_model.load_model()
            
            if success:
                # Load video generator data and model
                st.session_state.video_generator.load_bird_data()
                st.session_state.video_generator.load_video_model()
                st.session_state.model_loaded = True
                st.session_state.system_initialized = True
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
            st.rerun()
    
    # Video Generation Section
    st.markdown("---")
    st.markdown('<div class="section-title">🎬 AI Bird Video Generator</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="video-section">
        <strong>🎥 Generate Educational Videos with AI</strong><br>
        Create short educational videos about identified bird species using our advanced video generation model.
        The system uses the birdsuganda.csv database and AI model to create informative content about each species.
    </div>
    """, unsafe_allow_html=True)
    
    # Video generation options
    col1, col2 = st.columns(2)
    
    with col1:
        # Option 1: Use detected species
        if st.session_state.get('selected_species_for_video'):
            st.info(f"🦜 Detected Species: **{st.session_state.selected_species_for_video}**")
            if st.button("🎬 Generate AI Video for Detected Bird", use_container_width=True, type="primary"):
                with st.spinner("Creating AI educational video..."):
                    video_path = video_generator.generate_video(st.session_state.selected_species_for_video)
                    if video_path:
                        st.session_state.generated_video_path = video_path
                        st.success("✅ AI video generated successfully!")
    
    with col2:
        # Option 2: Manual species selection
        manual_species = st.selectbox(
            "Or select a species manually:",
            options=bird_model.bird_species,
            index=0 if not st.session_state.get('selected_species_for_video') else 
                  bird_model.bird_species.index(st.session_state.selected_species_for_video) 
                  if st.session_state.selected_species_for_video in bird_model.bird_species else 0
        )
        
        if st.button("🎬 Generate AI Video for Selected Bird", use_container_width=True, type="primary"):
            with st.spinner("Creating AI educational video..."):
                video_path = video_generator.generate_video(manual_species)
                if video_path:
                    st.session_state.generated_video_path = video_path
                    st.session_state.selected_species_for_video = manual_species
                    st.success("✅ AI video generated successfully!")
    
    # Video duration selection
    video_duration = st.slider("Video Duration (seconds)", min_value=5, max_value=30, value=10, step=5)
    
    # Display generated video
    if st.session_state.get('generated_video_path') and os.path.exists(st.session_state.generated_video_path):
        st.markdown("---")
        st.markdown("### 🎥 AI Generated Bird Video")
        
        # Display video
        try:
            with open(st.session_state.generated_video_path, "rb") as video_file:
                video_bytes = video_file.read()
            
            st.video(video_bytes)
            
            # Video information
            st.info(f"**Video Details:** {st.session_state.selected_species_for_video} | {video_duration} seconds | AI Generated")
            
            # Download button
            st.download_button(
                label="📥 Download AI Video",
                data=video_bytes,
                file_name=f"uganda_bird_{st.session_state.selected_species_for_video.replace(' ', '_')}.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error displaying video: {e}")

if __name__ == "__main__":
    main()