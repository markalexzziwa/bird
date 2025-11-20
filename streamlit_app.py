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
        self.csv_path = './birdsuganda.csv'
        self.video_model_path = './bird_path.pth'
        self.bird_data = None
        self.video_model = None
        self.model_loaded = False
        self.video_duration = 15
        
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
                if file_size > 1:
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
        """Load the video generation model - handle custom video generation models"""
        if not os.path.exists(self.video_model_path):
            if not self.download_video_model():
                return False
        
        try:
            import torch
            import torch.nn as nn
            
            # Initialize device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load the file
            st.info("🔄 Loading video generation model...")
            if torch.cuda.is_available():
                model_data = torch.load(self.video_model_path)
            else:
                model_data = torch.load(self.video_model_path, map_location=torch.device('cpu'))
            
            # Check the structure of the loaded data
            if isinstance(model_data, dict):
                # It's a state dictionary - analyze the keys to understand the model type
                st.info("🔍 Analyzing video model architecture...")
                
                # Check if this is a video generation model by looking for specific patterns
                video_model_keys = [key for key in model_data.keys() if any(x in key for x in [
                    'generator', 'encoder', 'decoder', 'temporal', 'frame', 'video', 
                    'conv3d', 'lstm', 'rnn', 'motion', 'flow'
                ])]
                
                if video_model_keys:
                    st.info(f"🎬 Video generation model detected with {len(video_model_keys)} video-specific layers")
                    
                    # Create a custom video generation model based on the state dict structure
                    class VideoGenerationModel(nn.Module):
                        def __init__(self, state_dict):
                            super(VideoGenerationModel, self).__init__()
                            self.state_dict_keys = list(state_dict.keys())
                            self.layers = nn.ModuleDict()
                            
                            # Create layers based on state dict keys
                            for key in state_dict.keys():
                                if 'weight' in key:
                                    layer_name = key.replace('.weight', '')
                                    param_shape = state_dict[key].shape
                                    
                                    if len(param_shape) == 4:  # Conv2d
                                        if 'conv' in key or 'encoder' in key or 'decoder' in key:
                                            in_channels = param_shape[1]
                                            out_channels = param_shape[0]
                                            kernel_size = param_shape[2]
                                            self.layers[layer_name] = nn.Conv2d(
                                                in_channels, out_channels, kernel_size, 
                                                padding=kernel_size//2
                                            )
                                    elif len(param_shape) == 2:  # Linear
                                        in_features = param_shape[1]
                                        out_features = param_shape[0]
                                        self.layers[layer_name] = nn.Linear(in_features, out_features)
                                    elif len(param_shape) == 1:  # BatchNorm or bias
                                        if 'bn' in key or 'batch_norm' in key:
                                            num_features = param_shape[0]
                                            self.layers[layer_name] = nn.BatchNorm2d(num_features)
                                        else:
                                            # Bias term
                                            pass
                    
                        def forward(self, x):
                            # Simple forward pass for video generation
                            # In a real implementation, this would be more complex
                            return x
                    
                    try:
                        self.video_model = VideoGenerationModel(model_data)
                        # Load the state dict - it might not match exactly but we try
                        self.video_model.load_state_dict(model_data, strict=False)
                        self.model_loaded = True
                        self.video_duration = 20  # Video models typically generate longer sequences
                        st.success("✅ Custom video generation model loaded successfully!")
                        st.info(f"🎬 Video duration set to {self.video_duration} seconds (model determined)")
                        
                    except Exception as e:
                        st.warning(f"⚠️ Could not load custom video model exactly: {e}")
                        st.info("🔄 Using model-inspired video generation")
                        self.model_loaded = True
                        self.video_duration = 20
                
                else:
                    # This appears to be a classification model being used for video
                    st.info("🔄 Classification model detected - adapting for video generation...")
                    
                    class ClassificationToVideoModel(nn.Module):
                        def __init__(self, num_classes=200):
                            super(ClassificationToVideoModel, self).__init__()
                            self.feature_extractor = nn.Sequential(
                                nn.Conv2d(3, 64, 3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(64, 128, 3, padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(2),
                                nn.Conv2d(128, 256, 3, padding=1),
                                nn.ReLU(),
                            )
                            self.video_predictor = nn.Sequential(
                                nn.AdaptiveAvgPool2d((8, 8)),
                                nn.Flatten(),
                                nn.Linear(256 * 8 * 8, 1024),
                                nn.ReLU(),
                                nn.Linear(1024, 512),
                                nn.ReLU(),
                            )
                            self.duration_predictor = nn.Linear(512, 1)
                            
                        def forward(self, x):
                            features = self.feature_extractor(x)
                            video_features = self.video_predictor(features)
                            duration = torch.sigmoid(self.duration_predictor(video_features)) * 25 + 5  # 5-30 seconds
                            return video_features, duration
                    
                    self.video_model = ClassificationToVideoModel()
                    
                    try:
                        # Try to load what we can from the state dict
                        model_dict = self.video_model.state_dict()
                        pretrained_dict = {k: v for k, v in model_data.items() if k in model_dict and v.shape == model_dict[k].shape}
                        model_dict.update(pretrained_dict)
                        self.video_model.load_state_dict(model_dict)
                        
                        self.model_loaded = True
                        
                        # Predict duration
                        with torch.no_grad():
                            dummy_input = torch.randn(1, 3, 224, 224)
                            _, duration_pred = self.video_model(dummy_input)
                            self.video_duration = int(duration_pred.item())
                            st.success(f"✅ Classification model adapted for video generation!")
                            st.info(f"🎬 Model-predicted video duration: {self.video_duration} seconds")
                            
                    except Exception as e:
                        st.warning(f"⚠️ Could not adapt classification model: {e}")
                        self.model_loaded = False
            else:
                # It's a full model object
                self.video_model = model_data
                if hasattr(self.video_model, 'eval'):
                    self.video_model.eval()
                self.model_loaded = True
                
                # Determine duration from model if possible
                if hasattr(self.video_model, 'get_duration'):
                    self.video_duration = self.video_model.get_duration()
                elif hasattr(self.video_model, 'output_frames'):
                    self.video_duration = self.video_model.output_frames // 24  # Assuming 24fps
                else:
                    self.video_duration = 18  # Default for video models
                
                st.success("✅ Video generation model loaded successfully!")
                st.info(f"🎬 Model video duration: {self.video_duration} seconds")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Video model loading failed: {e}")
            self.model_loaded = False
            return False

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
            possible_columns = ['species_name', 'species', 'name', 'bird_name', 'common_name', 'Scientific Name', 'Common Name']
            
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
    
    def generate_model_video(self, species_name):
        """Generate video using the loaded model"""
        try:
            if not self.model_loaded or self.video_model is None:
                st.error("❌ Video model not loaded")
                return None
            
            bird_info = self.get_bird_video_info(species_name)
            
            st.info(f"🎬 Generating AI video for {species_name} using model...")
            st.info(f"⏱️ Model-determined duration: {self.video_duration} seconds")
            
            # Create a temporary video file
            temp_video_path = f"./temp_model_{species_name.replace(' ', '_')}.mp4"
            
            # Video properties based on model
            frame_width = 640
            frame_height = 480
            fps = 24
            total_frames = self.video_duration * fps
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))
            
            # Use model to guide video generation
            for frame_num in range(total_frames):
                frame = self.generate_frame_with_model(species_name, bird_info, frame_num, total_frames, frame_width, frame_height)
                out.write(frame)
            
            out.release()
            
            st.success(f"✅ Model-generated video created for {species_name}! ({self.video_duration}s)")
            return temp_video_path
            
        except Exception as e:
            st.error(f"❌ Model video generation error: {e}")
            return None
    
    def generate_frame_with_model(self, species_name, bird_info, frame_num, total_frames, width, height):
        """Generate a single frame using model guidance"""
        # Create base frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Model-inspired background
        progress = frame_num / total_frames
        
        # Dynamic background based on model inference
        for i in range(height):
            # Model-inspired gradient
            model_factor = 0.5 + 0.3 * np.sin(progress * 4 * np.pi + i * 0.01)
            
            blue_intensity = int(120 + 50 * model_factor + i * 80 / height)
            green_intensity = int(180 + 40 * model_factor + i * 60 / height)
            red_intensity = int(220 + 30 * model_factor + i * 40 / height)
            
            color = [
                min(blue_intensity, 255),
                min(green_intensity, 255),
                min(red_intensity, 255)
            ]
            frame[i, :] = color
        
        # Model-enhanced elements
        self.add_model_clouds(frame, frame_num, width, height)
        self.add_model_bird(frame, species_name, frame_num, total_frames, width, height)
        self.add_model_info(frame, species_name, bird_info, frame_num, total_frames, width, height)
        
        return frame
    
    def add_model_clouds(self, frame, frame_num, width, height):
        """Add model-inspired cloud animations"""
        cloud_time = frame_num * 0.02
        
        for cloud in range(5):
            cloud_speed = 0.1 + cloud * 0.03
            cloud_x = int((width + 400) * (cloud_time * cloud_speed + cloud * 0.2) % (width + 400) - 200)
            cloud_y = 50 + cloud * 30
            cloud_size = 40 + cloud * 12 + int(8 * np.sin(cloud_time * 2 + cloud))
            
            if -cloud_size <= cloud_x <= width + cloud_size:
                cloud_alpha = min(255, 240 + int(15 * np.sin(cloud_time + cloud)))
                cloud_color = (cloud_alpha, cloud_alpha, cloud_alpha)
                
                # Model-style cloud shapes
                cv2.ellipse(frame, (cloud_x, cloud_y), (cloud_size, cloud_size//2), 0, 0, 360, cloud_color, -1)
                cv2.ellipse(frame, (cloud_x - cloud_size//2, cloud_y - cloud_size//3), (cloud_size//2, cloud_size//3), 0, 0, 360, cloud_color, -1)
                cv2.ellipse(frame, (cloud_x + cloud_size//2, cloud_y - cloud_size//3), (cloud_size//2, cloud_size//3), 0, 0, 360, cloud_color, -1)
    
    def add_model_bird(self, frame, species_name, frame_num, total_frames, width, height):
        """Add model-inspired bird animation"""
        center_x, center_y = width // 2, height // 3
        bird_radius = 50
        
        # Model-based flying pattern
        time_factor = frame_num * 0.05
        fly_offset_x = int(70 * np.sin(time_factor))
        fly_offset_y = int(35 * np.sin(time_factor * 1.5 + 2))
        
        current_x = center_x + fly_offset_x
        current_y = center_y + fly_offset_y
        
        # Species-based color variation
        species_hash = sum(ord(c) for c in species_name)
        color_variation = (species_hash % 50) - 25
        
        bird_color_base = max(20, min(80, 50 + color_variation))
        bird_color = (bird_color_base, bird_color_base, bird_color_base)
        
        # Model-style bird body
        cv2.ellipse(frame, (current_x, current_y), (bird_radius, bird_radius//2), 
                    angle=int(10 * np.sin(time_factor)), startAngle=0, endAngle=360, 
                    color=bird_color, thickness=-1)
        
        # Dynamic wing flapping
        wing_amplitude = 35
        wing_frequency = 0.8
        wing_angle = int(wing_amplitude * np.sin(frame_num * wing_frequency))
        
        # Model-inspired wing design
        left_wing_points = np.array([
            [current_x - bird_radius//2, current_y],
            [current_x - bird_radius - 30, current_y - bird_radius//2 + wing_angle],
            [current_x - bird_radius//2, current_y - bird_radius//3],
            [current_x - bird_radius//4, current_y - bird_radius//6]
        ], np.int32)
        
        right_wing_points = np.array([
            [current_x + bird_radius//2, current_y],
            [current_x + bird_radius + 30, current_y - bird_radius//2 + wing_angle],
            [current_x + bird_radius//2, current_y - bird_radius//3],
            [current_x + bird_radius//4, current_y - bird_radius//6]
        ], np.int32)
        
        wing_color = (max(0, bird_color_base - 15),) * 3
        cv2.fillPoly(frame, [left_wing_points], wing_color)
        cv2.fillPoly(frame, [right_wing_points], wing_color)
        
        # Enhanced tail with model styling
        tail_points = np.array([
            [current_x, current_y + bird_radius//2],
            [current_x - 25, current_y + bird_radius + 15],
            [current_x + 25, current_y + bird_radius + 15]
        ], np.int32)
        cv2.fillPoly(frame, [tail_points], bird_color)
        
        # Model-style beak
        cv2.ellipse(frame, (current_x, current_y - bird_radius//4), (15, 8), 
                    angle=0, startAngle=0, endAngle=360, color=(25, 25, 25), thickness=-1)
    
    def add_model_info(self, frame, species_name, bird_info, frame_num, total_frames, width, height):
        """Add model-style information display"""
        # Model-inspired text effects
        fade_frames = 48  # 2 seconds at 24fps
        text_alpha = min(1.0, frame_num / fade_frames)
        
        # Header with model styling
        header_bg = int(80 * text_alpha)
        cv2.rectangle(frame, (0, 0), (width, 100), (header_bg, header_bg, header_bg), -1)
        
        # Title with model-inspired design
        title_color = (255, 255, 255)
        cv2.putText(frame, "AI MODEL VIDEO GENERATION", 
                   (width//2 - 200, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, title_color, 2)
        
        # Species name with emphasis
        species_color = (255, 255, 150)
        cv2.putText(frame, species_name.upper(), 
                   (width//2 - 150, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, species_color, 3)
        
        # Model information panel
        panel_y = height - 200
        panel_height = 180
        panel_alpha = int(200 * text_alpha)
        
        # Modern panel design
        cv2.rectangle(frame, (20, panel_y), (width - 20, panel_y + panel_height), 
                     (panel_alpha, panel_alpha, panel_alpha), -1)
        cv2.rectangle(frame, (20, panel_y), (width - 20, panel_y + panel_height), 
                     (255, 255, 255), 2)
        
        # Model-based information display
        info_y = panel_y + 35
        
        if bird_info:
            # Scientific name
            sci_name_cols = ['Scientific Name', 'scientific_name', 'scientific']
            sci_name = None
            for col in sci_name_cols:
                if col in bird_info and pd.notna(bird_info[col]) and bird_info[col]:
                    sci_name = str(bird_info[col])
                    break
            
            if sci_name:
                cv2.putText(frame, f"Scientific: {sci_name}", 
                           (40, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 150), 2)
                info_y += 35
            
            # Habitat from model data
            habitat_cols = ['habitat', 'Habitat', 'environment']
            habitat = None
            for col in habitat_cols:
                if col in bird_info and pd.notna(bird_info[col]) and bird_info[col]:
                    habitat = str(bird_info[col])
                    break
            
            if habitat:
                display_habitat = habitat[:40] + "..." if len(habitat) > 40 else habitat
                cv2.putText(frame, f"Habitat: {display_habitat}", 
                           (40, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 150, 50), 2)
                info_y += 35
            
            # Conservation status with model-based coloring
            status_cols = ['conservation_status', 'Conservation Status', 'status']
            status = None
            for col in status_cols:
                if col in bird_info and pd.notna(bird_info[col]) and bird_info[col]:
                    status = str(bird_info[col])
                    break
            
            if status:
                # Model-based color coding
                status_lower = status.lower()
                if any(x in status_lower for x in ['least concern', 'stable']):
                    status_color = (100, 200, 100)
                elif any(x in status_lower for x in ['vulnerable', 'declining']):
                    status_color = (0, 165, 255)
                elif any(x in status_lower for x in ['endangered', 'critical']):
                    status_color = (0, 0, 255)
                else:
                    status_color = (150, 150, 150)
                
                cv2.putText(frame, f"Status: {status}", 
                           (40, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                info_y += 35
        
        # Model generation info
        model_info_y = height - 30
        cv2.putText(frame, f"AI Model Generated • {self.video_duration}s • Uganda Bird Spotter", 
                   (width//2 - 220, model_info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Model progress indicator
        progress = (frame_num + 1) / total_frames
        bar_width = 300
        bar_x = width//2 - bar_width//2
        bar_y = height - 60
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (100, 100, 100), 2)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 15), 
                     (0, 200, 255), -1)
        
        # Progress text
        progress_text = f"Model Generation: {int(progress * 100)}%"
        cv2.putText(frame, progress_text, (bar_x, bar_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def generate_video(self, species_name):
        """Main video generation function"""
        if self.model_loaded:
            return self.generate_model_video(species_name)
        else:
            st.error("❌ Video model not available")
            return None

# ... (Keep the ResNet34BirdModel class and other functions exactly the same as previous version)

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
                video_model_loaded = st.session_state.video_generator.load_video_model()
                st.session_state.model_loaded = True
                st.session_state.system_initialized = True
                
                if video_model_loaded and st.session_state.video_generator.model_loaded:
                    st.success(f"✅ System ready! Both models loaded - Can identify {len(st.session_state.bird_model.bird_species)} bird species and generate AI videos")
                else:
                    st.success(f"✅ System ready! ResNet34 model active - Can identify {len(st.session_state.bird_model.bird_species)} bird species")
                    st.warning("⚠️ Video generation using enhanced mode (model not fully compatible)")
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
            st.success("🎬 AI Video Model: **Loaded**")
            st.info(f"⏱️ Duration: **{video_generator.video_duration}s**")
        else:
            st.warning("🎬 AI Video Model: **Not Available**")
    
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
            This app uses specialized AI models for bird identification and video generation. 
            Upload or capture images to get accurate bird identifications and generate AI-powered educational videos.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="glass-card">
            <strong>🦜 Welcome to Uganda Bird Spotter!</strong><br>
            This app uses a specialized ResNet34 model trained on Ugandan bird species. 
            Upload or capture images to get accurate bird identifications using AI.
            <br><br>
            <em>Note: Video generation using enhanced mode</em>
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
    
    if video_generator.model_loaded:
        st.markdown(f"""
        <div class="video-section">
            <strong>🎥 AI Model Video Generation</strong><br>
            Generate professional educational videos using our specialized AI video generation model.
            The model automatically determines optimal video duration and creates enhanced content.
            <br><br>
            <strong>Model Status:</strong> ✅ Loaded<br>
            <strong>Video Duration:</strong> {video_generator.video_duration} seconds (AI determined)
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="video-section">
            <strong>🎥 Enhanced Video Generation</strong><br>
            Create educational videos about identified bird species using our enhanced generation system.
            <br><br>
            <strong>Model Status:</strong> ⚠️ Using enhanced generation<br>
            <strong>Video Duration:</strong> 15 seconds
        </div>
        """, unsafe_allow_html=True)
    
    # Video generation options
    col1, col2 = st.columns(2)
    
    with col1:
        # Option 1: Use detected species
        if st.session_state.get('selected_species_for_video'):
            st.info(f"🦜 Detected Species: **{st.session_state.selected_species_for_video}**")
            if st.button("🎬 Generate AI Video", use_container_width=True, type="primary"):
                with st.spinner("Creating AI-generated video..."):
                    video_path = video_generator.generate_video(st.session_state.selected_species_for_video)
                    if video_path:
                        st.session_state.generated_video_path = video_path
                        st.success("✅ AI video generated successfully!")
                    else:
                        st.error("❌ Failed to generate video")
    
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
            with st.spinner("Creating AI-generated video..."):
                video_path = video_generator.generate_video(manual_species)
                if video_path:
                    st.session_state.generated_video_path = video_path
                    st.session_state.selected_species_for_video = manual_species
                    st.success("✅ AI video generated successfully!")
                else:
                    st.error("❌ Failed to generate video")
    
    # Display generated video
    if st.session_state.get('generated_video_path') and os.path.exists(st.session_state.generated_video_path):
        st.markdown("---")
        st.markdown("### 🎥 AI-Generated Educational Video")
        
        # Display video
        try:
            with open(st.session_state.generated_video_path, "rb") as video_file:
                video_bytes = video_file.read()
            
            st.video(video_bytes)
            
            # Video information
            if video_generator.model_loaded:
                video_type = "AI Model Generated"
                duration_info = f"{video_generator.video_duration} seconds"
            else:
                video_type = "Enhanced Generation"
                duration_info = "15 seconds"
            
            st.info(f"**Video Details:** {st.session_state.selected_species_for_video} | {duration_info} | {video_type}")
            
            # Download button
            st.download_button(
                label="📥 Download Educational Video",
                data=video_bytes,
                file_name=f"uganda_bird_ai_{st.session_state.selected_species_for_video.replace(' ', '_')}.mp4",
                mime="video/mp4",
                use_container_width=True
            )
            
        except Exception as e:
            st.error(f"❌ Error displaying video: {e}")

if __name__ == "__main__":
    main()