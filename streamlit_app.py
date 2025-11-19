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
        width: 80px;
        height: 80px;
        border-radius: 16px;
        object-fit: cover;
        margin: 0 auto 20px auto;
        display: block;
    }
    .bird-list {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

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
        """Download model from Google Drive"""
        try:
            # Create models directory
            os.makedirs('./models', exist_ok=True)
            
            if not os.path.exists(self.model_path):
                st.info("📥 Downloading ResNet34 model from Google Drive...")
                
                # Google Drive file ID - YOU NEED TO UPDATE THIS WITH YOUR ACTUAL FILE ID
                file_id = "1yfiYcz6e2hWtQTXW6AZVU-iwSUjDP92y"  # Replace with your actual file ID
                
                # Method 1: Using gdown
                try:
                    import gdown
                    url = f'https://drive.google.com/uc?id={file_id}'
                    gdown.download(url, self.model_path, quiet=False)
                except ImportError:
                    st.warning("gdown not available, trying alternative download...")
                    # Method 2: Direct download
                    url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    urllib.request.urlretrieve(url, self.model_path)
                
                # Method 3: Using requests with session for large files
                if not os.path.exists(self.model_path):
                    st.info("Trying alternative download method...")
                    session = requests.Session()
                    response = session.get(f"https://docs.google.com/uc?export=download&id={file_id}", stream=True)
                    
                    # Handle large file confirmation
                    for key, value in response.cookies.items():
                        if key.startswith('download_warning'):
                            params = {'confirm': value}
                            response = session.get(
                                "https://docs.google.com/uc?export=download", 
                                params=params, 
                                stream=True
                            )
                    
                    # Download the file
                    with open(self.model_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=32768):
                            if chunk:
                                f.write(chunk)
            
            if os.path.exists(self.model_path):
                file_size = os.path.getsize(self.model_path) / (1024 * 1024)
                if file_size > 1:  # Ensure file is not empty/corrupted
                    st.success(f"✅ Model downloaded successfully! ({file_size:.1f} MB)")
                    return True
                else:
                    st.error("❌ Downloaded file is too small - may be corrupted")
                    return False
            else:
                st.error("❌ Failed to download model file")
                return False
                
        except Exception as e:
            st.error(f"❌ Download error: {e}")
            return False

    def download_model_fallback(self):
        """Alternative download method using different services"""
        try:
            st.info("🔄 Trying alternative download method...")
            
            # List of potential download URLs - YOU NEED TO ADD YOUR ACTUAL URLs
            download_urls = [
                # Add your actual file URLs here:
                # "https://your-cloud-storage.com/resnet34_bird_region_weights.pth",
                # "https://dropbox.com/s/yourfilelink/resnet34_bird_region_weights.pth",
                # "https://github.com/yourusername/yourrepo/raw/main/resnet34_bird_region_weights.pth",
            ]
            
            for url in download_urls:
                try:
                    st.info(f"Trying: {url}")
                    urllib.request.urlretrieve(url, self.model_path)
                    if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:  # >1MB
                        st.success("✅ Model downloaded via alternative method!")
                        return True
                except:
                    continue
            
            return False
            
        except Exception as e:
            st.error(f"❌ Alternative download failed: {e}")
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
            torch
            torchvision
            pillow
            numpy
            opencv-python-headless
            requests
            gdown
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
                if not self.download_model_fallback():
                    st.error("""
                    ❌ Could not download the model file.
                    
                    Please manually upload the model file to one of these services and update the download URL:
                    1. Google Drive (make shareable)
                    2. Dropbox
                    3. GitHub Releases
                    4. Any cloud storage
                    
                    Then update the download URLs in the code.
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

    # ... (keep the rest of your methods: detect_bird_regions, classify_bird_region, predict_bird_species)
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
                classifications.append(("Processing Error", 0.0))
        
        return detections, classifications, original_image

# ... (keep the BirdInformation class and other functions the same as before)

class BirdInformation:
    def __init__(self):
        self.bird_database = {}
    
    def load_bird_database(self, species_list):
        """Load bird information"""
        uganda_bird_info = {
            "African Fish Eagle": {
                "description": "A majestic bird of prey found near water bodies across Uganda.",
                "habitat": "Lakes, rivers, reservoirs",
                "diet": "Fish, waterfowl, small mammals",
                "fun_fact": "The African Fish Eagle's cry is often called 'the voice of Africa'.",
                "conservation_status": "Least Concern",
                "size": "63-75 cm",
                "wingspan": "175-210 cm"
            },
            "Grey Crowned Crane": {
                "description": "Uganda's national bird with golden crown of feathers.",
                "habitat": "Wetlands, grasslands",
                "diet": "Insects, seeds, small vertebrates",
                "fun_fact": "Perform elaborate dancing rituals during courtship.",
                "conservation_status": "Endangered",
                "size": "100-110 cm",
                "wingspan": "180-200 cm"
            },
            # ... (include other bird data)
        }
        
        updated_database = {}
        for species in species_list:
            if species in uganda_bird_info:
                updated_database[species] = uganda_bird_info[species]
            else:
                updated_database[species] = {
                    "description": f"The {species} is found in Uganda.",
                    "habitat": "Various habitats in Uganda",
                    "diet": "Varied diet",
                    "fun_fact": f"Contributes to Uganda's biodiversity.",
                    "conservation_status": "Protected",
                    "size": "Varies",
                    "wingspan": "Varies"
                }
        
        self.bird_database = updated_database
    
    def get_bird_information(self, species_name):
        return self.bird_database.get(species_name, {
            "description": f"Information about {species_name}",
            "habitat": "Various habitats",
            "diet": "Species-specific diet",
            "fun_fact": "Interesting facts about this species",
            "conservation_status": "Monitored",
            "size": "Data available",
            "wingspan": "Data available"
        })

def initialize_system():
    """Initialize the system"""
    if 'bird_model' not in st.session_state:
        st.session_state.bird_model = ResNet34BirdModel()
        st.session_state.bird_info = BirdInformation()
        st.session_state.detection_complete = False
        st.session_state.bird_detections = []
        st.session_state.bird_classifications = []
        st.session_state.current_image = None
        st.session_state.active_method = "upload"
        st.session_state.model_loaded = False
        st.session_state.system_initialized = False
    
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing Uganda Bird Spotter System..."):
            success = st.session_state.bird_model.load_model()
            
            if success:
                st.session_state.bird_info.load_bird_database(st.session_state.bird_model.bird_species)
                st.session_state.model_loaded = True
                st.session_state.system_initialized = True
                st.success(f"✅ System ready! Can identify {len(st.session_state.bird_model.bird_species)} bird species")
            else:
                st.error("❌ System initialization failed.")

def main():
    initialize_system()
    
    if not st.session_state.get('system_initialized', False):
        st.error("System failed to initialize. Please check the requirements.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🦅 Detectable Birds")
        st.markdown(f"**Total Species:** {len(st.session_state.bird_model.bird_species)}")
        
        st.markdown('<div class="bird-list">', unsafe_allow_html=True)
        for species in st.session_state.bird_model.bird_species:
            st.markdown(f"• {species}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="main-header">Uganda Bird Spotter</div>', unsafe_allow_html=True)
    
    # ... (rest of your main function remains the same)

if __name__ == "__main__":
    main()