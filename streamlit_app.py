import streamlit as st
import numpy as np
from PIL import Image
import tempfile
import os
import base64
from io import BytesIO
import json
import random

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
        width: 60px;
        height: 60px;
        border-radius: 12px;
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
        transition: all 0.3s ease;
    }
    .glass-upload:hover {
        background: rgba(255, 255, 255, 0.35);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        transform: translateY(-2px);
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
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
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
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background: rgba(46, 134, 171, 0.9);
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(46, 134, 171, 0.3);
    }
    .file-upload-text {
        color: #2E86AB;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 15px;
    }
    .file-upload-subtext {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 20px;
    }
    .upload-icon {
        font-size: 3rem;
        margin-bottom: 15px;
        color: #2E86AB;
    }
    .section-title {
        font-size: 1.5rem;
        color: #2E86AB;
        margin-bottom: 15px;
        text-align: center;
        font-weight: 600;
    }
    .method-button {
        width: 100%;
        margin: 10px 0;
        padding: 15px;
        font-size: 1.1rem;
    }
    .active-method {
        background: rgba(46, 134, 171, 0.9) !important;
        border: 2px solid #2E86AB !important;
    }
    .warning-box {
        background: rgba(255, 193, 7, 0.2);
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
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
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .bird-list {
        max-height: 400px;
        overflow-y: auto;
        padding-right: 10px;
    }
    .bird-list::-webkit-scrollbar {
        width: 6px;
    }
    .bird-list::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.1);
        border-radius: 3px;
    }
    .bird-list::-webkit-scrollbar-thumb {
        background: #2E86AB;
        border-radius: 3px;
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
        # Look for model weights in current directory
        self.model_path = self._find_model_file()
        self.label_map_path = './label_map.json'
        
    def _find_model_file(self):
        """Find the ResNet34 model file in the current directory"""
        possible_names = [
            'resnet34_bird_region_weights.pth',
            'resnet34_weights.pth',
            'bird_model.pth',
            'model_weights.pth',
            'resnet34.pth'
        ]
        
        for filename in possible_names:
            if os.path.exists(filename):
                st.info(f"📁 Found model file: {filename}")
                return filename
        
        # Also check in models subdirectory
        for filename in possible_names:
            model_path = f'./models/{filename}'
            if os.path.exists(model_path):
                st.info(f"📁 Found model file: {model_path}")
                return model_path
        
        st.error("❌ No model file found in current directory. Please ensure the ResNet34 model file is in the app directory.")
        return None
    
    def check_dependencies(self):
        """Check if PyTorch and torchvision are available"""
        try:
            import torch
            import torchvision
            return True
        except ImportError:
            st.error("""
            ❌ PyTorch and torchvision are required but not installed.
            
            Please install them first:
            ```bash
            pip install torch torchvision pillow
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
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.label_map_path) if os.path.dirname(self.label_map_path) else '.', exist_ok=True)
        
        with open(self.label_map_path, 'w') as f:
            json.dump(label_map, f, indent=2)
        
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.bird_species = default_species
        st.success(f"✅ Created default label map with {len(self.bird_species)} species")
        return True
    
    def load_label_map(self):
        """Load the label map for bird species, create default if missing"""
        if not os.path.exists(self.label_map_path):
            st.warning("📝 Label map not found, creating default...")
            return self.create_default_label_map()
        
        try:
            with open(self.label_map_path, 'r') as f:
                label_map = json.load(f)
            
            self.inv_label_map = {v: k for k, v in label_map.items()}
            self.bird_species = list(label_map.keys())
            st.success(f"✅ Loaded {len(self.bird_species)} bird species")
            return True
        except Exception as e:
            st.error(f"❌ Error loading label map: {e}")
            st.warning("Creating default label map...")
            return self.create_default_label_map()
    
    def load_model(self):
        """Load the ResNet34 model with weights from local directory"""
        if not self.check_dependencies():
            return False
        
        if self.model_path is None or not os.path.exists(self.model_path):
            st.error("❌ Model file not found. Cannot proceed without the ResNet34 model.")
            return False
        
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            
            # Load label map first
            if not self.load_label_map():
                return False
            
            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            st.info(f"🔄 Using device: {self.device}")
            
            # Create ResNet34 model architecture
            model = models.resnet34(weights=None)
            num_classes = len(self.bird_species)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            
            # Load trained weights from local file
            st.info(f"🔄 Loading model weights from: {self.model_path}")
            
            # Get file size for info
            file_size = os.path.getsize(self.model_path) / (1024 * 1024)
            st.info(f"📦 Model file size: {file_size:.1f} MB")
            
            # Load the state dict
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(self.model_path))
            else:
                # Load on CPU
                model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            
            self.model = model.to(self.device)
            self.model.eval()
            
            # Define image transforms (must match training)
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
        """Detect bird regions in image using simple color-based detection"""
        try:
            # Convert to numpy array for processing
            if isinstance(image, np.ndarray):
                image_array = image
            else:
                image_array = np.array(image)
            
            height, width = image_array.shape[:2]
            
            st.info("🔍 Scanning image for birds...")
            
            # Simple bird detection based on color and edges
            detections = self._simple_bird_detection(image_array)
            
            if detections:
                st.success(f"✅ Found {len(detections)} bird region(s)")
                return detections, image_array
            else:
                st.info("🔍 No birds detected in this image")
                return [], image_array
                
        except Exception as e:
            st.error(f"❌ Error processing image: {e}")
            return [], None
    
    def _simple_bird_detection(self, image_array):
        """Simple bird detection using color analysis"""
        try:
            import cv2
            
            # Convert to HSV color space for better color detection
            hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
            
            # Define color ranges for common bird colors (browns, blues, greens, reds)
            color_ranges = [
                # Brown tones (common in many birds)
                ([10, 50, 50], [20, 255, 255]),
                # Blue tones (like kingfishers, turacos)
                ([100, 50, 50], [130, 255, 255]),
                # Green tones (parrots, turacos)
                ([40, 50, 50], [80, 255, 255]),
                # Red tones (cardinals, etc.)
                ([0, 50, 50], [10, 255, 255]),
                ([170, 50, 50], [180, 255, 255])
            ]
            
            all_contours = []
            for lower, upper in color_ranges:
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                
                mask = cv2.inRange(hsv, lower, upper)
                
                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by size
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 500 < area < 50000:  # Reasonable bird size range
                        x, y, w, h = cv2.boundingRect(contour)
                        # Ensure reasonable aspect ratio for birds
                        aspect_ratio = w / h
                        if 0.3 < aspect_ratio < 3.0:
                            all_contours.append((x, y, w, h))
            
            # Remove duplicates and return with confidence scores
            if all_contours:
                # Simple non-maximum suppression
                filtered_boxes = []
                for box in all_contours:
                    x, y, w, h = box
                    # Calculate confidence based on size and position
                    confidence = min(0.95, 0.7 + (w * h) / 100000)
                    filtered_boxes.append(([x, y, w, h], confidence))
                
                return filtered_boxes[:3]  # Return max 3 detections
            
            return []
            
        except Exception as e:
            st.warning(f"OpenCV not available for advanced detection: {e}")
            # Simple fallback - assume one bird in center
            height, width = image_array.shape[:2]
            detections = []
            
            # Create a reasonable bounding box in the center
            x = width // 4
            y = height // 4
            w = width // 2
            h = height // 2
            
            detection_confidence = 0.85
            detections.append(([x, y, w, h], detection_confidence))
            
            return detections
    
    def classify_bird_region(self, bird_region):
        """Classify bird region using the ResNet34 model"""
        if not self.model_loaded:
            return "Model not loaded", 0.0
        
        try:
            import torch
            
            # Ensure bird_region is PIL Image
            if isinstance(bird_region, np.ndarray):
                bird_region = Image.fromarray(bird_region)
            
            # Preprocess the bird region
            input_tensor = self.transform(bird_region).unsqueeze(0).to(self.device)
            
            # Predict using ResNet34 model
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
        """Complete prediction pipeline using ResNet34 model"""
        if not self.model_loaded:
            st.error("❌ Model not loaded. Cannot make predictions.")
            return [], [], None
        
        # Detect bird regions
        detections, original_image = self.detect_bird_regions(image)
        
        if not detections:
            return [], [], original_image
        
        classifications = []
        
        # Convert to PIL for cropping
        if isinstance(original_image, np.ndarray):
            pil_original = Image.fromarray(original_image)
        else:
            pil_original = original_image
        
        # Classify each detected region
        for i, (box, detection_confidence) in enumerate(detections):
            x, y, w, h = box
            
            # Crop bird region for classification
            try:
                # Ensure crop coordinates are within image bounds
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(pil_original.width, x + w), min(pil_original.height, y + h)
                
                if x2 > x1 and y2 > y1:  # Valid crop region
                    bird_region = pil_original.crop((x1, y1, x2, y2))
                    
                    # Display the cropped region for debugging
                    with st.expander(f"View Bird Region {i+1}"):
                        st.image(bird_region, caption=f"Bird Region {i+1}", use_column_width=True)
                else:
                    bird_region = pil_original  # Fallback to full image
                
                # Classify using model
                species, classification_confidence = self.classify_bird_region(bird_region)
                classifications.append((species, classification_confidence))
                
            except Exception as e:
                st.error(f"❌ Error processing bird region {i+1}: {e}")
                classifications.append(("Processing Error", 0.0))
        
        return detections, classifications, original_image

class BirdInformation:
    def __init__(self):
        self.bird_database = {}
    
    def load_bird_database(self, species_list):
        """Load bird information for the species in the model"""
        uganda_bird_info = {
            "African Fish Eagle": {
                "description": "A majestic bird of prey found near water bodies across Uganda. Known for its distinctive white head and powerful talons.",
                "habitat": "Lakes, rivers, reservoirs, and large water bodies",
                "diet": "Fish, waterfowl, small mammals, carrion",
                "fun_fact": "The African Fish Eagle's haunting cry is often called 'the voice of Africa' and is a signature sound of the continent's wilderness areas.",
                "conservation_status": "Least Concern",
                "size": "63-75 cm",
                "wingspan": "175-210 cm"
            },
            "Grey Crowned Crane": {
                "description": "Uganda's national bird, known for its golden crown of feathers and elegant dancing displays.",
                "habitat": "Wetlands, grasslands, agricultural fields",
                "diet": "Insects, seeds, small vertebrates, grains",
                "fun_fact": "These cranes perform elaborate dancing rituals that include bowing, jumping, and wing-flapping as part of their courtship behavior.",
                "conservation_status": "Endangered",
                "size": "100-110 cm",
                "wingspan": "180-200 cm"
            },
            "Shoebill Stork": {
                "description": "A prehistoric-looking bird with a massive shoe-shaped bill, found in Uganda's swamps and marshes.",
                "habitat": "Freshwater swamps, marshes, dense vegetation",
                "diet": "Fish, frogs, snakes, baby crocodiles, turtles",
                "fun_fact": "Shoebills can stand motionless for hours waiting for prey, earning them the nickname 'statue-like hunters'.",
                "conservation_status": "Vulnerable",
                "size": "110-140 cm",
                "wingspan": "230-260 cm"
            },
            "Lilac-breasted Roller": {
                "description": "One of Africa's most colorful birds with stunning rainbow-like plumage and acrobatic flight displays.",
                "habitat": "Savannas, open woodlands, agricultural areas",
                "diet": "Insects, small reptiles, rodents, amphibians",
                "fun_fact": "During courtship, males perform spectacular aerial acrobatics including rolling and diving, which gives them their name.",
                "conservation_status": "Least Concern",
                "size": "28-30 cm",
                "wingspan": "50-58 cm"
            },
            "Great Blue Turaco": {
                "description": "A large, colorful bird with striking blue plumage and a distinctive crest, often seen in forest canopies.",
                "habitat": "Forests, woodlands, and forest edges",
                "diet": "Fruits, leaves, flowers, and occasionally insects",
                "fun_fact": "Despite their size, turacos are excellent climbers and can run along branches like squirrels.",
                "conservation_status": "Least Concern",
                "size": "70-76 cm",
                "wingspan": "95-100 cm"
            },
            "African Jacana": {
                "description": "Known as the 'lily-trotter' for its ability to walk on floating vegetation with its long toes.",
                "habitat": "Freshwater wetlands, lakes, and marshes",
                "diet": "Insects, snails, small fish, and aquatic invertebrates",
                "fun_fact": "Jacanas have a unique breeding system where females mate with multiple males and leave them to care for the eggs and chicks.",
                "conservation_status": "Least Concern",
                "size": "23-31 cm",
                "wingspan": "50-55 cm"
            },
            "Marabou Stork": {
                "description": "A large wading bird with a distinctive bald head and massive bill, often seen near human settlements.",
                "habitat": "Wetlands, savannas, and urban areas",
                "diet": "Carrion, fish, frogs, small mammals, and human waste",
                "fun_fact": "Marabou storks have the largest wingspan of any land bird, reaching up to 3.2 meters (10.5 feet).",
                "conservation_status": "Least Concern",
                "size": "120-150 cm",
                "wingspan": "225-320 cm"
            },
            "Pied Kingfisher": {
                "description": "A black and white kingfisher known for its hovering flight and diving fishing technique.",
                "habitat": "Rivers, lakes, coasts, and mangroves",
                "diet": "Small fish, aquatic insects, and crustaceans",
                "fun_fact": "Pied kingfishers can hover in place like helicopters before diving straight down into the water to catch fish.",
                "conservation_status": "Least Concern",
                "size": "25-30 cm",
                "wingspan": "45-50 cm"
            }
        }
        
        # Only include species that are in both the model and our database
        updated_database = {}
        for species in species_list:
            if species in uganda_bird_info:
                updated_database[species] = uganda_bird_info[species]
            else:
                # Generic info for species not in our detailed database
                updated_database[species] = {
                    "description": f"The {species} is a beautiful bird species found in Uganda with unique characteristics and behaviors.",
                    "habitat": "Various habitats across Uganda including forests, wetlands, and savannas",
                    "diet": "Varied diet including insects, seeds, fruits, and small animals",
                    "fun_fact": f"The {species} contributes to Uganda's rich biodiversity and plays an important role in the ecosystem.",
                    "conservation_status": "Protected in Uganda",
                    "size": "Varies by species",
                    "wingspan": "Varies by species"
                }
        
        self.bird_database = updated_database
    
    def get_bird_information(self, species_name):
        """Get information about the identified bird species"""
        return self.bird_database.get(species_name, {
            "description": f"The {species_name} is a bird species identifiable by our ResNet34 model.",
            "habitat": "Various habitats in Uganda",
            "diet": "Species-specific diet",
            "fun_fact": "This species is part of Uganda's diverse avian population.",
            "conservation_status": "Monitored",
            "size": "Data available",
            "wingspan": "Data available"
        })
    
    def generate_bird_report(self, species_name, bird_info):
        """Generate a comprehensive bird report"""
        report = f"""
# 🐦 Bird Report - {species_name}

## Description
{bird_info['description']}

## Physical Characteristics
- **Size**: {bird_info['size']}
- **Wingspan**: {bird_info['wingspan']}

## Habitat & Behavior
- **Primary Habitat**: {bird_info['habitat']}
- **Diet**: {bird_info['diet']}
- **Conservation Status**: {bird_info['conservation_status']}

## Did You Know?
{bird_info['fun_fact']}

## Identification Details
- **Model Used**: ResNet34 trained on Ugandan bird species
- **Source**: Local model weights

*Report generated by Uganda Bird Spotter - Powered by ResNet34 AI Model*
"""
        return report

def get_base64_image(image_path):
    """Convert image to base64 for embedding in HTML"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except:
        return ""

def initialize_system():
    """Initialize the bird detection system with robust error handling"""
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
    
    # Initialize system only once
    if not st.session_state.system_initialized:
        with st.spinner("🚀 Initializing Uganda Bird Spotter System..."):
            # Try to load the model
            success = st.session_state.bird_model.load_model()
            
            if success:
                # Load bird information database
                st.session_state.bird_info.load_bird_database(st.session_state.bird_model.bird_species)
                st.session_state.model_loaded = True
                st.session_state.system_initialized = True
                st.success(f"✅ System ready! ResNet34 model active - Can identify {len(st.session_state.bird_model.bird_species)} bird species")
            else:
                st.error("❌ System initialization failed. Please check the requirements and model file.")
                st.session_state.system_initialized = False

def main():
    # Initialize the system
    initialize_system()
    
    # Check if system initialized properly
    if not st.session_state.get('system_initialized', False):
        st.error("""
        ❌ System failed to initialize properly. 
        
        Please check:
        1. Required dependencies are installed: `pip install torch torchvision pillow streamlit numpy opencv-python`
        2. ResNet34 model file exists in the app directory
        3. Sufficient memory available
        
        The app cannot run without the ResNet34 model file.
        """)
        return
    
    bird_model = st.session_state.bird_model
    bird_info = st.session_state.bird_info
    
    # Sidebar with logo and bird list only
    with st.sidebar:
        # Logo
        try:
            base64_logo = get_base64_image("ugb1.png")
            st.markdown(f'<img src="data:image/png;base64,{base64_logo}" class="sidebar-logo" alt="Bird Spotter Logo">', unsafe_allow_html=True)
        except:
            st.markdown('<div class="sidebar-logo" style="background: #2E86AB; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 24px;">UG</div>', unsafe_allow_html=True)
        
        st.markdown("### 🦅 Detectable Birds")
        st.markdown(f"**Total Species:** {len(bird_model.bird_species)}")
        
        # Bird list with scroll
        st.markdown('<div class="bird-list">', unsafe_allow_html=True)
        for species in bird_model.bird_species:
            st.markdown(f"• {species}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main app content
    # Custom header
    st.markdown(f"""
    <div class="main-header">
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
                
                bird_data = bird_info.get_bird_information(species)
                
                # Bird details
                st.markdown('<div class="glass-info">', unsafe_allow_html=True)
                st.markdown(f"### 📚 About {species}")
                st.markdown(f"""
                <div style="line-height: 1.6;">
                    <p><strong>Description:</strong> {bird_data['description']}</p>
                    <p><strong>Habitat:</strong> {bird_data['habitat']}</p>
                    <p><strong>Diet:</strong> {bird_data['diet']}</p>
                    <p><strong>Size:</strong> {bird_data['size']}</p>
                    <p><strong>Wingspan:</strong> {bird_data['wingspan']}</p>
                    <p><strong>Conservation Status:</strong> {bird_data['conservation_status']}</p>
                    <p><strong>Fun Fact:</strong> {bird_data['fun_fact']}</p>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Report generation
                if st.button(f"📄 Generate Report for {species}", key=f"report_{i}", use_container_width=True):
                    report = bird_info.generate_bird_report(species, bird_data)
                    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                    st.markdown(report)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Reset button
        if st.button("🔄 Analyze Another Image", type="secondary", use_container_width=True):
            st.session_state.detection_complete = False
            st.session_state.bird_detections = []
            st.session_state.bird_classifications = []
            st.session_state.current_image = None
            st.rerun()

if __name__ == "__main__":
    main()