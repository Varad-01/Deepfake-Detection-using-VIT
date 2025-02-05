import streamlit as st

# Set page configuration - This must be the first Streamlit command!
st.set_page_config(
    page_title="Deepfake Detection System",
    page_icon="🎥",
    layout="wide"
)

import torch
from vit_pytorch import ViT
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import tempfile
import os
import plotly.graph_objects as go
from datetime import datetime

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .deepfake {
        background-color: #ffebee;
        color: #c62828;
    }
    .authentic {
        background-color: #e8f5e9;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the ViT model with caching"""
    model = ViT(
        image_size=224,
        patch_size=32,
        num_classes=1,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )
    
    # Load pre-trained weights
    weights_path = "as_model_0.770.pt"  # Update this path
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    
    return model

def preprocess_frame(frame):
    """Preprocess video frame for model input"""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    return preprocess(pil_image).unsqueeze(0)

def process_video(video_path, model, progress_bar):
    """Process video frames and return predictions"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    predictions = []
    scores = []
    frames_processed = 0
    
    model.eval()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = preprocess_frame(frame)
            output = model(processed_frame)
            score = torch.sigmoid(output).cpu().numpy()[0][0]
            pred = 1 if score > 0.5 else 0
            
            predictions.append(pred)
            scores.append(score)
            
            frames_processed += 1
            progress_bar.progress(frames_processed / total_frames)
    
    cap.release()
    return predictions, scores, fps

def create_prediction_plot(scores, fps):
    """Create interactive plot using plotly"""
    time_points = [i/fps for i in range(len(scores))]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_points,
        y=scores,
        mode='lines',
        name='Confidence Score'
    ))
    
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="red",
        annotation_text="Decision Threshold"
    )
    
    fig.update_layout(
        title="Deepfake Detection Confidence Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Confidence Score",
        hovermode='x'
    )
    
    return fig

def main():
    st.title("🎥 Deepfake Detection System")
    
    # Sidebar
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Detection Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Adjust the threshold for deepfake detection"
    )
    
    # Load model
    try:
        with st.spinner("Loading model..."):
            model = load_model()
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov'],
        help="Upload a video file to analyze for deepfake detection"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        try:
            # Create columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Processing Video")
                progress_bar = st.progress(0)
                predictions, scores, fps = process_video(video_path, model, progress_bar)
                
                # Calculate statistics
                total_frames = len(predictions)
                positive_predictions = sum(predictions)
                deepfake_percentage = (positive_predictions/total_frames*100)
                avg_confidence = np.mean(scores) * 100
                
                # Display verdict
                if deepfake_percentage > 50:
                    st.markdown("""
                        <div class="verdict-box deepfake">
                            <h2>⚠️ DEEPFAKE DETECTED</h2>
                            <p>This video is likely manipulated</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="verdict-box authentic">
                            <h2>✅ AUTHENTIC VIDEO</h2>
                            <p>This video appears to be genuine</p>
                        </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Detection Statistics")
                st.metric("Frames Analyzed", total_frames)
                st.metric("Deepfake Confidence", f"{deepfake_percentage:.1f}%")
                st.metric("Average Score", f"{avg_confidence:.1f}%")
            
            # Show prediction plot
            st.subheader("Analysis Over Time")
            fig = create_prediction_plot(scores, fps)
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional information
            with st.expander("Detailed Analysis"):
                st.write(f"- Total frames analyzed: {total_frames}")
                st.write(f"- Frames classified as deepfake: {positive_predictions}")
                st.write(f"- Frames classified as authentic: {total_frames - positive_predictions}")
                st.write(f"- Video FPS: {fps:.2f}")
            
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        
        finally:
            # Cleanup temporary file
            os.unlink(video_path)
    
    # Footer
    st.markdown("---")
    st.markdown("Deepfake Detection System powered by Vision Transformer (ViT)")

if __name__ == "__main__":
    main()
