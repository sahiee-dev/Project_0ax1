import streamlit as st
import sys
import os

# Add parent dir to sys.path to allow importing from model and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import tempfile
from PIL import Image
import numpy as np

from model.loader import load_model
from model.inference import run_inference
from utils.processing import parse_results
from utils.localization import get_text

# Page Config
st.set_page_config(
    page_title="Advanced Weapon Detection System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'
    
if 'upload_mode' not in st.session_state:
    st.session_state.upload_mode = None

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

# ----------------- TECHNICAL DESIGN SYSTEM -----------------
st.markdown("""
    <style>
    /* Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Global Background */
    .stApp {
        background: #ffffff;
        color: #1a1a1a;
    }

    /* Headings */
    h1, h2, h3, h4 {
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #000000;
        margin-bottom: 1rem;
    }
    
    h1 {
        font-size: 3.5rem !important;
        line-height: 1.1;
        border-bottom: 2px solid #000000;
        padding-bottom: 1rem;
    }
    
    h2 {
        font-size: 2rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Technical Text Blocks */
    .tech-description {
        font-size: 1.05rem;
        line-height: 1.8;
        color: #333333;
        margin-bottom: 2.5rem;
        max-width: 900px;
    }

    .info-section {
        background: #f8f9fa;
        border-left: 4px solid #000000;
        padding: 2rem;
        margin: 2rem 0;
    }

    /* Buttons */
    .stButton>button {
        background: #000000;
        color: #ffffff;
        border: none;
        border-radius: 4px;
        padding: 1rem 2rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background: #333333;
        color: #ffffff;
        transform: translateY(-1px);
    }

    /* Selection Cards */
    .selection-card {
        background: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 2.5rem;
        border-radius: 0;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .selection-card:hover {
        border-color: #000000;
        box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    }

    .selection-header {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #000000;
        color: #ffffff !important;
        padding: 2rem;
        border-radius: 0;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #ffffff !important;
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem !important;
    }

    /* Status Indicators */
    .status-box {
        padding: 1rem 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        text-transform: uppercase;
        border: 2px solid;
    }
    
    .status-alert {
        background: #fffafa;
        border-color: #ff0000;
        color: #ff0000;
    }
    
    .status-secure {
        background: #fafffa;
        border-color: #008000;
        color: #008000;
    }
    
    </style>
    """, unsafe_allow_html=True)

# ----------------- LANDING PAGE -----------------
def landing_page():
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_main = st.columns([1, 10, 1])
    with col_main[1]:
        st.markdown("<h1>INTELLIGENT WEAPON DETECTION SYSTEM</h1>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-description">
        This high-fidelity surveillance framework utilizes state-of-the-art computer vision to identify potential weapon threats in real-time. 
        Developed as a robust security solution, the system integrates advanced neural architectures with specialized pre-processing 
        to ensure reliability in complex operational environments.
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("INITIALIZE SYSTEM PIPELINE"):
            navigate_to('app')

        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Methodology Section
        st.markdown("<h2>TECHNICAL METHODOLOGY</h2>", unsafe_allow_html=True)
        
        meth_col1, meth_col2 = st.columns(2)
        
        with meth_col1:
            st.markdown("""
            <div class="info-section">
                <h4 style="margin-top: 0;">YOLOv8 Neural Architecture</h4>
                <p style="font-size: 0.95rem; color: #444;">
                The core detection engine leverages the YOLOv8 (You Only Look Once) framework, an anchor-free detection paradigm. 
                Unlike previous iterations, YOLOv8 employs a highly efficient backbone that maximizes feature extraction through 
                C2f (Cross Stage Partial Bottleneck with two convolutions) modules. This allows for superior spatial understanding 
                and pixel-perfect localization of objects within a single forward pass.
                </p>
                <ul style="font-size: 0.9rem; color: #666; font-family: 'JetBrains Mono', monospace;">
                    <li>Decoupled Detection Heads</li>
                    <li>Global Contextual Pooling</li>
                    <li>Task-Aligned Assignment</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with meth_col2:
            st.markdown("""
            <div class="info-section" style="border-left-color: #444;">
                <h4 style="margin-top: 0;">Wavelet-Based Signal Refinement</h4>
                <p style="font-size: 0.95rem; color: #444;">
                To counteract the common challenges of CCTV feeds—specifically noise and variable lighting—this system implements 
                Discrete Wavelet Transformation (DWT). By decomposing input signals into multi-frequency sub-bands, we isolate 
                high-frequency edges characteristic of metallic weapon surfaces while suppressing low-frequency noise. This 
                dual-domain processing significantly reduces false-positive rates in low-visibility scenarios.
                </p>
                <ul style="font-size: 0.9rem; color: #666; font-family: 'JetBrains Mono', monospace;">
                    <li>Multi-Resolution Decomposition</li>
                    <li>Edge Feature Preservation</li>
                    <li>Atmospheric Noise Reduction</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-section" style="background: #ffffff; border: 1px solid #eee;">
            <p style="text-align: center; color: #888; font-size: 0.85rem; letter-spacing: 0.05em; margin: 0;">
            RESEARCH & DEVELOPMENT | AUTONOMOUS SURVEILLANCE PROTOCOL V2.0
            </p>
        </div>
        """, unsafe_allow_html=True)

# ----------------- MAIN APP -----------------
def main_app():
    # Sidebar
    with st.sidebar:
        st.markdown("<h4 style='color: #fff;'>SYSTEM CONTROL</h4>", unsafe_allow_html=True)
        if st.button("TERMINATE SESSION"):
            st.session_state.upload_mode = None
            navigate_to('landing')
            
        st.markdown("<br>", unsafe_allow_html=True)
        lang = st.selectbox("LOCALIZATION SCHEME", ["English", "Telugu"], index=0)
        lang_code = "en" if lang == "English" else "te"
        

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<h4 style='color: #fff; font-size: 0.9rem;'>INFERENCE CONFIGURATION</h4>", unsafe_allow_html=True)
        
        # Model Selection
        model_options = {
            "Standard (Nano)": "./runs/detect/Normal_Compressed/weights/best.pt",
            "High Accuracy (Small - DB)": "./models/db.onnx",
            "High Accuracy (Small - Haar)": "./models/haar.onnx"
        }
        selected_model_name = st.selectbox("MODEL ARCHITECTURE", list(model_options.keys()), index=0)
        model_path = model_options[selected_model_name]
        
        # Inference Parameters
        conf_threshold = st.slider("CONFIDENCE THRESHOLD", 0.0, 1.0, 0.25, 0.05)
        iou_threshold = st.slider("IOU THRESHOLD", 0.0, 1.0, 0.7, 0.05)
        use_tta = st.checkbox("ENABLE TTA (TEST TIME AUGMENTATION)", value=False, help="Increases accuracy but reduces speed.")

        st.markdown("---")
        st.markdown(f"""
        <div style='font-family: JetBrains Mono, monospace; font-size: 0.75rem; color: #888;'>
        MODEL: {selected_model_name.upper()}<br>
        BACKEND: CUDA/NVIDIA_RTX<br>
        LATENCY: &lt; 42MS
        </div>
        """, unsafe_allow_html=True)

    # model_path is now set by the selector above
    
    @st.cache_resource
    def get_model(path):
        return load_model(path)

    try:
        model = get_model(model_path)
    except Exception as e:
        st.error(f"SYSTEM FAULT DETECTED: {e}")
        st.stop()
        
    st.markdown(f"<h2>{get_text('app_title', lang_code).upper()}</h2>", unsafe_allow_html=True)

    # Mode Selection
    if st.session_state.upload_mode is None:
        st.markdown("<p style='font-family: JetBrains Mono, monospace; color: #666;'>SELECT INPUT PIPELINE FOR ANALYSIS</p>", unsafe_allow_html=True)
        
        mode_col1, mode_col2 = st.columns(2)
        
        with mode_col1:
            st.markdown("""
            <div class="selection-card">
                <div class="selection-header">Static Input Pipeline</div>
                <h3>IMAGE ANALYSIS</h3>
                <p style="color: #666; font-size: 0.9rem;">
                Execution protocol for individual frames. High-precision inspection with detailed per-object metadata. 
                Supported for forensic evidence review and high-resolution verification.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("INITIATE IMAGE ANALYSIS", key="mode_image"):
                st.session_state.upload_mode = 'image'
                st.rerun()
                
        with mode_col2:
            st.markdown("""
            <div class="selection-card">
                <div class="selection-header">Dynamic Input Pipeline</div>
                <h3>VIDEO ANALYSIS</h3>
                <p style="color: #666; font-size: 0.9rem;">
                Real-time sequential frame processing. Implements frame aggregation logic to track threats across 
                temporal domains. Optimized for live monitoring and recorded stream review.
                </p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("INITIATE VIDEO ANALYSIS", key="mode_video"):
                st.session_state.upload_mode = 'video'
                st.rerun()
    
    # Image Mode
    elif st.session_state.upload_mode == 'image':
        if st.button("SELECT ALTERNATE PIPELINE"):
            st.session_state.upload_mode = None
            st.rerun()
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "INPUT SOURCE: SELECT IMAGE FILE (JPG/PNG)", 
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.8rem;'>RAW SOURCE DATA</p>", unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                
                if st.button("EXECUTE DETECTION ALGORITHM", width='stretch'):
                    with st.spinner("PROCESSING NEURAL LAYERS..."):
                        results = run_inference(model, image, conf=conf_threshold, iou=iou_threshold, augment=use_tta)
                        processed_data = parse_results(results)
                        st.session_state.processed_data_img = processed_data
            
            with img_col2:
                st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.8rem;'>INFERENCE OUTPUT</p>", unsafe_allow_html=True)
                if 'processed_data_img' in st.session_state:
                    data = st.session_state.processed_data_img
                    st.image(data['annotated_image'], use_container_width=True, channels="BGR")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    met_col1, met_col2 = st.columns(2)
                    met_col1.metric("WEAPONS DETECTED", data['total_weapons'])
                    
                    status_class = "status-alert" if data['total_weapons'] > 0 else "status-secure"
                    status_text = "STATUS: THREAT DETECTED" if data['total_weapons'] > 0 else "STATUS: SYSTEM SECURE"
                    
                    st.markdown(f'<div class="status-box {status_class}">{status_text}</div>', unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.8rem;'>DATA TELEMETRY</p>", unsafe_allow_html=True)
                    st.json({
                        "processing_engine": "YOLOv8_Engine",
                        "detection_instances": data['total_weapons'],
                        "classification_breakdown": data['counts'],
                        "inference_status": "Complete"
                    })

    # Video Mode
    elif st.session_state.upload_mode == 'video':
        if st.button("SELECT ALTERNATE PIPELINE"):
            st.session_state.upload_mode = None
            st.rerun()
        
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "INPUT SOURCE: SELECT VIDEO FILE (MP4/AVI/MOV)", 
            type=["mp4", "avi", "mov"]
        )
        
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            if st.button("EXECUTE STREAM ANALYSIS"):
                 vid_col1, vid_col2 = st.columns(2)
                 
                 with vid_col1:
                     st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.8rem;'>INPUT STREAM FEED</p>", unsafe_allow_html=True)
                     input_placeholder = st.empty()
                     
                 with vid_col2:
                     st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.8rem;'>REAL-TIME TELEMETRY OVERLAY</p>", unsafe_allow_html=True)
                     output_placeholder = st.empty()
                 
                 st.markdown("---")
                 metric_col1, metric_col2 = st.columns(2)
                 frames_metric = metric_col1.empty()
                 weapons_metric = metric_col2.empty()
                 
                 progress_bar = st.progress(0)
                 
                 cap = cv2.VideoCapture(tfile.name)
                 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                 
                 frames_processed = 0
                 frames_with_weapon = 0
                 unique_weapon_ids = set()
                 
                 while cap.isOpened():
                     ret, frame = cap.read()
                     if not ret:
                         break
                     
                     frames_processed += 1
                     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                     input_placeholder.image(frame_rgb, use_container_width=True)
                     
                     results = run_inference(model, frame, conf=conf_threshold, iou=iou_threshold, augment=use_tta, track=True)
                     processed_data = parse_results(results)
                     
                     output_placeholder.image(processed_data['annotated_image'], channels="BGR", use_container_width=True)
                     
                     weapon_count = processed_data['total_weapons']
                     if weapon_count > 0:
                         frames_with_weapon += 1
                         # Add tracking IDs to unique set
                         for det in processed_data['detections']:
                             if det.get('track_id') is not None:
                                 unique_weapon_ids.add(det['track_id'])
                     
                     if total_frames > 0:
                         progress_bar.progress(min(frames_processed / total_frames, 1.0))
                          
                     frames_metric.metric("FRAMES WITH THREATS", frames_with_weapon)
                     weapons_metric.metric("TOTAL UNIQUE WEAPONS", len(unique_weapon_ids))
                     
                 cap.release()
                 progress_bar.progress(1.0)
                 st.markdown('<div class="status-box status-secure">PIPELINE ANALYSIS COMPLETE</div>', unsafe_allow_html=True)
                 
                 st.markdown("<br>", unsafe_allow_html=True)
                 st.markdown("<p style='font-family: JetBrains Mono; font-size: 0.8rem;'>AGGREGATE SESSION REPORT</p>", unsafe_allow_html=True)
                 st.json({
                     "session_metrics": {
                         "frames_analyzed": frames_processed,
                         "threat_occurrence_rate": f"{(frames_with_weapon/frames_processed)*100:.2f}%" if frames_processed > 0 else "0%",
                         "total_unique_weapons": len(unique_weapon_ids)
                     },
                     "final_security_status": "CRITICAL_THREAT_DETECTED" if frames_with_weapon > 0 else "SESSION_SECURE"
                 })
            
            else:
                st.video(tfile.name)

# Router
if st.session_state.page == 'landing':
    landing_page()
else:
    main_app()
