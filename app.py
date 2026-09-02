import os
import io
import zipfile
import tempfile
import time
from datetime import datetime
import json
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import plotly.express as px
import plotly.graph_objects as go
import torch

# ------------------------------
# Configuration and Constants
# ------------------------------
MODEL_PATH = "models/best.pt"
OUTPUT_DIR = "outputs"
IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "images")
VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "videos")
CSV_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "csv")

# Create output directories
for dir_path in [IMAGE_OUTPUT_DIR, VIDEO_OUTPUT_DIR, CSV_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Class names and colors (adjust based on your training)
CLASS_NAMES = {
    0: "HAIRNET",
    1: "NO HAIRNET"
}
CLASS_COLORS = {
    0: (0, 255, 0),      # Green for HAIRNET (BGR)
    1: (0, 0, 255)       # Red for NO HAIRNET (BGR)
}
CLASS_COLORS_RGB = {
    0: (0, 255, 0),      # Green for HAIRNET (RGB)
    1: (255, 0, 0)       # Red for NO HAIRNET (RGB)
}

# ------------------------------
# Page Configuration and Custom CSS
# ------------------------------
st.set_page_config(
    page_title="Hairnet Detection System",
    page_icon="🪖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton>button {
        background-color: #2a5298;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #1e3c72;
        color: white;
    }
    .detection-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2a5298;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #555;
    }
    .stream-live-badge {
        display: inline-block;
        background-color: #d32f2f;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Model Loading (cached)
# ------------------------------
@st.cache_resource
def load_model(model_path: str, device: str = None) -> YOLO:
    """
    Load YOLO model from path, optionally specifying device.
    If device is None, use GPU if available else CPU.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO(model_path)
    model.to(device)
    return model

def get_available_devices():
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    return devices

# ------------------------------
# Helper Functions
# ------------------------------
def annotate_image(image: np.ndarray, results, conf_threshold: float = 0.25) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.
    Only draws boxes with confidence above threshold.
    """
    annotated = image.copy()
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))

                # Draw rectangle
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                text = f"{label} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1,
                )
                # Draw label text
                cv2.putText(
                    annotated,
                    text,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
    return annotated

def get_detections_dataframe(results, conf_threshold: float = 0.25) -> pd.DataFrame:
    """
    Convert YOLO results to a pandas DataFrame with detection details.
    """
    data = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                label = CLASS_NAMES.get(cls_id, f"Class {cls_id}")
                data.append({
                    "Class": label,
                    "Confidence": round(conf, 3),
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1),
                    "Width": round(x2 - x1, 1),
                    "Height": round(y2 - y1, 1),
                    "Area": round((x2 - x1) * (y2 - y1), 1)
                })
    return pd.DataFrame(data)

def process_video(video_path: str, model: YOLO, output_path: str,
                  conf_threshold: float = 0.25, iou_threshold: float = 0.45,
                  progress_callback=None) -> list:
    """
    Process video frame by frame, annotate, save, and return detection stats per frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file.")

    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        frame_count = 1  # Avoid division by zero

    frame_idx = 0
    frame_stats = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)

        # Annotate frame
        annotated_frame = annotate_image(frame, results, conf_threshold)

        # Write to output
        out.write(annotated_frame)

        # Collect stats
        detections = get_detections_dataframe(results, conf_threshold)
        if not detections.empty:
            frame_summary = {
                "frame": frame_idx,
                "total_detections": len(detections),
                "hairnet_count": (detections["Class"] == "HAIRNET").sum(),
                "no_hairnet_count": (detections["Class"] == "NO HAIRNET").sum(),
            }
        else:
            frame_summary = {
                "frame": frame_idx,
                "total_detections": 0,
                "hairnet_count": 0,
                "no_hairnet_count": 0,
            }
        frame_stats.append(frame_summary)

        frame_idx += 1
        if progress_callback:
            progress_callback(frame_idx / frame_count)

    cap.release()
    out.release()
    return frame_stats

def save_detections_csv(detections_df: pd.DataFrame, file_path: str):
    """Save detections DataFrame to CSV."""
    detections_df.to_csv(file_path, index=False)

def process_single_image_bytes(file_bytes, filename, model, conf_threshold, iou_threshold):
    """
    Run detection on a single in-memory image (bytes) and return a result dict.
    Used by the batch/multiple-image pipeline.
    """
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = model(image_bgr, conf=conf_threshold, iou=iou_threshold, verbose=False)
    annotated_bgr = annotate_image(image_bgr, results, conf_threshold)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections_df = get_detections_dataframe(results, conf_threshold)
    if not detections_df.empty:
        detections_df.insert(0, "Filename", filename)

    return {
        "filename": filename,
        "original_rgb": image_np,
        "annotated_bgr": annotated_bgr,
        "annotated_rgb": annotated_rgb,
        "detections_df": detections_df,
    }

def build_zip_of_images(results_list):
    """
    Build an in-memory ZIP file containing the annotated images from a
    list of results produced by process_single_image_bytes.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for res in results_list:
            ok, buf = cv2.imencode(".png", res["annotated_bgr"])
            if ok:
                out_name = f"annotated_{res['filename']}"
                zf.writestr(out_name, buf.tobytes())
    zip_buffer.seek(0)
    return zip_buffer

# ------------------------------
# Session State Initialization
# ------------------------------
if 'history' not in st.session_state:
    st.session_state.history = []  # list of dicts: {timestamp, type, filename, stats}

if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []  # list of output file paths

if 'streaming' not in st.session_state:
    st.session_state.streaming = False

if 'stream_cap' not in st.session_state:
    st.session_state.stream_cap = None

if 'stream_stats' not in st.session_state:
    st.session_state.stream_stats = {"total": 0, "hairnet": 0, "no_hairnet": 0, "frames": 0}

# ------------------------------
# UI Components
# ------------------------------
def display_metrics(detections_df: pd.DataFrame):
    """Display summary metrics as cards."""
    if detections_df.empty:
        st.info("No detections found.")
        return
    total = len(detections_df)
    hairnet = (detections_df["Class"] == "HAIRNET").sum()
    no_hairnet = (detections_df["Class"] == "NO HAIRNET").sum()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">'
                    f'<div class="metric-value">{total}</div>'
                    '<div class="metric-label">Total Detections</div>'
                    '</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">'
                    f'<div class="metric-value">{hairnet}</div>'
                    '<div class="metric-label">Wearing Hairnet</div>'
                    '</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">'
                    f'<div class="metric-value">{no_hairnet}</div>'
                    '<div class="metric-label">Not Wearing Hairnet</div>'
                    '</div>', unsafe_allow_html=True)

def display_detection_chart(detections_df: pd.DataFrame):
    """Display a bar chart of detection counts."""
    if detections_df.empty:
        return
    counts = detections_df["Class"].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    fig = px.bar(counts, x="Class", y="Count", color="Class",
                 color_discrete_map={"HAIRNET": "green", "NO HAIRNET": "red"},
                 title="Detection Counts")
    st.plotly_chart(fig, use_container_width=True)

def add_to_history(entry: dict):
    """Add an entry to session history."""
    st.session_state.history.append(entry)

# ------------------------------
# Main Application
# ------------------------------
def main():
    # Header
    st.markdown('<div class="main-header"><h1>🪖 Hairnet Detection System</h1>'
                '<p>AI-powered compliance monitoring for workplace safety</p></div>',
                unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Settings")

    # Device selection
    devices = get_available_devices()
    if len(devices) > 1:
        device = st.sidebar.selectbox("Device", devices, index=1)
    else:
        device = devices[0]
        st.sidebar.info(f"Running on {device.upper()}")

    # Confidence threshold
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    # IoU threshold
    iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.45, 0.05)

    # Load model
    try:
        model = load_model(MODEL_PATH, device)
        st.sidebar.success(f"Model loaded on **{device.upper()}**")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.stop()

    # Model info
    with st.sidebar.expander("Model Information"):
        st.write(f"Model path: `{MODEL_PATH}`")
        st.write(f"Classes: {list(CLASS_NAMES.values())}")
        st.write("Input size: 640x640 (default)")
        st.write("Framework: Ultralytics YOLO")

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(["Detection", "Analytics Dashboard", "About"])

    # ------------------------------
    # Tab 1: Detection
    # ------------------------------
    with tab1:
        st.header("Detection")
        app_mode = st.radio(
            "Select Input Source",
            ["Image", "Multiple Images", "Video", "Multiple Videos", "Live Stream (URL)", "Webcam"],
            horizontal=True
        )

        # ------------------------------------------------------------
        # Single Image
        # ------------------------------------------------------------
        if app_mode == "Image":
            st.subheader("Upload Image")
            uploaded_file = st.file_uploader("Choose an image...",
                                             type=["jpg", "jpeg", "png", "bmp"],
                                             key="image_uploader")

            if uploaded_file is not None:
                # Read image
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                # Run inference
                with st.spinner("Detecting..."):
                    results = model(image_bgr, conf=conf_threshold, iou=iou_threshold,
                                    verbose=False)

                # Annotate
                annotated_bgr = annotate_image(image_bgr, results, conf_threshold)
                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                # Display side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original Image**")
                    st.image(image, use_column_width=True)
                with col2:
                    st.markdown("**Annotated Image**")
                    st.image(annotated_rgb, use_column_width=True)

                # Detections DataFrame
                detections_df = get_detections_dataframe(results, conf_threshold)
                st.markdown("### Detection Results")
                display_metrics(detections_df)
                display_detection_chart(detections_df)

                if not detections_df.empty:
                    st.dataframe(detections_df, use_container_width=True)

                    # Save and download
                    output_filename = f"annotated_{uploaded_file.name}"
                    output_path = os.path.join(IMAGE_OUTPUT_DIR, output_filename)
                    cv2.imwrite(output_path, annotated_bgr)

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        with open(output_path, "rb") as f:
                            st.download_button(
                                label="Download Annotated Image",
                                data=f,
                                file_name=output_filename,
                                mime="image/png"
                            )
                    with col_dl2:
                        csv_filename = f"detections_{uploaded_file.name.split('.')[0]}.csv"
                        csv_path = os.path.join(CSV_OUTPUT_DIR, csv_filename)
                        save_detections_csv(detections_df, csv_path)
                        with open(csv_path, "rb") as f:
                            st.download_button(
                                label="Download Detections (CSV)",
                                data=f,
                                file_name=csv_filename,
                                mime="text/csv"
                            )

                    # Add to history
                    add_to_history({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "image",
                        "filename": uploaded_file.name,
                        "stats": {
                            "total": len(detections_df),
                            "hairnet": (detections_df["Class"] == "HAIRNET").sum(),
                            "no_hairnet": (detections_df["Class"] == "NO HAIRNET").sum()
                        }
                    })
                else:
                    st.warning("No detections above confidence threshold.")

        # ------------------------------------------------------------
        # Multiple Images (batch)
        # ------------------------------------------------------------
        elif app_mode == "Multiple Images":
            st.subheader("Upload Multiple Images")
            uploaded_files = st.file_uploader(
                "Choose one or more images...",
                type=["jpg", "jpeg", "png", "bmp"],
                accept_multiple_files=True,
                key="multi_image_uploader"
            )

            if uploaded_files:
                st.info(f"{len(uploaded_files)} image(s) selected.")

                if st.button("Run Detection on All Images"):
                    results_list = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    for idx, uf in enumerate(uploaded_files):
                        status_text.text(f"Processing {uf.name} ({idx + 1}/{len(uploaded_files)})...")
                        file_bytes = uf.read()
                        res = process_single_image_bytes(
                            file_bytes, uf.name, model, conf_threshold, iou_threshold
                        )
                        results_list.append(res)
                        progress_bar.progress((idx + 1) / len(uploaded_files))

                    progress_bar.empty()
                    status_text.empty()
                    st.success(f"Processed {len(results_list)} image(s).")

                    # Combine all detections into one DataFrame
                    all_dfs = [r["detections_df"] for r in results_list if not r["detections_df"].empty]
                    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

                    st.markdown("### Combined Detection Results")
                    display_metrics(combined_df)
                    display_detection_chart(combined_df)

                    # Show each image result in an expander
                    st.markdown("### Per-Image Results")
                    for res in results_list:
                        with st.expander(f"📷 {res['filename']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Original**")
                                st.image(res["original_rgb"], use_column_width=True)
                            with col2:
                                st.markdown("**Annotated**")
                                st.image(res["annotated_rgb"], use_column_width=True)
                            if not res["detections_df"].empty:
                                st.dataframe(res["detections_df"], use_column_width=True)
                            else:
                                st.warning("No detections above confidence threshold.")

                    # Downloads: ZIP of annotated images + combined CSV
                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        zip_buffer = build_zip_of_images(results_list)
                        st.download_button(
                            label="Download All Annotated Images (ZIP)",
                            data=zip_buffer,
                            file_name=f"annotated_images_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip"
                        )
                    with col_dl2:
                        if not combined_df.empty:
                            csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label="Download Combined Detections (CSV)",
                                data=csv_bytes,
                                file_name=f"detections_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )

                    # Add each image to history
                    for res in results_list:
                        df = res["detections_df"]
                        add_to_history({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": "image (batch)",
                            "filename": res["filename"],
                            "stats": {
                                "total": len(df),
                                "hairnet": (df["Class"] == "HAIRNET").sum() if not df.empty else 0,
                                "no_hairnet": (df["Class"] == "NO HAIRNET").sum() if not df.empty else 0
                            }
                        })

        # ------------------------------------------------------------
        # Single Video
        # ------------------------------------------------------------
        elif app_mode == "Video":
            st.subheader("Upload Video")
            uploaded_file = st.file_uploader("Choose a video...",
                                             type=["mp4", "avi", "mov", "mkv"],
                                             key="video_uploader")

            if uploaded_file is not None:
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_video_path = tmp_file.name

                st.markdown("**Original Video**")
                st.video(uploaded_file)

                if st.button("Process Video"):
                    output_filename = f"annotated_{uploaded_file.name}"
                    output_path = os.path.join(VIDEO_OUTPUT_DIR, output_filename)

                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    def update_progress(progress):
                        progress_bar.progress(progress)
                        status_text.text(f"Processing... {progress*100:.1f}%")

                    with st.spinner("Processing video... This may take a while."):
                        frame_stats = process_video(
                            tmp_video_path, model, output_path,
                            conf_threshold, iou_threshold,
                            progress_callback=update_progress
                        )

                    progress_bar.empty()
                    status_text.empty()
                    st.success("Video processing complete!")

                    # Display annotated video
                    st.markdown("**Annotated Video**")
                    st.video(output_path)

                    # Download button
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Annotated Video",
                            data=f,
                            file_name=output_filename,
                            mime="video/mp4"
                        )

                    # Frame stats
                    if frame_stats:
                        df_frames = pd.DataFrame(frame_stats)
                        st.markdown("### Frame-wise Detection Summary")
                        st.dataframe(df_frames, use_container_width=True)

                        # Plot total detections per frame
                        fig = px.line(df_frames, x="frame", y=["hairnet_count", "no_hairnet_count"],
                                      labels={"value": "Count", "variable": "Class"},
                                      title="Detections per Frame")
                        st.plotly_chart(fig, use_container_width=True)

                        # Add to history
                        total_det = sum(s['total_detections'] for s in frame_stats)
                        hairnet_total = sum(s['hairnet_count'] for s in frame_stats)
                        no_hairnet_total = sum(s['no_hairnet_count'] for s in frame_stats)
                        add_to_history({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "type": "video",
                            "filename": uploaded_file.name,
                            "stats": {
                                "total": total_det,
                                "hairnet": hairnet_total,
                                "no_hairnet": no_hairnet_total,
                                "frames": len(frame_stats)
                            }
                        })
                    else:
                        st.info("No frames processed.")

                    # Clean up temp file
                    os.unlink(tmp_video_path)

        # ------------------------------------------------------------
        # Multiple Videos (batch)
        # ------------------------------------------------------------
        elif app_mode == "Multiple Videos":
            st.subheader("Upload Multiple Videos")
            uploaded_files = st.file_uploader(
                "Choose one or more videos...",
                type=["mp4", "avi", "mov", "mkv"],
                accept_multiple_files=True,
                key="multi_video_uploader"
            )

            if uploaded_files:
                st.info(f"{len(uploaded_files)} video(s) selected.")

                if st.button("Process All Videos"):
                    overall_progress = st.progress(0)
                    overall_status = st.empty()
                    all_frame_stats = []

                    for v_idx, uf in enumerate(uploaded_files):
                        overall_status.text(f"Video {v_idx + 1}/{len(uploaded_files)}: {uf.name}")

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                            tmp_file.write(uf.read())
                            tmp_video_path = tmp_file.name

                        output_filename = f"annotated_{uf.name}"
                        output_path = os.path.join(VIDEO_OUTPUT_DIR, output_filename)

                        video_progress = st.progress(0)

                        def update_progress(progress, bar=video_progress):
                            bar.progress(progress)

                        with st.expander(f"🎬 {uf.name}", expanded=False):
                            with st.spinner(f"Processing {uf.name}..."):
                                frame_stats = process_video(
                                    tmp_video_path, model, output_path,
                                    conf_threshold, iou_threshold,
                                    progress_callback=update_progress
                                )

                            st.video(output_path)
                            with open(output_path, "rb") as f:
                                st.download_button(
                                    label=f"Download Annotated Video - {uf.name}",
                                    data=f,
                                    file_name=output_filename,
                                    mime="video/mp4",
                                    key=f"dl_{v_idx}"
                                )

                            if frame_stats:
                                df_frames = pd.DataFrame(frame_stats)
                                df_frames.insert(0, "source_video", uf.name)
                                st.dataframe(df_frames, use_container_width=True)
                                all_frame_stats.append(df_frames)

                                total_det = sum(s['total_detections'] for s in frame_stats)
                                hairnet_total = sum(s['hairnet_count'] for s in frame_stats)
                                no_hairnet_total = sum(s['no_hairnet_count'] for s in frame_stats)
                                add_to_history({
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "type": "video (batch)",
                                    "filename": uf.name,
                                    "stats": {
                                        "total": total_det,
                                        "hairnet": hairnet_total,
                                        "no_hairnet": no_hairnet_total,
                                        "frames": len(frame_stats)
                                    }
                                })

                        os.unlink(tmp_video_path)
                        overall_progress.progress((v_idx + 1) / len(uploaded_files))

                    overall_status.empty()
                    overall_progress.empty()
                    st.success(f"Processed {len(uploaded_files)} video(s).")

                    if all_frame_stats:
                        combined_frames_df = pd.concat(all_frame_stats, ignore_index=True)
                        csv_bytes = combined_frames_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Combined Frame Stats (CSV)",
                            data=csv_bytes,
                            file_name=f"video_batch_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

        # ------------------------------------------------------------
        # Live Stream via URL (RTSP / HTTP / CCTV / IP camera)
        # ------------------------------------------------------------
        elif app_mode == "Live Stream (URL)":
            st.subheader("Live Stream Detection (RTSP / CCTV / IP Camera URL)")
            st.caption(
                "Enter a stream URL such as `rtsp://user:pass@192.168.1.10:554/stream1`, "
                "an HTTP MJPEG URL, or `0` for a locally connected camera index."
            )

            stream_url = st.text_input(
                "Stream URL",
                value=st.session_state.get("stream_url_value", ""),
                placeholder="rtsp://192.168.1.10:554/stream1"
            )
            st.session_state.stream_url_value = stream_url

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                frame_skip = st.number_input("Process every Nth frame", min_value=1, max_value=30, value=2)
            with col_b:
                max_display_width = st.number_input("Display width (px)", min_value=320, max_value=1920, value=800)
            with col_c:
                record_stream = st.checkbox("Record annotated stream to file", value=False)

            start_col, stop_col = st.columns(2)
            start_clicked = start_col.button("▶ Start Stream", disabled=st.session_state.streaming)
            stop_clicked = stop_col.button("⏹ Stop Stream", disabled=not st.session_state.streaming)

            if start_clicked and stream_url:
                cap_source = int(stream_url) if stream_url.strip().isdigit() else stream_url
                cap = cv2.VideoCapture(cap_source)
                if not cap.isOpened():
                    st.error("Could not open the stream. Check the URL, network access, and credentials.")
                else:
                    st.session_state.stream_cap = cap
                    st.session_state.streaming = True
                    st.session_state.stream_stats = {"total": 0, "hairnet": 0, "no_hairnet": 0, "frames": 0}
                    if record_stream:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                        rec_path = os.path.join(
                            VIDEO_OUTPUT_DIR,
                            f"stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                        )
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        st.session_state.stream_writer = cv2.VideoWriter(rec_path, fourcc, 15, (w, h))
                        st.session_state.stream_record_path = rec_path
                    else:
                        st.session_state.stream_writer = None
                    st.rerun()
            elif start_clicked and not stream_url:
                st.warning("Please enter a stream URL first.")

            if stop_clicked:
                st.session_state.streaming = False
                if st.session_state.stream_cap is not None:
                    st.session_state.stream_cap.release()
                    st.session_state.stream_cap = None
                if st.session_state.get("stream_writer") is not None:
                    st.session_state.stream_writer.release()
                    st.session_state.stream_writer = None
                    st.success(f"Recording saved to {st.session_state.get('stream_record_path', '')}")
                st.rerun()

            if st.session_state.streaming and st.session_state.stream_cap is not None:
                st.markdown('Streaming <span class="stream-live-badge">● LIVE</span>', unsafe_allow_html=True)
                frame_placeholder = st.empty()
                metrics_placeholder = st.empty()

                cap = st.session_state.stream_cap
                ret, frame = cap.read()

                if not ret:
                    st.error("Stream ended or connection lost.")
                    st.session_state.streaming = False
                    cap.release()
                    st.session_state.stream_cap = None
                    if st.session_state.get("stream_writer") is not None:
                        st.session_state.stream_writer.release()
                        st.session_state.stream_writer = None
                else:
                    st.session_state.stream_stats["frames"] += 1
                    do_infer = (st.session_state.stream_stats["frames"] % frame_skip == 0)

                    if do_infer:
                        results = model(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)
                        annotated_frame = annotate_image(frame, results, conf_threshold)
                        detections_df = get_detections_dataframe(results, conf_threshold)

                        st.session_state.stream_stats["total"] += len(detections_df)
                        if not detections_df.empty:
                            st.session_state.stream_stats["hairnet"] += int((detections_df["Class"] == "HAIRNET").sum())
                            st.session_state.stream_stats["no_hairnet"] += int((detections_df["Class"] == "NO HAIRNET").sum())
                    else:
                        annotated_frame = frame

                    if st.session_state.get("stream_writer") is not None:
                        st.session_state.stream_writer.write(annotated_frame)

                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    h, w = annotated_rgb.shape[:2]
                    if w > max_display_width:
                        scale = max_display_width / w
                        annotated_rgb = cv2.resize(annotated_rgb, (int(w * scale), int(h * scale)))

                    frame_placeholder.image(annotated_rgb, use_container_width=True)

                    stats = st.session_state.stream_stats
                    with metrics_placeholder.container():
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Frames Read", stats["frames"])
                        c2.metric("Total Detections", stats["total"])
                        c3.metric("Hairnet", stats["hairnet"])
                        c4.metric("No Hairnet", stats["no_hairnet"])

                    time.sleep(0.03)
                    st.rerun()
            else:
                st.info("Enter a stream URL and click **Start Stream** to begin live detection.")
                if st.session_state.stream_stats["frames"] > 0:
                    st.markdown("### Last Session Summary")
                    stats = st.session_state.stream_stats
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Frames Read", stats["frames"])
                    c2.metric("Total Detections", stats["total"])
                    c3.metric("Hairnet", stats["hairnet"])
                    c4.metric("No Hairnet", stats["no_hairnet"])
                    add_to_history({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "type": "live stream",
                        "filename": stream_url or "stream",
                        "stats": {
                            "total": stats["total"],
                            "hairnet": stats["hairnet"],
                            "no_hairnet": stats["no_hairnet"],
                            "frames": stats["frames"]
                        }
                    })
                    st.session_state.stream_stats = {"total": 0, "hairnet": 0, "no_hairnet": 0, "frames": 0}

        elif app_mode == "Webcam":
            st.subheader("Live Webcam Detection")
            st.warning("Allow camera access when prompted. Detection runs in real‑time.")
            
           

            # Define a video processor that runs YOLO on each frame
            class HairnetVideoProcessor(VideoProcessorBase):
                def __init__(self, model, conf_threshold, iou_threshold):
                    self.model = model
                    self.conf_threshold = conf_threshold
                    self.iou_threshold = iou_threshold

                def recv(self, frame):
                    # Convert frame to numpy array (BGR)
                    img = frame.to_ndarray(format="bgr24")
                    # Run inference
                    results = self.model(img, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                    # Annotate
                    annotated = annotate_image(img, results, self.conf_threshold)
                    # Return annotated frame
                    return frame.from_ndarray(annotated, format="bgr24")

            # RTC configuration (TURN/STUN is optional for local webcam; we use default)
            rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

            # Start the webcam stream
            webrtc_ctx = webrtc_streamer(
                key="webcam-detection",
                video_processor_factory=lambda: HairnetVideoProcessor(model, conf_threshold, iou_threshold),
                rtc_configuration=rtc_config,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

            # Optional: show detection stats from the last processed frame
            if webrtc_ctx.state.playing:
                st.info("🎥 Streaming live – detection applied to every frame.")
            else:
                st.info("Click 'Start' to begin webcam detection.")
    # ------------------------------
    # Tab 2: Analytics Dashboard
    # ------------------------------
    with tab2:
        st.header("Analytics Dashboard")
        if not st.session_state.history:
            st.info("No detection history yet. Process images or videos to see analytics.")
        else:
            # Convert history to DataFrame
            history_df = pd.DataFrame(st.session_state.history)
            history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
            history_df['total'] = history_df['stats'].apply(lambda x: x['total'])
            history_df['hairnet'] = history_df['stats'].apply(lambda x: x['hairnet'])
            history_df['no_hairnet'] = history_df['stats'].apply(lambda x: x['no_hairnet'])

            st.markdown("### Detection History")
            st.dataframe(history_df[['timestamp', 'type', 'filename', 'total', 'hairnet', 'no_hairnet']],
                         use_container_width=True)

            # Aggregate stats
            total_all = history_df['total'].sum()
            hairnet_all = history_df['hairnet'].sum()
            no_hairnet_all = history_df['no_hairnet'].sum()
            compliance_rate = (hairnet_all / total_all * 100) if total_all > 0 else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Detections", total_all)
            col2.metric("Wearing Hairnet", hairnet_all)
            col3.metric("Not Wearing", no_hairnet_all)
            col4.metric("Compliance Rate", f"{compliance_rate:.1f}%")

            # Pie chart
            if total_all > 0:
                fig_pie = go.Figure(data=[go.Pie(labels=['HAIRNET', 'NO HAIRNET'],
                                                 values=[hairnet_all, no_hairnet_all],
                                                 marker_colors=['green', 'red'])])
                fig_pie.update_layout(title="Overall Detection Distribution")
                st.plotly_chart(fig_pie, use_container_width=True)

            # Timeline of detections per file
            fig_timeline = px.bar(history_df, x='timestamp', y=['hairnet', 'no_hairnet'],
                                  barmode='stack',
                                  labels={'value': 'Count', 'variable': 'Class'},
                                  title="Detections Over Time")
            st.plotly_chart(fig_timeline, use_container_width=True)

            # Clear history button
            if st.button("Clear History"):
                st.session_state.history = []
                st.rerun()

    # ------------------------------
    # Tab 3: About
    # ------------------------------
    with tab3:
        st.header("About This Application")
        st.markdown("""
        ### Hairnet Detection System
        This application uses a YOLO (You Only Look Once) object detection model to
        identify whether workers are wearing hairnets in images, videos, live streams,
        or a live webcam feed.

        **Features:**
        - Real-time detection with adjustable confidence and IoU thresholds
        - Single **and batch (multiple)** image detection
        - Single **and batch (multiple)** video processing
        - **Live stream detection** from an RTSP / HTTP / CCTV / IP camera URL
        - Annotated output with bounding boxes and class labels
        - Detection analytics and compliance metrics
        - Export results to CSV, and ZIP export for batch images
        - Historical tracking of processed files and streams

        **Model Information:**
        - Architecture: YOLOv8 (custom trained)
        - Classes: HAIRNET, NO HAIRNET
        - Input resolution: 640x640

        **Usage:**
        1. Select input source (Image, Multiple Images, Video, Multiple Videos, Live Stream, Webcam)
        2. Adjust confidence/IoU thresholds if needed
        3. Upload file(s), or enter a stream URL and click **Start Stream**
        4. View annotated results and download if desired
        5. Check Analytics Dashboard for aggregate statistics

        **Live Stream Notes:**
        - Works with RTSP URLs (`rtsp://...`), HTTP MJPEG URLs, or a local camera index (`0`, `1`, ...)
        - Use "Process every Nth frame" to trade off detection frequency vs. performance on CPU
        - Optionally record the annotated stream to an MP4 file in `outputs/videos`

        **Note:** For best performance, use a GPU (CUDA) if available.
        """)

if __name__ == "__main__":
    main()
