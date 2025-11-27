import os
import tempfile

import numpy as np
import joblib
import streamlit as st
from PIL import Image
import cv2

# ============ CONFIG ============
MODEL_FILENAME = "traffic_model.pkl"
IMAGE_SIZE = (224, 224)          # change if your model expects different size
NUM_FRAMES_TO_SAMPLE = 8         # video frames to sample for averaging
VIDEO_EXTENSIONS = ["mp4", "avi", "mov", "mkv", "webm"]
IMAGE_EXTENSIONS = ["png", "jpg", "jpeg", "bmp"]


# ============ MODEL LOADING ============
@st.cache_resource
def load_model():
    """Load the pickled model from local file."""
    model_path = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{MODEL_FILENAME}' not found in repo.")
    model = joblib.load(model_path)
    return model


# ============ PREPROCESSING HELPERS ============
def preprocess_image_for_model(pil_image):
    """
    Convert a PIL image into a numpy array acceptable to the model.
    Default: RGB -> resize -> scale to [0,1] -> flatten.
    CHANGE this if your training preprocessing was different.
    """
    img = pil_image.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    flattened = arr.flatten().reshape(1, -1)       # (1, H*W*3)
    return flattened


def preprocess_numeric_features(feature_list):
    """
    Convert numeric list to 2D array: shape (1, n_features).
    Make sure number & order of features match training.
    """
    arr = np.asarray(feature_list, dtype=np.float32).reshape(1, -1)
    return arr


def extract_frames_from_video_file(file_bytes, num_frames=NUM_FRAMES_TO_SAMPLE):
    """
    Save uploaded video bytes to temp file, then use OpenCV to
    sample 'num_frames' frames across the video.
    Returns list of PIL.Image objects.
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        cap.release()
        os.remove(tmp_path)
        raise RuntimeError("Failed to open video file.")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    if total_frames <= 0:
        # fallback: read sequentially
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        total_frames = len(frames)
        if total_frames == 0:
            cap.release()
            os.remove(tmp_path)
            raise RuntimeError("No frames found in video.")
        indices = np.linspace(0, total_frames - 1, min(num_frames, total_frames), dtype=int)
        sampled = [frames[i] for i in indices]
    else:
        num_to_sample = min(num_frames, total_frames)
        indices = np.linspace(0, total_frames - 1, num_to_sample, dtype=int)
        sampled = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            sampled.append(frame)

    cap.release()
    os.remove(tmp_path)

    pil_images = []
    for f in sampled:
        # BGR -> RGB
        if len(f.shape) == 2:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
        else:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(f_rgb))

    return pil_images


# ============ PREDICTION HELPERS ============
def predict_single(model, X):
    """
    Single-input prediction wrapper.
    Returns dict with 'prediction' and optional 'probabilities'.
    """
    out = {}
    preds = model.predict(X)
    try:
        out["prediction"] = preds.tolist()
    except Exception:
        out["prediction"] = preds

    if hasattr(model, "predict_proba"):
        try:
            out["probabilities"] = model.predict_proba(X).tolist()
        except Exception:
            out["probabilities"] = None

    return out


def predict_on_frames(model, pil_images):
    """
    Predict on multiple frames and combine:
      - If predict_proba exists: average probabilities, final class = argmax(mean)
      - Else: try numeric average, else majority vote.
    Returns dict with combined result and per-frame details.
    """
    per_frame_preds = []
    per_frame_probs = []

    for img in pil_images:
        X = preprocess_image_for_model(img)

        # Prefer predict_proba if available
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)      # (1, n_classes)
            per_frame_probs.append(probs[0])
            pred = int(np.argmax(probs, axis=1)[0])
            per_frame_preds.append(pred)
        else:
            pred = model.predict(X)
            try:
                val = pred.tolist()
            except Exception:
                val = pred
            per_frame_preds.append(val)

    result = {}

    if len(per_frame_probs) > 0:
        avg_probs = np.mean(np.vstack(per_frame_probs), axis=0)
        final_class = int(np.argmax(avg_probs))
        result["method"] = "average_probabilities"
        result["final_class"] = final_class
        result["average_probabilities"] = avg_probs.tolist()
    else:
        # Try numeric average for regressors
        try:
            numeric_preds = np.array([np.squeeze(p) for p in per_frame_preds], dtype=float)
            final_value = float(np.mean(numeric_preds))
            result["method"] = "average_regression"
            result["final_value"] = final_value
        except Exception:
            # Fallback: majority vote on discrete labels
            try:
                flat_preds = [
                    p[0] if isinstance(p, (list, tuple, np.ndarray)) else p
                    for p in per_frame_preds
                ]
                values, counts = np.unique(flat_preds, return_counts=True)
                final_class = values[np.argmax(counts)]
                result["method"] = "majority_vote"
                result["final_class"] = final_class
            except Exception as e:
                result["method"] = "failed_combination"
                result["error"] = str(e)

    # Attach per-frame prediction data
    def to_plain(x):
        return x.tolist() if hasattr(x, "tolist") else x

    result["per_frame_predictions"] = [to_plain(p) for p in per_frame_preds]

    if len(per_frame_probs) > 0:
        result["per_frame_probabilities"] = [p.tolist() for p in per_frame_probs]

    return result


# ============ STREAMLIT UI ============
st.set_page_config(page_title="Traffic Detection – Video & Image", layout="centered")
st.title("Traffic Detection – Video / Image / Numeric Demo")

st.sidebar.header("Configuration")
st.sidebar.write("This app loads a `.pkl` model and runs inference on:")
st.sidebar.markdown("- 📹 Uploaded **video** (multi-frame average)  \n- 🖼️ Single **image**  \n- 🔢 **Numeric features**")

try:
    model = load_model()
    st.sidebar.success("Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    st.stop()

mode = st.radio(
    "Choose input type:",
    ("Video (multi-frame)", "Single image", "Numeric features"),
)

st.markdown("---")

# -------- VIDEO MODE --------
if mode == "Video (multi-frame)":
    st.header("Upload a traffic video")
    video_file = st.file_uploader(
        "Upload a video file",
        type=VIDEO_EXTENSIONS,
        key="video_uploader",
    )

    num_frames = st.slider(
        "Number of frames to sample from video",
        min_value=2,
        max_value=24,
        value=NUM_FRAMES_TO_SAMPLE,
        step=2,
    )

    if video_file is not None:
        st.video(video_file)
        if st.button("Run prediction on video"):
            with st.spinner("Extracting frames and running predictions..."):
                try:
                    frames = extract_frames_from_video_file(
                        video_file.read(),
                        num_frames=num_frames,
                    )
                    st.write(f"Sampled frames: {len(frames)}")
                    result = predict_on_frames(model, frames)
                except Exception as e:
                    st.error(f"Video processing / prediction failed: {e}")
                else:
                    st.success("Prediction complete.")
                    # Show summarized result
                    if "final_class" in result:
                        st.subheader("Final predicted class")
                        st.write(result["final_class"])
                    if "final_value" in result:
                        st.subheader("Final predicted value (regression)")
                        st.write(result["final_value"])

                    st.subheader("Raw model output")
                    st.json(result)

# -------- IMAGE MODE --------
elif mode == "Single image":
    st.header("Upload a traffic image")
    img_file = st.file_uploader(
        "Upload an image file",
        type=IMAGE_EXTENSIONS,
        key="image_uploader",
    )

    if img_file is not None:
        pil_img = Image.open(img_file)
        st.image(pil_img, caption="Uploaded image", use_container_width=True)

        if st.button("Run prediction on image"):
            with st.spinner("Running prediction..."):
                try:
                    X = preprocess_image_for_model(pil_img)
                    result = predict_single(model, X)
                except Exception as e:
                    st.error(f"Image processing / prediction failed: {e}")
                else:
                    st.success("Prediction complete.")
                    st.subheader("Raw model output")
                    st.json(result)

# -------- NUMERIC MODE --------
else:
    st.header("Enter numeric features")
    st.caption("Adjust number/meaning/order of features to match your training.")

    # Example: 3 features — change labels & count as per your model
    f1 = st.number_input("Feature 1", value=0.0)
    f2 = st.number_input("Feature 2", value=0.0)
    f3 = st.number_input("Feature 3", value=0.0)

    features = [f1, f2, f3]
    st.write("Feature vector:", features)

    if st.button("Run prediction on numeric features"):
        with st.spinner("Running prediction..."):
            try:
                Xn = preprocess_numeric_features(features)
                result = predict_single(model, Xn)
            except Exception as e:
                st.error(f"Numeric prediction failed: {e}")
            else:
                st.success("Prediction complete.")
                st.subheader("Raw model output")
                st.json(result)

st.markdown("---")
st.caption(
    "Note: Make sure **traffic_model.pkl** is in the same folder as `app.py`, "
    "and adjust `IMAGE_SIZE` & preprocessing if your training used different settings."
)
