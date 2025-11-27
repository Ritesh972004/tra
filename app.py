import os
import joblib
import numpy as np
from PIL import Image

# Default image size used for preprocessing. Change to match training.
IMAGE_SIZE = (224, 224)

def load_model(model_path="traffic_model.pkl"):
    """
    Load a pickled model (joblib/pickle). Raises FileNotFoundError if missing.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    model = joblib.load(model_path)
    return model

def preprocess_pil_image(pil_image, image_size=IMAGE_SIZE):
    """
    Convert a PIL.Image to the model input expected format.
    Default pipeline: RGB -> resize -> scale to [0,1] -> flatten -> shape (1, N)
    Edit this function to exactly match your training preprocessing if needed.
    """
    img = pil_image.convert("RGB")
    img = img.resize(image_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    X = arr.flatten().reshape(1, -1)
    return X

def predict_on_frames(model, pil_images):
    """
    Given a model and a list of PIL images, run per-frame prediction and combine results.
    - If model supports predict_proba(): average probabilities across frames and choose argmax.
    - Else try numeric averaging (regression).
    - Else majority vote.
    Returns a dict with combined result and per-frame details.
    """
    per_frame_preds = []
    per_frame_probs = []

    for pil in pil_images:
        X = preprocess_pil_image(pil)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)  # shape (1, n_classes)
            per_frame_probs.append(probs[0])
            pred_class = int(np.argmax(probs, axis=1)[0])
            per_frame_preds.append(pred_class)
        else:
            pred = model.predict(X)
            # convert to python scalar/list-friendly object
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
        # try numeric average (regression)
        try:
            numeric_preds = np.array([np.squeeze(p) for p in per_frame_preds], dtype=float)
            final_value = float(np.mean(numeric_preds))
            result["method"] = "average_regression"
            result["final_value"] = final_value
        except Exception:
            # fallback to majority vote
            try:
                flat_preds = [p[0] if isinstance(p, (list, tuple, np.ndarray)) else p for p in per_frame_preds]
                vals, counts = np.unique(flat_preds, return_counts=True)
                final_class = vals[np.argmax(counts)]
                result["method"] = "majority_vote"
                result["final_class"] = final_class
            except Exception as e:
                result["method"] = "failed_combination"
                result["error"] = str(e)

    # attach per-frame details (JSON-friendly)
    result["per_frame_predictions"] = [ (p.tolist() if hasattr(p, "tolist") else p) for p in per_frame_preds ]
    if len(per_frame_probs) > 0:
        result["per_frame_probabilities"] = [ p.tolist() for p in per_frame_probs ]

    return result
