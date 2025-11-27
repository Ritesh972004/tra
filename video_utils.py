# video_utils.py
import tempfile
import imageio.v3 as iio
import numpy as np
from PIL import Image
import os

NUM_FRAMES_TO_SAMPLE = 8
IMAGE_SIZE = (224, 224)  # change if needed

def extract_frames_from_video_bytes_imageio(video_bytes, num_frames=NUM_FRAMES_TO_SAMPLE):
    """
    Save video bytes to a temp file, read frames with imageio (pyav/ffmpeg),
    sample `num_frames` evenly across the video, return list of PIL.Image RGB.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        tmp.write(video_bytes)
        tmp.flush()
        tmp.close()

        # try opening with pyav backend first; fallback otherwise
        try:
            reader = iio.imopen(tmp.name, plugin="pyav")
        except Exception:
            reader = iio.imopen(tmp.name)

        frames = []
        # read all frames into memory (usually OK for short videos)
        for frame in reader:
            frames.append(np.asarray(frame))
        reader.close()

        total = len(frames)
        if total == 0:
            raise RuntimeError("No frames could be read from the uploaded video.")

        k = min(num_frames, total)
        indices = np.linspace(0, total - 1, k, dtype=int)

        sampled = []
        for idx in indices:
            arr = frames[int(idx)]
            # normalize channels: grayscale -> RGB, RGBA -> RGB
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[..., :3]
            # convert to uint8 just in case and to PIL
            pil = Image.fromarray(arr.astype("uint8"), mode="RGB")
            pil = pil.resize(IMAGE_SIZE)
            sampled.append(pil)

        return sampled
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass
