import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from PIL import Image

# Cache resources using streamlit's caching
@st.cache_resource
def load_resources():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    specs = cv2.imread('glass.png', -1)
    cigar = cv2.imread('cigar.png', -1)
    return face_cascade, specs, cigar

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    """Optimized transparent overlay using vectorized operations"""
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    h, w, _ = overlay.shape
    rows, cols, _ = src.shape
    y, x = pos[0], pos[1]
    
    # Early return if overlay is out of bounds
    if x + h > rows or y + w > cols:
        return src
    
    # Vectorized alpha blending
    alpha = overlay[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=-1)
    overlay_region = src[x:x+h, y:y+w]
    overlay_colors = overlay[:, :, :3]
    
    src[x:x+h, y:y+w] = (alpha * overlay_colors + (1 - alpha) * overlay_region).astype(np.uint8)
    return src

# Class to maintain state and cache intermediate results
class FaceProcessor:
    def __init__(self):
        self.face_cascade, self.specs_ori, self.cigar_ori = load_resources()
        self.prev_faces = None
        self.skip_frames = 0
        self.max_skip_frames = 2  # Process every 3rd frame
        
    def detect_faces(self, frame):
        self.skip_frames = (self.skip_frames + 1) % (self.max_skip_frames + 1)
        
        # Use previous face locations for skipped frames
        if self.skip_frames != 0 and self.prev_faces is not None:
            faces = self.prev_faces
        else:
            # Convert to grayscale for faster detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.2, 
                minNeighbors=5, 
                minSize=(120, 120), 
                maxSize=(350, 350),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            self.prev_faces = faces

        for (x, y, w, h) in faces:
            if h <= 0 or w <= 0:
                continue
                
            # Calculate regions once
            glass_symin = int(y + 1.5 * h / 5)
            glass_symax = int(y + 2.5 * h / 5)
            sh_glass = glass_symax - glass_symin

            cigar_symin = int(y + 4 * h / 6)
            cigar_symax = int(y + 5.5 * h / 6)
            sh_cigar = cigar_symax - cigar_symin

            # Resize accessories once per face
            specs = cv2.resize(self.specs_ori, (w, sh_glass), interpolation=cv2.INTER_LINEAR)
            cigar = cv2.resize(self.cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_LINEAR)

            # Apply overlays
            frame[glass_symin:glass_symax, x:x+w] = transparentOverlay(
                frame[glass_symin:glass_symax, x:x+w], 
                specs
            )
            frame[cigar_symin:cigar_symax, x:x+w] = transparentOverlay(
                frame[cigar_symin:cigar_symax, x:x+w],
                cigar,
                (int(w/2), int(sh_cigar/2))
            )
        
        return frame

# Initialize processor with state
face_processor = FaceProcessor()

# Optimized callback function
def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    img = face_processor.detect_faces(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Thug Life Generator")
    
    # Load and display image
    image = Image.open('thug.jpg')
    st.image(image, width=420)

    # Load audio once
    with open("snoop.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3", loop=True, autoplay=True)

    # Configure WebRTC with optimized settings
    webrtc_ctx = webrtc_streamer(
        key="opencv-filter",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 640},
                "height": {"ideal": 480},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
            "bundlePolicy": "max-bundle",
            "iceCandidatePoolSize": 1
        }
    )

    st.markdown("""
        - Thug Life Generator by Daniele Grotti, follow me on [LinkedIn](https://www.linkedin.com/in/daniele-grotti/)
    """)

if __name__ == "__main__":
    main()