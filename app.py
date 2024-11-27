import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase
from PIL import Image
import logging
import threading
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache resources using streamlit's caching
@st.cache_resource
def load_resources():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        specs = cv2.imread('glass.png', -1)
        cigar = cv2.imread('cigar.png', -1)
        if specs is None or cigar is None:
            raise ValueError("Failed to load overlay images")
        return face_cascade, specs, cigar
    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        raise

class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.face_cascade = None
        self.specs_ori = None
        self.cigar_ori = None
        self.prev_faces = None
        self.skip_frames = 0
        self.max_skip_frames = 2
        self.init_resources()

    def init_resources(self):
        try:
            self.face_cascade, self.specs_ori, self.cigar_ori = load_resources()
        except Exception as e:
            logger.error(f"Failed to initialize resources: {str(e)}")
            st.error("Failed to initialize video processing resources")
            raise

    def transparentOverlay(self, src: np.ndarray, overlay: np.ndarray, pos=(0, 0), scale=1) -> np.ndarray:
        try:
            overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            h, w, _ = overlay.shape
            rows, cols, _ = src.shape
            y, x = pos[0], pos[1]

            if x + h > rows or y + w > cols:
                return src

            alpha = overlay[:, :, 3] / 255.0
            alpha = np.expand_dims(alpha, axis=-1)
            overlay_region = src[x:x+h, y:y+w]
            overlay_colors = overlay[:, :, :3]

            src[x:x+h, y:y+w] = (alpha * overlay_colors + (1 - alpha) * overlay_region).astype(np.uint8)
            return src
        except Exception as e:
            logger.error(f"Error in transparentOverlay: {str(e)}")
            return src

    def process_faces(self, frame: np.ndarray) -> np.ndarray:
        try:
            with self._lock:
                self.skip_frames = (self.skip_frames + 1) % (self.max_skip_frames + 1)

                if self.skip_frames != 0 and self.prev_faces is not None:
                    faces = self.prev_faces
                else:
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

                    # Calculate regions
                    glass_symin = int(y + 1.5 * h / 5)
                    glass_symax = int(y + 2.5 * h / 5)
                    sh_glass = glass_symax - glass_symin

                    cigar_symin = int(y + 4 * h / 6)
                    cigar_symax = int(y + 5.5 * h / 6)
                    sh_cigar = cigar_symax - cigar_symin

                    # Safe region checks
                    if (glass_symin < 0 or glass_symax > frame.shape[0] or
                        cigar_symin < 0 or cigar_symax > frame.shape[0] or
                        x < 0 or x + w > frame.shape[1]):
                        continue

                    # Resize and apply overlays
                    specs = cv2.resize(self.specs_ori, (w, sh_glass), interpolation=cv2.INTER_LINEAR)
                    cigar = cv2.resize(self.cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_LINEAR)

                    frame[glass_symin:glass_symax, x:x+w] = self.transparentOverlay(
                        frame[glass_symin:glass_symax, x:x+w].copy(),
                        specs
                    )
                    frame[cigar_symin:cigar_symax, x:x+w] = self.transparentOverlay(
                        frame[cigar_symin:cigar_symax, x:x+w].copy(),
                        cigar,
                        (int(w/2), int(sh_cigar/2))
                    )

                return frame
        except Exception as e:
            logger.error(f"Error in process_faces: {str(e)}")
            return frame

    def recv(self, frame: av.VideoFrame) -> Union[av.VideoFrame, None]:
        try:
            img = frame.to_ndarray(format="bgr24")
            img = cv2.flip(img, 1)
            img = self.process_faces(img)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logger.error(f"Error in recv: {str(e)}")
            return frame

def main():
    st.title("Thug Life Generator")
    
    try:
        image = Image.open('thug.jpg')
        st.image(image, width=420)

        with open("snoop.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", loop=True, autoplay=True)

        # WebRTC configuration with multiple STUN servers for redundancy
        rtc_configuration = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:stun1.l.google.com:19302"]},
                {"urls": ["stun:stun2.l.google.com:19302"]},
                {"urls": ["stun:stun3.l.google.com:19302"]},
                {"urls": ["stun:stun4.l.google.com:19302"]}
            ],
            "bundlePolicy": "max-bundle",
            "iceCandidatePoolSize": 10,
            "iceTransportPolicy": "all"
        }

        webrtc_ctx = webrtc_streamer(
            key="thug-life-filter",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 30}
                },
                "audio": False
            },
            async_processing=True
        )

        st.markdown("""
            - Thug Life Generator by Daniele Grotti, follow me on [LinkedIn](https://www.linkedin.com/in/daniele-grotti/)
        """)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        st.error("An error occurred while starting the application. Please refresh the page and try again.")

if __name__ == "__main__":
    main()