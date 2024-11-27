import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer, VideoProcessorBase
from PIL import Image
import logging
import threading
from typing import Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_resources():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    specs = cv2.imread('glass.png', -1)
    cigar = cv2.imread('cigar.png', -1)
    return face_cascade, specs, cigar

class VideoProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.face_cascade = None
        self.specs_ori = None
        self.cigar_ori = None
        self.load_resources()

    def load_resources(self):
        self.face_cascade, self.specs_ori, self.cigar_ori = load_resources()

    def process(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(120, 120),
            maxSize=(350, 350)
        )

        for (x, y, w, h) in faces:
            # Calculate regions
            glass_symin = int(y + 1.5 * h / 5)
            glass_symax = int(y + 2.5 * h / 5)
            sh_glass = glass_symax - glass_symin

            cigar_symin = int(y + 4 * h / 6)
            cigar_symax = int(y + 5.5 * h / 6)
            sh_cigar = cigar_symax - cigar_symin

            # Check boundaries
            if (glass_symin >= 0 and glass_symax < img.shape[0] and 
                cigar_symin >= 0 and cigar_symax < img.shape[0] and 
                x >= 0 and x + w <= img.shape[1]):
                
                # Resize accessories
                specs = cv2.resize(self.specs_ori, (w, sh_glass))
                cigar = cv2.resize(self.cigar_ori, (w, sh_cigar))

                # Apply overlays with alpha blending
                for i in range(sh_glass):
                    for j in range(w):
                        if glass_symin + i < img.shape[0] and x + j < img.shape[1]:
                            alpha = float(specs[i][j][3] / 255.0)
                            img[glass_symin + i][x + j] = alpha * specs[i][j][:3] + (1 - alpha) * img[glass_symin + i][x + j]

                for i in range(sh_cigar):
                    for j in range(w):
                        if cigar_symin + i < img.shape[0] and x + j < img.shape[1]:
                            alpha = float(cigar[i][j][3] / 255.0)
                            img[cigar_symin + i][x + j] = alpha * cigar[i][j][:3] + (1 - alpha) * img[cigar_symin + i][x + j]

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("Thug Life Generator")
    
    try:
        image = Image.open('thug.jpg')
        st.image(image, width=420)

        with open("snoop.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", loop=True)

        # WebRTC configuration with TURN servers
        rtc_configuration = {
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {
                    "urls": ["turn:numb.viagenie.ca"],
                    "username": "webrtc@live.com",
                    "credential": "muazkh"
                }
            ],
            "iceTransportPolicy": "all",
            "bundlePolicy": "max-bundle",
            "rtcpMuxPolicy": "require",
            "iceCandidatePoolSize": 1
        }

        ctx = webrtc_streamer(
            key="thug-life",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_configuration,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={
                "video": {"width": 640, "height": 480, "frameRate": 30},
                "audio": False
            },
            async_processing=True
        )

        st.markdown("""
            - Thug Life Generator by Daniele Grotti, follow me on [LinkedIn](https://www.linkedin.com/in/daniele-grotti/)
        """)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        st.error("An error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()