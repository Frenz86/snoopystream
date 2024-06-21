import av
import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from PIL import Image

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
specs_ori = cv2.imread('glass.png', -1)
cigar_ori = cv2.imread('cigar.png', -1)

# Function to apply transparent overlay
def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape  # Size of foreground
    rows, cols, _ = src.shape  # Size of background Image
    y, x = pos[0], pos[1]  # Position of foreground/overlay image
    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # read the alpha channel
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

# Function to detect faces and apply overlays
def detect_faces(frame):
    faces = face_cascade.detectMultiScale(frame, 1.2, 5, 0, (120, 120), (350, 350))

    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            glass_symin = int(y + 1.5 * h / 5)
            glass_symax = int(y + 2.5 * h / 5)
            sh_glass = glass_symax - glass_symin

            cigar_symin = int(y + 4 * h / 6)
            cigar_symax = int(y + 5.5 * h / 6)
            sh_cigar = cigar_symax - cigar_symin

            face_glass_roi_color = frame[glass_symin:glass_symax, x:x + w]
            face_cigar_roi_color = frame[cigar_symin:cigar_symax, x:x + w]

            specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
            cigar = cv2.resize(cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_CUBIC)

            transparentOverlay(face_glass_roi_color, specs)
            transparentOverlay(face_cigar_roi_color, cigar, (int(w / 2), int(sh_cigar / 2)))
    return frame

def callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)
    img = detect_faces(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    image = Image.open('thug.jpg')
    st.image(image, width=420)

    audio_file = open("snoop.mp3", "rb")
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format="audio/mp3", loop=True, autoplay=True)

    webrtc_streamer(
                    key="opencv-filter",
                    mode=WebRtcMode.SENDRECV,
                    video_frame_callback=callback,
                    media_stream_constraints={"video": True, "audio": False},
                    async_processing=True,
                    )

    st.markdown("""
        - Thug Life Generator by Daniele Grotti, follow me on [LinkedIn](https://www.linkedin.com/in/daniele-grotti/)

        """)
if __name__ == "__main__":
    main()