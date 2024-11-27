import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Carica le immagini di occhiali e sigaro
specs_ori = cv2.imread('glass.png', -1)
cigar_ori = cv2.imread('cigar.png', -1)

def transparentOverlay(src, overlay, pos=(0, 0)):
    """Applica un overlay trasparente (con canale alpha) su un'immagine."""
    h, w, _ = overlay.shape
    rows, cols, _ = src.shape
    y, x = pos[0], pos[1]

    for i in range(h):
        for j in range(w):
            if x + i >= rows or y + j >= cols:
                continue
            alpha = float(overlay[i][j][3] / 255.0)  # Usa il canale alpha
            src[x + i][y + j] = alpha * overlay[i][j][:3] + (1 - alpha) * src[x + i][y + j]
    return src

def detect_faces(frame):
    """Rileva volti e aggiunge occhiali e sigaro."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=5, minSize=(120, 120), maxSize=(350, 350))

    for (x, y, w, h) in faces:
        if h > 0 and w > 0:
            # Coordinate e dimensioni degli occhiali (esatte come nel tuo codice)
            glass_symin = int(y + 1.5 * h / 5)
            glass_symax = int(y + 2.5 * h / 5)
            sh_glass = glass_symax - glass_symin
            face_glass_roi_color = frame[glass_symin:glass_symax, x:x + w]
            
            # Dimensioni e coordinate sigaro
            cigar_symin = int(y + 4 * h / 6)
            cigar_symax = int(y + 5.5 * h / 6)
            sh_cigar = cigar_symax - cigar_symin
            face_cigar_roi_color = frame[cigar_symin:cigar_symax, x:x + w]

            # Ridimensionamento degli overlay
            specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_CUBIC)
            cigar = cv2.resize(cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_CUBIC)

            # Posizionamento overlay
            transparentOverlay(face_glass_roi_color, specs)
            transparentOverlay(face_cigar_roi_color, cigar, pos=(int(w / 2), int(sh_cigar / 2)))
    
    return frame

def main():
    """Funzione principale Streamlit."""
    st.title("Thug Life Generator")
    image = Image.open('thug.jpg')
    st.image(image, width=420)

    if st.button('Start'):
        audio_file = open("snoop.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", loop=True, autoplay=True)

        FRAME_WINDOW = st.image([])

        # Inizializzazione webcam
        video_capture = cv2.VideoCapture(0)

        while True:
            ret, frame = video_capture.read()
            if not ret:
                st.error("Errore durante la cattura del frame.")
                break
            frame = cv2.flip(frame, 1)
            frame_with_faces = detect_faces(frame)
            FRAME_WINDOW.image(frame_with_faces, channels="BGR")
        
        video_capture.release()
        
    else:
        st.success("Clicca su 'Start' per iniziare il Thug Life!")

if __name__ == "__main__":
    main()
