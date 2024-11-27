import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Cache del classificatore e delle immagini per evitare ricaricamenti
@st.cache_resource
def load_resources():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    specs = cv2.imread('glass.png', -1)
    cigar = cv2.imread('cigar.png', -1)
    return face_cascade, specs, cigar

def transparentOverlay(src, overlay, pos=(0, 0), scale=1):
    # Utilizzo di numpy per operazioni vettorizzate invece del ciclo for
    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w, _ = overlay.shape
    rows, cols, _ = src.shape
    y, x = pos[0], pos[1]
    
    # Controlla i limiti dell'immagine
    if x + h > rows or y + w > cols:
        return src
        
    # Estrai il canale alpha e crea la maschera
    alpha = overlay[:, :, 3] / 255.0
    alpha = np.expand_dims(alpha, axis=-1)
    
    # Calcola la regione di sovrapposizione
    overlay_region = src[x:x+h, y:y+w]
    overlay_colors = overlay[:, :, :3]
    
    # Applica la sovrapposizione usando operazioni vettorizzate
    src[x:x+h, y:y+w] = (alpha * overlay_colors + (1 - alpha) * overlay_region).astype(np.uint8)
    
    return src

def detect_faces(frame, face_cascade, specs_ori, cigar_ori):
    # Converti in scala di grigi per una rilevazione più veloce
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5, 
        minSize=(120, 120), 
        maxSize=(350, 350)
    )

    for (x, y, w, h) in faces:
        # Calcola le dimensioni una sola volta
        glass_symin = int(y + 1.5 * h / 5)
        glass_symax = int(y + 2.5 * h / 5)
        sh_glass = glass_symax - glass_symin

        cigar_symin = int(y + 4 * h / 6)
        cigar_symax = int(y + 5.5 * h / 6)
        sh_cigar = cigar_symax - cigar_symin

        # Ridimensiona gli accessori una sola volta per faccia
        specs = cv2.resize(specs_ori, (w, sh_glass), interpolation=cv2.INTER_LINEAR)
        cigar = cv2.resize(cigar_ori, (w, sh_cigar), interpolation=cv2.INTER_LINEAR)

        # Applica gli overlay
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

def main():
    st.title("Thug Life Generator")
    
    # Carica le risorse una sola volta
    face_cascade, specs_ori, cigar_ori = load_resources()
    
    image = Image.open('thug.jpg')
    st.image(image, width=420)

    if st.button('Start'):
        # Carica l'audio una sola volta
        audio_file = open("snoop.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", loop=True, autoplay=True)

        FRAME_WINDOW = st.image([])
        video_capture = cv2.VideoCapture(0)
        
        # Imposta parametri ottimali per la webcam
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)

        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Errore nella cattura del frame dalla webcam.")
                    break
                    
                frame = cv2.flip(frame, 1)
                frame_with_faces = detect_faces(frame, face_cascade, specs_ori, cigar_ori)
                FRAME_WINDOW.image(frame_with_faces, channels="BGR")
                
        finally:
            video_capture.release()
    else:
        st.success("Clicca 'Start' per un Selfie Thug")

if __name__ == "__main__":
    main()