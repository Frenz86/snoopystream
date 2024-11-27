import cv2
import streamlit as st
from PIL import Image
import numpy as np

@st.cache_resource
def load_images():
    specs_ori = cv2.imread('glass.png', -1)
    cigar_ori = cv2.imread('cigar.png', -1)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return specs_ori, cigar_ori, face_cascade

def transparentOverlay(src, overlay, pos=(0, 0)):
    """
    Applica un overlay trasparente su un'immagine sorgente.
    Args:
        src: Immagine sorgente (ROI dove applicare l'overlay)
        overlay: Immagine con trasparenza da sovrapporre
        pos: Posizione (x, y) dove applicare l'overlay. Default (0, 0)
    """
    # Se l'overlay è vuoto o più grande della sorgente, ritorna la sorgente
    if overlay.shape[0] > src.shape[0] or overlay.shape[1] > src.shape[1]:
        return src

    # Separa il canale alpha dall'overlay
    overlay_alpha = overlay[:, :, 3] / 255.0
    overlay_3chan = overlay[:, :, :3]
    
    # Espandi gli assi per la trasmissione corretta
    alpha = overlay_alpha[..., np.newaxis]
    
    # Calcola la miscelazione usando il canale alpha
    try:
        # Formula: output = alpha * overlay + (1 - alpha) * background
        src[:] = overlay_3chan * alpha + src * (1.0 - alpha)
    except ValueError:
        # In caso di dimensioni non corrispondenti, ritorna l'immagine originale
        return src
        
    return src

def detect_faces(frame, specs_ori, cigar_ori, face_cascade):
    """Rileva volti e aggiunge occhiali e sigaro con posizionamento corretto."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100),
        maxSize=(400, 400)
    )

    for (x, y, w, h) in faces:
        # Occhiali
        glass_height = int(h / 5)
        glass_y = y + int(1.5 * glass_height)
        glass_roi = frame[glass_y:glass_y + glass_height, x:x + w]
        
        # Sigaro - posizionamento completamente rivisto
        cigar_height = int(h / 6)  # Altezza del sigaro
        cigar_width = w // 3  # Larghezza del sigaro
        
        # Nuove coordinate per il sigaro - spostate a sinistra e in alto
        mouth_y = y + int(3.4 * h / 4)  # Spostato ancora più in alto
        mouth_x = x + int(w / 1.7)  # Spostato molto più a sinistra
        
        # Calcolo posizione finale del sigaro
        cigar_y = mouth_y - cigar_height // 3  # Aggiustato per posizione più alta
        cigar_x = mouth_x - cigar_width // 2  # Centrato rispetto al nuovo punto
        
        # Verifica che le ROI siano valide
        if glass_roi.size > 0 and cigar_y + cigar_height <= frame.shape[0] and cigar_x + cigar_width <= frame.shape[1]:
            # Ridimensiona gli overlay
            specs = cv2.resize(specs_ori, (w, glass_height), interpolation=cv2.INTER_LINEAR)
            cigar = cv2.resize(cigar_ori, (cigar_width, cigar_height), interpolation=cv2.INTER_LINEAR)
            
            # Applica gli overlay
            transparentOverlay(glass_roi, specs)
            
            # Definisci e applica la ROI per il sigaro
            cigar_roi = frame[cigar_y:cigar_y + cigar_height, cigar_x:cigar_x + cigar_width]
            if cigar_roi.size > 0:
                transparentOverlay(cigar_roi, cigar)
    
    return frame

def main():
    """Funzione principale Streamlit ottimizzata."""
    st.title("Thug Life Generator")
    
    specs_ori, cigar_ori, face_cascade = load_images()
    
    image = Image.open('thug.jpg')
    st.image(image, width=420)

    if st.button('Start'):
        audio_file = open("snoop.mp3", "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3", loop=True)

        FRAME_WINDOW = st.image([])
        
        video_capture = cv2.VideoCapture(0)
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    st.error("Errore durante la cattura del frame.")
                    break
                    
                frame = cv2.flip(frame, 1)
                frame_with_faces = detect_faces(frame, specs_ori, cigar_ori, face_cascade)
                FRAME_WINDOW.image(frame_with_faces, channels="BGR")
                
        finally:
            video_capture.release()
    else:
        st.success("Clicca su 'Start' per iniziare il Thug Life!")

if __name__ == "__main__":
    main()