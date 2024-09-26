import cv2
import mediapipe as mp
import requests

def control_lights(state):
    requests.post("https://maker.ifttt.com/trigger/" + state + "/with/key/iUT3Wz1etb3LLJK274uVDX0YPSOz9m5zdIOJmIbQGyS")

# Inicializa MediaPipe Hands y el módulo de dibujo.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Cargar clasificadores en cascada para detección de rostros y ojos.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Inicia captura de video desde la cámara.
cap = cv2.VideoCapture(0)

# Función para determinar si la mano está abierta.
def is_hand_open(landmarks):
    middle_finger_tip_y = landmarks[12].y
    middle_finger_base_y = landmarks[9].y
    return middle_finger_tip_y < middle_finger_base_y

# Inicializa el modelo de MediaPipe Hands.
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=1) as hands:
    light_on = False  # Estado inicial de la luz

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("No se pudo obtener el frame.")
            break

        # Detección de rostros
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
        eyes_detected = 0  # Contador para los ojos detectados

        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y + h, x:x + w]  # Región de interés en escala de grises
            roi_color = frame[y:y + h, x:x + w]  # Región de interés en color

            # Detección de ojos dentro de la región del rostro
            eyes = eye_cascade.detectMultiScale(roi_gray)
            eyes_detected += len(eyes)  # Contar ojos detectados

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)  # Dibujar rectángulo alrededor de los ojos

        # Inicializa la imagen a mostrar
        image = frame.copy()  # Copia del frame original para mostrar
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        small_image = cv2.resize(image, (320, 240))
        results = hands.process(small_image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Solo detectar la mano si se están mirando a la cámara
        if eyes_detected >= 2:
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if is_hand_open(hand_landmarks.landmark):
                        if not light_on:  # Enciende la luz solo si está apagada
                            control_lights("light_on")
                            light_on = True  # Actualiza el estado de la luz

                    else:
                        if light_on:  # Apaga la luz solo si está encendida
                            control_lights("light_off")  # Opcional: Apagar la luz también puede enviar una señal
                            light_on = False  # Actualiza el estado de la luz
        else:
            pass

        cv2.imshow('Detección de Mano', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
