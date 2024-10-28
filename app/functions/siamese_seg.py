import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.models import load_model # type: ignore
from IOU_calc import IOU_calc

def segment_video(video_path, model_path='../models/2siames_org_inst.h5'):
    # Cargar el modelo entrenado
    model = load_model(model_path, custom_objects={'IOU_calc': IOU_calc})

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return None
    
    segmented_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Redimensionar el cuadro al tamaño de entrada del modelo
        resized_frame = cv2.resize(frame, (128, 128))
        input_frame = np.expand_dims(resized_frame, axis=0) / 255.0  # Normalización

        # Realizar predicción de segmentación
        instrument_mask, background_mask = model.predict(input_frame)

        # Procesar y guardar la máscara de segmentación
        instrument_mask = (instrument_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        background_mask = (background_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        
        # Redimensionar máscaras al tamaño original del cuadro
        instrument_mask = cv2.resize(instrument_mask, (frame.shape[1], frame.shape[0]))
        background_mask = cv2.resize(background_mask, (frame.shape[1], frame.shape[0]))

        # Agregar las máscaras de segmentación al resultado
        segmented_frames.append((frame, instrument_mask, background_mask))

    # Liberar el video y retornar los cuadros segmentados
    cap.release()
    return segmented_frames
