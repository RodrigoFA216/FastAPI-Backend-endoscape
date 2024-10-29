import cv2 #type: ignore
import numpy as np #type: ignore
from tensorflow.keras.models import load_model #type: ignore
from app.functions.metrics import IOU_calc

def segment_video_to_output(video_path, model_path='2siames_org_inst.h5', output_path='../temp/videos/segmented_output.mp4'):
    # Cargar el modelo entrenado
    model = load_model(model_path, custom_objects={'IOU_calc': IOU_calc})

    # Abrir el video de entrada
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: No se pudo abrir el video.")
        return None
    
    # Obtener las dimensiones del video original
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Configurar el video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codificación para el formato .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Redimensionar el cuadro al tamaño de entrada del modelo
        resized_frame = cv2.resize(frame, (128, 128))
        input_frame = np.expand_dims(resized_frame, axis=0) / 255.0  # Normalización

        # Realizar predicción de segmentación
        instrument_mask, background_mask = model.predict(input_frame)

        # Procesar y redimensionar la máscara de segmentación al tamaño original del cuadro
        instrument_mask = (instrument_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
        instrument_mask = cv2.resize(instrument_mask, (frame_width, frame_height))

        # Superponer la máscara al cuadro original
        mask_colored = cv2.applyColorMap(instrument_mask, cv2.COLORMAP_JET)
        overlay_frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

        # Escribir el cuadro con la segmentación en el video de salida
        out.write(overlay_frame)

    # Liberar recursos
    cap.release()
    out.release()
    print(f"Video segmentado guardado en: {output_path}")
