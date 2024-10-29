import cv2  # type: ignore
import numpy as np  # type: ignore
from tensorflow.keras.models import load_model  # type: ignore
from app.functions.metrics import IOU_calc

def segment_image_to_output(image_path, model_path='2siames_org_inst.h5', output_path='../temp/videos/segmented_output.png'):
    # Cargar el modelo entrenado
    model = load_model(model_path, custom_objects={'IOU_calc': IOU_calc})

    # Cargar la imagen de entrada
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: No se pudo abrir la imagen.")
        return None

    # Obtener las dimensiones de la imagen original
    frame_height, frame_width = frame.shape[:2]

    # Redimensionar la imagen al tamaño de entrada del modelo
    resized_frame = cv2.resize(frame, (128, 128))
    input_frame = np.expand_dims(resized_frame, axis=0) / 255.0  # Normalización

    # Realizar predicción de segmentación
    instrument_mask, background_mask = model.predict(input_frame)

    # Procesar y redimensionar la máscara de segmentación al tamaño original de la imagen
    instrument_mask = (instrument_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255
    instrument_mask = cv2.resize(instrument_mask, (frame_width, frame_height))

    # Superponer la máscara a la imagen original
    mask_colored = cv2.applyColorMap(instrument_mask, cv2.COLORMAP_JET)
    overlay_frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

    # Guardar la imagen segmentada
    cv2.imwrite(output_path, overlay_frame)
    print(f"Imagen segmentada guardada en: {output_path}")
