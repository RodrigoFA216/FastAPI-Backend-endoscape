from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
import cv2 # type: ignore

#from app.schemas.item_scheme import ItemScheme

router = APIRouter()

# Carpetas de archivos temporales
imgFolder = "app/temp/img/"
vidFolder = "app/temp/vid/"

# Formatos válidos
imgFormats = (".png", ".jpg", ".bmp", ".dcm")
vidFormats = (".mp4", ".avi", ".mov", ".wmv", ".mkv")


@router.post("/API/Get/Image/Detection", tags=["Get", "Image"])
async def reciveImage(file: UploadFile = File(...)):
    try:
        if file.filename[-4:] in imgFormats:
            return JSONResponse(
                content={"message": "Reciving data", "file": file.filename},
                status_code=202
            )
        else:
            return JSONResponse(
                content={"Error": "La extención del archivo no es válida"},
                status_code=415,
            )
    except:
        return JSONResponse(
            content={"Error": "Algo Falló con el archivo"}, status_code=200
        )

@router.post("/API/Get/Video/Detection", tags=["Get", "Video"])
async def reciveImage(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        videos_dir = os.path.join(os.path.dirname(__file__), "../temp/videos")
        processed_dir = os.path.join(os.path.dirname(__file__), "../temp/processed_videos")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)
        
        if file.filename[-4:] in vidFormats:
            file_path = os.path.join(videos_dir, file.filename)
            
            # Guardar el archivo en el directorio
            with open(file_path, 'wb') as video_file:
                video_file.write(await file.read())
            
            # Procesar el video con OpenCV
            processed_file_path = os.path.join(processed_dir, f"processed_{file.filename}")
            process_video(file_path, processed_file_path)

            # Agregar tarea en segundo plano para eliminar los archivos temporales después de la respuesta
            background_tasks.add_task(os.remove, file_path)
            background_tasks.add_task(os.remove, processed_file_path)
            
            return FileResponse(processed_file_path, filename=f"processed_{file.filename}", media_type=file.content_type)
        else:
            return JSONResponse(
                content={"Error": "La extensión del archivo no es válida"},
                status_code=415,
            )
    except Exception as e:
        return JSONResponse(
            content={"Error": f"Algo Falló con el archivo: {str(e)}"}, 
            status_code=500
        )

# Función para procesar el video usando OpenCV
def process_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    
    # Obtener propiedades del video original
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el video de salida
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Crear el objeto para escribir el video procesado
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Aplicar algún procesamiento a cada frame (ej. detección de bordes)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_frame = cv2.Canny(gray_frame, 100, 200)  # Detección de bordes
        
        # Escribir el frame procesado en el archivo de salida
        out.write(processed_frame)
    
    cap.release()