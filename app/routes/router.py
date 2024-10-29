from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
import os
import shutil
import cv2 # type: ignore
from app.functions.siamese_seg import segment_video_to_output  
from app.functions.siamese_seg_images import segment_image_to_output  

#from app.schemas.item_scheme import ItemScheme

router = APIRouter()

# Carpetas de archivos temporales
imgFolder = "app/temp/img/"
vidFolder = "app/temp/vid/"

# Formatos válidos
imgFormats = (".png", ".jpg", ".bmp", ".dcm")
vidFormats = ['.mp4', '.avi', '.mov', '.mkv']


@router.post("/API/Get/Image/Detection/old", tags=["Get", "Image"])
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

@router.post("/API/Get/Video/Detection/old", tags=["Get", "Video"])
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

@router.post("/API/Get/Video/Detection", tags=["Get", "Video", "Siamese"])
async def receive_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Directorios de videos
        videos_dir = os.path.join(os.path.dirname(__file__), "../temp/videos")
        processed_dir = os.path.join(os.path.dirname(__file__), "../temp/processed_videos")
        os.makedirs(videos_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Validación de formato de video
        if file.filename[-4:].lower() in vidFormats:
            input_path = os.path.join(videos_dir, file.filename)
            output_path = os.path.join(processed_dir, f"segmented_{file.filename}")

            # Guardar el archivo de video en el directorio temporal
            with open(input_path, 'wb') as video_file:
                video_file.write(await file.read())
            
            # Llamar a la función de segmentación
            segment_video_to_output(input_path, model_path='app/models/2siames_org_inst.h5', output_path=output_path)

            # Agregar tareas para eliminar archivos temporales después de la respuesta
            background_tasks.add_task(os.remove, input_path)
            background_tasks.add_task(os.remove, output_path)
            
            # Retornar el archivo de video segmentado
            return FileResponse(output_path, filename=f"segmented_{file.filename}", media_type=file.content_type)
        else:
            return JSONResponse(
                content={"Error": "La extensión del archivo no es válida"},
                status_code=415,
            )
    except Exception as e:
        return JSONResponse(
            content={"Error": f"Algo falló con el archivo: {str(e)}"}, 
            status_code=500
        )

@router.post("/API/Get/Image/Detection", tags=["Get", "Image", "Siamese"])
async def receive_image(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    try:
        # Directorios de imágenes
        images_dir = os.path.join(os.path.dirname(__file__), "../temp/images")
        processed_dir = os.path.join(os.path.dirname(__file__), "../temp/processed_images")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Validación de formato de imagen (puedes ajustar los formatos según necesites)
        if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            input_path = os.path.join(images_dir, file.filename)
            output_path = os.path.join(processed_dir, f"segmented_{file.filename}")

            # Guardar el archivo de imagen en el directorio temporal
            with open(input_path, 'wb') as image_file:
                image_file.write(await file.read())

            # Llamar a la función de segmentación de imagen
            segment_image_to_output(input_path, model_path='app/models/2siames_org_inst.h5', output_path=output_path)

            # Agregar tareas para eliminar archivos temporales después de la respuesta
            background_tasks.add_task(os.remove, input_path)
            background_tasks.add_task(os.remove, output_path)

            # Retornar el archivo de imagen segmentada
            return FileResponse(output_path, filename=f"segmented_{file.filename}", media_type="image/png")
        else:
            return JSONResponse(
                content={"Error": "La extensión del archivo no es válida"},
                status_code=415,
            )
    except Exception as e:
        return JSONResponse(
            content={"Error": f"Algo falló con el archivo: {str(e)}"}, 
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