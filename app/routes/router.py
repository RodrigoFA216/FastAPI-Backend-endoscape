from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
import os

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
async def reciveImage(file: UploadFile = File(...)):
    try:
        videos_dir = os.path.join(os.path.dirname(__file__), "../temp/videos")
        
        os.makedirs(videos_dir, exist_ok=True)
        
        if file.filename[-4:] in vidFormats:
            file_path = os.path.join(videos_dir, file.filename)
            
            with open(file_path, 'wb') as video_file:
                video_file.write(await file.read())
            
            return JSONResponse(
                content={"message": "Reciving data", "file": file.filename},
                status_code=202
            )
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

