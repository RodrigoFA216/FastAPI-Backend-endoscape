from fastapi import FastAPI
from batabase import get_all_detections_db, create_detection_id_db
from models import Detection

app = FastAPI()

@app.get('/api/testfunctionality')
def read_root():
    return {
        "message": "All Working well",
        "Errors": [],
        }

@app.get('/api/detections')
async def get_detections():
    detections = await get_all_detections_db()
    return detections

@app.post('/api/detections')
async def get_detections(detection: Detection):
    print(detection)
    return {
        "message": "Create detection",
        "Errors": [],
        }

@app.get('/api/detections/{id}')
async def get_detection():
    return {
        "message": "Single detection",
        "Errors": [],
        }

@app.put('/api/detections/{id}')
async def update_detection():
    return {
        "message": "Update detection",
        "Errors": [],
        }


@app.delete('/api/detections/{id}')
async def delete_detection():
    return {
        "message": "Delete detection",
        "Errors": [],
        }