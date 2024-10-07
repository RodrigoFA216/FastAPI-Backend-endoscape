from motor.motor_asyncio import AsyncIOMotorClient
from models import Detection

client = AsyncIOMotorClient('mongodb://localhost')
database = client.detectiondatabase
collection = database.detections

async def get_detection_id_db(id):
    detection = await collection.find_one({'id': id})
    return detection

async def get_all_detections_db():
    detections = []
    cursor = collection.find({})
    async for document in cursor:
        detections.append(Detection(**document))
    return detections

async def create_detection_id_db(detection):
    new = await collection.insert_one(detection)
    detection_gen = await collection.find_one({'id': new.inserted_id})
    return detection_gen

async def update_detection_id_db(id: str, detection):
    await collection.update_one({'id': id}, {'$set': detection})
    detection_gen = await collection.find_one({'id': id})
    return detection_gen

async def delete_detection_id_db(id: str):
    await collection.delete_one({'id': id})
    return True