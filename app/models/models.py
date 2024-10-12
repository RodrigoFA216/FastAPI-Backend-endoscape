from pydantic import BaseModel
from typing import Optional

class Detection(BaseModel):
    id: Optional[str] 
    title: str
    notations: Optional[str] = None
    completed: bool = False
    archive: Optional[str] #Se Necesita cambiar por file y remover opcionalCXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX