from typing import List
from pydantic import BaseModel

from app.enums import Engine


class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str


class EngineResult(BaseModel):
    engine: Engine
    boxes: List[Box]
    inference_time_ms: float
    num_boxes: int
    
class PredictionsResponse(BaseModel):
    results: List[EngineResult]