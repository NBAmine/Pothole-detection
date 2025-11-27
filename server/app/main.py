from fastapi import FastAPI, HTTPException, UploadFile
import cv2
import numpy as np
from app.engines.onnx_engine import infer_onnx
from app.engines.openvino_engine import infer_openvino
from app.engines.pytorch_engine import infer_pytorch
from app.enums import Engine
from app.schemas.prediction import EngineResult, PredictionsResponse

app = FastAPI()


@app.post(
    "/predict/image",
    summary="Single image inference",
    description="Upload an image and get YOLO detections (all engines)",
    response_model=PredictionsResponse,
)
async def predict(file: UploadFile):

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(400, "File is not an image")

    content = await file.read()

    img_arr = np.frombuffer(content, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)  # decode image (BGR)

    if img_bgr is None:
        raise HTTPException(400, "Failed to decode image")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    all_results = []

    # Original
    boxes, latency = infer_pytorch(img_rgb, "original")
    all_results.append(
        EngineResult(
            engine=Engine.ORIGINAL,
            boxes=boxes,
            inference_time_ms=latency,
            num_boxes=len(boxes),
        )
    )

    # Pruned
    boxes, latency = infer_pytorch(img_rgb, "pruned")
    all_results.append(
        EngineResult(
            engine=Engine.PRUNED,
            boxes=boxes,
            inference_time_ms=latency,
            num_boxes=len(boxes),
        )
    )

    # ONNX
    boxes, latency = infer_onnx(img_rgb)
    all_results.append(
        EngineResult(
            engine=Engine.ONNX,
            boxes=boxes,
            inference_time_ms=latency,
            num_boxes=len(boxes),
        )
    )
    
    # OpenVino
    boxes, latency = infer_openvino(img_rgb)
    all_results.append(
        EngineResult(
            engine=Engine.OPENVINO,
            boxes=boxes,
            inference_time_ms=latency,
            num_boxes=len(boxes),
        )
    )

    return {"results": all_results}
