from ultralytics.models import YOLO
import time

# Load once at startup
original_model = YOLO("app/models/original.pt")
pruned_model = YOLO("app/models/pruned.pt")


def infer_pytorch(img, model_type: str = "original", conf: float = 0.25):

    model = original_model if model_type == "original" else pruned_model

    t0 = time.time()
    results = model.predict(img, conf=conf, iou=0.45, imgsz=640, verbose=False)[0]
    t1 = time.time()

    inference_time_ms = (t1 - t0) * 1000

    if results.boxes is None or len(results.boxes) == 0:
        return [], inference_time_ms

    print(" ============ pt ============")
    print(results.boxes)
    
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes.append(
            {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": box.conf.item(),
                "class_id": int(box.cls.item()),
                "class_name": results.names[int(box.cls.item())],
            }
        )

    return boxes, inference_time_ms
