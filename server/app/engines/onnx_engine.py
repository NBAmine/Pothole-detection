import onnxruntime as ort
import numpy as np
import cv2
import time

session = ort.InferenceSession(
    "app/models/model.onnx", providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

def non_max_suppression(boxes, iou_threshold: float =0.45):
    """
    Many of the predictions correspond to the same pothole, so this functios keeps only the best bounding box for each object.

    Args:
        boxes: bounding boxes
        iou_threshold (float, optional): Defaults to 0.45.

    Returns:
        boxes: Best boxes
    """
    if not boxes or len(boxes) == 0:
        return boxes

    # Sort by confidence descending
    boxes.sort(key=lambda x: x["confidence"], reverse=True)
    keep = []

    for i, box in enumerate(boxes):
        
        x1_i, y1_i, x2_i, y2_i = box["x1"], box["y1"], box["x2"], box["y2"]
        keep_box = True

        for kept_box in keep:
            
            x1_k, y1_k, x2_k, y2_k = kept_box["x1"], kept_box["y1"], kept_box["x2"], kept_box["y2"]

            # Compute IoU
            inter_x1 = max(x1_i, x1_k)
            inter_y1 = max(y1_i, y1_k)
            inter_x2 = min(x2_i, x2_k)
            inter_y2 = min(y2_i, y2_k)

            inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            area_k = (x2_k - x1_k) * (y2_k - y1_k)
            union_area = area_i + area_k - inter_area

            if union_area > 0:
                boxes_iou = inter_area / union_area
                if boxes_iou > iou_threshold:
                    keep_box = False
                    break

        if keep_box:
            keep.append(box)

    return keep

def infer_onnx(img: np.ndarray, conf_threshold: float = 0.25, iou_threshold: float = 0.45):

    h, w = img.shape[:2]
    
    # Preprocess
    img = cv2.resize(img, (640, 640))

    # ONNX expects shape [Channels, Height, Width]
    img = img.transpose(2, 0, 1)

    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    t0 = time.time()
    outputs = session.run(None, {"images": img})
    t1 = time.time()

    inference_time_ms = (t1 - t0) * 1000
    
    predictions = outputs[0]                 # shape (1, 5, 8400)
    predictions = predictions.squeeze()      # type: ignore # shape (5, 8400)
    predictions = predictions.T              # shape (8400, 5)
    
    print(" ============ ONNX ============")
    
    boxes = []
    
    for pred in predictions:
        x_center, y_center, width, height, confidence = pred
        
        if confidence > conf_threshold:
            
            # Convert normalized center coords to absolute pixel coords
            x1 = (x_center - width / 2) * w
            y1 = (y_center - height / 2) * h
            x2 = (x_center + width / 2) * w
            y2 = (y_center + height / 2) * h

            boxes.append({
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "confidence": float(confidence),
                "class_id": 0,
                "class_name": "pothole"
            })

    # Apply NMS to remove duplicates
    boxes = non_max_suppression(boxes, iou_threshold)
    
    return boxes, inference_time_ms
