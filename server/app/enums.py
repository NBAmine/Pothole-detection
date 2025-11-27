from enum import StrEnum, auto

class Engine(StrEnum):
    ORIGINAL = auto()
    PRUNED   = auto()
    ONNX     = auto()
    NCNN     = auto()
    OPENVINO = auto()