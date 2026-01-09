import onnx
from onnx2torch import convert
import torch

onnx_model_path = "D:/workspace/PythonBot/PUBG.onnx"
# You can pass the path to the onnx model to convert it or...
onnx_model = onnx.load(onnx_model_path)

torch_model_1 = convert(onnx_model)