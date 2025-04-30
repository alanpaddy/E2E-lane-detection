import onnxruntime as ort
import numpy as np
import time

def benchmark():
    session = ort.InferenceSession("resnet18_lane_detector.onnx")
    input_name = session.get_inputs()[0].name

    dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

    start = time.time()
    for _ in range(100):
        _ = session.run(None, {input_name: dummy_input})
    end = time.time()

    print(f"✅ Average inference time: {(end - start)/100:.6f} seconds")

if __name__ == "__main__":
    benchmark()
