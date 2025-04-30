import torch
import torchvision.models as models

def export():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "resnet18_lane_detector.onnx",
        input_names=['input'],
        output_names=['output'],
        opset_version=11
    )
    print("✅ ONNX model exported as resnet18_lane_detector.onnx")

if __name__ == "__main__":
    export()
