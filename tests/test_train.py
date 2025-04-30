import unittest
import torch
from torchvision import models

class TestModel(unittest.TestCase):
    def test_resnet18_output_shape(self):
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 2))

if __name__ == "__main__":
    unittest.main()
