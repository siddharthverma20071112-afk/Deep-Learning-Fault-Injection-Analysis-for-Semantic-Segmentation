from pathlib import Path
import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50,
    DeepLabV3_ResNet50_Weights,
)

output = Path("dlModels/PASCAL_VOC/pretrained/deeplabv3_resnet50.pth")
output.parent.mkdir(parents=True, exist_ok=True)

print("Downloading pretrained DeepLabV3-ResNet50 weights...")
model = deeplabv3_resnet50(
    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
)

model.eval()
torch.save(model, output)

print(f"Checkpoint saved to: {output.resolve()}")
