import torch
from torchreid import models, utils
import cv2
from torchvision import transforms


class Encoder:
    def __init__(self):
        # Build model
        self.model = models.build_model(
            name="osnet_x1_0",
            num_classes=1000,
            pretrained = True
        )


        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # H, W
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print("Encoder loaded.")

    def encode(self, img):
        # img is OpenCV BGR
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        tensor = self.transform(img_rgb)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor)

        return embedding.cpu().numpy().flatten()
