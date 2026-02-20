import io
import numpy as np
import cv2
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import time
import uuid
from src.utils.logger import logger

# ===== same model as training =====
IMAGE_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "unet_kidney.pth"


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.d1 = DoubleConv(3, 64)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(128, 256)
        self.u2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c2 = DoubleConv(256, 128)
        self.u1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(self.p1(d1))
        bridge = self.bridge(self.p2(d2))
        u2 = self.u2(bridge)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.c2(u2)
        u1 = self.u1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.c1(u1)
        return torch.sigmoid(self.out(u1))


# ===== load model =====
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

app = FastAPI(title="Kidney Tumor Segmentation API")


@app.get("/")
def home():
    return {"message": "Kidney Tumor API running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    h, w, _ = image_np.shape

    image_resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))
    image_resized = image_resized / 255.0
    image_resized = np.transpose(image_resized, (2, 0, 1))
    image_tensor = (
        torch.tensor(image_resized, dtype=torch.float32)
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():
        pred = model(image_tensor)[0, 0].cpu().numpy()

    pred_mean = float(pred.mean())
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    _, buffer = cv2.imencode(".png", pred_mask)

    latency = time.time() - start_time

    # ⭐ MONITORING LOG
    logger.info(
        f"request_id={request_id} "
        f"height={h} width={w} "
        f"pred_mean={pred_mean:.4f} "
        f"latency={latency:.4f}"
    )

    return {
        "request_id": request_id,
        "mask_bytes": buffer.tobytes().hex(),
    }