import io
import base64
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, render_template
from skimage.restoration import denoise_wavelet
import torch
from torchvision import transforms
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# ----- U-Net Definition -----
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1), nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.output = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.pool(x1)
        x2 = self.encoder2(x2)
        x3 = self.pool(x2)
        x3 = self.middle(x3)
        x4 = self.upconv1(x3)
        x4 = torch.cat([x4, x2], dim=1)
        x4 = self.decoder1(x4)
        x5 = self.upconv2(x4)
        x5 = torch.cat([x5, x1], dim=1)
        x5 = self.decoder2(x5)
        return self.output(x5)

# ----- LLFlowNet Definition -----
class LLFlowNet(nn.Module):
    def __init__(self):
        super(LLFlowNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(), nn.Conv2d(64, 3, 3, 1, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ----- Load models -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

unet_model = UNet().to(device)
unet_model.load_state_dict(torch.load('yolov5500.pth', map_location=device))
unet_model.eval()

llflow_model = LLFlowNet().to(device)
llflow_model.load_state_dict(torch.load('llflow_model.pth', map_location=device))
llflow_model.eval()

retinex_model = tf.keras.models.load_model('retinexnet_model.h5', custom_objects={'mse': tf.keras.losses.MeanSquaredError()})

transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])

# ----- Helper: In-memory to base64 -----
def img_to_base64(img_array):
    _, buffer = cv2.imencode('.png', img_array)
    return base64.b64encode(buffer).decode('utf-8')

# ----- Preprocessing -----
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    denoised = denoise_wavelet(gray, mode='soft', wavelet_levels=3, rescale_sigma=True)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply((denoised * 255).astype(np.uint8))
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

def enhance_with_unet(image):
    pil_image = Image.fromarray(image).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = unet_model(tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
    return (np.clip(output, 0, 1) * 255).astype(np.uint8)

def enhance_with_llflow(image):
    pil_image = Image.fromarray(image).convert("RGB")
    tensor = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = llflow_model(tensor).squeeze(0).cpu().permute(1, 2, 0).numpy()
    return (output * 255).astype(np.uint8)

def enhance_with_retinex(image):
    pil_image = Image.fromarray(image).resize((256, 256))
    img_array = img_to_array(pil_image) / 255.0
    output = retinex_model.predict(np.expand_dims(img_array, 0))[0]
    return np.clip(output * 255, 0, 255).astype(np.uint8)

@app.route("/", methods=["GET", "POST"])
def upload_image():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            image = Image.open(file.stream).convert("RGB")
            image_np = np.array(image)

            # Process
            original = cv2.resize(image_np, (256, 256))
            pipeline = preprocess_image(original)
            unet_img = enhance_with_unet(original)
            llflow_img = enhance_with_llflow(original)
            retinex_img = enhance_with_retinex(original)

            # Canny edges
            def canny_count(img):
                edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), 50, 150)
                return edges, int(np.sum(edges > 0))

            original_canny, count_orig = canny_count(original)
            pipeline_canny, count_pipe = canny_count(pipeline)
            retinex_canny, count_retinex = canny_count(retinex_img)

            # Render
            return render_template("index.html",
                original=img_to_base64(original),
                pipeline=img_to_base64(pipeline),
                unet=img_to_base64(unet_img),
                llflow=img_to_base64(llflow_img),
                retinex=img_to_base64(retinex_img),
                canny_original=img_to_base64(original_canny),
                canny_pipeline=img_to_base64(pipeline_canny),
                canny_retinex=img_to_base64(retinex_canny),
                count_original=count_orig,
                count_pipeline=count_pipe,
                count_retinex=count_retinex
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)