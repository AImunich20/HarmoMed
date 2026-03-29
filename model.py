import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

def predict_external_image(model, image_path, device, size=256):
    model.eval()
    
    # 1. โหลดด้วย OpenCV เพื่อให้ได้ BGR เหมือนตอนเทรน
    img = cv2.imread(image_path)
    if img is None:
        print("Error: หาไฟล์รูปไม่เจอ")
        return
        
    # 2. Preprocess ให้เหมือน HarmoMedDataset เป๊ะๆ
    img = cv2.resize(img, (size, size))
    img = img / 255.0  # สเกลค่าเป็น 0-1
    
    # 3. เตรียม Tensor (H, W, C) -> (C, H, W) -> (B, C, H, W)
    inp = torch.tensor(img).permute(2, 0, 1).float()
    inp = inp.unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(inp)
    
    # 4. แปลงกลับเป็น Numpy เพื่อแสดงผล
    # (B, C, H, W) -> (C, H, W) -> (H, W, C)
    out_img = out.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # 5. แสดงผล (เนื่องจากเป็น BGR ต้องสลับเป็น RGB เพื่อให้ plt แสดงสีถูก)
    input_rgb = img[:, :, ::-1]      # รูป Input ดั้งเดิม
    output_rgb = out_img[:, :, ::-1] # รูปที่ AI ทำออกมา
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_rgb)
    plt.title("Input Image")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(output_rgb)
    plt.title("Model Prediction")
    plt.axis("off")
    plt.show()
    cv2.imwrite("predicted_output.jpg", out_img[:, :, ::-1] * 255)  # บันทึกผลลัพธ์เป็น BGR

# เรียกใช้งาน
predict_external_image("model.pth", "tar.jpg", torch.device)