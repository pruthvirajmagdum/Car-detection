import json
import cv2
import torch
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
import model_utils
import os

with open("class_names.json") as f:
    class_names = json.load(f)

# Initializing YOLO model
yolo_model = YOLO("yolov8s.pt")

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

resnet = torch.hub.load('pytorch/vision:v0.15.2', 'resnet18', pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, len(class_names))
resnet.load_state_dict(torch.load("resnet_indian_car.pth", map_location=device))
resnet.to(device)
resnet.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


#video capturing
# Ask user to select input type
print("Select video input source:")
print("1: Webcam")
print("2: Mobile Camera (IP webcam)")
print("3: CCTV (RTSP link)")
print("4: Video file")

choice = input("Enter your choice (1/2/3/4): ")

if choice == '1':
    cap = cv2.VideoCapture(0)  # Default webcam
elif choice == '2':
    ip = input("Enter IP camera URL (e.g., http://192.168.11.190:8080/video): ")
    cap = cv2.VideoCapture(ip)
elif choice == '3':
    rtsp = input("Enter RTSP link (e.g., rtsp://user:pass@192.168.1.10:554/stream): ")
    cap = cv2.VideoCapture(rtsp)
elif choice == '4':
    file_path = input("Enter video file path (e.g., video.mp4): ")
    cap = cv2.VideoCapture(file_path)
else:
    print("Invalid choice! Exiting.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            yolo_class_name = yolo_model.names[class_id]
            if yolo_class_name.lower() not in ['car', 'truck', 'bus']:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()

            if conf > 0.5:
                car_img = frame[y1:y2, x1:x2]
                if car_img.size == 0:
                    continue

                pil_img = Image.fromarray(cv2.cvtColor(car_img, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = resnet(input_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, pred = torch.max(probabilities, 1)
                    confidence = confidence.item()

                    if confidence > 0.6:
                        car_class = class_names[pred.item()]
                    else:
                        car_class = "Unknown"
            
                car_color = model_utils.detect_color(car_img)

                label = f"{yolo_class_name} | {car_class} | {car_color} | {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Car Detection & Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
