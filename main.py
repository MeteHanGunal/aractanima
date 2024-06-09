import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
#version 0.1.0
# Modeli yükle
model = load_model('updated_car_classification_model.h5')

# YOLO modelini yükle
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

vehicle_classes = ["car", "bus", "truck"]

car_models = {
    0: "Audi",
    1: "Hyundai Creta",
    2: "Mahindra Scorpio",
    3: "Rolls Royce",
    4: "Swift",
    5: "Tata Safari",
    6: "Toyota Innova"
}

cap = cv2.VideoCapture(0)

detecting = False

root = tk.Tk()
root.title("Araç Tanıma Uygulaması")

label = tk.Label(root)
label.grid(row=0, column=0, columnspan=2)


def start_detection():
    global detecting
    detecting = True
    threading.Thread(target=detect).start()


def stop_detection():
    global detecting
    detecting = False


def detect():
    while detecting:
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı")
            return

        height, width, channels = frame.shape

        if detecting:
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.7 and classes[class_id] in vehicle_classes:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label_text = str(classes[class_ids[i]])
                    if label_text in vehicle_classes:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        car_img = frame[y:y + h, x:x + w]
                        if car_img.size == 0:
                            continue
                        car_img = cv2.resize(car_img, (224, 224))
                        car_img = car_img / 255.0
                        car_img_array = np.expand_dims(car_img, axis=0)
                        prediction = model.predict(car_img_array)
                        predicted_class = np.argmax(prediction)
                        model_name = car_models.get(predicted_class, "Unknown")
                        cv2.putText(frame, f'{model_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)

        label.imgtk = imgtk
        label.configure(image=imgtk)

        label.update_idletasks()


start_button = tk.Button(root, text="Tanımayı Başlat", command=start_detection)
start_button.grid(row=1, column=0)

stop_button = tk.Button(root, text="Tanımayı Durdur", command=stop_detection)
stop_button.grid(row=1, column=1)

root.mainloop()

cap.release()
cv2.destroyAllWindows()
