import cv2
import pyttsx3
import numpy as np

engine = pyttsx3.init()

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')

cap = cv2.VideoCapture(1)

prev_nearest_object = None

while True:
    ret, frame = cap.read()

    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getUnconnectedOutLayersNames()

    outputs = net.forward(layer_names)

    objects = []
    locations = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = scores.argmax()
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w/2), int(center_y - h/2)

                objects.append(classes[class_id])
                locations.append((x, y, x + w, y + h))

    nearest_object = None
    min_distance = float('inf')
    center_x, center_y = width // 2, height // 2

    for i, (x1, y1, x2, y2) in enumerate(locations):
        object_center_x = (x1 + x2) // 2
        object_center_y = (y1 + y2) // 2
        distance = ((center_x - object_center_x) ** 2 + (center_y - object_center_y) ** 2) ** 0.5

        if distance < min_distance:
            min_distance = distance
            nearest_object = objects[i]

    object_height = 1.0
    focal_length = 1000.0

    if nearest_object:
        distance_meters = (object_height * focal_length) / min_distance
    else:
        distance_meters = None

    if nearest_object != prev_nearest_object:
        if nearest_object and distance_meters is not None:
            engine.say(f"There is a {nearest_object} in front of you at a distance of {distance_meters:.2f} meters")
        else:
            engine.say("No objects detected")
        engine.runAndWait()
        prev_nearest_object = nearest_object

    for i, (x1, y1, x2, y2) in enumerate(locations):
        if objects[i] == nearest_object:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{objects[i]} - {distance_meters:.2f}m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
