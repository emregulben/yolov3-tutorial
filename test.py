import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

output_layers = net.getUnconnectedOutLayersNames()
colors = np.random.uniform(0, 255, size=(len(classes), 3))  # size=(how many colors, for blue-green-red)

# Load image
img = cv2.imread("myimage.jpg")
img = cv2.resize(img, None, fx=1.5, fy=1.5)
height, width, channels = img.shape

# Detect image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Show info on screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            #cv2.circle(img, (center_x, center_y), 10, (0, 255, 0), 2)    # Center

            # Rectangle coordinates
            x = int(center_x - (w / 2))
            y = int(center_y - (h / 2))

            #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)    # Rectangles

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)   # Get indices of meaningful boxes
font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indices:

        x, y, w, h = boxes[i]
        label = str(classes[int(class_ids[i])])
        conf = round(confidences[i], 2)
        text = label + ' ' + str(conf)
        color = colors[i]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, text, (x, y + 15), font, 1, color, 2)


cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

