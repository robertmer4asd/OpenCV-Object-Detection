import numpy as np
import cv2

prototxt_path = 'deploy.prototxt'
model_path = 'MobileNetSSD_deploy.caffemodel'
min_confidence = 0.2

classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
np.random.seed(34732)
colors = np.random.uniform(0, 255, size=(len(classes), 3))
person_detected_flag = False
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
cap = cv2.VideoCapture(1)

while True:
    _, image = cap.read()
    image = cv2.flip(image, 1)  # Horizontal flip to mirror the image
    height, width = image.shape[0], image.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()

    screen_center_x = width // 2  # Center of the screen
    center_threshold = 50  # Threshold for considering person in the center

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]
        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])
            if classes[class_index] == 'person' and confidence > 0.8:
                person_detected_flag = True
                upper_left_x = int(detected_objects[0, 0, i, 3] * width)
                upper_left_y = int(detected_objects[0, 0, i, 4] * height)
                lower_right_x = int(detected_objects[0, 0, i, 5] * width)
                lower_right_y = int(detected_objects[0, 0, i, 6] * height)

                person_center_x = (upper_left_x + lower_right_x) // 2
                person_center_y = (upper_left_y + lower_right_y) // 2

                if person_center_x < screen_center_x - center_threshold:
                    position_text = "Left"
                elif person_center_x > screen_center_x + center_threshold:
                    position_text = "Right"
                else:
                    position_text = "Center"

                focal_length = 800
                real_person_height = 180
                image_person_height = lower_right_y - upper_left_y
                distance_to_person = (focal_length * real_person_height) / image_person_height

                if distance_to_person > 400:
                    distance_text = "Far"
                else:
                    distance_text = "Close"

                cv2.putText(image, f"Position: {position_text}", (upper_left_x, upper_left_y + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, f"Distance: {distance_text}", (upper_left_x, upper_left_y + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, f"X: {person_center_x}", (upper_left_x, upper_left_y + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, f"Y: {person_center_y}", (upper_left_x, upper_left_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            prediction_text = f"{classes[class_index]} : {confidence:.2f}"
            cv2.rectangle(image, (upper_left_x, upper_left_y), (lower_right_x, lower_right_y), colors[class_index], 3)
            cv2.putText(image, prediction_text,
                        (upper_left_x, upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[class_index], 2)

    cv2.imshow("Detected Objects", image)
    cv2.waitKey(5)

    print("Person detected:", person_detected_flag)

    if person_detected_flag:
        print("Person detected!")
        person_detected_flag = False

cv2.destroyAllWindows()
cap.release()
