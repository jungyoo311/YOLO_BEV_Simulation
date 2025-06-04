import cv2
import numpy as np
import torch
import math
from ultralytics import YOLO

# Load yolo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolo11n.pt')
model.to(device)
yolo_classes = model.names

# Load the input video
video = cv2.VideoCapture('videos/hrv_vid.mp4')
output_filename = 'outtut.mp4'
width, height = 1280, 720
videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
def overlay_transparent(background, foreground, angle, x, y, objSize=50):
    foreground = cv2.resize(foreground, (objSize, objSize))
    
    res = background
    return res
def simulate_object(background, object_class, x, y):
    object_img = cv2.imread(f'assets/MyCar.png', cv2.IMREAD_UNCHANGED)
    if object_img is None:
        return background
    object_img = cv2.resize(object_img, (100, 100))
    # background[y:y+100, x:x+100] 
    return background
def add_ego_vehicle_overlay():
    return True

track = True
frame_cnt = 0
centroid_prev_frame = []
tracking_objects = []
tracking_id = 0
# each frame
while True:
    # capture frame-by-frame
    ret, frame = video.read()
    if not ret:
        print("Can't receive frame.Exiting ...")
        break
    frame = cv2.resize(frame, (width, height))
    frame_cnt += 1
    image = np.zeros((height, width, 3), dtype=np.uint8)
    simulated_image = image.copy()
    transformed_image_with_centroids = image.copy()
    transformed_image_to_sim = image.copy()
    simObjs = image.copy()

    results = model.track(frame, verbose=False, device=device)
    for predictions in results:
        if predictions is None:
            continue
        if predictions.boxes is None or predictions.boxes.id is None:
            continue
        objs = []
        centroid_curr_frame = [] # (centroid_x, centroid_y), class

        # object detection
        detections = predictions.boxes
        if predictions.boxes is not None:
            for bbox in predictions.boxes:
                for scores, classes, coords, id in zip(bbox.conf, bbox.cls, bbox.xyxy, bbox.id):
                    xmin, ymin = coords[0], coords[1]
                    xmax, ymax = coords[2], coords[3]
                    centroid_x, centroid_y = int(xmin + xmax) // 2, int(ymin + ymax) // 2
                    box_color = (0,0,225)
                    # draw box
                    cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), box_color, 2)
                    if int(classes) in [0,1,2,3,5,7] and scores >= 0.3:
                        # draw bbox on the frame
                        object_label = f"{int(classes)}: {scores:.2f}"
                        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), box_color, 2)
                        cv2.putText(frame, object_label, (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 1)
                        centroid_curr_frame.append([(centroid_x, centroid_y), yolo_classes[int(classes)]])
                        if track:
                            objs.append([(centroid_x, centroid_y), yolo_classes[int(classes)]])
        # object tracking
        if track:
            if frame_cnt <= 2:
                for pt1, class_id in centroid_curr_frame:
                    for pt2, class_id in centroid_prev_frame:
                        dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
                        if dist < 50 : # how did figure out this figure
                            tracking_objects[tracking_id] = pt1, class_id
                            tracking_id += 1
            else:
                tracking_objects_copy = tracking_objects.copy()
                for obj_id, pt2 in tracking_objects_copy.items():
                    objects_exists = False
                    for pt1, class_id in centroid_curr_frame:
                        dist = math.hypot(pt2[0][0] - pt1[0], pt2[0][1] - pt1[1])
                        if dist < 20: # how did find this figure
                            tracking_objects[obj_id] = pt1, class_id
                            objects_exists = True
                            continue
                    if not objects_exists:
                        tracking_objects.pop(obj_id)
            for obj_id, pt1 in tracking_objects.items():
                cv2.circle(frame, pt1[0], 3, (0,255,255), -1)
                cv2.putText(frame, str(obj_id)+' '+str(pt1[1]), pt1[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
                if track:
                    objs.append([pt1[0], pt1[0]])
            centroid_prev_frame = centroid_curr_frame.copy()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
videoOut.release()
cv2.destroyAllWindows()
