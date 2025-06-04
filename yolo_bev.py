import cv2
import numpy as np
import torch
from ultralytics import YOLO
import math

# Load yolo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolo11n.pt')
model.to(device)
yolo_classes = model.names

# Load the input video
video = cv2.VideoCapture('videos/hrv_vid.mp4')
output_filename = 'oyolon_output.mp4'
width, height = 1280, 720
videoOut = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))

def overlay_transparent(background, foreground, angle, x, y, objSize=50):
    foreground = cv2.resize(foreground, (objSize, objSize))
    # get the shape of the foreground image
    rows, cols, channels = foreground.shape
    # calcuate the center of the foreground image
    center_x, center_y = int(cols/2), int(rows/2)
    # rotate the foreground image
    M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1)
    foreground = cv2.warpAffine(foreground, M, (cols, rows)) # applies the rotation to the foreground image
    for row in range(rows):
        for col in range(cols):
            if x + row < background.shape[0] and y + col < background.shape[1]:
                alpha = foreground[row, col, 3] / 255.0
                background[x + row, y + col] = alpha * foreground[row, col, :3] + (1 - alpha) * background[x + row, y + col]
    res = background
    return res

def simulate_object(background, object_class, x, y):
    object_img = cv2.imread(f'assets/{object_class}.png', cv2.IMREAD_UNCHANGED)
    if object_img is None:
        return background
    object_img = cv2.resize(object_img, (100, 100))
    background[y:y+100, x:x+100] = overlay_transparent(background[y:y+100, x:x+100], object_img, 0, 0, 0)
    return background

def add_ego_vehicle_overlay(background):
    overlay_img = cv2.imread('assets/MyCar.png', cv2.IMREAD_UNCHANGED)
    rows, cols, _ = overlay_img.shape
    x, y = 550, background.shape[0] - 200
    
    # Overlay the image onto the background
    overlay_img = overlay_transparent(background[y:y+rows, x:x+cols], overlay_img, 0, 0, 0, objSize=250)
    background[y:y+rows, x:x+cols] = overlay_img
    return background

def plot_object_bev(transformed_image_with_centroids, src_pts, dst_pts, objs):
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    persObjs = []
    for obj in objs:
        if obj:
            # numpy array of centroid coordinates
            centroid_coords = np.array([list(obj[0])], dtype=np.float32)
            # apply the perspective transformations to be centroid coord
            transformed_coords = cv2.perspectiveTransform(centroid_coords.reshape(-1, 1, 2), M)
            transformed_coords_ = tuple(transformed_coords[0][0].astype(int))

            # Draw a circle at the transformed centroid location
            cv2.circle(transformed_image_with_centroids, transformed_coords_, radius=3, color=(0, 255, 0), thickness=-1)
            cv2.circle(transformed_image_with_centroids, transformed_coords_, radius=12, color=(255, 255, 255), thickness=1)
            class_text = f"Class: {obj[1]}"
            cv2.putText(transformed_image_with_centroids, class_text, (transformed_coords_[0] + 10, transformed_coords_[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            persObjs.append([transformed_coords_, obj[1]])

    return transformed_image_with_centroids, persObjs

track = True
frame_cnt = 0
centroid_prev_frame = []
tracking_objects = {}
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
    
    #####################
    ##        BEV      ##
    #####################
    # Define the source points (region of interest) in the original image
    x1, y1 = 10, 720  # Top-left point
    x2, y2 = 530, 400  # Top-right point
    x3, y3 = 840, 400  # Bottom-right point
    x4, y4 = 1270, 720  # Bottom-left point
    src_points = np.float32([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    # Draw the source points on the image (in red)
    # cv2.polylines(frame, [src_points.astype(int)], isClosed=True, color=(0, 0, 255), thickness=2)

    # # Define the destination points (desired output perspective)
    u1, v1 = 370, 720  # Top-left point
    u2, v2 = 0+150, 0  # Top-right point
    u3, v3 = 1280-150, 0  # Bottom-right point
    u4, v4 = 900, 720  # Bottom-left point
    dst_points = np.float32([[u1, v1], [u2, v2], [u3, v3], [u4, v4]])
    # # Draw the destination points on the image (in blue)
    # cv2.polylines(frame, [dst_points.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)
    # print(transformed_image_with_centroids)
    # perspectives plot and objs
    transformed_image_with_centroids, persObjs_ = plot_object_bev(transformed_image_with_centroids, src_points ,dst_points , objs)

    # fix here
    ### plot objs overlays
    for persObj_ in persObjs_:
        # print(simObjs)
        simObjs = simulate_object(transformed_image_to_sim, persObj_[1], persObj_[0][0], persObj_[0][1])
    # Add the car_img overlay to the simulated image
    simulated_image = add_ego_vehicle_overlay(simObjs)

    videoOut.write(simulated_image)
    # Display the simulated image and frame
    cv2.imshow("Video", frame)
    if track:
        cv2.imshow("Simulated Objects", simulated_image)
        cv2.imshow('Transformed Frame', transformed_image_with_centroids)
    # cv2.imwrite('test.jpg', simulated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
videoOut.release()
cv2.destroyAllWindows()
