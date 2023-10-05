

import cv2
import os
import supervision
import supervision as sv
import argparse

HOME = os.getcwd()
print(HOME)
SOURCE_VIDEO_PATH = cv2.VideoCapture(1)
#SOURCE_VIDEO_PATH = "/home/user/env/pg5.mp4"
TARGET_VIDEO_PATH = "/home/user/env/dnm14.avi"

from typing import List
import numpy as np
from yolox.tracker.byte_tracker import BYTETracker, STrack
from onemetric.cv.utils.iou import box_iou_batch
from yolox.tracker.basetrack import BaseTrack
from dataclasses import dataclass
from tqdm.notebook import tqdm
from tqdm import tqdm
from supervision.geometry.core import Point
from supervision.geometry.core import Tuple
supervision.utils.notebook.plot_image
COLORS = sv.ColorPalette.default()

from supervision.draw.color import ColorPalette
from supervision.utils.video import VideoInfo
from supervision.utils.video import get_video_frames_generator
from supervision.utils.video import VideoSink
#from supervision.notebook.utils import show_frame_in_notebook
from supervision.detection.annotate import Detections, BoxAnnotator
#from supervision.detection.line_counter import LineCounterAnnotator

# check if camera opens
if not SOURCE_VIDEO_PATH.isOpened():
    raise IOError("Cannot open CAMERA")

#VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
@dataclass(frozen=True)
class BYTETrackerArgs:
    track_thresh: float = 0.25
    track_buffer: int = 50
    match_thresh: float = 0.8
    aspect_ratio_thresh: float = 3.0
    min_box_area: float = 1.0
    mot20: bool = False

############Tracking utils Begins

# converts Detections into format that can be consumed by match_detections_with_tracks function
def detections2boxes(detections: Detections) -> np.ndarray:
    return np.hstack((
        detections.xyxy,
        detections.confidence[:, np.newaxis]
    ))

# converts List[STrack] into format that can be consumed by match_detections_with_tracks function
def tracks2boxes(tracks: List[STrack]) -> np.ndarray:
    return np.array([
        track.tlbr
        for track
        in tracks
    ], dtype=float)

# matches our bounding boxes with predictions
def match_detections_with_tracks(
    detections: Detections,
    tracks: List[STrack]
) -> Detections:
    if not np.any(detections.xyxy) or len(tracks) == 0:
        return np.empty((0,))

    tracks_boxes = tracks2boxes(tracks=tracks)
    iou = box_iou_batch(tracks_boxes, detections.xyxy)
    track2detection = np.argmax(iou, axis=1)

    tracker_ids = [None] * len(detections)

    for tracker_index, detection_index in enumerate(track2detection):
        if iou[tracker_index, detection_index] != 0:
            tracker_ids[detection_index] = tracks[tracker_index].track_id
    #print("here is in tracker functions id ",tracks)

    return tracker_ids
#######Tracking utils Ends-----

#######Load Pre trained model Yolov8 model--
MODEL = "/home/user/env/last.pt"
from ultralytics import YOLO
model = YOLO(MODEL)
model.fuse()
# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names
# class_ids of interest - car, motorcycle, bus and truck (for me pig,head,tail)
CLASS_ID = [0]
################################################

def parse_arguments()->argparse.Namespace:
    parser = argparse.ArgumentParser(description ="Yolov8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1024,768],
        nargs =2,
        type=int
    )
    args = parser.parse_args()
    return args

##########PREDICT, ANNOTATE, TRACK WHOLE VIDEO -------
# create BYTETracker instance
byte_tracker = BYTETracker(BYTETrackerArgs())
# create VideoInfo instance
##video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
# create frame generator
##generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

# create instance of BoxAnnotator and LineCounterAnnotator
box_annotator = BoxAnnotator(color=COLORS, thickness=2, text_thickness=1, text_scale=1)
#########ROI
polygon = np.array([[521, 250],[521, 290],[621, 290],[621, 250],[621, 250]])
#polygon = np.array([[1634, 518],[1386, 522],[1382, 618],[1630, 614],[1630, 518]])
#video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)
#zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=video_info.resolution_wh)
args = parse_arguments()
zone = sv.PolygonZone(polygon=polygon, frame_resolution_wh=tuple(args.webcam_resolution))
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.blue(), thickness=6)
################
#tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)

#variables for me
pre_tracker_id = [0]
lost_id = [0]

##################ROI
#polygon = np.array([[1634, 518],[1386, 522],[1382, 618],[1630, 614],[1630, 518]])
#polygon = np.array([[1529, 886],[1593, 886],[1589, 818],[1525, 818],[1525, 886]])
#polygon = np.array([[1393, 902],[1389, 802],[1613, 802],[1613, 898],[1389, 898]])
#polygon = np.array([[1265, 810],[1245, 250],[1561, 246],[1585, 802],[1265, 810]])
#product randm number, later for read number from scanner

#######################ROI
def is_point_inside_polygon(point, polygon):
    # Check if a point is inside a polygon using the "ray casting" algorithm
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def get_unique_identifier(detection, polygon):
    if (
        len(detection) >= 4
        and detection[0] is not None
        and detection[1] is  None
        and detection[2] is not None
        and detection[3] is not None
    ):
        center_x = (detection[0][0] + detection[0][2]) / 2
        center_y = (detection[0][1] + detection[0][3]) / 2
        class_id = detection[4]

        unique_identifier = f"{center_x}_{center_y}_{class_id}"

        # Check if the center of the detection is inside the polygon
        if is_point_inside_polygon((center_x, center_y), polygon):
            x = (class_id, "found in ROI ")

            ###Changin Tracking ID
            updated_tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
            )
            target_track_id_to_change = class_id

            for track in updated_tracks:
              # Check if the current track has the target tracking ID
              if track.track_id == target_track_id_to_change:
                  # Modify the tracking ID to the new value (Read from scanner)
                  track.track_id = "scanned"  # Replace with your desired new tracking ID
                  continue
                #track.track_id = "scanned"
            return x #return unique_identifier
            
        else:
            #print("Detection is outside the polygon")
            return None
    else:
        # Handle the case where the detection doesn't have enough elements or contains None values
        #print("Invalid detection format:", detection)
        return None

############################

def product_num():
  num = random.randrange(200,300)
  return num


# open target video file
#with VideoSink(TARGET_VIDEO_PATH, args) as sink:
    # loop over video frames
    #for frame in tqdm(generator, total=video_info.total_frames):
    
    #loop over webcam frames###########################
fourcc = cv2.VideoWriter_fourcc(*'XVID')    
result = cv2.VideoWriter(TARGET_VIDEO_PATH, fourcc, 3.0, (640,480))
while True:
    ret, frame = SOURCE_VIDEO_PATH.read()
    if not ret:
        print("error retrieving frame!")
        break

    img= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#################################### web cam frames

    # model prediction on single frame and conversion to supervision Detections
    results = model(frame)[0]
    detections = Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )
    # filtering out detections with unwanted classes
    mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
    #detections.filter(mask=mask, inplace=True)
    detections = detections[detections.class_id == 0]

    #Every 'run' the id numbers are increasing
    tracks = byte_tracker.update(
        output_results=detections2boxes(detections=detections),
        img_info=frame.shape,
        img_size=frame.shape
    )
    
    #print(tracker_id)
    tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
    detections.tracker_id = np.array(tracker_id)

    #print(len(pre_tracker_id)," previous ID's  =",pre_tracker_id)
    #print(len(tracker_id)," Current ID's  = ",tracker_id)

    #Checking every Tracker ID's list with previous ID's list
    i = (int)
    for i in range(0, len(pre_tracker_id)):
        idflag = False
        for j in range(0,len(tracker_id)):
        #print("if",pre_tracker_id[i]," != ", tracker_id[j])
            if(pre_tracker_id[i]!= tracker_id[j]):
                idflag = True
            else:
                idflag = False
                break
        if idflag == True:
            lost_id.append(pre_tracker_id[i])

    #print("lost ids = ",lost_id)
    lost_id = [0]
    t = 0
    ##############ROI
    for detection in detections:
        unique_identifier = get_unique_identifier(detection, polygon)
        if unique_identifier:
            print(t," = Inside the area ", unique_identifier)
            t+=1
    ###################

    '''labels = [
        f"#{tracker_id} {CLASS_NAMES_DICT[class_id]}"
        for _, confidence, class_id, tracker_id
        in detections
    ]'''
    labels = [f"#{tracker_id}" for tracker_id in detections.tracker_id]
    #to check same id or not
    pre_tracker_id = tracker_id
    # annotate and display frame
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = zone_annotator.annotate(scene=frame)
    result.write(frame)
    cv2.imshow('frame', frame)
    
    c = cv2.waitKey(1)
    if c == 27:
        break

    #sink.write_frame(frame)
SOURCE_VIDEO_PATH.release()
result.release()
cv2.destroyAllWindows()