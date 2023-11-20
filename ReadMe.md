# Object Detection and Tracking on Pigs farm

## Project Overview
This project focuses on real-time detection and tracking of pigs, their heads, and tails using the YOLOv8 model in combination with OpenCV and other libraries. Each pig is assigned a unique identification number (ID) for tracking purposes.
Example video: https://drive.google.com/file/d/1TbA8ZPOuwIwb0pBFf5PlVxpToBVNsMrX/view?usp=sharing

## Key Features:
- **Pig, Head, and Tail Detection:** Utilizes YOLOv8 to accurately detect and track pigs, their heads, and tails in real time.
- **Unique ID Tracking:** Assigns and maintains unique ID numbers to individual pigs for continuous tracking across frames.
- **RFID Integration:** Integrates with an RFID reader to cross-verify the detected pig's ID with the read RFID data, ensuring consistency and accuracy in tracking.

## Functionality:
Detects and tracks pigs, their heads, and tails using designated class IDs.
Manages the tracking IDs persistently and updates them when pigs are relocated or lost within the specified region.
Compares the RFID ID read by the RFID reader with the assigned ID of the tracked pig, head, or tail within a specific location.
## Usage:
The script requires a webcam or video file input to perform real-time pig, head, and tail detection.
Users can define the ROI and adjust parameters related to tracking thresholds and RFID integration for their specific setup.
This project serves as an efficient framework for real-time monitoring and tracking of pigs, their heads, and tails within a defined region, integrating RFID verification to ensure accurate identification and tracking.


## Installation
There is Jupyter Notebooks that you can see which libraries and installations required to run the project on Google Colab. Also all required files folders are here for run this project on NVIDIA Jetson Nano. Also i will try to write all required libraries below but always may you face with any library problem. If then you should make some search on google :) or for any help mail me please. nlyusufcetin@gmail.com
There is already trained file for using pig detection "last.pt". This file has produced on colab with following Yolov8 Jupyter Notebooks train object detection.
  
 **List of required libraries for python**
 - !pip install ultralytics
 - !git clone https://github.com/ifzhang/ByteTrack.git
  (After installation of ByteTrack, inside the folder there is requirements.txt which should install) !pip3 install -q -r requirements.txt
- !python3 setup.py -q develop
- !pip install -q cython_bbox
- !pip install -q onemetric
- !pip install pip install -q loguru lap thop
- !pip install supervision==0.14.0
- !pip install opencv-python-headless
- OpenCV (cv2): For computer vision tasks like video processing and frame manipulation.
- Installation: pip install opencv-python

- NumPy: For numerical operations and array handling.
- Installation: pip install numpy

- Supervision: External library/module used for various geometric operations, video handling, and annotations.
- Please check the specific installation method or source for the Supervision library you're using. It might not be publicly available or might have specific installation instructions.

- TQDM: For progress bars and monitoring loops.
- Installation: pip install tqdm

- Yolox: For tracking algorithms.
- Check the specific installation instructions for the Yolox library or module you're utilizing.
dataclasses: For creating immutable data structures.
- Included in Python standard library (Python 3.7 and later)

**Please note:**

Some of the modules or libraries might be custom or specific to your project and might not have direct installations through pip.
Ensure to install the exact versions compatible with your project requirements.
If using custom or external modules (like specific trackers or detectors), refer to their documentation or source for installation instructions.
Installing these libraries should provide the necessary functionality and dependencies to run the provided code without import errors.

