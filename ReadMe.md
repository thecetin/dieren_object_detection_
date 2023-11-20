# Object Detection and Tracking on Pigs farm

# Project Overview
This project focuses on real-time detection and tracking of pigs, their heads, and tails using the YOLOv8 model in combination with OpenCV and other libraries. Each pig is assigned a unique identification number (ID) for tracking purposes.

# Key Features:
- **Pig, Head, and Tail Detection:** Utilizes YOLOv8 to accurately detect and track pigs, their heads, and tails in real time.
- **Unique ID Tracking:** Assigns and maintains unique ID numbers to individual pigs for continuous tracking across frames.
- **RFID Integration:** Integrates with an RFID reader to cross-verify the detected pig's ID with the read RFID data, ensuring consistency and accuracy in tracking.

# Functionality:
Detects and tracks pigs, their heads, and tails using designated class IDs.
Manages the tracking IDs persistently and updates them when pigs are relocated or lost within the specified region.
Compares the RFID ID read by the RFID reader with the assigned ID of the tracked pig, head, or tail within a specific location.
# Usage:
The script requires a webcam or video file input to perform real-time pig, head, and tail detection.
Users can define the ROI and adjust parameters related to tracking thresholds and RFID integration for their specific setup.
This project serves as an efficient framework for real-time monitoring and tracking of pigs, their heads, and tails within a defined region, integrating RFID verification to ensure accurate identification and tracking.



## Installation

-- OpenCV (cv2): For computer vision tasks like video processing and frame manipulation.

Installation: pip install opencv-python
NumPy: For numerical operations and array handling.

Installation: pip install numpy
Supervision: External library/module used for various geometric operations, video handling, and annotations.

Please check the specific installation method or source for the Supervision library you're using. It might not be publicly available or might have specific installation instructions.
TQDM: For progress bars and monitoring loops.

Installation: pip install tqdm
Yolox: For tracking algorithms.

Check the specific installation instructions for the Yolox library or module you're utilizing.
dataclasses: For creating immutable data structures.

Included in Python standard library (Python 3.7 and later)
Please note:

Some of the modules or libraries might be custom or specific to your project and might not have direct installations through pip.
Ensure to install the exact versions compatible with your project requirements.
If using custom or external modules (like specific trackers or detectors), refer to their documentation or source for installation instructions.
Installing these libraries should provide the necessary functionality and dependencies to run the provided code without import errors.

