# Ph sensitive dipstick detection

The repository contains the code to detect images of ph sensitive dipsticks. The detection is built on OpenCV with a simple streamlit app. For the automatic detection of sizes of objects in the image aruco markers are required in the background image.

## Installation

The app works using python 3.11, the dependencies are specified in the `requirements.txt` file

## Using the script

To run the streamlit app use: `streamlit run /path/to/view_detections.py`. The detection logic is implemented in the `ph_strips.py` file


