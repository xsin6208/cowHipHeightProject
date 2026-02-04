This was a group project in which we produced an end to end product which is detailed in the presentation and report. I did the majority of the backend for the cow detection module for segmentation and hip height determination which can be seen in this folder code files. 

# COWPal Back End
This is the back end for CowPal, for monitoring cattle measurements. This back end provides the work to extract cows from imagery, determine the hip point, and calculate the hip height of the cow. This is stored in a json file for the fron end to read

## How it works
The system is build using a ROS2 network. The general pipeline is obataining image and depth data from the NYX660 Time of Flight (ToF) camera, aligning the RGB image with the depth map, and segmenting the cow to locate the hop point. This point is reprojected into 3D space, and the normal distance between the hip point and the ground plane is calculate

## How to Run
To run the entire system, make sure to clone this repository into your ROS2 Workspace, and build the program. Then, launch the file using
```
ros2 launch cowpal cowpal_launch.py
```