#!/usr/bin/env python3
import rclpy
from rclpy.lifecycle import Node as LifeCycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from sensor_msgs.msg import CompressedImage
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from ament_index_python.packages import get_package_share_directory
from cowpal.msg import CowInfo
import os
import yaml
import torch
from groundPlane import GroundPlane
from cowDetectionModel import cowDetectionModel
import numpy as np
import cv2
from ament_index_python.packages import get_package_share_directory

PKG_SHARE = get_package_share_directory('cowpal')
CONFIG_PATH = os.path.join(PKG_SHARE, 'config', 'ground_plane.yaml')

# Device to use for model training (Will need to be updated for AI hat)
device = torch.device("cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"))

class heightObtainor(LifeCycleNode):
    """
    This node handles incoming data, aligns the rgb and depth imagery,
    segments the cow, and returns the hip height 
    """
    def __init__(self):
        """
        Initial setup for the node
        """
        super().__init__("HeightObtainor")

        self.image = None
        self.depth = None

        self.img_sub = None
        self.depth_sub = None
        self.sync_subs = None

        self.no_cow = 0
        self.cow_id = 1
        self.heights = []
        self.detected = False

        self.depth_man = GroundPlane()

        self.activate = False

        self.cow_data_pub = self.create_publisher(CowInfo, "/cow/data", 10)

        self.get_logger().info("Height Obtainor Node has begun")
        self.cowDetection = cowDetectionModel(device)
        
    def on_configure(self, state: State):
        """
        Sets up the callback processes, these are synced using a synchronise
        so the depth map and imagery are aligned
        """
        self.get_logger().info("Node is being Configured")
        qos_profile = QoSProfile(
                reliability = QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10
            )
        
        self.img_sub = Subscriber(
            self,
            CompressedImage,
            '/nyx660/rgb/image_raw/compressed',
            qos_profile=qos_profile
        )

        self.depth_sub = Subscriber(
            self,
            CompressedImage,
            '/nyx660/depth/image_raw/compressedDepth',
            qos_profile=qos_profile
        )

        self.sync_subs = ApproximateTimeSynchronizer(
            [self.img_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )

        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state):
        """
        Sets activation status to true to determine hip heights, reads in the 
        yaml file for updated ground plane 
        """
        self.get_logger().info("Node is being activated")
        self.activate = True

        # Finding the root directory for config files
        try:
            pkg_share = get_package_share_directory('cowpal') 
            config_path = os.path.join(pkg_share, 'config', 'nyx660.yaml')
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.rgb_int = config["rgb_camera"]
            self.depth_int = config["depth_camera"]
            self.rgb_ext = config["extrinsics"]["rgb_to_base_link"]
            self.depth_ext = config["extrinsics"]["depth_to_base_link"]
        except Exception as e:
            self.get_logger().error(f"Could load file: {e}")

        self.depth_man.precompute_rgb_alignment((480, 640),
                                                self.rgb_int,
                                                self.depth_int,
                                                self.rgb_ext,
                                                self.depth_ext,
                                                False)

        self.sync_subs.registerCallback(self.callback)

        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state):
        """
        Sets activation status to false, this is usually during
        calibration cycles so no wrong information is sent
        """
        # Setting nodes to None to stop callbacks
        self.get_logger().info("Node has been deactivated")
        self.activate = False
        return TransitionCallbackReturn.SUCCESS
    

    def callback(self, rgb_msg, depth_msg):
        """
        callback is a ros node callback function that gets triggered whenever
        a depth and rgb image come in. It coordinates the segmentation, height calculation
        and publishing to the website wrapper
        Args:
            rgb_msg: CompressedImage data
            depth_msg: CompressedDepth data
        """
        self.get_logger().info("Received Callback")
        # If not currently active
        if not self.activate:
            return
        
        # Decoding rgb data
        np_arr = np.frombuffer(rgb_msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Decoding depth data
        raw_depth = depth_msg.data[12:]
        depth_frame = cv2.imdecode(np.frombuffer(raw_depth, np.uint8), cv2.IMREAD_UNCHANGED)

        # Updating depth map for ground plane for image alignment
        self.depth_man.update_depth(depth_frame)
        img = self.depth_man.align_rgb(img)

        # Shifting image if using pre-computed mappin values
        img = np.roll(img, shift=8, axis=1) 

        # Getting index for hip point on depth map
        x, y = self.cowDetection.hipHeightPipeline(img, depth_frame, useYolo=True)
        x = int(x)
        y = int(y)

        # If no cow is detected
        if x == -1 or y == -1:
            self.no_cow += 1
            self.get_logger().info("No cow detected")
            # No cow has been present for long enough to consider it a new cow (lack of RFID scanner)
            if self.no_cow > 3 and self.detected:
                self.publish_data()

                self.detected = False
                self.cow_id += 1

            return
        
        # Resetting
        self.no_cow = 0
        self.detected = True


        # Obtaining height of cow
        height = self.get_height(depth_frame, x, y)

        # Checking valid height (this occurs where dept map gets no return)
        if height < 1:
            self.get_logger().info("The distance is too small")
            return
        
        self.heights.append(height)
        self.get_logger().info(f"The cow is {height} tall!!!!")

    
    def remove_outliers(self):
        """
        Uses inter-quartile range to remove outliers from the height array
        """
        q1 = np.percentile(self.heights, 25)
        q3 = np.percentile(self.heights, 75)
        iqr = q3 - q1
        
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        
        filtered = [height for height in self.heights if lower <= height <= upper]
        self.heights = filtered

    def publish_data(self):
        """
        Publishes hip height and id for the website
        """
        self.remove_outliers()

        height = sum(self.heights)/len(self.heights)
        self.heights = []

        msg = CowInfo()
        msg.id = self.cow_id
        msg.height = height

        self.cow_data_pub.publish(msg)


    
    def get_height(self, depth_map, x, y):
        # Updating plane data in case calibration was triggered
        with open(CONFIG_PATH, 'r') as f:
            config = yaml.safe_load(f)

        plane_normal = config['plane']['normal']  
        plane_offset = config['plane']['offset']  

        # Convert to [a, b, c, d]
        plane = plane_normal + [plane_offset]

        # Determine point in 3d space
        point = self.depth_man.reproject_point(depth_map, np.array(self.depth_int['K']).reshape(3, 3), x, y)

        return self.depth_man.get_normal_distance(point, plane)
        



def main():
    rclpy.init()
    node = heightObtainor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        node.destroy_node()

if __name__=='__main__':
    main()