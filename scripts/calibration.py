#!/usr/bin/env python3
import numpy as np
import cv2
import rclpy
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
import os
from groundPlane import GroundPlane
from rclpy.lifecycle import Node as LifeCycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

class Calibration(LifeCycleNode):
    """
    This is the Calibration Lifecycle Node, it's main goal is to run RANSAC when
    trigger via the frontend
    """
    def __init__(self):
        """
        Init sets up the parametes needed
        """
        super().__init__("Calibrator")

        self.fx = 414.09
        self.fy = 447.47
        self.cx = 339.3
        self.cy = 239.3

        self.groundPlane = GroundPlane()
        
        self.depth = None
        self.plane = None

        self.get_logger().info("Calibration LifeCycle Node Started")

    def on_configure(self, state: State) -> TransitionCallbackReturn:
        """
        on_configure sets up the callbacks for when new depth data is received
        """
        self.get_logger().info("Calibration node is Configuring")
        try:
            qos_profile = QoSProfile(
                reliability = QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10
            )

            self.depth_sub = self.create_subscription(
                CompressedImage,
                "/nyx660/depth/image_raw/compressedDepth",
                self.callback,
                qos_profile
            )
        except Exception as e:
            self.get_logger().error(f"Exception in on_configure: {e}")
            return TransitionCallbackReturn.FAILURE
        return TransitionCallbackReturn.SUCCESS
    
    def on_activate(self, state: State) -> TransitionCallbackReturn:
        """
        on_activate triggers the calibration process, the latest depth map is used
        and the depth map is converted to a 3D Point cloud. RANSAC is used to 
        fit the ground plane to the cloud. Results are stored in a yaml file
        """
        self.get_logger().info("Activating Calibration Node")
        try:
            if self.depth is None:
                self.get_logger().info("Calibration Failed: No depth data")
                return TransitionCallbackReturn.SUCCESS
            
            point_cloud = self.groundPlane.convert_depth_to_point_cloud(
                self.depth,
                self.fx,
                self.fy,
                self.cx,
                self.cy)
            
            inliers, plane = self.groundPlane.fit_ground_plane(point_cloud)
            self.plane = plane
            data = {
                "plane": {
                    "normal": self.plane[:3].tolist(),
                    "offset": float(self.plane[3])
                }
            }

            path = os.path.expanduser('~/ros2_ws/src/CowPal/config/ground_plane.yaml')

            with open (path, 'w') as f:
                yaml.dump(data, f)

        except Exception as e:
            self.get_logger().error(f"Error: {e}")
            return TransitionCallbackReturn.FAILURE

        return TransitionCallbackReturn.SUCCESS
        

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        """
        Places the node in the deactivated state, idly waits until calibration
        is triggered again
        """
        self.get_logger().info("Deactivating Calibration")
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        """
        Clear any variables used, and destroy the subscription process
        """
        self.depth = None
        self.plane = None
        self.destroy_subscription(self.depth_sub)
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state) -> TransitionCallbackReturn:
        """
        Notifies the user via terminal that the system is being shut down
        """
        self.get_logger().info("Shutting Down Calibration Node")
        return TransitionCallbackReturn.SUCCESS
        

    def callback(self, depth_msg):
        """
        Updates for the latest depth map received, incoming data is compressed,
        so this process decodes it
        """
        raw_data = depth_msg.data[12:]      # Ignoring the first 12 header bytes 
        depth_img = np.frombuffer(raw_data, np.uint8)
        self.depth = cv2.imdecode(depth_img, cv2.IMREAD_UNCHANGED)



def main():
    rclpy.init()
    node = Calibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()
        node.destroy_node()

if __name__=='__main__':
    main()