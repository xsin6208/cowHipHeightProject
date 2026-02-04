#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from cowpal.msg import CowInfo
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import numpy as np
import socketio
import json
import os
import cv2
import time

filepath = os.path.expanduser("~/Documents/COMP3888/front_end")

class WebsiteWriter(Node):
    """
    This node reads acts as the interface betwee the ros nodes and the website
    """
    def __init__(self):
        """
        Sets up ros2 subscription callbacks for new information
        """
        super().__init__('website_writer')

        self.sio = socketio.Client()
        try:
            self.sio.connect("http://127.0.0.1:5000")
            
        except Exception as e:
            self.get_logger().error(f"Couldn't connect to socket: {e}")

        self.info_sub = self.create_subscription(
            CowInfo,
            "/cow/data",
            self.info_callback,
            10
        )
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.img_sub = self.create_subscription(
            CompressedImage,
            '/nyx660/rgb/image_raw/compressed',
            self.img_callback,
            qos_profile
        )
        
    def info_callback(self, msg: CowInfo):
        """
        Callback for hip height and cow ID
        """
        id = msg.id
        height = msg.height

        self.get_logger().info(f"Received: Cow {id} and Height {height}")

        self.write_and_notify(id, {"height": height})

    def img_callback(self, msg: CompressedImage):
        """
        Callback for new images, which are stored to files for website viewing
        """
        temp_filename = os.path.join(filepath, "image/feed_tmp.jpg")
        filename = os.path.join(filepath, "image/feed.jpg")

        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        width = 640
        height = 480
        cv_image = cv2.resize(cv_image, (width, height))

        cv2.imwrite(temp_filename, cv_image)

        os.replace(temp_filename, filename)

        time.sleep(1)

    def write_and_notify(self, cow_id: str, cow_data: dict):
        """
        Writes hip height information to the json file and notifies the
        webiste through a socket call
        """
        # Load existing JSON
        filename = os.path.join(filepath, "cow_data.json")
        if os.path.exists(filename):
            with open(filename, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}
        else:
            data = {}

        # Update/insert record
        data[cow_id] = cow_data

        # Write back to file
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

        # Emit updated record
        self.sio.emit("file_update", {"cow_id": cow_id, "cow_data": cow_data})


def main():
    rclpy.init()

    node = WebsiteWriter()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.sio.disconnect()
        node.destroy_node()
        rclpy.shutdown()
    

if __name__=='__main__':
    main()