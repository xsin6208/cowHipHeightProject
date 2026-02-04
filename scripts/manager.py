#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from lifecycle_msgs.srv import ChangeState, GetState
from lifecycle_msgs.msg import Transition
import os
import time
import json

class Manager(Node):
    """
    This class manages the two main nodes for cow segmentation and calibration.
    When calibration is triggered, it deactivates the main node, and re-activates it when 
    calibration is complete
    """
    def __init__(self):
        """
        Initialises the services for each node managed
        """
        # Extension of ROS2 Node
        super().__init__("LifecycleManager")

        # Lifecycle nodes to be managed throughout execution
        self.lifecycle_clients = {}
        self.targets = ['Calibrator', 'HeightObtainor']

        # Creating the getting and setting services for each lifecycle node
        self.get_logger().info("Waiting to connect to lifecycle nodes")
        for node in self.targets:
            change_cli = self.create_client(ChangeState, f'/{node}/change_state')

            get_cli = self.create_client(GetState, f'/{node}/get_state')

            self.lifecycle_clients[node] = {
                'change' : change_cli,
                'get' : get_cli
            }

            change_cli.wait_for_service()
            get_cli.wait_for_service()

        self.get_logger().info("Connected to lifecycle nodes")
        # File path to be monitored that triggers calibration
        self.watch_file = os.path.expanduser("~/Documents/COMP3888/front_end/trigger.json")

    
    def on_startup(self):
        """
        Puts all lifecycle nodes into the configure state
        """
        # Add in logic to make the main node be activate
        self.get_logger().info("Setting up lifecycle nodes")
        self.get_logger().info(f"{self.targets}")
        for node in self.targets:
            self.change_state(node, 'configure')

        self.change_state('HeightObtainor', 'activate')
        

    def run(self):
        """
        Continues to check for json file if calibration has been triggered, if so it 
        puts calibration node into active configuration then deactivates it
        """
        self.get_logger().info("Running")
        while rclpy.ok():
            # If calibration has been triggered
            if self.check_trigger():
                self.get_logger().info("Calibration has been triggered")
                self.change_state('HeightObtainor', 'deactivate')
                self.change_state('Calibrator', 'activate')
                self.finished_calibration()
                self.change_state('Calibrator', 'deactivate')
                self.change_state('HeightObtainor', 'activate')

            # Preventing it from taking up too much CPU usage by busy waiting
            time.sleep(0.1)

    def finished_calibration(self):
        """
        Write to calibration file that calibration has been completed
        """
        if os.path.exists(self.watch_file):
            try:
                with open(self.watch_file, 'w') as f:
                    data = {"calibrate": False}
                    json.dump(data, f)
            except Exception as e:
                self.get_logger().warn(f"Couldn't open file: {e}")

    def check_trigger(self):
        """
        Checks json file if calibration has been triggered

        Returns:
            Boolean: bool type as to whether or not triggered
        """
        if os.path.exists(self.watch_file):
            try:
                with open(self.watch_file, 'r') as f:
                    data = json.load(f)
                if data.get("calibrate"):
                    return True
                
            except Exception as e:
                self.get_logger().warn(f"Couldn't open file: {e}")
                
        return False

    def change_state(self, node_name, target_state):
        """
        Changes state of lifecycle node to desired state

        Args:
            node_name (str): name of lifecycle node
            target_state (str): desired state for node
        """

        # Getting change service for client
        client = self.lifecycle_clients[node_name]['change']

        # Obtaining transition id 
        req = ChangeState.Request()
        if target_state == 'activate':
            req.transition.id = Transition.TRANSITION_ACTIVATE
        
        elif target_state == 'configure':
            req.transition.id = Transition.TRANSITION_CONFIGURE

        elif target_state == 'deactivate':
            req.transition.id = Transition.TRANSITION_DEACTIVATE

        elif target_state == 'shutdown':
            req.transition.id = Transition.TRANSITION_ACTIVE_SHUTDOWN

        # Calling to service to trigger transition
        future = client.call_async(req)

        # Waiting for transition to finish
        rclpy.spin_until_future_complete(self, future)

        self.get_logger().info(f"Node {node_name} has succesfully been transitioned")


def main():
    rclpy.init()

    manager = Manager()

    try:
        manager.on_startup()
        manager.run()
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
