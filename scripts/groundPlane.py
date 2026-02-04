import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as Rot

class GroundPlane:
    """
    GroundPlane reprojects depth maps to point clouds, determines ground planes, and finds the distance between a point and the ground plane
    """
    def __init__(self, threshold=0.03, max_iter=10000):
        '''
        Initialises GroundPlane for accuracy

        Args:
            threshold (float): Threshold for points to be considered inliers
            max_iter (int): Max number of iterations for RANSAC
        '''
        self.threshold = threshold
        self.max_iter = max_iter

        self.depth = None

        self.map_x = None
        self.map_y = None

    def update_depth(self, depth_map):
        self.depth = depth_map
        

    def convert_depth_to_point_cloud(self, depth_map, fx, fy, cx, cy):
        '''
        Takes in a depth map and converts it to a point cloud, does not consider the extrinsics and world frame
        Args:
            depth_map (array): 2D array containing depth information
            fx (float): focal length in x direction
            fy (float): focal length in y direction
            cx (float): centre of image in x
            cy (float): centre of imag in y
        Returns:
            point cloud normalised to meters
        '''

        height, width = depth_map.shape
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))

        z = depth_map
        x = (u_coords - cx) * z / fx
        y = (v_coords - cy) * z / fy

        cloud = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
        return cloud/1000.0
    
    def reproject_point(self, depth_map, K, u, v):
        '''
        Reprojects a single depth point to a point in space from the cameras perspective
        Args:
            depth_map (array): 2D array containing depth information
            fx (float): focal length in x direction
            fy (float): focal length in y direction
            cx (float): centre of image in x
            cy (float): centre of imag in y
            u (int): x index of depth map
            y (int): y index of depth map
        Return:
            x, y, z coordinates of point
        '''
        # Getting focal length and centre points
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]

        # Finding 3D point
        z = depth_map[v, u] / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        return x, y, z
    
    def get_normal_distance(self, point, plane):
        '''
        Returns normal distance between a point and a plane
        Args:
            point (array): array containing [x,y,z] coordinates
            plane (array): array containing [a, b, c, d] of plane

        Returns:
            normal distance
        '''
        x, y, z = point
        a, b, c, d = plane

        num = a*x + b*y + c*z + d
        den = math.sqrt(a*a + b*b + c*c)

        return abs(num/den)

    def fit_ground_plane(self, point_cloud):
        """
        finds the ground plane given a point cloud (largest flat area)
        Args:
            point_cloud: point cloud of area
        Returns:
            number of inliers and plane equation in form [A, B, C, D]
        """
        best_plane = None
        best_inliers=np.array([])

        for _ in range(self.max_iter):
            indices = np.random.choice(point_cloud.shape[0], 3, replace=False)
            p1, p2, p3 = point_cloud[indices]

            normal_plane = np.cross(p2 - p1, p3 - p1)
            if np.linalg.norm(normal_plane) == 0:
                continue
            normal_plane = normal_plane / np.linalg.norm(normal_plane)

            offset = -np.dot(normal_plane, p1)
            distances = np.abs(point_cloud @ normal_plane + offset)

            inliers = np.where(distances < self.threshold)[0]

            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = np.append(normal_plane, offset)

        return best_inliers, best_plane
    
    def precompute_rgb_alignment(self, depth_shape, rgb_int, depth_int, rgb_ext, depth_ext, live = False):
        """
        Precompute rgb alignment determins the mapping from rgb to depth, which is always the same (at least pretty much the same lol)

        Args:
            depth_shape: shape of the depth map
            rgb_int: RGB Camera Intrinsic Information
            depth_int: Depth Camera intrinsic information
            rgb_ext: RGB to base extrinsic information
            depth_ext: Depth to base extrinsic information

        Returns:
            map_x and map_y
        """
        
        K_rgb = np.array(rgb_int['K']).reshape(3,3)
        K_depth = np.array(depth_int['K']).reshape(3,3)

        # Translation matrix for rgb
        t_rgb = np.array(rgb_ext['translation']).reshape(3,1)
        r_rgb = Rot.from_quat([rgb_ext['rotation_quaternion'][1], rgb_ext['rotation_quaternion'][2], rgb_ext['rotation_quaternion'][3], rgb_ext['rotation_quaternion'][0]]).as_matrix()
        T_rgb = np.eye(4)
        T_rgb[:3,:3] = r_rgb
        T_rgb[:3,3:] = t_rgb

        # Translation matrix for depth
        t_depth = np.array(depth_ext['translation']).reshape(3,1)
        r_depth = Rot.from_quat([depth_ext['rotation_quaternion'][1], depth_ext['rotation_quaternion'][2], depth_ext['rotation_quaternion'][3], depth_ext['rotation_quaternion'][0]]).as_matrix()
        T_depth = np.eye(4)
        T_depth[:3,:3] = r_depth
        T_depth[:3,3:] = t_depth

        # Depth to RGB Translation 
        T_depth_rgb = np.linalg.inv(T_rgb) @ T_depth
        
        height, width = depth_shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        if not live:
            # Converting depth map to point cloud
            z = np.ones_like(u, dtype=np.float32)
            x = (u - K_depth[0,2]) * z / K_depth[0,0] # x = (u - cx) * z / fx
            y = (v - K_depth[1, 2]) * z / K_depth[1,1] # y = (v - cy) * z / fy
        else:
            z = self.depth.astype(np.float32) / 1000.0 # Convert mm to m

            # Converting depth map to point cloud
            valid = (z > 0)
            x = np.where(valid, (u - K_depth[0,2]) * z / K_depth[0,0], 0)
            y = np.where(valid, (v - K_depth[1,2]) * z / K_depth[1,1], 0)
        depth_points = np.stack((x, y, z), axis=-1).reshape(-1,3).T

        rgb_points = T_depth_rgb[:3,:3] @ depth_points + T_depth_rgb[:3, 3].reshape(3,1)

        u_rgb = (rgb_points[0,:] / rgb_points[2,:]) * K_rgb[0,0] + K_rgb[0,2]
        v_rgb = (rgb_points[1,:] / rgb_points[2,:]) * K_rgb[1,1] + K_rgb[1,2]

        self.map_x = u_rgb.reshape(height, width).astype(np.float32)
        self.map_y = v_rgb.reshape(height, width).astype(np.float32)

        return self.map_x, self.map_y
    
    def align_rgb(self, img):
        """
        Align rgb image to the depth map using precomputed mapping
        Args:
            img: RGB Image
        """
        return cv2.remap(img, self.map_x, self.map_y, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)
