#!/usr/bin/env python3

import numpy as np
from scipy.spatial import distance, transform
from scipy.ndimage import binary_dilation
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray ###
from geometry_msgs.msg import Point ###

class MultiLanePurePursuit(Node):
    def __init__(self):
        super().__init__('multi_lane_pure_pursuit_node')

        # ===== Parameters =====
        self.is_real = False
        self.map_name = 'E1_out2_refined1'
        self.L = 1.0  # Lookahead distance
        self.steering_gain = 0.5
        self.ref_speed = 4.0
        self.max_sight = 4.0
        self.steering_limit = 0.4  # radians

        # Multi-lane parameters
        self.lane_offsets = [-0.6, -0.3, 0.0, 0.3, 0.6]
        self.lanes = []
        self.current_lane_idx = 2  # Start with the center lane
        self.hysteresis_counter = 0
        self.hysteresis_threshold = 3

        # ===== Local Occupancy Grid Parameters (LiDAR-based) =====
        self.grid_res = 0.05  # [m/cell]
        self.grid_forward = 2.2  # [m]
        self.grid_side = 1.0  # [m]
        self.inflate_radius_m = 0.15  # [m]
        self.inflate_iters = max(1, int(self.inflate_radius_m / self.grid_res))

        self.grid_w = int(self.grid_forward / self.grid_res)
        self.grid_h = int(self.grid_side / self.grid_res)
        self.grid_y_offset = self.grid_h // 2

        # ===== Topics/Publishers =====
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        self.sub_pose = self.create_subscription(Odometry, odom_topic, self.pose_callback, 1)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.drive_msg = AckermannDriveStamped()

        # Waypoints
        map_path = os.path.abspath(os.path.join('src', "f1tenth-software-stack", 'csv_data'))
        csv_data = np.loadtxt(f"{map_path}/{self.map_name}.csv", delimiter=',', skiprows=1)
        self.waypoints = csv_data[:, 0:2]
        self.numWaypoints = self.waypoints.shape[0]
        self._generate_lanes()
        self.active_waypoints = self.lanes[self.current_lane_idx]

        # Viz
        visualization_topic = '/visualization_marker_array'
        self.pub_vis = self.create_publisher(MarkerArray, visualization_topic, 1)
        self.markerArray = MarkerArray()

        self.visualization_init()

        # LiDAR
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Current state
        self.currX = 0.0
        self.currY = 0.0
        self.rot = np.eye(3)
        self.have_pose = False

        # Latest local grid
        self.local_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)

    def visualization_init(self):
        # Colors
        self.colors = [
            (0.0, 1.0, 0.0, 1.0),  # Green
            (0.5, 0.5, 1.0, 1.0),  # Light Blue
            (1.0, 1.0, 1.0, 1.0),  # White
            (1.0, 0.5, 0.5, 1.0),  # Light Red
            (1.0, 0.0, 0.0, 1.0),  # Red
        ]

        # Lane markers
        for i, lane in enumerate(self.lanes):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.type = Marker.POINTS
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = self.colors[i]
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.id = i
            marker.points = [Point(x=p[0], y=p[1], z=0.0) for p in lane]
            self.markerArray.markers.append(marker)

        # Active lane marker
        self.active_lane_marker = Marker()
        self.active_lane_marker.header.frame_id = 'map'
        self.active_lane_marker.type = Marker.POINTS
        self.active_lane_marker.color.r, self.active_lane_marker.color.g, self.active_lane_marker.color.b, self.active_lane_marker.color.a = (1.0, 1.0, 0.0, 1.0) # Yellow
        self.active_lane_marker.scale.x = 0.1
        self.active_lane_marker.scale.y = 0.1
        self.active_lane_marker.id = len(self.lanes)
        self.markerArray.markers.append(self.active_lane_marker)

        # Target marker
        self.targetMarker = Marker()
        self.targetMarker.header.frame_id = 'map'
        self.targetMarker.type = Marker.POINTS
        self.targetMarker.color.r = 1.0
        self.targetMarker.color.a = 1.0
        self.targetMarker.scale.x = 0.2
        self.targetMarker.scale.y = 0.2
        self.targetMarker.id = len(self.lanes) + 1
        self.markerArray.markers.append(self.targetMarker)


    # ============ Pure Pursuit ==================
    def pose_callback(self, msg):
        self.currX = msg.pose.pose.position.x
        self.currY = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True

        self._select_best_lane()
        self.drive_to_target_pure_pursuit()

    def drive_to_target_pure_pursuit(self):
        # Find the target point
        target_point = self.get_target_point(self.L)

        # Transform the target point to the car's coordinate frame
        translated_target_point = self.translate_point(target_point)

        # Calculate curvature/steering angle
        y = translated_target_point[1]
        gamma = self.steering_gain * (2 * y / self.L**2)
        angle = np.clip(gamma, -self.steering_limit, self.steering_limit)

        velocity = self.ref_speed

        self.drive_msg.drive.steering_angle = angle
        self.drive_msg.drive.speed = velocity
        self.pub_drive.publish(self.drive_msg)
        print(f"[Pure Pursuit] steer={round(angle, 3)}, speed={velocity:.2f}")

        # Update and publish visualization markers
        now = self.get_clock().now().to_msg()
        for marker in self.markerArray.markers:
            marker.header.stamp = now
        
        self.active_lane_marker.points = [Point(x=p[0], y=p[1], z=0.0) for p in self.active_waypoints]
        self.targetMarker.points = [Point(x=float(target_point[0]), y=float(target_point[1]), z=0.0)]
        
        self.pub_vis.publish(self.markerArray)

    def get_target_point(self, lookahead_dist):
        curr_pos = np.array([self.currX, self.currY]).reshape((1, 2))
        distances = distance.cdist(curr_pos, self.active_waypoints, 'euclidean').reshape((-1))
        closest_index = np.argmin(distances)

        point_index = closest_index
        dist = distances[point_index]

        while dist < lookahead_dist:
            point_index = (point_index + 1) % len(self.active_waypoints)
            dist = distances[point_index]

        return self.active_waypoints[point_index]

    def translate_point(self, target_point):
        # Create a 4x4 homogeneous transformation matrix
        H = np.zeros((4, 4))
        H[0:3, 0:3] = self.rot.T  # Transpose of rotation matrix for world to car frame
        H[0:3, 3] = -self.rot.T @ np.array([self.currX, self.currY, 0.0])
        H[3, 3] = 1.0

        # Convert target point to homogeneous coordinates
        target_homogeneous = np.array([target_point[0], target_point[1], 0.0, 1.0])

        # Apply the transformation
        transformed_target = H @ target_homogeneous

        return transformed_target[0:3]


    # ========================= Lane Generation and Selection =========================
    def _generate_lanes(self):
        """
        Generates parallel lanes based on the centerline waypoints.
        """
        self.lanes = []
        for offset in self.lane_offsets:
            lane = []
            for i in range(self.numWaypoints):
                p1 = self.waypoints[i]
                p2 = self.waypoints[(i + 1) % self.numWaypoints]
                
                # Tangent vector
                tangent = p2 - p1
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm > 0:
                    tangent = tangent / tangent_norm
                
                # Normal vector
                normal = np.array([-tangent[1], tangent[0]])
                
                new_point = p1 + offset * normal
                lane.append(new_point)
            self.lanes.append(np.array(lane))

    def _select_best_lane(self):
        """
        Selects the best lane based on collision checking and a cost function.
        """
        best_lane_idx = self.current_lane_idx
        min_cost = float('inf')

        for i, lane in enumerate(self.lanes):
            is_colliding = self._check_lane_for_collision(lane)
            if not is_colliding:
                cost = abs(self.lane_offsets[i]) # Simple cost: prefer center lane
                if cost < min_cost:
                    min_cost = cost
                    best_lane_idx = i
        
        # Hysteresis
        if best_lane_idx != self.current_lane_idx:
            self.hysteresis_counter += 1
            if self.hysteresis_counter >= self.hysteresis_threshold:
                self.current_lane_idx = best_lane_idx
                print(f"[Lane Change] Switched to lane {self.current_lane_idx} (offset: {self.lane_offsets[self.current_lane_idx]}m)")
                self.hysteresis_counter = 0
        else:
            self.hysteresis_counter = 0
            
        self.active_waypoints = self.lanes[self.current_lane_idx]

    def _check_lane_for_collision(self, lane):
        """
        Checks if a given lane collides with obstacles in the local grid.
        """
        # Find the closest point on the lane to the car
        dists = distance.cdist(np.array([[self.currX, self.currY]]), lane[:, 0:2]).reshape(-1)
        closest_idx = np.argmin(dists)

        # Check a horizon ahead
        check_horizon_m = 2.0
        check_horizon_indices = int(check_horizon_m / 0.1) # Assuming waypoints are ~0.1m apart

        for i in range(check_horizon_indices):
            point_idx = (closest_idx + i) % len(lane)
            point = lane[point_idx]
            
            # Transform point to local frame
            local_point = self._map_point_to_base_local(point)
            
            # Check if the point is within the grid
            gx, gy, in_bounds = self._local_point_to_grid(local_point[0], local_point[1])
            
            if in_bounds and self.local_grid[gy, gx] > 0:
                return True # Collision
        
        return False # No collision


    # ========================= Local Grid Helpers =========================
    def scan_callback(self, scan_msg: LaserScan):
        self._build_local_grid_from_scan(scan_msg)

    def _build_local_grid_from_scan(self, scan: LaserScan):
        if not self.have_pose:
            return

        grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
        
        angle = scan.angle_min
        for r in scan.ranges:
            if np.isinf(r) or np.isnan(r) or r <= 0.0 or r > self.max_sight:
                angle += scan.angle_increment
                continue

            x = r * np.cos(angle)
            y = r * np.sin(angle)
            gi, gj, inb = self._local_point_to_grid(x, y)
            if inb:
                grid[gj, gi] = 1  # hit

            angle += scan.angle_increment

        grid = binary_dilation(grid, iterations=self.inflate_iters).astype(np.uint8) * 100
        self.local_grid = grid

    def _local_point_to_grid(self, x_local, y_local):
        if x_local < 0.0 or x_local > self.grid_forward:
            return 0, 0, False
        if abs(y_local) > (self.grid_side / 2.0):
            return 0, 0, False
        i = int(x_local / self.grid_res)
        j = int(self.grid_y_offset - (y_local / self.grid_res))
        inb = (0 <= i < self.grid_w) and (0 <= j < self.grid_h)
        return i, j, inb

    def _map_point_to_base_local(self, pt_xy):
        R_wb = self.rot  # body->world
        R_bw = R_wb.T  # world->body
        p_w = np.array([pt_xy[0] - self.currX, pt_xy[1] - self.currY, 0.0])
        p_b = R_bw @ p_w
        return np.array([p_b[0], p_b[1]])


def main(args=None):
    rclpy.init(args=args)
    node = MultiLanePurePursuit()
    print("[INFO] Multi Lane Pure Pursuit Node initialized")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
