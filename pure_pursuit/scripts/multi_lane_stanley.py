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

class MultiLaneStanley(Node):
    def __init__(self):
        super().__init__('multi_lane_stanley_node')

        # ===== Parameters =====
        self.is_real = False
        self.map_name = 'E1_out2_refined1'
        self.L = 1.0
        self.steering_gain = 0.5
        self.ref_speed = 4.0
        self.max_sight = 4.0

        # Stanley parameters
        self.K_E = 0.7  # Crosstrack error gain
        self.K_H = 1.0  # Heading error gain
        self.steering_limit = 0.4  # radians
        self.velocity_percentage = 1.0

        # Multi-lane parameters
        self.lane_offsets = [-0.8, -0.6, 0.0, 0.6, 0.8]
        self.lanes = []
        self.current_lane_idx = 2  # Start with the center lane
        self.hysteresis_counter = 0
        self.hysteresis_threshold = 3

        # ===== Local Occupancy Grid Parameters (LiDAR-based) =====
        self.grid_res = 0.05  # [m/cell]
        self.grid_forward = 3.0  # [m]
        self.grid_side = 2.0  # [m]
        self.inflate_radius_m = 0.30  # [m]
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


    # ============ Pure Pursuit / Stanley ==================
    def pose_callback(self, msg):
        self.currX = msg.pose.pose.position.x
        self.currY = msg.pose.pose.position.y
        quat = msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True

        self._select_best_lane()
        self.drive_to_target_stanley()

    def drive_to_target_stanley(self):
        """
        Using the stanley method derivation. 
        """
        # Get current heading from rotation matrix
        current_heading = transform.Rotation.from_matrix(self.rot).as_euler('xyz')[2]

        # Front axle position
        front_x = self.currX + self.L * np.cos(current_heading)
        front_y = self.currY + self.L * np.sin(current_heading)
        front_axle_pos = np.array([front_x, front_y])

        # Find closest point on path to front and rear axles
        closest_point_front_car, closest_point_front_world, _ = self._get_closest_point_on_path(front_axle_pos)
        _, closest_point_rear_world, _ = self._get_closest_point_on_path(np.array([self.currX, self.currY]))

        path_heading = np.arctan2(closest_point_front_world[1] - closest_point_rear_world[1], 
                                  closest_point_front_world[0] - closest_point_rear_world[0])

        # Normalize angles
        if current_heading < 0: current_heading += 2 * np.pi
        if path_heading < 0: path_heading += 2 * np.pi

        # Crosstrack error (y-value in car frame, relative to front axle)
        crosstrack_error = closest_point_front_car[1]
        crosstrack_error_term = np.arctan2(self.K_E * crosstrack_error, self.ref_speed)
        
        # Heading error
        heading_error = path_heading - current_heading
        if heading_error > np.pi: heading_error -= 2 * np.pi
        elif heading_error < -np.pi: heading_error += 2 * np.pi
        heading_error *= self.K_H

        # Stanley controller formula
        angle = heading_error + crosstrack_error_term
        angle = np.clip(angle, -self.steering_limit, self.steering_limit)
        
        velocity = self.ref_speed * self.velocity_percentage

        self.drive_msg.drive.steering_angle = angle
        self.drive_msg.drive.speed = velocity
        self.pub_drive.publish(self.drive_msg)
        print(f"[Stanley] steer={round(angle, 3)}, speed={velocity:.2f}, cte={crosstrack_error:.2f}, he={heading_error:.2f}")

        # Update and publish visualization markers
        now = self.get_clock().now().to_msg()
        for marker in self.markerArray.markers:
            marker.header.stamp = now
        
        self.active_lane_marker.points = [Point(x=p[0], y=p[1], z=0.0) for p in self.active_waypoints]
        self.targetMarker.points = [Point(x=float(closest_point_front_world[0]), y=float(closest_point_front_world[1]), z=0.0)]
        
        self.pub_vis.publish(self.markerArray)

    def _get_closest_point_on_path(self, pos):
        # Find closest waypoint to pos
        dists = distance.cdist(pos.reshape(1, -1), self.active_waypoints[:, 0:2], 'euclidean').reshape(-1)
        closest_idx = np.argmin(dists)

        # Consider two segments around the closest waypoint
        prev_idx = (closest_idx - 1 + len(self.active_waypoints)) % len(self.active_waypoints)
        next_idx = (closest_idx + 1) % len(self.active_waypoints)

        # Segment 1: prev_idx to closest_idx
        p1_seg1 = self.active_waypoints[prev_idx, 0:2]
        p2_seg1 = self.active_waypoints[closest_idx, 0:2]
        l2_seg1 = np.sum((p1_seg1 - p2_seg1)**2)
        if l2_seg1 == 0.0:
            closest_point1 = p1_seg1
        else:
            t = max(0, min(1, np.dot(pos - p1_seg1, p2_seg1 - p1_seg1) / l2_seg1))
            closest_point1 = p1_seg1 + t * (p2_seg1 - p1_seg1)
        dist1 = np.linalg.norm(pos - closest_point1)

        # Segment 2: closest_idx to next_idx
        p1_seg2 = self.active_waypoints[closest_idx, 0:2]
        p2_seg2 = self.active_waypoints[next_idx, 0:2]
        l2_seg2 = np.sum((p1_seg2 - p2_seg2)**2)
        if l2_seg2 == 0.0:
            closest_point2 = p1_seg2
        else:
            t = max(0, min(1, np.dot(pos - p1_seg2, p2_seg2 - p1_seg2) / l2_seg2))
            closest_point2 = p1_seg2 + t * (p2_seg2 - p1_seg2)
        dist2 = np.linalg.norm(pos - closest_point2)

        if dist1 < dist2:
            closest_point_world = closest_point1
            path_segment_start = p1_seg1
            path_segment_end = p2_seg1
        else:
            closest_point_world = closest_point2
            path_segment_start = p1_seg2
            path_segment_end = p2_seg2

        # Transform closest point to car frame (origin at pos)
        yaw = transform.Rotation.from_matrix(self.rot).as_euler('xyz')[2]
        R_wc = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        R_cw = R_wc.T
        
        p_relative = closest_point_world - pos
        closest_point_car = R_cw @ p_relative

        # Path heading
        path_vector = path_segment_end - path_segment_start
        path_heading = np.arctan2(path_vector[1], path_vector[0])

        return closest_point_car, closest_point_world, path_heading

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
    node = MultiLaneStanley()
    print("[INFO] Multi Lane Stanley Node initialized")
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
