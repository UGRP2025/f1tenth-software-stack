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
from geometry_msgs.msg import Point, PointStamped, QuaternionStamped, TransformStamped, PoseStamped

class MultiLanePurePursuit(Node):
    def __init__(self):
        super().__init__('multi_lane_pure_pursuit_node')

        # ===== Parameters =====
        self.is_real = False
        self.map_name = 'E1_out2_refined'
        self.steering_gain = 0.5
        self.speed_reducing_rate = 0.6
        self.max_sight = 4.0
        self.steering_limit = 0.35  # radians

        # Speed-dependent lookahead
        self.lookahead_norm = 2.0 # Lookahead for normal speed
        self.lookahead_slow = 1.8 # Lookahead for decelerated speed
        self.wheelbase = self.lookahead_norm  # [m]

        # Multi-lane parameters
        self.lane_offsets = [-0.8, -0.4, 0.0, 0.4, 0.8]
        self.lanes = []
        self.current_lane_idx = 2  # Start with the center lane
        self.hysteresis_counter = 0
        self.hysteresis_threshold = 3
        self.opposite_lane_penalty = 1.0  # Penalty for choosing opposite lane
        self.lane_switch_timer = 0
        self.lane_switch_cooldown = 30 # cycles
        self.safety_threshold = 0.3 # meters

        # ===== Topics/Publishers =====
        odom_topic = '/pf/viz/inferred_pose' if self.is_real else '/ego_racecar/odom'
        self.sub_pose = self.create_subscription(PoseStamped if self.is_real else Odometry, odom_topic, self.pose_callback, 1)

        self.pub_drive = self.create_publisher(AckermannDriveStamped, '/drive', 1)
        self.drive_msg = AckermannDriveStamped()

        # Waypoints
        map_path = os.path.abspath(os.path.join('src', "f1tenth-software-stack", 'csv_data'))
        csv_data = np.loadtxt(f"{map_path}/{self.map_name}.csv", delimiter=';', skiprows=1)
        self.waypoints = csv_data[:, 1:3]
        if self.is_real:
            self.ref_speed = csv_data[:, 5] * 0.6
        else:
            self.ref_speed = 4.0
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
        self.centerline_closest_index = 0

        # Obstacle points from LiDAR
        self.obstacle_points = None

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
    def pose_callback(self, pose_msg):
        self.currX = pose_msg.pose.position.x if self.is_real else pose_msg.pose.pose.position.x
        self.currY = pose_msg.pose.position.y if self.is_real else pose_msg.pose.pose.position.y
        quat = pose_msg.pose.orientation if self.is_real else pose_msg.pose.pose.orientation
        quat = [quat.x, quat.y, quat.z, quat.w]
        self.rot = transform.Rotation.from_quat(quat).as_matrix()
        self.have_pose = True

        # Find closest waypoint on centerline for speed profile
        curr_pos = np.array([self.currX, self.currY]).reshape((1, 2))
        distances = distance.cdist(curr_pos, self.waypoints, 'euclidean').reshape((-1))
        self.centerline_closest_index = np.argmin(distances)

        decelerate = self._select_best_lane()
        self.drive_to_target_pure_pursuit(decelerate)

    def drive_to_target_pure_pursuit(self, decelerate=False):
        # Determine lookahead distance and velocity
        if self.is_real:
            base_velocity = self.ref_speed[self.centerline_closest_index]
        else:
            base_velocity = self.ref_speed

        if decelerate:
            lookahead_dist = self.lookahead_slow
            velocity = base_velocity * self.speed_reducing_rate
            print("[Deceleration] Obstacle prompted lane change, reducing speed.")
        else:
            lookahead_dist = self.lookahead_norm
            velocity = base_velocity

        # Find the target point
        target_point = self.get_target_point(lookahead_dist)

        # Transform the target point to the car's coordinate frame
        translated_target_point = self.translate_point(target_point)

        # Calculate curvature/steering angle
        y = translated_target_point[1]
        gamma = self.steering_gain * (2 * y / self.wheelbase**2)
        angle = np.clip(gamma, -self.steering_limit, self.steering_limit)

        self.drive_msg.drive.steering_angle = angle
        self.drive_msg.drive.speed = velocity
        self.pub_drive.publish(self.drive_msg)
        print(f"[Pure Pursuit] steer={round(angle, 3)}, speed={velocity:.2f}, lookahead={lookahead_dist:.1f}")

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
        Selects the best lane based on clearance from obstacles.
        Returns whether to decelerate.
        """
        original_lane_idx = self.current_lane_idx

        if self.lane_switch_timer > 0:
            self.lane_switch_timer -= 1

        # Calculate clearance for all lanes
        lane_clearances = [self._get_lane_clearance(lane) for lane in self.lanes]

        # If current lane is safe enough and we are in a cooldown period, stay in the lane.
        if lane_clearances[self.current_lane_idx] > self.safety_threshold and self.lane_switch_timer > 0:
            self.active_waypoints = self.lanes[self.current_lane_idx]
            return False

        # Choose the lane with the largest clearance
        best_lane_idx_by_clearance = np.argmax(lane_clearances)
        
        # Hysteresis and lane change logic
        lane_changed = False
        if best_lane_idx_by_clearance != self.current_lane_idx:
            self.hysteresis_counter += 1
            if self.hysteresis_counter >= self.hysteresis_threshold:
                self.current_lane_idx = best_lane_idx_by_clearance
                print(f"[Lane Change] Switched to lane {self.current_lane_idx} (offset: {self.lane_offsets[self.current_lane_idx]}m) with clearance {lane_clearances[self.current_lane_idx]:.2f}m")
                self.hysteresis_counter = 0
                lane_changed = True
                self.lane_switch_timer = self.lane_switch_cooldown # Reset cooldown on lane change
        else:
            self.hysteresis_counter = 0
            
        self.active_waypoints = self.lanes[self.current_lane_idx]

        # Determine if deceleration is needed
        original_lane_was_unsafe = lane_clearances[original_lane_idx] < self.safety_threshold
        decelerate = (lane_changed and original_lane_was_unsafe) or (lane_clearances[self.current_lane_idx] < self.safety_threshold)
        return decelerate

    def _get_lane_clearance(self, lane):
        """
        Calculates the minimum clearance of a lane from obstacles.
        """
        if self.obstacle_points is None or len(self.obstacle_points) == 0:
            return float('inf') # No obstacles, lane is perfectly clear

        # Find the closest point on the lane to the car
        dists = distance.cdist(np.array([[self.currX, self.currY]]), lane[:, 0:2]).reshape(-1)
        closest_idx = np.argmin(dists)

        # Check a horizon ahead
        check_horizon_m = 4.0
        
        p1 = lane[0]
        p2 = lane[1]
        waypoint_spacing = np.linalg.norm(p2-p1)
        if waypoint_spacing < 0.01: waypoint_spacing = 0.1

        check_horizon_indices = int(check_horizon_m / waypoint_spacing)

        min_clearance = float('inf')

        lane_points_to_check = []
        for i in range(check_horizon_indices):
            point_idx = (closest_idx + i) % len(lane)
            point_on_lane_world = lane[point_idx]
            
            # Transform lane point to local frame
            point_on_lane_local = self._map_point_to_base_local(point_on_lane_world)
            lane_points_to_check.append(point_on_lane_local)

        if not lane_points_to_check:
            return min_clearance

        # Calculate distances from all relevant lane points to all obstacle points
        all_dists = distance.cdist(np.array(lane_points_to_check), self.obstacle_points)
        
        min_clearance = np.min(all_dists)
        
        return min_clearance


    # ========================= LiDAR Helpers =========================
    def scan_callback(self, scan_msg: LaserScan):
        if not self.have_pose:
            return

        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment

        # Filter out invalid ranges and those too far away
        valid_indices = np.where((ranges > scan_msg.range_min) & (ranges < self.max_sight))[0]
        
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]

        # Convert to cartesian coordinates (in car's frame)
        x = valid_ranges * np.cos(valid_angles)
        y = valid_ranges * np.sin(valid_angles)
        
        self.obstacle_points = np.vstack((x, y)).T

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
