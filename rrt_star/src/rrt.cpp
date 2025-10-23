// This file contains the class definition of tree nodes and RRT
// Before you start, please read: https://arxiv.org/pdf/1105.1186.pdf
// Make sure you have read through the header file as well
#include "rrt/rrt.h"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include <Eigen/Geometry>
#include <Eigen/Dense>
#include "std_msgs/msg/bool.hpp"
#include <math.h>
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <algorithm>


// Destructor of the RRT class
RRT::~RRT() {
    // Do something in here, free up used memory, print message, etc.
    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "RRT shutting down");
}

// Constructor of the RRT class
RRT::RRT(): rclcpp::Node("rrt_node"), gen((std::random_device())()) {
    // ROS publishers
    grid_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rrt_occugrid_rviz",1);
    rrt_rviz = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_node_connections_rviz",1);
    rrt_path_rviz = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_path_connections_rviz",1);
    goal_pub = this->create_publisher<visualization_msgs::msg::Marker>("/rrt_goal_point_rviz",1);
    node_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/rrt_all_nodes_rviz",1);
    path_pub = this->create_publisher<visualization_msgs::msg::MarkerArray>("/rrt_path_nodes_rviz",1);
    grid_path_pub = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/rrt__path_occugrid_rviz",1);
    use_rrt_pub = this->create_publisher<std_msgs::msg::Bool>("/use_rrt",1);
    local_goal_pub = this->create_publisher<nav_msgs::msg::Odometry>("/rrt_local_goal",1);
    spline_points_pub = this->create_publisher<geometry_msgs::msg::PoseArray>("/rrt_spline_points",1);
    drive_pub_ = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("/drive", 1);

    // ROS subscribers
    string pose_topic = "ego_racecar/odom";
    string scan_topic = "/scan";
    string global_goal_topic = "/global_goal_pure_pursuit";
    string standoff_topic = "/standoff_dist_pure_pursuit";
    string use_rrt_topic = "/use_rrt";

    pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        pose_topic, 1, std::bind(&RRT::pose_callback, this, std::placeholders::_1));

    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        scan_topic, 1, std::bind(&RRT::scan_callback, this, std::placeholders::_1));

    global_goal_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        global_goal_topic, 1, std::bind(&RRT::global_goal_callback, this, std::placeholders::_1));

    pure_pursuit_standoff_ = this->create_subscription<std_msgs::msg::Float32>(
        standoff_topic, 1, std::bind(&RRT::standoff_callback, this, std::placeholders::_1));

    use_rrt_sub_ = this->create_subscription<std_msgs::msg::Bool>(
        use_rrt_topic, 1, std::bind(&RRT::use_rrt_callback, this, std::placeholders::_1));

    previous_time = rclcpp::Clock().now();

    RCLCPP_INFO(rclcpp::get_logger("RRT"), "%s\n", "Created new RRT Object.");

    // Load waypoints and create spline for pure pursuit
    load_waypoints("E1_out2_refined1");
    create_global_spline();

    this->declare_parameter<double>("wheelbase", 0.33);
    this->declare_parameter<double>("max_steering_angle", 0.4);
    this->declare_parameter<double>("max_speed", 3.5);
    this->declare_parameter<double>("min_speed", 1.0);

    this->get_parameter("wheelbase", wheelbase_);
    this->get_parameter("max_steering_angle", max_steering_angle_);
    this->get_parameter("max_speed", max_speed_);
    this->get_parameter("min_speed", min_speed_);

    this->declare_parameter<int>("rrt_max_iter", 500);
    this->declare_parameter<double>("rrt_max_expansion_dist", 0.5);
    this->declare_parameter<double>("rrt_goal_threshold", 0.1);
    this->declare_parameter<double>("rrt_near_gamma", 2.0);

    this->get_parameter("rrt_max_iter", rrt_max_iter_);
    this->get_parameter("rrt_max_expansion_dist", rrt_max_expansion_dist_);
    this->get_parameter("rrt_goal_threshold", rrt_goal_threshold_);
    this->get_parameter("rrt_near_gamma", rrt_near_gamma_);

    this->declare_parameter<double>("rrt_update_rate", 0.04);
    this->declare_parameter<int>("rrt_activation_hits", 8);
    this->declare_parameter<double>("rrt_sample_radius", 3.0);
    this->declare_parameter<double>("rrt_goal_bias", 0.1);

    this->get_parameter("rrt_update_rate", rrt_update_rate_);
    this->get_parameter("rrt_activation_hits", rrt_activation_hits_);
    this->get_parameter("rrt_sample_radius", rrt_sample_radius_);
    this->get_parameter("rrt_goal_bias", rrt_goal_bias_);

    this->declare_parameter<double>("lookahead_distance", 3.0);
    this->get_parameter("lookahead_distance", lookahead_distance_);
}

void RRT::load_waypoints(const std::string& map_name) {
    std::string file_name = ament_index_cpp::get_package_share_directory("rrt_star") + "/" + map_name + ".csv";
    std::ifstream file(file_name);
    if (!file.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open waypoint file: %s", file_name.c_str());
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.front() == '#') continue;
        std::stringstream ss(line);
        std::string value_str;
        std::vector<double> row;
        while (std::getline(ss, value_str, ',')) {
            try {
                row.push_back(std::stod(value_str));
            } catch (const std::invalid_argument& e) {
                RCLCPP_ERROR(this->get_logger(), "Failed to convert string to double: %s", value_str.c_str());
            }
        }
        if(row.size() >= 2) {
            waypoints_.push_back(row);
        }
    }
    file.close();
    RCLCPP_INFO(this->get_logger(), "Loaded %zu waypoints.", waypoints_.size());
}

void RRT::create_global_spline() {
    if (waypoints_.size() < 2) {
        RCLCPP_ERROR(this->get_logger(), "Not enough waypoints to create a spline.");
        return;
    }

    std::vector<double> s, x, y;
    s.push_back(0.0);
    x.push_back(waypoints_[0][0]);
    y.push_back(waypoints_[0][1]);

    // Calculate arc length (s) for parameterization
    for (size_t i = 1; i < waypoints_.size(); ++i) {
        double dx = waypoints_[i][0] - waypoints_[i-1][0];
        double dy = waypoints_[i][1] - waypoints_[i-1][1];
        s.push_back(s.back() + std::sqrt(dx*dx + dy*dy));
        x.push_back(waypoints_[i][0]);
        y.push_back(waypoints_[i][1]);
    }

    // Create splines for x and y as a function of arc length s
    x_spline_.set_points(s, x);
    y_spline_.set_points(s, y);

    // Generate dense points along the spline for easier lookup
    global_spline_points_.clear();
    double total_length = s.back();
    for (double i = 0; i < total_length; i += 0.1) { // 10cm resolution
        RRT_Node p;
        p.x = x_spline_(i);
        p.y = y_spline_(i);
        global_spline_points_.push_back(p);
    }
    RCLCPP_INFO(this->get_logger(), "Global spline created with %zu points.", global_spline_points_.size());
}

void RRT::pure_pursuit_control() {
    if (global_spline_points_.empty()) {
        RCLCPP_WARN(this->get_logger(), "Global spline not available for Pure Pursuit.");
        return;
    }

    // 1. Find the closest point on the path to the vehicle
    int closest_idx = get_closest_point(global_spline_points_, current_car_pose.pose.pose.position);

    // 2. Find the goal point on the path, starting the search from the closest point
    RRT_Node global_goal_point = find_goal_point_on_path(global_spline_points_, closest_idx, lookahead_distance_);

    // 3. Transform the goal point to the vehicle's frame
    geometry_msgs::msg::Point goal_point_global_msg;
    goal_point_global_msg.x = global_goal_point.x;
    goal_point_global_msg.y = global_goal_point.y;
    goal_point_global_msg.z = 0.0;
    geometry_msgs::msg::Point transformed_goal = transform_point(goal_point_global_msg, current_car_pose.pose.pose);

    // 4. Calculate the steering angle
    double L_squared = lookahead_distance_ * lookahead_distance_;
    double y_local = transformed_goal.y;
    double Lwb = wheelbase_; // F1TENTH ~0.33 m
    double curvature = (2.0 * y_local) / L_squared;
    double steer = std::atan(Lwb * curvature);
    steer = std::max(-max_steering_angle_, std::min(steer, max_steering_angle_));

    double v = std::max(min_speed_, std::min( max_speed_ - 0.5*std::abs(steer)/max_steering_angle_ * 2.0, max_speed_ ));

    // 5. Create and publish the drive message
    auto drive_msg = std::make_unique<ackermann_msgs::msg::AckermannDriveStamped>();
    drive_msg->header.stamp = this->get_clock()->now();
    drive_msg->header.frame_id = "base_link"; // Or whatever the robot base frame is
    drive_msg->drive.steering_angle = static_cast<float>(steer);
    drive_msg->drive.speed = v;
    drive_pub_->publish(std::move(drive_msg));

    // Also publish the global goal for RRT to use if it needs to take over
    global_goal.pose.pose.position.x = global_goal_point.x;
    global_goal.pose.pose.position.y = global_goal_point.y;
    global_goal.pose.pose.position.z = 0.0;
}

int RRT::get_closest_point(const std::vector<RRT_Node>& points, const geometry_msgs::msg::Point& current_pos) {
    int closest_idx = 0;
    double min_dist = std::numeric_limits<double>::max();

    for (size_t i = 0; i < points.size(); ++i) {
        double dx = points[i].x - current_pos.x;
        double dy = points[i].y - current_pos.y;
        double dist = dx*dx + dy*dy; // Use squared distance for efficiency
        if (dist < min_dist) {
            min_dist = dist;
            closest_idx = i;
        }
    }
    return closest_idx;
}

RRT_Node RRT::find_goal_point_on_path(const std::vector<RRT_Node>& path, int& start_index, double lookahead_distance) {
    if (path.empty()) {
        RCLCPP_WARN(this->get_logger(), "Cannot find goal point on an empty path.");
        return RRT_Node(); // Return a default node
    }

    // Start searching from the given start_index to ensure we move forward along the path
    for (size_t i = start_index; i < path.size(); ++i) {
        double dx = path[i].x - current_car_pose.pose.pose.position.x;
        double dy = path[i].y - current_car_pose.pose.pose.position.y;
        double dist = std::sqrt(dx*dx + dy*dy);

        if (dist >= lookahead_distance) {
            start_index = i; // Update start_index for the next search cycle
            return path[i];
        }
    }

    // If no point is found far enough ahead (e.g., at the end of the path),
    // return the last point of the path as the goal.
    start_index = path.size() - 1;
    return path.back();
}

geometry_msgs::msg::Point RRT::transform_point(const geometry_msgs::msg::Point& point, const geometry_msgs::msg::Pose& pose) {
    Eigen::Quaterniond q(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = q.toRotationMatrix();
    T.translation() << pose.position.x, pose.position.y, pose.position.z;
    Eigen::Vector3d pg(point.x, point.y, point.z);
    Eigen::Vector3d pl = T.inverse() * pg;

    geometry_msgs::msg::Point transformed_point;
    transformed_point.x = pl.x();
    transformed_point.y = pl.y();
    transformed_point.z = pl.z();

    return transformed_point;
}

void RRT::use_rrt_callback(const std_msgs::msg::Bool::ConstSharedPtr use_rrt_msg) {
    use_rrt_ = use_rrt_msg->data;
}
void RRT::standoff_callback(const std_msgs::msg::Float32::ConstSharedPtr l_dist){
    lookahead_distance_ = l_dist->data;
}

void RRT::global_goal_callback(const nav_msgs::msg::Odometry::ConstSharedPtr goal_msg){
    global_goal = *goal_msg;
}

// Bresenham's Line Algorithm for a given origin and goal point in 2D space
std::vector<std::vector<int>> RRT::bresenhams_line_algorithm(int goal_point[2], int origin_point[2]){
    try{
        // Initialize start and end coordinates
        int x1 = origin_point[0];
        int y1 = origin_point[1];
        int x2 = goal_point[0];
        int y2 = goal_point[1];

        // Calculate the differences between the coordinates
        int y_diff = y2 - y1;
        int x_diff = x2 - x1;

        // Swap the coordinates if the slope is steep
        if (abs(y_diff) >= abs(x_diff)){
            x1 = origin_point[1];
            y1 = origin_point[0];
            x2 = goal_point[1];
            y2 = goal_point[0];
        }

        int intermediate;
        if(x1 > x2){
            intermediate = x1;
            x1 = x2;
            x2 = intermediate;

            intermediate = y1;
            y1 = y2;
            y2 = intermediate;
        }

        y_diff = y2 - y1;
        x_diff = x2 - x1;

        // Initialize the error term to half the difference in x
        int error = int(x_diff / 2);
        float ystep=-1;
        if(y1 < y2){
            ystep=1;
        }

        int y = y1;
        std::vector<std::vector<int>> output;
        for(int x=x1; x < x2+1 ;x++){
            std::vector<int> coords{x,y};
            // Swap back the coordinates if the slope was steep
            if (abs(y_diff) > abs(x_diff)){
                coords[0] = y;
                coords[1] = x;
            }
            output.push_back(coords);
            error -= abs(y_diff);
            if(error < 0){
                y+=ystep;
                error+=x_diff;
            }
        }
        if(abs(goal_point[0] - output.back()[0]) > 1 && abs(origin_point[0] - output.back()[0]) > 1){
            std::vector<std::vector<int>> newoutput;
            for(int i=0;i<output.size();i++){
                std::vector<int> newcoords{output[i][1],output[i][0]};
                newoutput.push_back(newcoords);
            }
            return newoutput;
        }
        else{
            return output;
        }    
    }
    catch(...){
        std::cout<<"bresenhams failed"<<std::endl;
    }
   
}

// Function to check if RRT should be activated based on obstacle data. Utilize the Bresenhams line algorithm
void RRT::check_to_activate_rrt(std::vector<signed char> &obstacle_data){
    try{
        // Convert current car pose orientation to a quaternion    
        Eigen::Quaterniond q;
        q.x()= current_car_pose.pose.pose.orientation.x;
        q.y()= current_car_pose.pose.pose.orientation.y;
        q.z()= current_car_pose.pose.pose.orientation.z;
        q.w()= current_car_pose.pose.pose.orientation.w;

        auto rotation_mat = q.normalized().toRotationMatrix();

        // Calculate the goal's local coordinates relative to the current car position
        float x_goal = global_goal.pose.pose.position.x - current_car_pose.pose.pose.position.x;
        float y_goal = global_goal.pose.pose.position.y - current_car_pose.pose.pose.position.y;

        Eigen::Vector3d shift_coords(x_goal, y_goal, 0);
        Eigen::Vector3d local_goal_ = rotation_mat.inverse() * shift_coords;

        // Define the goal and origin points in the grid
        int goal_point[2]={(local_goal_[0]/resolution)+center_x,(local_goal_[1]/resolution)+center_y};
        int origin_point[2]={center_x, center_y};
        std::vector<std::vector<int>> grid_interp_points;
       
        grid_interp_points = bresenhams_line_algorithm(goal_point,origin_point);

        //Make Interp Points Wider
        //Add 9 cm to each side
        int add_val_x = abs((0.09 / resolution));
        int add_val_y = abs((0.09 / resolution));

        if(add_val_x==0){
            add_val_x=1;
        }
        if(add_val_y==0){
            add_val_y=1;
        }

        // Expand the list of interpolation points to account for resolution
        int size_val= grid_interp_points.size();
        for(int i=0;i<size_val;i++){
            for(int j=-add_val_y;j<add_val_y;j++){
                for(int k=-add_val_x;k<add_val_x;k++){
                    if(grid_interp_points[i][0]+k >0 && grid_interp_points[i][0]+k <occu_grid_x_size){
                        if( grid_interp_points[i][1]+j >0 && grid_interp_points[i][1]+j <occu_grid_y_size){
                            int x_val = grid_interp_points[i][0]+k;
                            int y_val = grid_interp_points[i][1]+j;
                            std::vector<int> add_points{x_val,y_val};
                            grid_interp_points.push_back(add_points);
                        }
                    }
                }
            }
        }
        //Make Current Position Wider, in case there is an obstacle adjacent to the car
        for(int i=-5;i<5;i++){
            for(int j=-5;j<5;j++){
                if(i+center_x >0 && i+center_x <occu_grid_x_size){
                    if(j+center_y >0 && j+center_y <occu_grid_y_size){
                        int x_val = center_x+i;
                        int y_val = center_y+j;
                        std::vector<int> add_points{x_val,y_val};
                        grid_interp_points.push_back(add_points);
                    }
                }
            }
        }
        // Mark the grid points in the occupancy grid
        std::vector<signed char> listed_data(occu_grid_y_size * occu_grid_x_size);
        for(int i=0;i<grid_interp_points.size();i++){
            listed_data[((grid_interp_points[i][1])* occu_grid_x_size) + (grid_interp_points[i][0])]=100;
            occu_grid_flat[((grid_interp_points[i][1])* occu_grid_x_size) + (grid_interp_points[i][0])]=100;
        }
        bool rrt_use_it=false;
        int hit_count=0;
        for(int i=0;i<listed_data.size();i++){
            if(listed_data[i]==100 && i < obstacle_data.size() && obstacle_data[i]==100){
                hit_count++;
            }
        }
        if(hit_count >= rrt_activation_hits_){ // param: activation_hits
           
            rrt_use_it = true;
        }  

        auto new_grid= nav_msgs::msg::OccupancyGrid();

        new_grid.info.resolution=  resolution;
        new_grid.info.width=occu_grid_x_size;
        new_grid.info.height=occu_grid_y_size;

        std::string frame_id="map";
        new_grid.header.frame_id=frame_id;
        new_grid.header.stamp=rclcpp::Clock().now();
        new_grid.info.origin = current_car_pose.pose.pose;

        // Adjust the origin of the grid to match the current car position
        Eigen::Vector3d shift_coords1(center_x * resolution, center_y* resolution, 0);
        Eigen::Vector3d shift_in_global_coords = rotation_mat * shift_coords1;

        new_grid.info.origin.position.x-= shift_in_global_coords[0];
        new_grid.info.origin.position.y-= shift_in_global_coords[1];

        new_grid.data= listed_data;
        grid_path_pub->publish(new_grid);

        // Publish whether RRT should be used based on obstacle detection
        auto use_rrt= std_msgs::msg::Bool();
        use_rrt.data = rrt_use_it;
        use_rrt_pub->publish(use_rrt);
    }
    catch(...){
        std::cout<<"RRT ACTIVATION FAILED"<<std::endl;
    }
   
}

void RRT::scan_callback(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan_msg) {
    // The scan callback, update your occupancy grid here
    // Args:
    //    scan_msg (*LaserScan): pointer to the incoming scan message
    // Returns:
    //
    // TODO: update your occupancy grid
    int x_scan;
    int y_scan;
    std::memset(occu_grid, 0, sizeof occu_grid);
    std::vector<signed char> listed_data(occu_grid_y_size * occu_grid_x_size);
    for(int i=0; i<scan_msg->ranges.size(); i++){
        if (std::isnan(scan_msg->ranges[i])==false && std::isinf(scan_msg->ranges[i])==false && scan_msg->ranges[i]!=0){

            x_scan = scan_msg->ranges[i] * cos(scan_msg->angle_increment * i + scan_msg->angle_min) / resolution;
            y_scan = scan_msg->ranges[i] * sin(scan_msg->angle_increment * i + scan_msg->angle_min) / resolution;

            //Make the scans show up larger on the occupancy grid for visualization.
            for(int j=-2 + x_scan;j<2+ x_scan;j++){
                for(int k=-2 + y_scan;k<2+ y_scan;k++){
                    if(j+center_x >0 && j+center_x <occu_grid_x_size){
                        if(k+center_y >0 && k+center_y <occu_grid_y_size){
                            occu_grid[(j+center_x)][occu_grid_y_size-(k+center_y)]=100;
                            listed_data[((k  + center_y)* occu_grid_x_size) + (j + center_x)]=100;
                            occu_grid_flat[((k  + center_y)* occu_grid_x_size) + (j + center_x)]=100;
                        }
                    }
                }
            }
        }
    }

    auto new_grid= nav_msgs::msg::OccupancyGrid();

    new_grid.info.resolution=  resolution;
    new_grid.info.width=occu_grid_x_size;
    new_grid.info.height=occu_grid_y_size;

    std::string frame_id="map";
    new_grid.header.frame_id=frame_id;
    new_grid.header.stamp=rclcpp::Clock().now();
    new_grid.info.origin = current_car_pose.pose.pose;

    Eigen::Quaterniond q;
    q.x()= current_car_pose.pose.pose.orientation.x;
    q.y()= current_car_pose.pose.pose.orientation.y;
    q.z()= current_car_pose.pose.pose.orientation.z;
    q.w()= current_car_pose.pose.pose.orientation.w;

    auto rotation_mat = q.normalized().toRotationMatrix();

    // Apply rotation to shift them in global frame
    Eigen::Vector3d shift_coords(center_x * resolution, center_y* resolution, 0);
    Eigen::Vector3d shift_in_global_coords = rotation_mat * shift_coords;

    new_grid.info.origin.position.x-= shift_in_global_coords[0];
    new_grid.info.origin.position.y-= shift_in_global_coords[1];

    new_grid.data= listed_data;
    grid_pub->publish(new_grid);

    check_to_activate_rrt(listed_data);
}

std::vector<RRT_Node> RRT::perform_rrt(){
    try{
        std::cout<<"-----Starting RRT-----"<<std::endl;
        bool continue_searching = true;
        int loop_count=0;
        int increment = 2;
        int increment_val = 2;
        int max_loops=20;
        float x_goal;
        float y_goal;

        std::vector<RRT_Node> output_path;

        //Keep bringing the goal closer to the car until a viable path is found
        while(continue_searching == true){
            if(loop_count > max_loops){
                continue_searching= false;
            }
            std::vector<RRT_Node> tree;
            std::vector<double> sampled_point;
            std::vector<double> parent_vec;
            std::vector<RRT_Node> path;
            // TODO: fill in the RRT main loop
            // Add starting pose to the tree
            struct RRT_Node x_0 = {0.0, 0.0};
            x_0.parent = -1; //set the parent of the base node to -1 for detecting path completion in find_path()
            tree.push_back(x_0); //add to tree

            //Declare variables
            std::vector<double> x_rand;
            int x_nearest;
            bool collision;
            bool goal_flag;
            //For debugging purposes the goal point is just directly in front of the car

            Eigen::Quaterniond q;
            q.x()= current_car_pose.pose.pose.orientation.x;
            q.y()= current_car_pose.pose.pose.orientation.y;
            q.z()= current_car_pose.pose.pose.orientation.z;
            q.w()= current_car_pose.pose.pose.orientation.w;
            auto rotation_mat = q.normalized().toRotationMatrix();
       
            //Take the pure_pursuit goal in the initial loop
            if(loop_count==0){
                x_goal = global_goal.pose.pose.position.x - current_car_pose.pose.pose.position.x;
                y_goal = global_goal.pose.pose.position.y - current_car_pose.pose.pose.position.y;
            }
            else{//If pure_pursuit goal not drivable, start bringing the goal closer to the car along the spline
                float spline_index=-1000;
                for(int i=0;i<spline_points.size();i++){
                    if(abs(global_goal.pose.pose.position.x - spline_points[i][0]) < 0.1 && abs(global_goal.pose.pose.position.y - spline_points[i][1]) < 0.1){
                        spline_index=i;
                        break;
                    }
                }
                if (spline_index ==-1000){
                    //std::cout<<"No path was found, resorting back to original goal"<<std::endl;
                    x_goal = global_goal.pose.pose.position.x - current_car_pose.pose.pose.position.x;
                    y_goal = global_goal.pose.pose.position.y - current_car_pose.pose.pose.position.y;
                }
                else{
                    //Make sure the chosen spline index is a realistic value
                    if(spline_index - increment < 0){
                    increment = increment + spline_points.size();
                    }
                    if(spline_index - increment >= spline_points.size()){
                    increment = increment - spline_points.size();
                    }

                    x_goal = spline_points[spline_index - increment][0] - current_car_pose.pose.pose.position.x;
                    y_goal = spline_points[spline_index - increment][1] - current_car_pose.pose.pose.position.y;
                }
                float distance_to_goal = sqrt(pow(global_goal.pose.pose.position.x - current_car_pose.pose.pose.position.x,2)+pow(global_goal.pose.pose.position.y - current_car_pose.pose.pose.position.y,2));
                //std::cout<<"distance="<<distance_to_goal<<std::endl;
                //Don't let the closer goal end up behind the car
                if(abs(distance_to_goal) > 0.25){
                    increment+=increment_val;
                }
            }

            Eigen::Vector3d shift_coords(x_goal, y_goal, 0);
            Eigen::Vector3d local_goal_ = rotation_mat.inverse() * shift_coords;

            x_goal = local_goal_[0];
            y_goal = local_goal_[1];

            //std::cout<<"Local goal x: "<<x_goal <<", Local goal y: "<< y_goal <<std::endl;
            std::vector<RRT_Node> final_path;


            for (int i = 0; i < rrt_max_iter_; i++){
            struct RRT_Node x_new; //New node
            x_rand = sample(static_cast<double>(x_goal), static_cast<double>(y_goal)); //Create sampled node "x_rand"
            x_nearest = nearest(tree, x_rand); //closest neigbor in tree
            x_new = steer(tree[x_nearest], x_rand); //get new point
            x_new.parent = x_nearest; //Record the parent and add to the tree
            const auto current_node_index = tree.size();

            collision = check_collision(tree[x_nearest], x_new); //Check if the random node collides with the wall
            if(collision == false){
                x_new.cost = cost(tree, x_new);
                const auto near_neighbour_indices = near(tree, x_new); // Start performing RRT*
                std::vector<bool> is_near_neighbor_collided;
                int best_neighbor = x_new.parent; // initialize with the current parent of x_new
                for (const int near_node_index: near_neighbour_indices) { // Check every neighbor node in vicinity of current node
                    if (check_collision(tree[near_node_index], x_new)) { // check if clear connectivity exists between new node and nearest neighbor
                        is_near_neighbor_collided.push_back(true); // maintain true/false vector of the neighbors
                        continue; // evaluate other neighbors instead
                    }
                    is_near_neighbor_collided.push_back(false);

                    double cost = tree[near_node_index].cost + line_cost(tree[near_node_index], x_new); // claculate line cost for each new neighbor and add it to parent cost

                    if (cost < x_new.cost) {
// if there is a node in vicinity that has a clear connectivity, then update the cost, parent and store the best neighbor index
                        x_new.cost = cost;
                        x_new.parent = near_node_index;
                        best_neighbor = near_node_index;
                    }
                }

                for (int i = 0; i < near_neighbour_indices.size(); i++) {
                    int ni = near_neighbour_indices[i];
                    if (is_near_neighbor_collided[i] || ni == best_neighbor) { // ensure that nearest neighbor isnt colliding with new node, or isnt self
                        continue;
                    }
   // Reroute other neighbors to connect with the newest node if they are have a lower cost than their connection scheme.
   // This allows more efficient and directed path planning, with a well definied distance cost
                    if (tree[ni].cost > x_new.cost + line_cost(x_new, tree[ni])) {
                        tree[ni].parent = current_node_index;
                    }
                }

            tree.emplace_back(x_new);

            goal_flag = is_goal(x_new, x_goal, y_goal);
            if (goal_flag == true){
                std::cout<<"new point x: "<<x_new.x <<", new point y: "<< x_new.y <<std::endl;          
                found_path=true;
                path = find_path(tree, x_new);  // Find the path
                //Publish visualization stuff for RVIZ
                update_rrt_path_lines(tree, path);
                update_rrt_rviz(tree);

                update_goal_point(x_goal, y_goal);
                std::cout<<"***path found***"<<std::endl;
                output_path = path;
                continue_searching = false;
                break;
            }
            }
        }

        loop_count++;
        if(loop_count > max_loops){
            found_path = false;
            std::cout<<"No Path Found"<<std::endl;
        }

        }
        return output_path;
    }
    catch(...){
        std::cout<<"IN THE CATCH, RRT FUNCTION FAILED"<<std::endl;
    }
}

void RRT::pose_callback(const nav_msgs::msg::Odometry::ConstSharedPtr pose_msg) {
    current_car_pose = *pose_msg;

    if (use_rrt_) {
        RCLCPP_INFO_ONCE(this->get_logger(), "RRT mode activated.");
        // RRT LOGIC
        rclcpp::Time current_time = rclcpp::Clock().now();
        if((current_time - previous_time).seconds() > rrt_update_rate_){
            previous_time = current_time;
            found_path = false;
            final_path_output = perform_rrt();
        }

        if (found_path && !final_path_output.empty()) {
            // Transform the local RRT path to the global frame
            std::vector<RRT_Node> global_rrt_path;
            Eigen::Quaterniond q(current_car_pose.pose.pose.orientation.w, current_car_pose.pose.pose.orientation.x, current_car_pose.pose.pose.orientation.y, current_car_pose.pose.pose.orientation.z);
            auto rotation_mat = q.normalized().toRotationMatrix();

            for (const auto& local_node : final_path_output) {
                Eigen::Vector3d local_point(local_node.x, local_node.y, 0);
                Eigen::Vector3d global_point_offset = rotation_mat * local_point;
                
                RRT_Node global_node;
                global_node.x = current_car_pose.pose.pose.position.x + global_point_offset.x();
                global_node.y = current_car_pose.pose.pose.position.y + global_point_offset.y();
                global_rrt_path.push_back(global_node);
            }

            // Follow the RRT path using pure pursuit logic on the global path
            int closest_idx_rrt = get_closest_point(global_rrt_path, current_car_pose.pose.pose.position);
            RRT_Node rrt_goal_point = find_goal_point_on_path(global_rrt_path, closest_idx_rrt, lookahead_distance_ / 1.5); // Use a shorter lookahead for tighter control

            geometry_msgs::msg::Point goal_point_global_msg;
            goal_point_global_msg.x = rrt_goal_point.x;
            goal_point_global_msg.y = rrt_goal_point.y;
            goal_point_global_msg.z = 0.0;
            geometry_msgs::msg::Point transformed_goal = transform_point(goal_point_global_msg, current_car_pose.pose.pose);

            double lookahead_rrt = lookahead_distance_ / 1.5;
            double L_squared = lookahead_rrt * lookahead_rrt;
            double y_local = transformed_goal.y;
            double Lwb = wheelbase_; // F1TENTH ~0.33 m
            double curvature = (2.0 * y_local) / L_squared;
            double steer = std::atan(Lwb * curvature);
            steer = std::max(-max_steering_angle_, std::min(steer, max_steering_angle_));

            auto drive_msg = std::make_unique<ackermann_msgs::msg::AckermannDriveStamped>();
            drive_msg->header.stamp = this->get_clock()->now();
            drive_msg->header.frame_id = "base_link";
            drive_msg->drive.steering_angle = static_cast<float>(steer);
            drive_msg->drive.speed = 0.8f; // Slower speed for obstacle avoidance
            drive_pub_->publish(std::move(drive_msg));
        } else {
            // If RRT is active but no path is found, stop the car for safety.
            RCLCPP_WARN(this->get_logger(), "RRT active, but no path found. Stopping car.");
            auto drive_msg = std::make_unique<ackermann_msgs::msg::AckermannDriveStamped>();
            drive_msg->header.stamp = this->get_clock()->now();
            drive_msg->header.frame_id = "base_link";
            drive_msg->drive.steering_angle = 0.0f;
            drive_msg->drive.speed = 0.0f;
            drive_pub_->publish(std::move(drive_msg));
        }

    } else {
        RCLCPP_INFO_ONCE(this->get_logger(), "Pure Pursuit mode activated.");
        // PURE PURSUIT LOGIC
        pure_pursuit_control();
    }
}

std::vector<double> RRT::sample(double goal_x, double goal_y) {
    // This method returns a sampled point from the free space
    // You should restrict so that it only samples a small region
    // of interest around the car's current position
    // Args:
    // Returns:
    //     sampled_point (std::vector<double>): the sampled point in free space
    std::uniform_real_distribution<double> uni(0.0,1.0);
    if (uni(gen) < rrt_goal_bias_) {
        std::vector<double> goal_point = {goal_x, goal_y};
        return goal_point; // 10% goal bias (goal in local frame)
    }
    
    // forward cone sampling
    double R = rrt_sample_radius_;          // param: sampling radius [m]
    double th = (uni(gen)-0.5) * (M_PI/2.0); // +/-45°
    double r = std::uniform_real_distribution<double>(0.3, R)(gen);
    
    std::vector<double> sampled_point = { r*std::cos(th), r*std::sin(th) };
    return sampled_point;
}

int RRT::nearest(std::vector<RRT_Node> &tree, std::vector<double> &sampled_point) {
    // This method returns the nearest node on the tree to the sampled point
    // Args:
    //     tree (std::vector<RRT_Node>): the current RRT tree
    //     sampled_point (std::vector<double>): the sampled point in free space
    // Returns:
    //     nearest_node (int): index of nearest node on the tree

    int nearest_node = 0;
    float min_dist = 1000.0;

    double x = sampled_point[0];
    double y = sampled_point[1];
    double cur_x;
    double cur_y;
    float cur_dist;

    for(int i = 0; i < tree.size(); i++){
      //Calculate the distance between sampled point and tree node
      cur_x = tree[i].x;
      cur_y = tree[i].y;
      cur_dist = std::sqrt(std::pow(cur_x - x, 2) + std::pow(cur_y - y, 2));

      //Check if less than the best distance
      if (cur_dist < min_dist){
        min_dist = cur_dist;
        nearest_node = i;
      }
    }

    return nearest_node; //return parent point
}

RRT_Node RRT::steer(RRT_Node &nearest_node, std::vector<double> &sampled_point) {
    // The function steer:(x,y)->z returns a point such that z is “closer”
    // to y than x is. The point z returned by the function steer will be
    // such that z minimizes ||z−y|| while at the same time maintaining
    //||z−x|| <= max_expansion_dist, for a prespecified max_expansion_dist > 0

    // basically, expand the tree towards the sample point (within a max dist)

    // Args:
    //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
    //    sampled_point (std::vector<double>): the sampled point in free space
    // Returns:
    //    new_node (RRT_Node): new node created from steering

    RRT_Node new_node;

    double dx = sampled_point[0] - nearest_node.x;
    double dy = sampled_point[1] - nearest_node.y;
    double angle = atan2(dy,dx);

    double dx_new = rrt_max_expansion_dist_ * cos(angle);
    double dy_new = rrt_max_expansion_dist_ * sin(angle);

    new_node.x = dx_new + nearest_node.x;
    new_node.y = dy_new + nearest_node.y;


    return new_node;
}

bool RRT::check_collision(RRT_Node &nearest_node, RRT_Node &new_node) {
    // This method returns a boolean indicating if the path between the
    // nearest node and the new node created from steering is collision free
    // Args:
    //    nearest_node (RRT_Node): nearest node on the tree to the sampled point
    //    new_node (RRT_Node): new node created from steering
    // Returns:
    //    collision (bool): true if in collision, false otherwise

    float x0_f = center_x + (nearest_node.x / resolution) + 0.5; //added 0.5 to round to the proper number
    float y0_f = center_y - (nearest_node.y / resolution) + 0.5;

    float x1_f = center_x + (new_node.x / resolution) + 0.5;
    float y1_f = center_y - (new_node.y / resolution) + 0.5;

    int dx, dy, sx, sy, error, e2;
    int x0 = (int)x0_f;
    int y0 = (int)y0_f;
    int x1 = (int)x1_f;
    int y1 = (int)y1_f;


    dx = abs(x1 - x0);
    if (x0 < x1){
      sx = 1;
    } else{
      sx = -1;
    }
    dy = -abs(y1 - y0);
    if (y0 < y1){
      sy = 1;
    } else{
      sy = -1;
    }
    error = dx + dy;

    bool collision = false;
    while(1){
        if(!(0<=x0 && x0<occu_grid_x_size && 0<=y0 && y0<occu_grid_y_size)) {
          collision = true;
          break;
        }
        if(occu_grid[x0][y0] > 70){
          collision = true;
          break;
        }
        if(x0 == x1 && y0 == y1){
          break;
        }
        e2 = 2 * error;
        if(e2 >= dy){
            if(x0 == x1){
              break;
            }
            error = error + dy;
            x0 = x0 + sx;
        }
        if(e2 <= dx){
            if(y0 == y1){
              break;
            }
            error = error + dx;
            y0 = y0 + sy;
        }
    }
    return collision;
}

bool RRT::is_goal(RRT_Node &node, double goal_x, double goal_y) {
    // This method checks if the latest node added to the tree is close
    // enough (defined by goal_threshold) to the goal so we can terminate
    // the search and find a path
    // Args:
    //   latest_added_node (RRT_Node): latest addition to the tree
    //   goal_x (double): x coordinate of the current goal
    //   goal_y (double): y coordinate of the current goal
    // Returns:
    //   close_enough (bool): true if node close enough to the goal

    bool close_enough = false;
    double node_x = node.x;
    double node_y = node.y;

    if(sqrt(pow(goal_x-node_x,2) + pow(goal_y-node_y,2)) < rrt_goal_threshold_){
        close_enough = true;
    }
    return close_enough;
}

 std::vector<RRT_Node> RRT::find_path(std::vector<RRT_Node> &tree, RRT_Node &latest_added_node) {
    // This method traverses the tree from the node that has been determined
    // as goal
    // Args:
    //   latest_added_node (RRT_Node): latest addition to the tree that has been
    //      determined to be close enough to the goal
    // Returns:
    //   path (std::vector<RRT_Node>): the vector that represents the order of
    //      of the nodes traversed as the found path

    std::vector<RRT_Node> found_path;
    auto current_node = latest_added_node;  //initialize with the last point in parent array which corresponds to nearest node in rrt tree
    found_path.push_back(current_node); //store the nearest node

    while(1){ //terminate loop when stop loop is true
        int parent = current_node.parent;
        if(parent == -1){
          break;
        }
        current_node = tree[parent];
        std::cout<<"X, Y: "<<current_node.x<<", "<<current_node.y<<std::endl;
        found_path.push_back(current_node); //store the nearest node
    }
    return found_path;
}

// RRT* methods
double RRT::cost(std::vector<RRT_Node> &tree, RRT_Node &newnode) {
    // This method returns the cost associated with a node
    // Args:
    //    tree (std::vector<RRT_Node>): the current tree
    //    node (RRT_Node): the node the cost is calculated for
    // Returns:
    //    cost (double): the cost value associated with the node

    double cost = 0;
    cost = tree[newnode.parent].cost + line_cost(tree[newnode.parent], newnode);

    return cost;
}

double RRT::line_cost(RRT_Node &n1, RRT_Node &n2) {
    // This method returns the cost of the straight line path between two nodes
    // Args:
    //    n1 (RRT_Node): the RRT_Node at one end of the path
    //    n2 (RRT_Node): the RRT_Node at the other end of the path
    // Returns:
    //    cost (double): the cost value associated with the path

    double line_cost = 0;
    double n1x = n1.x;
    double n1y = n1.y;
    double n2x = n2.x;
    double n2y = n2.y;      

    line_cost = std::sqrt(std::pow(n1x - n2x, 2) + std::pow(n1y - n2y, 2));

    return line_cost;
}

std::vector<int> RRT::near(std::vector<RRT_Node> &tree, RRT_Node &node) {
    // This method returns the set of Nodes in the neighborhood of a
    // node.
    // Args:
    //   tree (std::vector<RRT_Node>): the current tree
    //   node (RRT_Node): the node to find the neighborhood for
    // Returns:
    //   neighborhood (std::vector<int>): the index of the nodes in the neighborhood

    std::vector<int> neighborhood;
    double gamma = rrt_near_gamma_;
    int d = 2;
    double num_nodes = static_cast<double>(tree.size());
    double r_near = gamma * std::pow(std::log(std::max(num_nodes, 2.0)) / std::max(num_nodes, 2.0), 1.0/d);
    double search_radius = std::max(0.5, std::min(r_near, 2.0));

    for(size_t i=0; i<tree.size(); i++){
        const double distance = std::sqrt(std::pow(node.x - tree[i].x, 2) + std::pow(node.y - tree[i].y, 2));
        if(distance < search_radius){
            neighborhood.push_back(i);
        }
    }
    return neighborhood;
}

//VIZUALIZATION
void RRT::update_rrt_rviz(std::vector<RRT_Node> &tree){
        auto message = visualization_msgs::msg::Marker();

        Eigen::Quaterniond q;
        q.x()= current_car_pose.pose.pose.orientation.x;
        q.y()= current_car_pose.pose.pose.orientation.y;
        q.z()= current_car_pose.pose.pose.orientation.z;
        q.w()= current_car_pose.pose.pose.orientation.w;
        auto rotation_mat = q.normalized().toRotationMatrix();

        Eigen::Vector3d shift_coords(tree[0].x, tree[0].y, 0);
        Eigen::Vector3d shift_in_global_coords = rotation_mat.inverse() * shift_coords;
        message.header.frame_id="map";
        message.type= visualization_msgs::msg::Marker::LINE_LIST;
        message.action = visualization_msgs::msg::Marker::ADD;
        message.scale.x= 0.050;
        message.pose.position.x= current_car_pose.pose.pose.position.x;
        message.pose.position.y= current_car_pose.pose.pose.position.y;
        message.pose.position.z=0.0;
        message.color.a=1.0;
        message.color.r=0.0;
        message.color.b=1.0;
        message.color.g=0.0;
        message.pose.orientation.x=0.0;
        message.pose.orientation.y=0.0;
        message.pose.orientation.z=0.0;
        message.pose.orientation.w=1.0;
        message.lifetime.nanosec=int(1e8);

        for(int i=1;i<tree.size();i++){
            message.header.stamp = rclcpp::Clock().now();
            message.id=i;

            Eigen::Vector3d shift_coords_pt1(float(tree[tree[i].parent].x), float(tree[tree[i].parent].y), 0);
            Eigen::Vector3d shift_in_global_coords_pt1 = rotation_mat * shift_coords_pt1;

            auto point1 = geometry_msgs::msg::Point();
            point1.x=shift_in_global_coords_pt1[0];
            point1.y=shift_in_global_coords_pt1[1];
            point1.z=0.0;

            Eigen::Vector3d shift_coords_pt2(float(tree[i].x), float(tree[i].y), 0);
            Eigen::Vector3d shift_in_global_coords_pt2 = rotation_mat * shift_coords_pt2;

            message.points.push_back(point1);

            auto point2=geometry_msgs::msg::Point();
            point2.x=shift_in_global_coords_pt2[0];
            point2.y=shift_in_global_coords_pt2[1];
            point2.z=0.0;

            message.points.push_back(point2);
            rrt_rviz->publish(message);
        }
}

void RRT::update_rrt_path_lines(std::vector<RRT_Node> &tree, std::vector<RRT_Node> &path){
        auto message = visualization_msgs::msg::Marker();

        Eigen::Quaterniond q;
        q.x()= current_car_pose.pose.pose.orientation.x;
        q.y()= current_car_pose.pose.pose.orientation.y;
        q.z()= current_car_pose.pose.pose.orientation.z;
        q.w()= current_car_pose.pose.pose.orientation.w;
        auto rotation_mat = q.normalized().toRotationMatrix();

        Eigen::Vector3d shift_coords(path[0].x, path[0].y, 0);
        Eigen::Vector3d shift_in_global_coords = rotation_mat * shift_coords;
        message.header.frame_id="map";
        message.type= visualization_msgs::msg::Marker::LINE_LIST;
        message.action = visualization_msgs::msg::Marker::ADD;
        message.scale.x= 0.050;
        message.pose.position.x= current_car_pose.pose.pose.position.x;
        message.pose.position.y= current_car_pose.pose.pose.position.y;
        message.pose.position.z=0.0;
        message.color.a=1.0;
        message.color.r=0.0;
        message.color.b=0.0;
        message.color.g=1.0;
        message.pose.orientation.x=0.0;
        message.pose.orientation.y=0.0;
        message.pose.orientation.z=0.0;
        message.pose.orientation.w=1.0;
        message.lifetime.nanosec=int(1e8);

        for(int i=1;i<path.size();i++){
            message.header.stamp = rclcpp::Clock().now();
            message.id=i;

            Eigen::Vector3d shift_coords_pt1(float(tree[path[i].parent].x), float(tree[path[i].parent].y), 0);
            Eigen::Vector3d shift_in_global_coords_pt1 = rotation_mat * shift_coords_pt1;

            auto point1 = geometry_msgs::msg::Point();
            point1.x=shift_in_global_coords_pt1[0];
            point1.y=shift_in_global_coords_pt1[1];
            point1.z=0.0;

            Eigen::Vector3d shift_coords_pt2(float(path[i].x), float(path[i].y), 0);
            Eigen::Vector3d shift_in_global_coords_pt2 = rotation_mat * shift_coords_pt2;

            message.points.push_back(point1);

            auto point2=geometry_msgs::msg::Point();
            point2.x=shift_in_global_coords_pt2[0];
            point2.y=shift_in_global_coords_pt2[1];
            point2.z=0.0;

            message.points.push_back(point2);
            rrt_path_rviz->publish(message);
        }
}

void RRT::update_path(std::vector<RRT_Node> &path){
        auto message = visualization_msgs::msg::MarkerArray();
        Eigen::Quaterniond q;
        q.x()= current_car_pose.pose.pose.orientation.x;
        q.y()= current_car_pose.pose.pose.orientation.y;
        q.z()= current_car_pose.pose.pose.orientation.z;
        q.w()= current_car_pose.pose.pose.orientation.w;
        auto rotation_mat = q.normalized().toRotationMatrix();

        for(int i=0; i<path.size();i++){
            //Find position of marker
            Eigen::Vector3d shift_coords(path[i].x, path[i].y, 0);
            Eigen::Vector3d shift_in_global_coords = rotation_mat * shift_coords;

            auto marker = visualization_msgs::msg::Marker();
            marker.header.frame_id="map";
            marker.type= visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x= 0.1;
            marker.scale.y= 0.1;
            marker.scale.z= 0.1;
            marker.pose.position.x= current_car_pose.pose.pose.position.x + shift_in_global_coords[0] ;
            marker.pose.position.y= current_car_pose.pose.pose.position.y + shift_in_global_coords[1];
            marker.pose.position.z=0.0;
            marker.color.a=1.0;
            marker.color.r=1.0;
            marker.color.b=0.0;
            marker.color.g=0.0;
            marker.pose.orientation.x=0.0;
            marker.pose.orientation.y=0.0;
            marker.pose.orientation.z=0.0;
            marker.pose.orientation.w=1.0;
            marker.lifetime.nanosec=int(1e8);
            marker.header.stamp = rclcpp::Clock().now();
            marker.id=i;

            message.markers.push_back(marker);
        }
        path_pub->publish(message);
}

void RRT::update_nodes(std::vector<RRT_Node> &tree){
        auto message = visualization_msgs::msg::MarkerArray();
        Eigen::Quaterniond q;
        q.x()= current_car_pose.pose.pose.orientation.x;
        q.y()= current_car_pose.pose.pose.orientation.y;
        q.z()= current_car_pose.pose.pose.orientation.z;
        q.w()= current_car_pose.pose.pose.orientation.w;
        auto rotation_mat = q.normalized().toRotationMatrix();
        //std::cout<<"Size of tree: "<<tree.size()<<std::endl;

        for(int i=0; i<tree.size();i++){
            //Find position of marker
            Eigen::Vector3d shift_coords(tree[i].x, tree[i].y, 0);
            Eigen::Vector3d shift_in_global_coords = rotation_mat * shift_coords;

            auto marker = visualization_msgs::msg::Marker();
            marker.header.frame_id="map";
            marker.type= visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x= 0.1;
            marker.scale.y= 0.1;
            marker.scale.z= 0.1;
            marker.pose.position.x= shift_in_global_coords[0] + current_car_pose.pose.pose.position.x;
            marker.pose.position.y= shift_in_global_coords[1] + current_car_pose.pose.pose.position.y;
            marker.pose.position.z=0.0;
            marker.color.a=1.0;
            marker.color.r=0.0;
            marker.color.b=0.0;
            marker.color.g=1.0;
            marker.pose.orientation.x=0.0;
            marker.pose.orientation.y=0.0;
            marker.pose.orientation.z=0.0;
            marker.pose.orientation.w=1.0;
            marker.lifetime.nanosec=int(1e8);
            marker.header.stamp = rclcpp::Clock().now();
            marker.id=i;

            message.markers.push_back(marker);
        }
        node_pub->publish(message);
}

// call this function directly in pose_callback
void RRT::update_goal_point(float goal_point_x, float goal_point_y){
        auto message = visualization_msgs::msg::Marker();

        Eigen::Quaterniond q;
        q.x()= current_car_pose.pose.pose.orientation.x;
        q.y()= current_car_pose.pose.pose.orientation.y;
        q.z()= current_car_pose.pose.pose.orientation.z;
        q.w()= current_car_pose.pose.pose.orientation.w;
        auto rotation_mat = q.normalized().toRotationMatrix();

        Eigen::Vector3d shift_coords(goal_point_x, goal_point_y, 0);
        Eigen::Vector3d shift_in_global_coords = rotation_mat * shift_coords;

        message.header.frame_id="map";
        message.type= visualization_msgs::msg::Marker::SPHERE;
        message.action = visualization_msgs::msg::Marker::ADD;
        message.scale.x= 0.25;
        message.scale.y= 0.25;
        message.scale.z= 0.25;
        message.pose.position.x=  current_car_pose.pose.pose.position.x + shift_in_global_coords[0] ;
        message.pose.position.y=  current_car_pose.pose.pose.position.y + shift_in_global_coords[1] ;
        message.pose.position.z=0.0;
        message.color.a=1.0;
        message.color.r=0.0;
        message.color.b=0.0;
        message.color.g=1.0;
        message.pose.orientation.x=0.0;
        message.pose.orientation.y=0.0;
        message.pose.orientation.z=0.0;
        message.pose.orientation.w=1.0;
        message.lifetime.nanosec=int(1e8);

        message.header.stamp = rclcpp::Clock().now();
        message.id=0;

        goal_pub->publish(message);
}
