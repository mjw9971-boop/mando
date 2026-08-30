"""
Privileged driving agent used for data collection.
Drives by accessing the simulator directly.

# VTD: CARLA 원본(DriveLM pdm_lite, Apache 2.0)을 VTD 어댑터 위에서 돌린다.
# 수정한 줄에는 전부 `# VTD:` 주석이 있다. 판단 로직(min 중재·IDM·OBB·forecast·
# 조향)은 원문 그대로다. 원본 대비 diff: git show <phase3 원본 커밋>..HEAD
"""

import math
import numpy as np
from scipy.integrate import RK45

import vtd_adapter.carla_types as carla                      # VTD: carla → 어댑터 shim
from vtd_adapter.carla_types import RoadOption               # VTD: agents.navigation 대체
from lateral_controller import LateralPIDController
from config import GlobalConfig
from kinematic_bicycle_model import KinematicBicycleModel
# VTD: 제거된 원본 의존 — os/ujson/datetime/pathlib/gzip(데이터 수집),
# CarlaDataProvider/leaderboard(시뮬레이터 프레임워크), nav_planner.RoutePlanner
# (_command_planner — save() 전용), PrivilegedRoutePlanner(→ vtd_adapter.route.
# VtdRoutePlanner 주입), transfuser_utils(3개 함수 전부 제거 경로),
# ScenarioLogger, LongitudinalLinearRegressionController(→ VtdLongitudinalController 주입)


class AutoPilot:                                             # VTD: leaderboard 상속 제거
    """
    Privileged driving agent used for data collection.
    Drives by accessing the simulator directly.
    """

    def setup(self, world, world_map, waypoint_planner,
              longitudinal_controller, ego_vehicle, config=None):
        """
        Set up the autonomous agent for the CARLA simulation.

        # VTD: 센서/CarlaDataProvider 대신 어댑터 객체를 주입받는다 —
        #   world                  vtd_adapter.world.VtdWorld
        #   world_map              vtd_adapter.map.VtdMap
        #   waypoint_planner       vtd_adapter.route.VtdRoutePlanner
        #   longitudinal_controller vtd_adapter.control.VtdLongitudinalController
        #   ego_vehicle            vtd_adapter.actor.VtdEgo
        # 데이터 수집(save_path/datagen/histogram/tp_stats/recording)은 제거.
        """
        self.step = -1
        self.initialized = True                              # VTD: _init 불필요 (주입 완료)

        self.config = config if config is not None else GlobalConfig()   # VTD

        # Dynamics models
        self.ego_model = KinematicBicycleModel(self.config)
        self.vehicle_model = KinematicBicycleModel(self.config)

        # Configuration
        self.visualize = 0                                   # VTD: 렌더링 없음 (debug 는 no-op)

        self.walker_close = False
        self.distance_to_walker = np.inf
        self.stop_sign_close = False
        self.waiting_ticks_at_stop_sign = 0

        # To avoid failing the ActorBlockedTest, the agent has to move at least 0.1 m/s every 179 ticks
        self.ego_blocked_for_ticks = 0

        # Controllers
        self._turn_controller = LateralPIDController(self.config)

        # VTD: 판단은 next_traffic_light(정지선 기반)만 쓴다. CARLA 신호등 전처리
        # (t_u.get_traffic_light_waypoints)는 close_traffic_lights 데이터 수집
        # 전용이라 빈 리스트로 무력화 — ego_agent_affected_by_red_light 원문이
        # 그대로 동작한다.
        self.list_traffic_lights = []

        # Initialize controls
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0

        # Angle to the next waypoint, normalized in [-1, 1] corresponding to [-90, 90]
        self.angle = 0.0
        self.stop_sign_hazard = False
        self.traffic_light_hazard = False
        self.walker_hazard = False
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None                     # VTD: save() 제거로 여기서 초기화
        self.walker_affecting_id = None                      # VTD
        self.walker_close_id = None                          # VTD
        self.junction = False
        self.aim_wp = None  # Waypoint the expert is steering towards
        self.remaining_route = None  # Remaining route
        self.remaining_route_original = None  # Remaining original route
        self.close_traffic_lights = []
        self.close_stop_signs = []
        self.was_at_stop_sign = False
        self.cleared_stop_sign = False
        self.visible_walker_ids = []
        self.walker_past_pos = {}  # Position of walker in the last frame

        # VTD: 어댑터 주입 (원본은 CarlaDataProvider.get_map()/_init 에서 얻는다)
        self._world = world
        self.world_map = world_map
        self._waypoint_planner = waypoint_planner
        self._longitudinal_controller = longitudinal_controller
        self._vehicle = ego_vehicle

    # VTD: toggle_recording()/_init()/sensors() 제거 — 시뮬레이터 녹화·
    # CARLA 초기화·센서 스펙은 VTD 경로에 없다 (setup 주입으로 대체).

    def tick_autopilot(self, input_data):
        """
        Get the current state of the vehicle from the input data and the vehicle's sensors.

        # VTD: IMU/speedometer 센서 대신 어댑터(VtdEgo — CARLA 프레임)에서 읽는다.
        # input_data 는 쓰지 않는다 (원본 시그니처 유지용).

        Returns:
            dict: A dictionary containing the vehicle's position (GPS), speed, and compass heading.
        """
        # VTD: 속도 = EgoSpeedEstimator 추정값 (9910 에 자차 속도 필드가 없다)
        speed = self._vehicle.get_velocity().length()

        # VTD: compass = CARLA 프레임 yaw [rad] — preprocess_compass(-90° 보정) 불필요
        compass = np.deg2rad(self._vehicle.get_transform().rotation.yaw)

        # Get the vehicle's position from its location
        position = self._vehicle.get_location()
        gps = np.array([position.x, position.y, position.z])

        # Create a dictionary containing the vehicle's state
        vehicle_state = {
            "gps": gps,
            "speed": speed,
            "compass": compass,
        }

        return vehicle_state

    def run_step(self, input_data, timestamp, sensors=None, plant=False):
        """
        Run a single step of the agent's control loop.

        Args:
            input_data (dict): Input data for the current step.
            timestamp (float): Timestamp of the current step.
            sensors (list, optional): List of sensor objects. Default is None.
            plant (bool, optional): Flag indicating whether to run the plant simulation or not. Default is False.

        Returns:
            If plant is False, it returns the control commands (steer, throttle, brake).
            If plant is True, it returns the driving data for the current step.
        """
        self.step += 1

        # VTD: _init 불필요 (setup 주입), plant/데이터 수집 경로 제거
        control = self._get_control(input_data, plant)

        return control

    def _get_control(self, input_data, plant):
        """
        Compute the control commands and save the driving data for the current frame.

        Args:
            input_data (dict): Input data for the current frame.
            plant (object): The plant object representing the vehicle dynamics.

        Returns:
            tuple: A tuple containing the control commands (steer, throttle, brake) and the driving data.
        """
        tick_data = self.tick_autopilot(input_data)
        ego_position = tick_data["gps"]

        # Waypoint planning and route generation
        (
            route_np,
            route_wp,
            _,
            distance_to_next_traffic_light,
            next_traffic_light,
            distance_to_next_stop_sign,
            next_stop_sign,
            speed_limit,
        ) = self._waypoint_planner.run_step(ego_position)

        # Extract relevant route information
        self.remaining_route = route_np[self.config.tf_first_checkpoint_distance :][
            :: self.config.points_per_meter
        ]
        self.remaining_route_original = self._waypoint_planner.original_route_points[
            self._waypoint_planner.route_index :
        ][self.config.tf_first_checkpoint_distance :][:: self.config.points_per_meter]

        # Get the current speed and target speed
        ego_speed = tick_data["speed"]
        target_speed = min(
            speed_limit * self.config.ratio_target_speed_limit, 72.0 / 3.6
        )  # merge the two last speed bins

        # Reduce target speed if there is a junction ahead
        for i in range(
            min(self.config.max_lookahead_to_check_for_junction, len(route_wp))
        ):
            if route_wp[i].is_junction:
                target_speed = min(target_speed, self.config.max_speed_in_junction)
                break

        # Get the list of vehicles in the scene
        actors = self._world.get_actors()
        vehicles = list(actors.filter("*vehicle*"))

        # Manage route obstacle scenarios and adjust target speed
        target_speed_route_obstacle, keep_driving, speed_reduced_by_obj = (
            self._manage_route_obstacle_scenarios(
                target_speed, ego_speed, route_wp, vehicles, route_np
            )
        )

        # In case the agent overtakes an obstacle, keep driving in case the opposite lane is free instead of using idm
        # and the kinematic bicycle model forecasts
        if keep_driving:
            brake, target_speed = False, target_speed_route_obstacle
        else:
            brake, target_speed, speed_reduced_by_obj = self.get_brake_and_target_speed(
                plant,
                route_np,
                distance_to_next_traffic_light,
                next_traffic_light,
                distance_to_next_stop_sign,
                next_stop_sign,
                vehicles,
                actors,
                target_speed,
                speed_reduced_by_obj,
            )

        target_speed = min(target_speed, target_speed_route_obstacle)

        # Determine if the ego vehicle is at a junction
        ego_vehicle_waypoint = self.world_map.get_waypoint(self._vehicle.get_location())
        self.junction = ego_vehicle_waypoint.is_junction

        # Compute throttle and brake control
        throttle, control_brake = self._longitudinal_controller.get_throttle_and_brake(
            brake, target_speed, ego_speed
        )

        # Compute steering control
        steer = self._get_steer(route_np, ego_position, tick_data["compass"], ego_speed)

        # Create the control command
        control = carla.VehicleControl()
        control.steer = steer + self.config.steer_noise * np.random.randn()
        control.throttle = throttle
        control.brake = float(brake or control_brake)
        # VTD: VtdLongitudinalController 는 throttle 자리에 accel [m/s²] 를 낸다
        # (phase0 §0-4 (b)). 9910 송신은 이 값을 그대로 targetAccel 로 쓴다.
        control.accel = throttle

        # Apply brake if the vehicle is stopped to prevent rolling back
        if (
            control.throttle == 0
            and ego_speed < self.config.minimum_speed_to_prevent_rolling_back
        ):
            control.brake = 1

        # Apply throttle if the vehicle is blocked for too long
        ego_velocity = self._vehicle.get_velocity().length()   # VTD: CarlaDataProvider 대체
        if ego_velocity < 0.1:
            self.ego_blocked_for_ticks += 1
        else:
            self.ego_blocked_for_ticks = 0

        if self.ego_blocked_for_ticks >= self.config.max_blocked_ticks:
            control.throttle = 1
            control.brake = 0

        # Save control commands and target speed
        self.steer = control.steer
        self.throttle = control.throttle
        self.brake = control.brake

        # VTD: command planner(target_point — TF++ 데이터 수집 전용)와 save()
        # 데이터 수집을 제거 — 틱 기록은 vtd_adapter.logger 가 한다.

        # VTD: 한국 대회 규칙 계층 (team_code/kr_rules.py) — route_end 정지
        # 후보를 min() 중재 뒤에 덧댄다 (CARLA 리더보드는 결승선 통과로 끝나
        # PDM 에 종점 정지 개념이 없다). 판단 원문은 무수정.
        control, target_speed = self.kr_rules.apply(control, target_speed, self)

        return control


    def _manage_route_obstacle_scenarios(
        self, target_speed, ego_speed, route_waypoints, list_vehicles, route_points
    ):
        """
        # VTD: CARLA 시나리오 전용 원문(InvadingTurn/Accident/…TwoWays/
        # YieldToEmergencyVehicle — CarlaDataProvider.active_scenarios 하드코딩)
        # 640줄을 stub 으로 대체. 코스에 해당 시나리오가 없고, 정차 차량 추월은
        # phase4 에서 kr_rules 가 VtdRoutePlanner.shift_route_around_actors 로
        # 발동한다. 반환 형태는 원문과 동일.
        """
        return target_speed, False, [target_speed, None, None, None]

    # VTD: save()(measurements 데이터 수집)·destroy() 제거 —
    # 틱 기록은 vtd_adapter.logger(run_*.jsonl 스키마)가 한다.

    def _get_steer(
        self, route_points, current_position, current_heading, current_speed
    ):
        """
        Calculate the steering angle based on the current position, heading, speed, and the route points.

        Args:
            route_points (numpy.ndarray): An array of (x, y) coordinates representing the route points.
            current_position (tuple): The current position (x, y) of the vehicle.
            current_heading (float): The current heading angle (in radians) of the vehicle.
            current_speed (float): The current speed of the vehicle (in m/s).

        Returns:
            float: The calculated steering angle.
        """
        speed_scale = self.config.lateral_pid_speed_scale
        speed_offset = self.config.lateral_pid_speed_offset

        # Calculate the lookahead index based on the current speed
        speed_in_kmph = current_speed * 3.6
        lookahead_distance = speed_scale * speed_in_kmph + speed_offset
        lookahead_distance = np.clip(
            lookahead_distance,
            self.config.lateral_pid_default_lookahead,
            self.config.lateral_pid_maximum_lookahead_distance,
        )
        lookahead_index = int(min(lookahead_distance, route_points.shape[0] - 1))

        # Get the target point from the route points
        target_point = route_points[lookahead_index]

        # Calculate the angle between the current heading and the target point
        angle_unnorm = self._get_angle_to(
            current_position, current_heading, target_point
        )
        normalized_angle = angle_unnorm / 90

        self.aim_wp = target_point
        self.angle = normalized_angle

        # Calculate the steering angle using the turn controller
        steering_angle = self._turn_controller.step(
            route_points, current_speed, current_position, current_heading
        )
        steering_angle = round(steering_angle, 3)

        return steering_angle

    def _compute_target_speed_idm(
        self,
        desired_speed,
        leading_actor_length,
        ego_speed,
        leading_actor_speed,
        distance_to_leading_actor,
        s0=4.0,
        T=0.5,
    ):
        """
        Compute the target speed for the ego vehicle using the Intelligent Driver Model (IDM).

        Args:
            desired_speed (float): The desired speed of the ego vehicle.
            leading_actor_length (float): The length of the leading actor (vehicle or obstacle).
            ego_speed (float): The current speed of the ego vehicle.
            leading_actor_speed (float): The speed of the leading actor.
            distance_to_leading_actor (float): The distance to the leading actor.
            s0 (float, optional): The minimum desired net distance.
            T (float, optional): The desired time headway.

        Returns:
            float: The computed target speed for the ego vehicle.
        """
        a = self.config.idm_maximum_acceleration  # Maximum acceleration [m/s²]
        b = (
            self.config.idm_comfortable_braking_deceleration_high_speed
            if ego_speed > self.config.idm_comfortable_braking_deceleration_threshold
            else self.config.idm_comfortable_braking_deceleration_low_speed
        )  # Comfortable deceleration [m/s²]
        delta = self.config.idm_acceleration_exponent  # Acceleration exponent

        t_bound = self.config.idm_t_bound

        def idm_equations(t, x):
            """
            Differential equations for the Intelligent Driver Model.

            Args:
                t (float): Time.
                x (list): State variables [position, speed].

            Returns:
                list: Derivatives of the state variables.
            """
            ego_position, ego_speed = x

            speed_diff = ego_speed - leading_actor_speed
            s_star = s0 + ego_speed * T + ego_speed * speed_diff / 2.0 / np.sqrt(a * b)
            # The maximum is needed to avoid numerical unstabilities
            s = max(
                0.1,
                distance_to_leading_actor
                + t * leading_actor_speed
                - ego_position
                - leading_actor_length,
            )
            dvdt = a * (1.0 - (ego_speed / desired_speed) ** delta - (s_star / s) ** 2)

            return [ego_speed, dvdt]

        # Set the initial conditions
        y0 = [0.0, ego_speed]

        # Integrate the differential equations using RK45
        rk45 = RK45(fun=idm_equations, t0=0.0, y0=y0, t_bound=t_bound)
        while rk45.status == "running":
            rk45.step()

        # The target speed is the final speed obtained from the integration
        target_speed = rk45.y[1]

        # Clip the target speed to non-negative values
        return np.clip(target_speed, 0, np.inf)

    def is_near_lane_change(self, ego_velocity, route_points):
        """
        Computes if the ego agent is/was close to a lane change maneuver.

        Args:
            ego_velocity (float): The current velocity of the ego agent in m/s.
            route_points (numpy.ndarray): An array of locations representing the planned route.

        Returns:
            bool: True if the ego agent is close to a lane change, False otherwise.
        """
        # Calculate the braking distance based on the ego velocity
        braking_distance = (
            ((ego_velocity * 3.6) / 10.0) ** 2 / 2.0
        ) + self.config.braking_distance_calculation_safety_distance

        # Determine the number of waypoints to look ahead based on the braking distance
        look_ahead_points = max(
            self.config.minimum_lookahead_distance_to_compute_near_lane_change,
            min(
                route_points.shape[0],
                self.config.points_per_meter * int(braking_distance),
            ),
        )
        current_route_index = self._waypoint_planner.route_index
        max_route_length = len(self._waypoint_planner.commands)

        from_index = max(
            0, current_route_index - self.config.check_previous_distance_for_lane_change
        )
        to_index = min(max_route_length - 1, current_route_index + look_ahead_points)
        # Iterate over the points around the current position, checking for lane change commands
        for i in range(from_index, to_index, 1):
            if self._waypoint_planner.commands[i] in (
                RoadOption.CHANGELANELEFT,
                RoadOption.CHANGELANERIGHT,
            ):
                return True

        return False

    def predict_other_actors_bounding_boxes(
        self,
        plant,
        actor_list,
        ego_vehicle_location,
        num_future_frames,
        near_lane_change,
    ):
        """
        Predict the future bounding boxes of actors for a given number of frames.

        Args:
            plant (bool): Whether to use PlanT.
            actor_list (list): A list of actors (e.g., vehicles) in the simulation.
            ego_vehicle_location (carla.Location): The current location of the ego vehicle.
            num_future_frames (int): The number of future frames to predict.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.

        Returns:
            dict: A dictionary mapping actor IDs to lists of predicted bounding boxes for each future frame.
        """
        predicted_bounding_boxes = {}

        if not plant:
            # Filter out nearby actors within the detection radius, excluding the ego vehicle
            nearby_actors = [
                actor
                for actor in actor_list
                if actor.id != self._vehicle.id
                and actor.get_location().distance(ego_vehicle_location)
                < self.config.detection_radius
            ]

            # If there are nearby actors, calculate their future bounding boxes
            if nearby_actors:
                # Get the previous control inputs (steering, throttle, brake) for the nearby actors
                previous_controls = [actor.get_control() for actor in nearby_actors]
                previous_actions = np.array(
                    [
                        [control.steer, control.throttle, control.brake]
                        for control in previous_controls
                    ]
                )

                # Get the current velocities, locations, and headings of the nearby actors
                velocities = np.array(
                    [actor.get_velocity().length() for actor in nearby_actors]
                )
                locations = np.array(
                    [
                        [
                            actor.get_location().x,
                            actor.get_location().y,
                            actor.get_location().z,
                        ]
                        for actor in nearby_actors
                    ]
                )
                headings = np.deg2rad(
                    np.array(
                        [actor.get_transform().rotation.yaw for actor in nearby_actors]
                    )
                )

                # Initialize arrays to store future locations, headings, and velocities
                future_locations = np.empty(
                    (num_future_frames, len(nearby_actors), 3), dtype="float"
                )
                future_headings = np.empty(
                    (num_future_frames, len(nearby_actors)), dtype="float"
                )
                future_velocities = np.empty(
                    (num_future_frames, len(nearby_actors)), dtype="float"
                )

                # Forecast the future locations, headings, and velocities for the nearby actors
                for i in range(num_future_frames):
                    locations, headings, velocities = (
                        self.vehicle_model.forecast_other_vehicles(
                            locations, headings, velocities, previous_actions
                        )
                    )
                    future_locations[i] = locations.copy()
                    future_velocities[i] = velocities.copy()
                    future_headings[i] = headings.copy()

                # Convert future headings to degrees
                future_headings = np.rad2deg(future_headings)

                # Calculate the predicted bounding boxes for each nearby actor and future frame
                for actor_idx, actor in enumerate(nearby_actors):
                    predicted_actor_boxes = []

                    for i in range(num_future_frames):
                        # Calculate the future location of the actor
                        location = carla.Location(
                            x=future_locations[i, actor_idx, 0].item(),
                            y=future_locations[i, actor_idx, 1].item(),
                            z=future_locations[i, actor_idx, 2].item(),
                        )

                        # Calculate the future rotation of the actor
                        rotation = carla.Rotation(
                            pitch=0, yaw=future_headings[i, actor_idx], roll=0
                        )

                        # Get the extent (dimensions) of the actor's bounding box
                        extent = actor.bounding_box.extent
                        # Otherwise we would increase the extent of the bounding box of the vehicle
                        extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)

                        # Adjust the bounding box size based on velocity and lane change maneuver to adjust for
                        # uncertainty during forecasting
                        s = (
                            self.config.high_speed_min_extent_x_other_vehicle_lane_change
                            if near_lane_change
                            else self.config.high_speed_min_extent_x_other_vehicle
                        )
                        extent.x *= (
                            self.config.slow_speed_extent_factor_ego
                            if future_velocities[i, actor_idx]
                            < self.config.extent_other_vehicles_bbs_speed_threshold
                            else max(
                                s,
                                self.config.high_speed_min_extent_x_other_vehicle
                                * float(i)
                                / float(num_future_frames),
                            )
                        )
                        extent.y *= (
                            self.config.slow_speed_extent_factor_ego
                            if future_velocities[i, actor_idx]
                            < self.config.extent_other_vehicles_bbs_speed_threshold
                            else max(
                                self.config.high_speed_min_extent_y_other_vehicle,
                                self.config.high_speed_extent_y_factor_other_vehicle
                                * float(i)
                                / float(num_future_frames),
                            )
                        )

                        # Create the bounding box for the future frame
                        bounding_box = carla.BoundingBox(location, extent)
                        bounding_box.rotation = rotation

                        # Append the bounding box to the list of predicted bounding boxes for this actor
                        predicted_actor_boxes.append(bounding_box)

                    # Store the predicted bounding boxes for this actor in the dictionary
                    predicted_bounding_boxes[actor.id] = predicted_actor_boxes

                if self.visualize == 1:
                    for (
                        actor_idx,
                        actors_forecasted_bounding_boxes,
                    ) in predicted_bounding_boxes.items():
                        for bb in actors_forecasted_bounding_boxes:
                            self._world.debug.draw_box(
                                box=bb,
                                rotation=bb.rotation,
                                thickness=0.1,
                                color=self.config.other_vehicles_forecasted_bbs_color,
                                life_time=self.config.draw_life_time,
                            )

        return predicted_bounding_boxes

    def compute_target_speed_wrt_leading_vehicle(
        self,
        initial_target_speed,
        predicted_bounding_boxes,
        near_lane_change,
        ego_location,
        rear_vehicle_ids,
        leading_vehicle_ids,
        speed_reduced_by_obj,
        plant,
    ):
        """
        Compute the target speed for the ego vehicle considering the leading vehicle.

        Args:
            initial_target_speed (float): The initial target speed for the ego vehicle.
            predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            ego_location (carla.Location): The current location of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance]
                for the object that caused the most speed reduction, or None if no speed reduction.
            plant (bool): Whether to use plant.

        Returns:
            float: The target speed considering the leading vehicle.
        """
        target_speed_wrt_leading_vehicle = initial_target_speed

        if not plant:
            for vehicle_id, _ in predicted_bounding_boxes.items():
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    # Vehicle is in front of the ego vehicle
                    ego_speed = self._vehicle.get_velocity().length()
                    vehicle = self._world.get_actor(vehicle_id)
                    other_speed = vehicle.get_velocity().length()
                    distance_to_vehicle = ego_location.distance(vehicle.get_location())

                    # Compute the target speed using the IDM
                    target_speed_wrt_leading_vehicle = min(
                        target_speed_wrt_leading_vehicle,
                        self._compute_target_speed_idm(
                            desired_speed=initial_target_speed,
                            leading_actor_length=vehicle.bounding_box.extent.x * 2,
                            ego_speed=ego_speed,
                            leading_actor_speed=other_speed,
                            distance_to_leading_actor=distance_to_vehicle,
                            s0=self.config.idm_leading_vehicle_minimum_distance,
                            T=self.config.idm_leading_vehicle_time_headway,
                        ),
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed_wrt_leading_vehicle
                    ):
                        speed_reduced_by_obj = [
                            target_speed_wrt_leading_vehicle,
                            vehicle.type_id,
                            vehicle.id,
                            distance_to_vehicle,
                        ]

            if self.visualize == 1:
                for vehicle_id in predicted_bounding_boxes.keys():
                    # check if vehicle is in front of the ego vehicle
                    if vehicle_id in leading_vehicle_ids and not near_lane_change:
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0
                        )
                        self._world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config.leading_vehicle_color,
                            life_time=self.config.draw_life_time,
                        )
                    elif vehicle_id in rear_vehicle_ids:
                        vehicle = self._world.get_actor(vehicle_id)
                        extent = vehicle.bounding_box.extent
                        bb = carla.BoundingBox(vehicle.get_location(), extent)
                        bb.rotation = carla.Rotation(
                            pitch=0, yaw=vehicle.get_transform().rotation.yaw, roll=0
                        )
                        self._world.debug.draw_box(
                            box=bb,
                            rotation=bb.rotation,
                            thickness=0.5,
                            color=self.config.trailing_vehicle_color,
                            life_time=self.config.draw_life_time,
                        )

        return target_speed_wrt_leading_vehicle, speed_reduced_by_obj

    def compute_target_speeds_wrt_all_actors(
        self,
        initial_target_speed,
        ego_bounding_boxes,
        predicted_bounding_boxes,
        near_lane_change,
        leading_vehicle_ids,
        rear_vehicle_ids,
        speed_reduced_by_obj,
        nearby_walkers,
        nearby_walkers_ids,
    ):
        """
        Compute the target speeds for the ego vehicle considering all actors (vehicles, bicycles,
        and pedestrians) by checking for intersecting bounding boxes.

        Args:
            initial_target_speed (float): The initial target speed for the ego vehicle.
            ego_bounding_boxes (list): A list of bounding boxes for the ego vehicle at different future frames.
            predicted_bounding_boxes (dict): A dictionary mapping actor IDs to lists of predicted bounding boxes.
            near_lane_change (bool): Whether the ego vehicle is near a lane change maneuver.
            leading_vehicle_ids (list): A list of IDs for vehicles in front of the ego vehicle.
            rear_vehicle_ids (list): A list of IDs for vehicles behind the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type,
                object ID, distance] for the object that caused the most speed reduction, or None if
                no speed reduction.
            nearby_walkers (dict): A list of predicted bounding boxes of nearby pedestrians.
            nearby_walkers_ids (list): A list of IDs for nearby pedestrians.

        Returns:
            tuple: A tuple containing the target speeds for bicycles, pedestrians, vehicles, and the updated
                speed_reduced_by_obj list.
        """
        target_speed_bicycle = initial_target_speed
        target_speed_pedestrian = initial_target_speed
        target_speed_vehicle = initial_target_speed
        ego_vehicle_location = self._vehicle.get_location()
        hazard_color = self.config.ego_vehicle_forecasted_bbs_hazard_color
        normal_color = self.config.ego_vehicle_forecasted_bbs_normal_color
        color = normal_color

        # Iterate over the ego vehicle's bounding boxes and predicted bounding boxes of other actors
        for i, ego_bounding_box in enumerate(ego_bounding_boxes):
            for vehicle_id, bounding_boxes in predicted_bounding_boxes.items():
                # Skip leading and rear vehicles if not near a lane change
                if vehicle_id in leading_vehicle_ids and not near_lane_change:
                    continue
                elif vehicle_id in rear_vehicle_ids and not near_lane_change:
                    continue
                else:
                    # Check if the ego bounding box intersects with the predicted bounding box of the actor
                    intersects_with_ego = self.check_obb_intersection(
                        ego_bounding_box, bounding_boxes[i]
                    )
                    ego_speed = self._vehicle.get_velocity().length()

                    if intersects_with_ego:
                        blocking_actor = self._world.get_actor(vehicle_id)

                        # Handle the case when the blocking actor is a bicycle
                        if (
                            "base_type" in blocking_actor.attributes
                            and blocking_actor.attributes["base_type"] == "bicycle"
                        ):
                            other_speed = blocking_actor.get_velocity().length()
                            distance_to_actor = ego_vehicle_location.distance(
                                blocking_actor.get_location()
                            )

                            # Compute the target speed for bicycles using the IDM
                            target_speed_bicycle = min(
                                target_speed_bicycle,
                                self._compute_target_speed_idm(
                                    desired_speed=initial_target_speed,
                                    leading_actor_length=blocking_actor.bounding_box.extent.x
                                    * 2,
                                    ego_speed=ego_speed,
                                    leading_actor_speed=other_speed,
                                    distance_to_leading_actor=distance_to_actor,
                                    s0=self.config.idm_bicycle_minimum_distance,
                                    T=self.config.idm_bicycle_desired_time_headway,
                                ),
                            )

                            # Update the object causing the most speed reduction
                            if (
                                speed_reduced_by_obj is None
                                or speed_reduced_by_obj[0] > target_speed_bicycle
                            ):
                                speed_reduced_by_obj = [
                                    target_speed_bicycle,
                                    blocking_actor.type_id,
                                    blocking_actor.id,
                                    distance_to_actor,
                                ]

                        # Handle the case when the blocking actor is not a bicycle
                        else:
                            self.vehicle_hazard = True  # Set the vehicle hazard flag
                            self.vehicle_affecting_id = vehicle_id  # Store the ID of the vehicle causing the hazard
                            color = hazard_color  # Change the following colors from green to red (no hazard to hazard)
                            target_speed_vehicle = (
                                0  # Set the target speed for vehicles to zero
                            )
                            distance_to_actor = blocking_actor.get_location().distance(
                                ego_vehicle_location
                            )

                            # Update the object causing the most speed reduction
                            if (
                                speed_reduced_by_obj is None
                                or speed_reduced_by_obj[0] > target_speed_vehicle
                            ):
                                speed_reduced_by_obj = [
                                    target_speed_vehicle,
                                    blocking_actor.type_id,
                                    blocking_actor.id,
                                    distance_to_actor,
                                ]

            # Iterate over nearby pedestrians and check for intersections with the ego bounding box
            for pedestrian_bb, pedestrian_id in zip(nearby_walkers, nearby_walkers_ids):
                if self.check_obb_intersection(ego_bounding_box, pedestrian_bb[i]):
                    color = hazard_color
                    ego_speed = self._vehicle.get_velocity().length()
                    blocking_actor = self._world.get_actor(pedestrian_id)
                    distance_to_actor = ego_vehicle_location.distance(
                        blocking_actor.get_location()
                    )

                    # Compute the target speed for pedestrians using the IDM
                    target_speed_pedestrian = min(
                        target_speed_pedestrian,
                        self._compute_target_speed_idm(
                            desired_speed=initial_target_speed,
                            leading_actor_length=0.5
                            + self._vehicle.bounding_box.extent.x,
                            ego_speed=ego_speed,
                            leading_actor_speed=0.0,
                            distance_to_leading_actor=distance_to_actor,
                            s0=self.config.idm_pedestrian_minimum_distance,
                            T=self.config.idm_pedestrian_desired_time_headway,
                        ),
                    )

                    # Update the object causing the most speed reduction
                    if (
                        speed_reduced_by_obj is None
                        or speed_reduced_by_obj[0] > target_speed_pedestrian
                    ):
                        speed_reduced_by_obj = [
                            target_speed_pedestrian,
                            blocking_actor.type_id,
                            blocking_actor.id,
                            distance_to_actor,
                        ]

            if self.visualize == 1:
                self._world.debug.draw_box(
                    box=ego_bounding_box,
                    rotation=ego_bounding_box.rotation,
                    thickness=0.1,
                    color=color,
                    life_time=self.config.draw_life_time,
                )

        return (
            target_speed_bicycle,
            target_speed_pedestrian,
            target_speed_vehicle,
            speed_reduced_by_obj,
        )

    def get_brake_and_target_speed(
        self,
        plant,
        route_points,
        distance_to_next_traffic_light,
        next_traffic_light,
        distance_to_next_stop_sign,
        next_stop_sign,
        vehicle_list,
        actor_list,
        initial_target_speed,
        speed_reduced_by_obj,
    ):
        """
        Compute the brake command and target speed for the ego vehicle based on various factors.

        Args:
            plant (bool): Whether to use PlanT.
            route_points (numpy.ndarray): An array of waypoints representing the planned route.
            distance_to_next_traffic_light (float): The distance to the next traffic light.
            next_traffic_light (carla.TrafficLight): The next traffic light actor.
            distance_to_next_stop_sign (float): The distance to the next stop sign.
            next_stop_sign (carla.StopSign): The next stop sign actor.
            vehicle_list (list): A list of vehicle actors in the simulation.
            actor_list (list): A list of all actors (vehicles, pedestrians, etc.) in the simulation.
            initial_target_speed (float): The initial target speed for the ego vehicle.
            speed_reduced_by_obj (list or None): A list containing [reduced speed, object type, object ID, distance]
                for the object that caused the most speed reduction, or None if no speed reduction.

        Returns:
            tuple: A tuple containing the brake command (bool), target speed (float), and the updated
                speed_reduced_by_obj list.
        """
        ego_speed = self._vehicle.get_velocity().length()
        target_speed = initial_target_speed

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_transform = self._vehicle.get_transform()

        # Calculate the global bounding box of the ego vehicle
        center_ego_bb_global = ego_vehicle_transform.transform(
            self._vehicle.bounding_box.location
        )
        ego_bb_global = carla.BoundingBox(
            center_ego_bb_global, self._vehicle.bounding_box.extent
        )
        ego_bb_global.rotation = ego_vehicle_transform.rotation

        if self.visualize == 1:
            self._world.debug.draw_box(
                box=ego_bb_global,
                rotation=ego_bb_global.rotation,
                thickness=0.1,
                color=self.config.ego_vehicle_bb_color,
                life_time=self.config.draw_life_time,
            )

        # Reset hazard flags
        self.stop_sign_close = False
        self.walker_close = False
        self.walker_close_id = None
        self.vehicle_hazard = False
        self.vehicle_affecting_id = None
        self.walker_hazard = False
        self.walker_affecting_id = None
        self.traffic_light_hazard = False
        self.stop_sign_hazard = False
        self.walker_hazard = False
        self.stop_sign_close = False

        # Compute if there will be a lane change close
        near_lane_change = self.is_near_lane_change(ego_speed, route_points)

        # Compute the number of future frames to consider for collision detection
        num_future_frames = int(
            self.config.bicycle_frame_rate
            * (
                self.config.forecast_length_lane_change
                if near_lane_change
                else self.config.default_forecast_length
            )
        )

        # Get future bounding boxes of pedestrians
        if not plant:
            nearby_pedestrians, nearby_pedestrian_ids = self.forecast_walkers(
                actor_list, ego_vehicle_location, num_future_frames
            )

        # Forecast the ego vehicle's bounding boxes for the future frames
        ego_bounding_boxes = self.forecast_ego_agent(
            ego_vehicle_transform,
            ego_speed,
            num_future_frames,
            initial_target_speed,
            route_points,
        )

        # Predict bounding boxes of other actors (vehicles, bicycles, etc.)
        predicted_bounding_boxes = self.predict_other_actors_bounding_boxes(
            plant,
            vehicle_list,
            ego_vehicle_location,
            num_future_frames,
            near_lane_change,
        )

        # Compute the leading and trailing vehicle IDs
        leading_vehicle_ids = self._waypoint_planner.compute_leading_vehicles(
            vehicle_list, self._vehicle.id
        )
        trailing_vehicle_ids = self._waypoint_planner.compute_trailing_vehicles(
            vehicle_list, self._vehicle.id
        )

        # Compute the target speed with respect to the leading vehicle
        target_speed_leading, speed_reduced_by_obj = (
            self.compute_target_speed_wrt_leading_vehicle(
                initial_target_speed,
                predicted_bounding_boxes,
                near_lane_change,
                ego_vehicle_location,
                trailing_vehicle_ids,
                leading_vehicle_ids,
                speed_reduced_by_obj,
                plant,
            )
        )

        # Compute the target speeds with respect to all actors (vehicles, bicycles, pedestrians)
        (
            target_speed_bicycle,
            target_speed_pedestrian,
            target_speed_vehicle,
            speed_reduced_by_obj,
        ) = self.compute_target_speeds_wrt_all_actors(
            initial_target_speed,
            ego_bounding_boxes,
            predicted_bounding_boxes,
            near_lane_change,
            leading_vehicle_ids,
            trailing_vehicle_ids,
            speed_reduced_by_obj,
            nearby_pedestrians,
            nearby_pedestrian_ids,
        )

        # VTD: BREAKOUT 크립 (kr_rules 규칙2) — 데드락 탈출의 마지막 단계다.
        # kr_rules 는 min() 에 후보를 덧대기만 하므로 "전진 하한" 을 만들 수
        # 없다: 0 을 만드는 것은 PDM 자신의 선행차 IDM(leading)과 OBB forecast
        # (vehicle/bicycle)다. 그래서 이 규칙만 예외적으로 판단(kr_rules)과
        # 소비(여기 두 줄)가 분리된다 — 황색 GO 훅과 같은 선례다.
        # **보행자 후보(target_speed_pedestrian)는 건드리지 않는다.**
        # 발동은 BREAKOUT 최종 단계(L4) 단독이고, 그 게이팅은 kr_rules 가 한다.
        if self.kr_rules.breakout_creep():
            target_speed_leading = initial_target_speed
            target_speed_bicycle = target_speed_vehicle = initial_target_speed

        # Compute the target speed with respect to the red light
        target_speed_red_light = self.ego_agent_affected_by_red_light(
            ego_vehicle_location,
            ego_speed,
            distance_to_next_traffic_light,
            next_traffic_light,
            route_points,
            initial_target_speed,
        )

        # Update the object causing the most speed reduction
        if (
            speed_reduced_by_obj is None
            or speed_reduced_by_obj[0] > target_speed_red_light
        ):
            speed_reduced_by_obj = [
                target_speed_red_light,
                None if next_traffic_light is None else next_traffic_light.type_id,
                None if next_traffic_light is None else next_traffic_light.id,
                distance_to_next_traffic_light,
            ]

        # Compute the target speed with respect to the stop sign
        target_speed_stop_sign = self.ego_agent_affected_by_stop_sign(
            ego_vehicle_location,
            ego_speed,
            next_stop_sign,
            initial_target_speed,
            actor_list,
        )
        # Update the object causing the most speed reduction
        if (
            speed_reduced_by_obj is None
            or speed_reduced_by_obj[0] > target_speed_stop_sign
        ):
            speed_reduced_by_obj = [
                target_speed_stop_sign,
                None if next_stop_sign is None else next_stop_sign.type_id,
                None if next_stop_sign is None else next_stop_sign.id,
                distance_to_next_stop_sign,
            ]

        # Compute the minimum target speed considering all factors
        target_speed = min(
            target_speed_leading,
            target_speed_bicycle,
            target_speed_vehicle,
            target_speed_pedestrian,
            target_speed_red_light,
            target_speed_stop_sign,
        )

        # Set the hazard flags based on the target speed and its cause
        if (
            target_speed == target_speed_pedestrian
            and target_speed_pedestrian != initial_target_speed
        ):
            self.walker_hazard = True
            self.walker_close = True
        elif (
            target_speed == target_speed_red_light
            and target_speed_red_light != initial_target_speed
        ):
            self.traffic_light_hazard = True
        elif (
            target_speed == target_speed_stop_sign
            and target_speed_stop_sign != initial_target_speed
        ):
            self.stop_sign_hazard = True
            self.stop_sign_close = True

        # Determine if the ego vehicle needs to brake based on the target speed
        brake = target_speed == 0
        return brake, target_speed, speed_reduced_by_obj

    def forecast_ego_agent(
        self,
        current_ego_transform,
        current_ego_speed,
        num_future_frames,
        initial_target_speed,
        route_points,
    ):
        """
        Forecast the future states of the ego agent using the kinematic bicycle model and assume their is no hazard to
        check subsequently whether the ego vehicle would collide.

        Args:
            current_ego_transform (carla.Transform): The current transform of the ego vehicle.
            current_ego_speed (float): The current speed of the ego vehicle in m/s.
            num_future_frames (int): The number of future frames to forecast.
            initial_target_speed (float): The initial target speed for the ego vehicle.
            route_points (numpy.ndarray): An array of waypoints representing the planned route.

        Returns:
            list: A list of bounding boxes representing the future states of the ego vehicle.
        """
        self._turn_controller.save_state()
        self._waypoint_planner.save()

        # Initialize the initial state without braking
        location = np.array(
            [
                current_ego_transform.location.x,
                current_ego_transform.location.y,
                current_ego_transform.location.z,
            ]
        )
        heading_angle = np.array([np.deg2rad(current_ego_transform.rotation.yaw)])
        speed = np.array([current_ego_speed])

        # Calculate the throttle command based on the target speed and current speed
        throttle = self._longitudinal_controller.get_throttle_extrapolation(
            initial_target_speed, current_ego_speed
        )
        steering = self._turn_controller.step(
            route_points, speed, location, heading_angle.item()
        )
        action = np.array([steering, throttle, 0.0]).flatten()

        future_bounding_boxes = []
        # Iterate over the future frames and forecast the ego agent's state
        for _ in range(num_future_frames):
            # Forecast the next state using the kinematic bicycle model
            location, heading_angle, speed = self.ego_model.forecast_ego_vehicle(
                location, heading_angle, speed, action
            )

            # Update the route and extrapolate steering and throttle commands
            extrapolated_route, _, _, _, _, _, _, _ = self._waypoint_planner.run_step(
                location
            )
            steering = self._turn_controller.step(
                extrapolated_route, speed, location, heading_angle.item()
            )
            throttle = self._longitudinal_controller.get_throttle_extrapolation(
                initial_target_speed, speed
            )
            action = np.array([steering, throttle, 0.0]).flatten()

            heading_angle_degrees = np.rad2deg(heading_angle).item()

            # Decrease the ego vehicles bounding box if it is slow and resolve permanent bounding box
            # intersectinos at collisions.
            # In case of driving increase them for safety.
            extent = self._vehicle.bounding_box.extent
            # Otherwise we would increase the extent of the bounding box of the vehicle
            extent = carla.Vector3D(x=extent.x, y=extent.y, z=extent.z)
            extent.x *= (
                self.config.slow_speed_extent_factor_ego
                if current_ego_speed < self.config.extent_ego_bbs_speed_threshold
                else self.config.high_speed_extent_factor_ego_x
            )
            extent.y *= (
                self.config.slow_speed_extent_factor_ego
                if current_ego_speed < self.config.extent_ego_bbs_speed_threshold
                else self.config.high_speed_extent_factor_ego_y
            )

            transform = carla.Transform(
                carla.Location(
                    x=location[0].item(), y=location[1].item(), z=location[2].item()
                )
            )

            ego_bounding_box = carla.BoundingBox(transform.location, extent)
            ego_bounding_box.rotation = carla.Rotation(
                pitch=0, yaw=heading_angle_degrees, roll=0
            )

            future_bounding_boxes.append(ego_bounding_box)

        self._turn_controller.load_state()
        self._waypoint_planner.load()

        return future_bounding_boxes

    def forecast_walkers(self, actors, ego_vehicle_location, number_of_future_frames):
        """
        Forecast the future locations of pedestrians in the vicinity of the ego vehicle assuming they
        keep their velocity and direction

        Args:
            actors (carla.ActorList): A list of actors in the simulation.
            ego_vehicle_location (carla.Location): The current location of the ego vehicle.
            number_of_future_frames (int): The number of future frames to forecast.

        Returns:
            tuple: A tuple containing two lists:
                - list: A list of lists, where each inner list contains the future bounding boxes for a pedestrian.
                - list: A list of IDs for the pedestrians whose locations were forecasted.
        """
        nearby_pedestrians_bbs, nearby_pedestrian_ids = [], []

        # Filter pedestrians within the detection radius
        pedestrians = [
            ped
            for ped in actors.filter("*walker*")
            if ped.get_location().distance(ego_vehicle_location)
            < self.config.detection_radius
        ]

        # If no pedestrians are found, return empty lists
        if not pedestrians:
            return nearby_pedestrians_bbs, nearby_pedestrian_ids

        # Extract pedestrian locations, speeds, and directions
        pedestrian_locations = np.array(
            [
                [ped.get_location().x, ped.get_location().y, ped.get_location().z]
                for ped in pedestrians
            ]
        )
        pedestrian_speeds = np.array(
            [ped.get_velocity().length() for ped in pedestrians]
        )
        pedestrian_speeds = np.maximum(pedestrian_speeds, self.config.min_walker_speed)
        pedestrian_directions = np.array(
            [
                [
                    ped.get_control().direction.x,
                    ped.get_control().direction.y,
                    ped.get_control().direction.z,
                ]
                for ped in pedestrians
            ]
        )

        # Calculate future pedestrian locations based on their current locations, speeds, and directions
        future_pedestrian_locations = (
            pedestrian_locations[:, None, :]
            + np.arange(1, number_of_future_frames + 1)[None, :, None]
            * pedestrian_directions[:, None, :]
            * pedestrian_speeds[:, None, None]
            / self.config.bicycle_frame_rate
        )

        # Iterate over pedestrians and calculate their future bounding boxes
        for i, ped in enumerate(pedestrians):
            bb, transform = ped.bounding_box, ped.get_transform()
            rotation = carla.Rotation(
                pitch=bb.rotation.pitch + transform.rotation.pitch,
                yaw=bb.rotation.yaw + transform.rotation.yaw,
                roll=bb.rotation.roll + transform.rotation.roll,
            )
            extent = bb.extent
            extent.x = max(
                self.config.pedestrian_minimum_extent, extent.x
            )  # Ensure a minimum width
            extent.y = max(
                self.config.pedestrian_minimum_extent, extent.y
            )  # Ensure a minimum length

            pedestrian_future_bboxes = []
            for j in range(number_of_future_frames):
                location = carla.Location(
                    future_pedestrian_locations[i, j, 0],
                    future_pedestrian_locations[i, j, 1],
                    future_pedestrian_locations[i, j, 2],
                )

                bounding_box = carla.BoundingBox(location, extent)
                bounding_box.rotation = rotation
                pedestrian_future_bboxes.append(bounding_box)

            nearby_pedestrian_ids.append(ped.id)
            nearby_pedestrians_bbs.append(pedestrian_future_bboxes)

        # Visualize the future bounding boxes of pedestrians (if enabled)
        if self.visualize == 1:
            for bbs in nearby_pedestrians_bbs:
                for bbox in bbs:
                    self._world.debug.draw_box(
                        box=bbox,
                        rotation=bbox.rotation,
                        thickness=0.1,
                        color=self.config.pedestrian_forecasted_bbs_color,
                        life_time=self.config.draw_life_time,
                    )

        return nearby_pedestrians_bbs, nearby_pedestrian_ids

    def ego_agent_affected_by_red_light(
        self,
        ego_vehicle_location,
        ego_vehicle_speed,
        distance_to_traffic_light,
        next_traffic_light,
        route_points,
        target_speed,
    ):
        """
        Handles the behavior of the ego vehicle when approaching a traffic light.

        Args:
            ego_vehicle_location (carla.Location): The ego vehicle location.
            ego_vehicle_speed (float): The current speed of the ego vehicle in m/s.
            distance_to_traffic_light (float): The distance from the ego vehicle to the next traffic light.
            next_traffic_light (carla.TrafficLight or None): The next traffic light in the route.
            route_points (numpy.ndarray): An array of (x, y, z) coordinates representing the route.
            target_speed (float): The target speed for the ego vehicle.

        Returns:
            float: The adjusted target speed for the ego vehicle.
        """

        self.close_traffic_lights.clear()

        for light, center, waypoints in self.list_traffic_lights:

            center_loc = carla.Location(center)
            if center_loc.distance(ego_vehicle_location) > self.config.light_radius:
                continue

            for wp in waypoints:
                # * 0.9 to make the box slightly smaller than the street to prevent overlapping boxes.
                length_bounding_box = carla.Vector3D(
                    (wp.lane_width / 2.0) * 0.9,
                    light.trigger_volume.extent.y,
                    light.trigger_volume.extent.z,
                )
                length_bounding_box = carla.Vector3D(1.5, 1.5, 0.5)

                bounding_box = carla.BoundingBox(
                    wp.transform.location, length_bounding_box
                )

                gloabl_rot = light.get_transform().rotation
                bounding_box.rotation = carla.Rotation(
                    pitch=gloabl_rot.pitch, yaw=gloabl_rot.yaw, roll=gloabl_rot.roll
                )

                affects_ego = (
                    next_traffic_light is not None and light.id == next_traffic_light.id
                )

                self.close_traffic_lights.append(
                    [bounding_box, light.state, light.id, affects_ego]
                )

                if self.visualize == 1:
                    if light.state == carla.libcarla.TrafficLightState.Red:
                        color = carla.Color(255, 0, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Yellow:
                        color = carla.Color(255, 255, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Green:
                        color = carla.Color(0, 255, 0, 255)
                    elif light.state == carla.libcarla.TrafficLightState.Off:
                        color = carla.Color(0, 0, 0, 255)
                    else:  # unknown
                        color = carla.Color(0, 0, 255, 255)

                    self._world.debug.draw_box(
                        box=bounding_box,
                        rotation=bounding_box.rotation,
                        thickness=0.1,
                        color=color,
                        life_time=0.051,
                    )

                    self._world.debug.draw_point(
                        wp.transform.location
                        + carla.Location(z=light.trigger_volume.location.z),
                        size=0.1,
                        color=color,
                        life_time=(1.0 / self.config.carla_fps) + 1e-6,
                    )

        if (
            next_traffic_light is None
            or next_traffic_light.state == carla.TrafficLightState.Green
            # VTD: 황색 GO 래치 / 교차로 통과 가드 (kr_rules) — "감속하지 말 것"이
            # 요지라 후보 덧대기로는 표현할 수 없다. 녹색일 때와 같은 메커니즘으로
            # 건너뛴다. 아래 IDM 본문은 무수정.
            or self.kr_rules.signal_release(self, distance_to_traffic_light)
        ):
            # No traffic light or green light, continue with the current target speed
            return target_speed

        # Compute the target speed using the IDM
        target_speed = self._compute_target_speed_idm(
            desired_speed=target_speed,
            leading_actor_length=0.0,
            ego_speed=ego_vehicle_speed,
            leading_actor_speed=0.0,
            distance_to_leading_actor=distance_to_traffic_light,
            s0=self.config.idm_red_light_minimum_distance,
            T=self.config.idm_red_light_desired_time_headway,
        )

        return target_speed

    def ego_agent_affected_by_stop_sign(
        self,
        ego_vehicle_location,
        ego_vehicle_speed,
        next_stop_sign,
        target_speed,
        actor_list,
    ):
        """
        # VTD: 한국 코스에 정지표지가 없다 — VtdRoutePlanner 가 next_stop_sign 을
        # 항상 None 으로 주므로 원문도 target_speed 를 그대로 돌려줬을 경로다.
        # trigger_volume 조회(우리 액터에 없음)를 피하려고 본문을 stub 으로 대체.
        """
        return target_speed

    def _dot_product(self, vector1, vector2):
        """
        Calculate the dot product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            float: The dot product of the two vectors.
        """
        return vector1.x * vector2.x + vector1.y * vector2.y + vector1.z * vector2.z

    def cross_product(self, vector1, vector2):
        """
        Calculate the cross product of two vectors.

        Args:
            vector1 (carla.Vector3D): The first vector.
            vector2 (carla.Vector3D): The second vector.

        Returns:
            carla.Vector3D: The cross product of the two vectors.
        """
        x = vector1.y * vector2.z - vector1.z * vector2.y
        y = vector1.z * vector2.x - vector1.x * vector2.z
        z = vector1.x * vector2.y - vector1.y * vector2.x

        return carla.Vector3D(x=x, y=y, z=z)

    def get_separating_plane(self, relative_position, plane_normal, obb1, obb2):
        """
        Check if there is a separating plane between two oriented bounding boxes (OBBs).

        Args:
            relative_position (carla.Vector3D): The relative position between the two OBBs.
            plane_normal (carla.Vector3D): The normal vector of the plane.
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if there is a separating plane, False otherwise.
        """
        # Calculate the projection of the relative position onto the plane normal
        projection_distance = abs(self._dot_product(relative_position, plane_normal))

        # Calculate the sum of the projections of the OBB extents onto the plane normal
        obb1_projection = (
            abs(
                self._dot_product(
                    obb1.rotation.get_forward_vector() * obb1.extent.x, plane_normal
                )
            )
            + abs(
                self._dot_product(
                    obb1.rotation.get_right_vector() * obb1.extent.y, plane_normal
                )
            )
            + abs(
                self._dot_product(
                    obb1.rotation.get_up_vector() * obb1.extent.z, plane_normal
                )
            )
        )

        obb2_projection = (
            abs(
                self._dot_product(
                    obb2.rotation.get_forward_vector() * obb2.extent.x, plane_normal
                )
            )
            + abs(
                self._dot_product(
                    obb2.rotation.get_right_vector() * obb2.extent.y, plane_normal
                )
            )
            + abs(
                self._dot_product(
                    obb2.rotation.get_up_vector() * obb2.extent.z, plane_normal
                )
            )
        )

        # Check if the projection distance is greater than the sum of the OBB projections
        return projection_distance > obb1_projection + obb2_projection

    def check_obb_intersection(self, obb1, obb2):
        """
        Check if two 3D oriented bounding boxes (OBBs) intersect.

        Args:
            obb1 (carla.BoundingBox): The first oriented bounding box.
            obb2 (carla.BoundingBox): The second oriented bounding box.

        Returns:
            bool: True if the two OBBs intersect, False otherwise.
        """
        relative_position = obb2.location - obb1.location

        # Check for separating planes along the axes of both OBBs
        if (
            self.get_separating_plane(
                relative_position, obb1.rotation.get_forward_vector(), obb1, obb2
            )
            or self.get_separating_plane(
                relative_position, obb1.rotation.get_right_vector(), obb1, obb2
            )
            or self.get_separating_plane(
                relative_position, obb1.rotation.get_up_vector(), obb1, obb2
            )
            or self.get_separating_plane(
                relative_position, obb2.rotation.get_forward_vector(), obb1, obb2
            )
            or self.get_separating_plane(
                relative_position, obb2.rotation.get_right_vector(), obb1, obb2
            )
            or self.get_separating_plane(
                relative_position, obb2.rotation.get_up_vector(), obb1, obb2
            )
        ):

            return False

        # Check for separating planes along the cross products of the axes of both OBBs
        if (
            self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_forward_vector(),
                    obb2.rotation.get_forward_vector(),
                ),
                obb1,
                obb2,
            )
            or self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_forward_vector(), obb2.rotation.get_right_vector()
                ),
                obb1,
                obb2,
            )
            or self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_forward_vector(), obb2.rotation.get_up_vector()
                ),
                obb1,
                obb2,
            )
            or self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_right_vector(), obb2.rotation.get_forward_vector()
                ),
                obb1,
                obb2,
            )
            or self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_right_vector(), obb2.rotation.get_right_vector()
                ),
                obb1,
                obb2,
            )
            or self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_right_vector(), obb2.rotation.get_up_vector()
                ),
                obb1,
                obb2,
            )
            or self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_up_vector(), obb2.rotation.get_forward_vector()
                ),
                obb1,
                obb2,
            )
            or self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_up_vector(), obb2.rotation.get_right_vector()
                ),
                obb1,
                obb2,
            )
            or self.get_separating_plane(
                relative_position,
                self.cross_product(
                    obb1.rotation.get_up_vector(), obb2.rotation.get_up_vector()
                ),
                obb1,
                obb2,
            )
        ):

            return False

        # If no separating plane is found, the OBBs intersect
        return True

    def _get_angle_to(self, current_position, current_heading, target_position):
        """
        Calculate the angle (in degrees) from the current position and heading to a target position.

        Args:
            current_position (list): A list of (x, y) coordinates representing the current position.
            current_heading (float): The current heading angle in radians.
            target_position (tuple or list): A tuple or list of (x, y) coordinates representing the target position.

        Returns:
            float: The angle (in degrees) from the current position and heading to the target position.
        """
        cos_heading = math.cos(current_heading)
        sin_heading = math.sin(current_heading)

        # Calculate the vector from the current position to the target position
        position_delta = target_position - current_position

        # Calculate the dot product of the position delta vector and the current heading vector
        aim_x = cos_heading * position_delta[0] + sin_heading * position_delta[1]
        aim_y = -sin_heading * position_delta[0] + cos_heading * position_delta[1]

        # Calculate the angle (in radians) from the current heading to the target position
        angle_radians = -math.atan2(-aim_y, aim_x)

        # Convert the angle from radians to degrees
        angle_degrees = np.float_(math.degrees(angle_radians))

        return angle_degrees

    def get_nearby_object(self, ego_vehicle_position, all_actors, search_radius):
        """
        Find actors, who's trigger boxes are within a specified radius around the ego vehicle.

        Args:
            ego_vehicle_position (carla.Location): The position of the ego vehicle.
            all_actors (list): A list of all actors.
            search_radius (float): The radius (in meters) around the ego vehicle to search for nearby actors.

        Returns:
            list: A list of actors within the specified search radius.
        """
        nearby_objects = []
        for actor in all_actors:
            try:
                trigger_box_global_pos = actor.get_transform().transform(
                    actor.trigger_volume.location
                )
            except:
                print(
                    "Warning! Error caught in get_nearby_objects. (probably AttributeError: actor.trigger_volume)"
                )
                print("Skipping this object.")
                continue

            # Convert the vector to a carla.Location for distance calculation
            trigger_box_global_pos = carla.Location(
                x=trigger_box_global_pos.x,
                y=trigger_box_global_pos.y,
                z=trigger_box_global_pos.z,
            )

            # Check if the actor's trigger volume is within the search radius
            if trigger_box_global_pos.distance(ego_vehicle_position) < search_radius:
                nearby_objects.append(actor)

        return nearby_objects
