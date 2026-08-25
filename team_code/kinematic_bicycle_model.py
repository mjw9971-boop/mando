"""
Kinematic bicycle model describing the motion of a car given its state and action.
"""

import numpy as np


class KinematicBicycleModel:
    """
    Kinematic bicycle model describing the motion of a car given its state and action.
    """

    def __init__(self, config):
        """
        Kinematic bicycle model describing the motion of a car given its state and action.
        Tuned parameters are taken from World on Rails.

        Args:
            config (GlobalConfig): Object of the config for hyperparameters.
        """
        self.config = config

        self.time_step = self.config.time_step
        self.front_wheel_base = self.config.front_wheel_base
        self.rear_wheel_base = self.config.rear_wheel_base
        self.steering_gain = self.config.steering_gain
        self.brake_acceleration = self.config.brake_acceleration
        self.throttle_acceleration = self.config.throttle_acceleration
        self.throttle_values = self.config.throttle_values
        self.brake_values = self.config.brake_values
        self.throttle_threshold_during_forecasting = (
            self.config.throttle_threshold_during_forecasting
        )

    def forecast_other_vehicles(self, locations, headings, speeds, actions):
        """
        Forecast the future states of other vehicles based on their current states and actions.
        Tuned parameters are taken from World on Rails.

        Args:
            locations (numpy.ndarray): Array of (x, y, z) coordinates representing the locations of other vehicles.
            headings (numpy.ndarray): Array of heading angles (in radians) for other vehicles.
            speeds (numpy.ndarray): Array of speeds (in m/s) for other vehicles.
            actions (numpy.ndarray): Array of actions (steer, throttle, brake) for other vehicles.

        Returns:
            tuple: A tuple containing the forecasted locations, headings, and speeds for other vehicles.
        """
        steers, throttles, brakes = (
            actions[:, 0],
            actions[:, 1],
            actions[:, 2].astype(np.uint8),
        )
        wheel_angles = self.steering_gain * steers
        slip_angles = np.arctan(
            self.rear_wheel_base
            / (self.front_wheel_base + self.rear_wheel_base)
            * np.tan(wheel_angles)
        )

        next_x = (
            locations[:, 0] + speeds * np.cos(headings + slip_angles) * self.time_step
        )
        next_y = (
            locations[:, 1] + speeds * np.sin(headings + slip_angles) * self.time_step
        )
        next_headings = (
            headings
            + speeds / self.rear_wheel_base * np.sin(slip_angles) * self.time_step
        )

        next_speeds = speeds + self.time_step * np.where(
            brakes, self.brake_acceleration, throttles * self.throttle_acceleration
        )
        next_speeds = np.maximum(0.0, next_speeds)

        next_locations = np.column_stack([next_x, next_y, locations[:, 2]])

        return next_locations, next_headings, next_speeds

    def forecast_ego_vehicle(self, location, heading, speed, action):
        """
        Forecast the future state of the ego vehicle based on its current state and action.

        Args:
            location (numpy.ndarray): Array of (x, y, z) coordinates representing the location of the ego vehicle.
            heading (float): Current heading angle (in radians) of the ego vehicle.
            speed (float): Current speed (in m/s) of the ego vehicle.
            action (numpy.ndarray): Action (steer, throttle, brake) for the ego vehicle.

        Returns:
            tuple: A tuple containing the forecasted location, heading, and speed for the ego vehicle.
        """
        steer, throttle, brake = action
        wheel_angle = self.steering_gain * steer
        slip_angle = np.arctan(
            self.rear_wheel_base
            / (self.front_wheel_base + self.rear_wheel_base)
            * np.tan(wheel_angle)
        )

        next_x = location[0] + speed * np.cos(heading + slip_angle) * self.time_step
        next_y = location[1] + speed * np.sin(heading + slip_angle) * self.time_step
        next_heading = (
            heading + speed / self.rear_wheel_base * np.sin(slip_angle) * self.time_step
        )

        # VTD: 원문은 CARLA Lincoln 에 캘리브레이션한 throttle/brake 다항식으로
        # 속도를 예측했다. 우리 종방향은 VtdLongitudinalController 가 accel
        # [m/s²] 를 직접 내므로(phase0 §0-4 (b)) throttle 자리가 accel 이다 —
        # 등가속 적분으로 대체. brake 분기는 ego forecast 에서 항상 0 이라
        # (forecast_ego_agent 가 action=[steer, accel, 0.0]) 같은 식으로 합친다.
        # 원 다항식은 git 이력(phase3 원본 커밋) / DriveLM 원본 저장소 참조.
        accel = self.brake_acceleration if brake else throttle
        next_speed = np.maximum(0.0, speed + accel * self.time_step)
        next_location = np.array([next_x[0], next_y[0], location[2]])

        return next_location, next_heading, next_speed
