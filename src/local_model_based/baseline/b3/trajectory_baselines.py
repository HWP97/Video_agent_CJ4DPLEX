"""동일한 관측 중심점 궤적의 유지/CV/Kalman 외삽. 좌표 단위는 pixel이다."""

import numpy as np


def forecast(history, future_times, method):
    times = np.asarray([point["time_s"] for point in history], dtype=float)
    xy = np.asarray([[point["x"], point["y"]] for point in history], dtype=float)
    if len(times) < 3 or np.any(np.diff(times) <= 0):
        raise ValueError("at_least_3_monotonic_observations_required")
    future_times = np.asarray(future_times, dtype=float)
    delta = future_times - times[-1]
    if np.any(delta <= 0):
        raise ValueError("future_times_must_follow_observations")
    if method == "hold":
        positions = np.repeat(xy[-1:], len(delta), axis=0)
    elif method == "constant_velocity":
        design = np.column_stack([times - times[-1], np.ones(len(times))])
        velocity = np.linalg.lstsq(design, xy, rcond=None)[0][0]
        positions = xy[-1] + delta[:, None] * velocity
    elif method == "kalman":
        state = np.array([xy[0, 0], xy[0, 1], 0.0, 0.0])
        covariance = np.diag([25.0, 25.0, 100.0, 100.0])
        observation = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        noise = np.eye(2) * 4.0
        for index in range(1, len(times)):
            dt = times[index] - times[index - 1]
            transition = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
            acceleration = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
            state = transition @ state
            covariance = transition @ covariance @ transition.T + acceleration @ acceleration.T * 9.0
            innovation = xy[index] - observation @ state
            residual_covariance = observation @ covariance @ observation.T + noise
            gain = np.linalg.solve(residual_covariance, observation @ covariance).T
            state = state + gain @ innovation
            covariance = (np.eye(4) - gain @ observation) @ covariance
        positions = state[:2] + delta[:, None] * state[2:]
    else:
        raise ValueError("unknown_method")
    return [{"time_s": float(timestamp), "x": float(point[0]), "y": float(point[1])}
            for timestamp, point in zip(future_times, positions)]
