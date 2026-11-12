"""표시 보간이 실제 시각을 유지하고 미래를 미리 노출하지 않는지 검사한다."""
import numpy as np
from dataclasses import replace

from b3_data import Case, Observation, Track, Truth, Score
from trajectory_selection import quality, passes, select_source

from b3_render import visible_path
from b3_projection import project, source_time


def test_interpolation_uses_physical_time_without_changing_source() -> None:
    points = np.array([[0., 2.], [4., 6.], [8., 10.]])
    times = np.array([3.2, 3.6, 4.])
    displayed = visible_path(points, times, 3.4)
    np.testing.assert_allclose(displayed, [[0., 2.], [2., 4.]])
    np.testing.assert_array_equal(points, [[0., 2.], [4., 6.], [8., 10.]])


def test_future_path_is_hidden_before_cutoff_and_exact_at_endpoint() -> None:
    points = np.array([[0., 0.], [1., 2.]])
    times = np.array([3.2, 3.6])
    assert visible_path(points, times, 3.19).shape == (0, 2)
    np.testing.assert_array_equal(visible_path(points, times, 3.6), points)


def test_centered_world_coordinates_project_to_native_video_pixels() -> None:
    points = np.array([[0., 0.], [2.1, 2.39], [-2.1, -2.39]])
    np.testing.assert_allclose(project(points, (720, 576)), [[360, 288], [460, 188], [260, 388]])


def test_future_half_speed_keeps_observation_and_forecast_boundary() -> None:
    np.testing.assert_allclose([source_time(t) for t in [0., 3.2, 4., 12.8]], [0., 3.2, 3.6, 8.])


def selection_case(start: int, person: int, error_px: float = 5.) -> Case:
    future = np.column_stack((np.arange(1, 13) * .2, np.zeros(12)))
    observation = Observation(sample_id=f"sample_{person}", primary_id=person, dt_s=.4, cutoff_s=3.2,
        cutoff_frame=start+80, tracks=[Track(pedestrian_id=person, points=[(start+10*i, .4*i, 0., 0.) for i in range(9)])],
        future_frames=[start+90+10*i for i in range(12)], future_times_s=[3.6+.4*i for i in range(12)])
    truth = Truth(sample_id=observation.sample_id, paths=[[(frame, person, *xy) for frame, xy in zip(observation.future_frames, future)]])
    score = Score(sample_id=observation.sample_id, ade=0., fde=0., collision_gt=0)
    return Case(observation.sample_id, "crowds_zara01", observation, truth,
                future+np.array([1., 0.]), future+np.array([error_px*.021, 0.]), score, score)


def test_joint_selection_avoids_a_low_error_interval_that_blocks_two_required_clips() -> None:
    available = [selection_case(i*210, i) for i in range(10)]
    bridge = selection_case(100, 10, 1.)
    inaccurate = selection_case(2400, 11, 30.)
    chosen = select_source([bridge, inaccurate, *available], 210)
    assert {case.sample_id for case in chosen} == {case.sample_id for case in available}


def test_quality_measures_anisotropic_native_pixels_and_rejects_offscreen_points() -> None:
    case = selection_case(0, 0)
    case = replace(case, ours=case.primary_truth+np.array([.063, .0956]))
    value = quality(case)
    np.testing.assert_allclose([value.mean_px, value.final_px, value.max_px], [5., 5., 5.])
    assert passes(value)
    shifted = case.primary_truth + [20., 0.]
    truth = Truth(sample_id=case.sample_id, paths=[[(frame, 0, *xy) for frame, xy in zip(case.observation.future_frames, shifted)]])
    outside = replace(case, truth=truth, ours=shifted)
    assert quality(outside).mean_px == 0
    assert not passes(quality(outside))
