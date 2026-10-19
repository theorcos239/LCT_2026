from .pipeline import record_axis_review
from .keypoints_backend import axis_from_contour_only, fuse_axis_with_contour


def test_review_state_is_not_suitable():
    out = {'flags': [], 'violations': []}
    record_axis_review(out, 'review')
    assert out['violations'] == ['undetermined']
    assert 'axis:needs_review' in out['flags']


def test_violation_policy_marks_axis():
    out = {'flags': [], 'violations': []}
    record_axis_review(out, 'violation')
    assert out['violations'] == ['undetermined', 'spine_axis']


def test_contour_only_uses_same_three_state_policy():
    out = axis_from_contour_only({'angle_deg': 5.0, 'curvature_mm': 0.0})
    assert out['state'] == 'undetermined'
    assert out['violated'] is None


def test_missing_model_points_require_review():
    out = axis_from_contour_only({'angle_deg': 0.0}, points_required=True)
    assert out['state'] == 'undetermined'
    assert out['violated'] is None
    assert 'axis:points_missing' in out['flags']


def test_scoliosis_like_contour_prevents_automatic_point_axis_violation():
    point = {'sources': {'theta_centers_deg': 10.0,
                         'theta_endplates_deg': 10.0},
             'curve': {'step_mm': 30.0}, 'angle_centers_deg': 10.0,
             'variance_params': {}, 'flags': []}
    contour = {'angle_deg': 10.0, 'curvature_mm': 8.0,
               'flags': ['axis:curved_suppressed']}
    out = fuse_axis_with_contour(point, contour)
    assert out['state'] == 'undetermined'
    assert out['violated'] is None
    assert 'scoliosis_like_contour' in out['reasons']
    assert 'axis:needs_review' in out['flags']
