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


def test_scoliosis_like_point_curve_prevents_automatic_violation():
    """Большой угол + сильная кривизна / segment_range → не автоматический брак."""
    from .keypoints_backend import axis_from_points

    # Дуга со скошенной хордой: |угол| > 5° и curvature > 12 мм.
    pts = {
        'l1_center': (110.0, 30.0),
        'l2_center': (150.0, 80.0),
        'l3_center': (130.0, 140.0),
        'l4_center': (90.0, 200.0),
        # Пластинки с тем же знаком наклона, иначе свёртка уводит угол к 0°.
        'l1_edge_left': (100.0, 28.0), 'l1_edge_right': (120.0, 32.0),
        'l2_edge_left': (140.0, 78.0), 'l2_edge_right': (160.0, 82.0),
        'l3_edge_left': (120.0, 138.0), 'l3_edge_right': (140.0, 142.0),
        'l4_edge_left': (80.0, 198.0), 'l4_edge_right': (100.0, 202.0),
    }
    out = axis_from_points(pts.get)
    assert out is not None
    assert 'axis:curved' in out['flags']
    assert out['violated'] is not True
    assert out['scoliosis']['present'] is True
    # Сильный segment_range гасит до OK; иначе остаётся review по кривизне.
    if out['scoliosis']['strength'] == 'strong':
        assert out['state'] == 'suitable'
        assert 'axis:scoliosis_suppressed' in out['flags']
    else:
        assert out['state'] == 'undetermined'
        assert 'scoliosis_like_curve' in out['reasons']


def test_report_scoliosis_column_in_extra():
    from .report import EXTRA
    assert 'scoliosis' in EXTRA


def test_point_clears_mild_contour_axis_fp():
    from .keypoints_backend import reconcile_contour_axis_with_points

    contour = {'angle_deg': 4.04, 'violated': True, 'text': 'ось...', 'flags': []}
    point = {'angle_deg': 1.88, 'angle_centers_deg': 2.19, 'flags': []}
    out = reconcile_contour_axis_with_points(contour, point)
    assert out['violated'] is False
    assert 'axis:point_clears_contour' in out['flags']


def test_point_over_tz_is_flag_only_without_disagree():
    """Точки ≥5° при контурной норме и согласии краёв — только флаг."""
    from .keypoints_backend import reconcile_contour_axis_with_points

    contour = {
        'angle_deg': 2.68, 'violated': False, 'text': '', 'flags': [],
        'disagreement_deg': 0.5,
    }
    point = {'angle_deg': 6.13, 'angle_centers_deg': 5.73, 'flags': []}
    out = reconcile_contour_axis_with_points(contour, point)
    assert out['violated'] is False
    assert 'axis:point_tilt_over_tz' in out['flags']


def test_point_overrides_contour_when_edges_disagree():
    """disagree>2° и θ_точек≥5° → брак по точкам (FN контура 2.7°/5.7°)."""
    from .keypoints_backend import reconcile_contour_axis_with_points

    contour = {
        'angle_deg': 2.68, 'violated': False, 'text': '',
        'flags': ['axis:estimates_disagree'], 'disagreement_deg': 3.1,
    }
    point = {'angle_deg': 5.71, 'angle_centers_deg': 5.72, 'flags': []}
    out = reconcile_contour_axis_with_points(contour, point)
    assert out['violated'] is True
    assert 'axis:point_overrides_contour' in out['flags']


def test_strong_scoliosis_clears_point_override():
    from .keypoints_backend import reconcile_contour_axis_with_points
    from dxa_qc.spine_shape import apply_scoliosis_to_axis

    contour = {
        'angle_deg': 5.65, 'violated': True, 'text': 'ось...', 'flags': [],
        'disagreement_deg': 1.0,
    }
    point = {'angle_deg': 8.4, 'flags': []}
    axis = reconcile_contour_axis_with_points(contour, point)
    assert axis['violated'] is True
    scoliosis = {
        'available': True, 'present': True, 'strength': 'strong',
        'segment_range_deg': 14.0, 'lateral_dev_mm': 2.0,
    }
    out = apply_scoliosis_to_axis(axis, scoliosis)
    assert out['violated'] is False
    assert 'axis:scoliosis_suppressed' in out['flags']

def test_point_reconcile_keeps_borderline_point_angle():
    """θ_точек ≥ clear (3°) — контурный брак не гасим (мягкая ось)."""
    from .keypoints_backend import reconcile_contour_axis_with_points

    contour = {'angle_deg': 4.26, 'violated': True, 'text': 'ось...', 'flags': []}
    point = {'angle_deg': 3.42, 'flags': []}
    out = reconcile_contour_axis_with_points(contour, point)
    assert out['violated'] is True
    assert 'axis:point_no_corroboration' not in out['flags']


def test_point_reconcile_keeps_true_positive():
    from .keypoints_backend import reconcile_contour_axis_with_points

    contour = {'angle_deg': 4.09, 'violated': True, 'text': 'ось...', 'flags': []}
    point = {'angle_deg': 4.48, 'flags': []}
    out = reconcile_contour_axis_with_points(contour, point)
    assert out['violated'] is True


def test_rotation_points_clear_normal_bulge():
    from .keypoints_backend import judge_rotation_with_points

    rot = {'violated': True, 'text': 'ротация...', 'flags': [], 'geometry_violated': True}
    out = judge_rotation_with_points(rot, 5.5)
    assert out['violated'] is False
    assert 'rotation:points_normal_bulge' in out['flags']


def test_rotation_points_keep_abnormal_bulge():
    from .keypoints_backend import judge_rotation_with_points

    rot = {'violated': True, 'text': 'ротация...', 'flags': []}
    out = judge_rotation_with_points(rot, 11.0)
    assert out['violated'] is True
    assert 'rotation:points_abnormal_bulge' in out['flags']
