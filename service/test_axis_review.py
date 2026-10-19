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
