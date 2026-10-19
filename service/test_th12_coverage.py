"""Половина тела Th12 по ключевым точкам — двухпороговое правило."""
from __future__ import annotations

from .keypoints_backend import (TH12_BODY_TO_STEP, TH12_HARD_FRAC,
                                apply_th12_to_position, record_th12_review,
                                th12_coverage_from_points)


def _get(points):
    return points.get


def test_both_endplates_is_ok():
    pts = {
        'th12_top': (100.0, 10.0),
        'th12_bottom': (100.0, 48.0),
        'l1_center': (100.0, 80.0),
        'l2_center': (100.0, 120.0),
    }
    out = th12_coverage_from_points(_get(pts), image_height=300)
    assert out['state'] == 'suitable'
    assert out['violated'] is False
    assert out['frac'] == 1.0
    assert out['reason'] == 'both_endplates'


def test_half_body_visible_is_ok():
    step = 40.0
    bot_y = 0.65 * TH12_BODY_TO_STEP * step
    pts = {
        'th12_bottom': (100.0, bot_y),
        'l1_center': (100.0, 80.0),
        'l2_center': (100.0, 80.0 + step),
    }
    out = th12_coverage_from_points(_get(pts), image_height=300)
    assert out['state'] == 'suitable'
    assert out['violated'] is False
    assert out['reason'] == 'bottom_only'


def test_mid_band_is_review_not_hard():
    """ТЗ < 50%, но выше hard — undetermined, не spine_position."""
    step = 40.0
    bot_y = 0.30 * TH12_BODY_TO_STEP * step  # между hard 0.13 и half 0.5
    pts = {
        'th12_bottom': (100.0, bot_y),
        'l1_center': (100.0, 80.0),
        'l2_center': (100.0, 80.0 + step),
    }
    out = th12_coverage_from_points(_get(pts), image_height=300)
    assert TH12_HARD_FRAC <= out['frac'] < 0.5
    assert out['state'] == 'undetermined'
    assert out['violated'] is None
    assert out['needs_review'] is True
    assert 'position:th12_review' in out['flags']


def test_hard_short_is_violation():
    step = 40.0
    bot_y = 0.08 * TH12_BODY_TO_STEP * step  # < 0.13
    pts = {
        'th12_bottom': (100.0, bot_y),
        'l1_center': (100.0, 80.0),
        'l2_center': (100.0, 80.0 + step),
    }
    out = th12_coverage_from_points(_get(pts), image_height=300)
    assert out['frac'] < TH12_HARD_FRAC
    assert out['state'] == 'unsuitable'
    assert out['violated'] is True
    assert 'position:th12_short' in out['flags']


def test_bottom_missing_is_review_not_hard():
    pts = {
        'l1_center': (100.0, 80.0),
        'l2_center': (100.0, 120.0),
    }
    out = th12_coverage_from_points(_get(pts), image_height=300)
    assert out['violated'] is None
    assert out['state'] == 'undetermined'
    assert out['reason'] == 'bottom_missing'
    assert out['needs_review'] is True
    assert 'position:th12_bottom_missing' in out['flags']


def test_prior_short_is_undetermined_not_violation():
    pts = {'th12_bottom': (100.0, 5.0)}
    out = th12_coverage_from_points(_get(pts), image_height=300)
    assert out['body_source'] == 'prior'
    assert out['state'] == 'undetermined'
    assert out['violated'] is None


def test_apply_hard_ors_into_position():
    position = {'violated': False, 'text': '', 'flags': [], 'iliac_area': 0.05}
    th12 = th12_coverage_from_points(_get({
        'th12_bottom': (100.0, 3.0),
        'l1_center': (100.0, 80.0),
        'l2_center': (100.0, 120.0),
    }), image_height=300)
    assert th12['violated'] is True
    out = apply_th12_to_position(position, th12)
    assert out['violated'] is True
    assert out['th12_violated'] is True


def test_apply_review_does_not_force_position():
    position = {'violated': False, 'text': '', 'flags': []}
    th12 = th12_coverage_from_points(_get({
        'th12_bottom': (100.0, 12.0),
        'l1_center': (100.0, 80.0),
        'l2_center': (100.0, 120.0),
    }), image_height=300)
    assert th12['violated'] is None
    out = apply_th12_to_position(position, th12)
    assert out['violated'] is False
    assert out['th12_needs_review'] is True


def test_record_th12_review_adds_undetermined():
    out = {'flags': [], 'violations': []}
    position = {'violated': False, 'th12_needs_review': True}
    record_th12_review(out, position)
    assert 'undetermined' in out['violations']
    assert 'position:th12_needs_review' in out['flags']


def test_record_th12_review_skipped_if_already_violated():
    out = {'flags': [], 'violations': ['spine_position']}
    position = {'violated': True, 'th12_needs_review': True}
    record_th12_review(out, position)
    assert out['violations'] == ['spine_position']


def test_apply_keeps_iliac_violation_text():
    position = {
        'violated': True,
        'text': 'укладка: не визуализированы верхние края подвздошных костей',
        'flags': ['position:iliac_missing_one_side'],
    }
    th12 = {
        'violated': True, 'frac': 0.08, 'flags': ['position:th12_short'],
        'text': 'укладка: видна малая доля тела Th12 (8% при hard-пороге 15%, норма ТЗ ≥50%)',
        'state': 'unsuitable', 'needs_review': False,
    }
    out = apply_th12_to_position(position, th12)
    assert 'подвздошных' in out['text']
    assert 'Th12' in out['text']
