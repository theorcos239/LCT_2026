from .criteria import axis_decision


def test_suitable_when_interval_below_limit():
    assert axis_decision(3.0, .5, 1.0, 3)["state"] == "suitable"


def test_unsuitable_when_interval_above_limit():
    out = axis_decision(7.0, .5, 1.0, 3)
    assert out["state"] == "unsuitable"
    assert out["violated"] is True


def test_borderline_is_undetermined_not_normal():
    out = axis_decision(5.0, .5, 1.0, 3)
    assert out["state"] == "undetermined"
    assert out["violated"] is None


def test_sources_disagree_is_undetermined():
    out = axis_decision(7.0, .2, 10.0, 3)
    assert out["state"] == "undetermined"
    assert "sources_disagree" in out["reasons"]


def test_missing_sources_is_undetermined():
    assert axis_decision(2.0, .2, 0.0, 1)["state"] == "undetermined"
