"""기록한 잔과 모델 예측의 비교 테스트. 보정(MODEL.md 5절) 전후의 오차를 이 비교로 잰다."""
import pytest

from dripcoffee import calibration as cal


@pytest.fixture
def compared():
    return {c.brew.brew_id: c for c in cal.compare_with_log(cal.load_brews())}


def test_every_cup_with_a_grind_size_is_compared(compared):
    # 표시가 있는 잔(연습, 막힘)도 비교에는 넣고, 뺄지는 쓰는 쪽이 표시를 보고 정한다
    assert {"20261009-1", "20261012-1", "20261012-2", "20261016-1", "20261016-2", "20261016-3"} <= set(compared)
    assert compared["20261016-2"].brew.flag == "막힘"


def test_cup_without_a_photo_borrows_from_the_same_bag_and_clicks(compared):
    # 20261012-2는 옛 봉투(9/28 로스팅). 같은 95클릭이어도 새 봉투(10/15) 잔 20261018-1의 1044 µm는 빌리지 않는다
    borrowed = compared["20261012-2"]
    assert borrowed.borrowed_grind and borrowed.grind_d50_um == pytest.approx(1255)
    assert not compared["20261012-1"].borrowed_grind


def test_errors_are_prediction_minus_actual(compared):
    c = compared["20261016-1"]  # 실제 2:38, 118 g
    assert c.end_error_s == pytest.approx(c.prediction.drawdown_end_s - 158)
    assert c.beverage_error_g == pytest.approx(c.prediction.beverage_g - 118)
    assert c.end_error_s < 0  # 보정 전 모델은 이 잔을 짧게 본다 (10/16 기록)


def test_values_not_measured_give_no_error(compared):
    assert compared["20261009-1"].beverage_error_g is None  # 이 잔은 음료 무게를 못 잼


def test_cup_without_any_grind_size_is_left_out():
    brews = cal.load_brews()
    for b in brews:
        if b.bean_id == "B02":
            b.grind_d50_um = None  # B02는 빌려 올 사진 측정도 없게 만든다
    ids = {c.brew.brew_id for c in cal.compare_with_log(brews)}
    assert "20261012-1" not in ids and "20261012-2" not in ids
    assert "20261016-1" in ids
