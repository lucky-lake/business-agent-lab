"""내 선호 영역(추천 목표) 계산 테스트. 설계는 docs/MODEL.md 9.1.

잔마다 모델로 (추출수율, 가수 후 농도) 위치를 계산하고, 맛 점수로 가중한 '내 선호 중심'을
골든컵 중심과 섞어 추천 목표를 정한다 (dripcoffee/recommend.py).
"""
import pytest

from dripcoffee import calibration as cal
from dripcoffee import recommend as rec


def point(ey, tds, score, brew_id="x"):
    return rec.BrewPoint(brew_id=brew_id, bean_id="B02", ey_pct=ey, tds_cup_pct=tds, score=score)


def test_without_liked_cups_the_target_is_the_golden_cup():
    target = rec.preference_target([point(15, 1.0, 2.0), point(16, 1.1, 2.5)])
    assert (target.ey_pct, target.tds_pct) == pytest.approx(rec.GOLDEN_CUP_CENTER)
    assert target.n_liked == 0 and target.personal_center is None


def test_better_cups_pull_harder():
    # 가중치 = 맛 점수 − 2.5 (2.5 이하는 0): 4.0점은 1.5, 3.0점은 0.5
    assert point(18, 1.3, 4.0).weight == pytest.approx(1.5)
    assert point(18, 1.3, 3.0).weight == pytest.approx(0.5)
    assert point(18, 1.3, 2.0).weight == 0
    target = rec.preference_target([point(16, 1.2, 4.0), point(22, 1.4, 3.0)])
    center_ey = (16 * 1.5 + 22 * 0.5) / 2.0
    assert target.personal_center[0] == pytest.approx(center_ey)


def test_target_moves_from_golden_cup_toward_my_center_as_records_grow():
    golden = rec.GOLDEN_CUP_CENTER
    few = rec.preference_target([point(17, 1.4, 4.0)] * 1)
    many = rec.preference_target([point(17, 1.4, 4.0)] * 15)
    # 목표 = (n × 내 선호 + 5 × 골든컵) ÷ (n + 5)
    assert few.ey_pct == pytest.approx((1 * 17 + 5 * golden[0]) / 6)
    assert many.ey_pct == pytest.approx((15 * 17 + 5 * golden[0]) / 20)
    assert abs(many.ey_pct - 17) < abs(few.ey_pct - 17)


def test_points_come_from_the_log_without_flagged_cups():
    points = rec.brew_points(cal.load_brews())
    ids = {p.brew_id for p in points}
    assert "20261009-1" not in ids  # 연습
    assert "20261016-2" not in ids  # 막힘
    assert {"20261012-1", "20261016-1"} <= ids
    # 사진이 없는 잔은 같은 원두·같은 클릭의 다른 잔 측정값을 빌려 쓴다
    assert "20261012-2" in ids
    for p in points:
        assert 5 < p.ey_pct < 30 and 0.3 < p.tds_cup_pct < 3


def test_cups_without_taste_scores_are_left_out():
    brews = cal.load_brews()
    for b in brews:
        b.taste = {k: None for k in b.taste}
    assert rec.brew_points(brews) == []


def test_borrowed_grind_size_comes_from_the_same_bag():
    # 20261012-2(옛 봉투, 사진 없음)는 같은 봉투 20261012-1의 1255 µm를 빌린다.
    # 같은 95클릭이어도 새 봉투 잔 20261018-1의 1044 µm는 섞지 않는다
    size, borrowed = cal.grind_sizes(cal.load_brews())["20261012-2"]
    assert borrowed and size == pytest.approx(1255)
