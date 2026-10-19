"""내 선호 영역(추천 목표) 계산 테스트. 설계는 docs/MODEL.md 9.1.

잔마다 모델로 (추출수율, 가수 후 농도) 위치를 계산하고, 맛 점수로 가중한 '내 선호 중심'을
골든컵 중심과 섞어 추천 목표를 정한다 (dripcoffee/recommend.py).
"""
import shutil

import pytest

from dripcoffee import DATA_DIR
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
    # 20261012-2(옛 봉투, 사진 없음)는 같은 봉투 20261012-1의 크기를 빌린다.
    # 같은 95클릭이어도 새 봉투 잔 20261018-1의 크기는 섞지 않는다
    brews = cal.load_brews()
    own = {b.brew_id: b.grind_d50_um for b in brews}
    size, borrowed = cal.grind_sizes(brews)["20261012-2"]
    assert borrowed and size == pytest.approx(own["20261012-1"])
    assert own["20261012-1"] != own["20261018-1"]


def test_pooled_grind_size_averages_the_same_bag_and_clicks(tmp_path):
    # 보정에서는 사진이 있는 잔도 같은 봉투·같은 클릭으로 잰 잔들의 평균을 쓴다.
    # B01 95클릭에 표시 없는 잔을 하나 더 넣으면 20261016-3과 평균을 낸다.
    path = tmp_path / "brews.csv"
    shutil.copy(DATA_DIR / "brews.csv", path)
    cal.append_brew({"brew_id": "20261020-1", "date": "2026-10-20", "bean_id": "B01", "roast_date": "2026-09-28",
                     "filter": "하리오 화이트", "grinder": "메버릭1", "grinder_clicks": 95, "grind_d50_um": 1200,
                     "dose_g": 20, "water_temp_c": 82, "pours": "0:40;60:80;120:40"}, path)
    brews = cal.load_brews(path)
    own = {b.brew_id: b.grind_d50_um for b in brews}
    size, borrowed = cal.grind_sizes(brews, pooled=True)["20261016-3"]
    assert not borrowed and size == pytest.approx((own["20261016-3"] + 1200) / 2)
    assert cal.grind_sizes(brews)["20261016-3"] == (own["20261016-3"], False)
    # 표시가 있는 잔(20261009-1 연습, 촘촘히 편 사진)은 평균에 넣지 않고, 다른 클릭(105)은 섞지 않는다
    assert cal.grind_sizes(brews, pooled=True)["20261009-1"][0] == pytest.approx(size)
    assert cal.grind_sizes(brews, pooled=True)["20261016-1"] == (own["20261016-1"], False)
