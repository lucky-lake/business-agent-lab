"""그라인더 클릭 힌트 테스트. 목표 크기에 맞는 클릭의 시작점을 알려 준다.

같은 봉투의 내 사진 기록이 있으면 그걸로, 없으면 인터넷 클릭표로 알려 준다. 클릭은 시작점일 뿐이고, 갈고 나서
사진으로 실제 크기를 재고 그에 맞춰 레시피를 바꾼다(한 잔 내리기 3단계). 그 잔을 기록하면 다음 힌트에 쓰인다.
"""
import shutil

import pytest

from dripcoffee import DATA_DIR
from dripcoffee import calibration as cal


@pytest.fixture(scope="module")
def brews():
    return cal.load_brews()


def test_same_bag_record_comes_first(brews):
    hint = cal.click_hint(1150, "메버릭1", "B02", "2026-10-15", brews)  # 새 봉투 95클릭 = 1176 µm
    assert hint.clicks == 93 and hint.source == "내 기록" and hint.near.row["roast_date"] == "2026-10-15"
    assert cal.click_hint(1314, "메버릭1", "B01", "2026-09-28", brews).clicks == 95  # 95클릭 1314 µm 기록이 있다


def test_other_bags_of_the_same_bean_are_not_used(brews):
    # 옛 봉투 기록(95클릭 1406 µm)만 있는 크기를 새 봉투에 물으면 내 기록이 아니라 클릭표를 쓴다
    hint = cal.click_hint(1406, "메버릭1", "B02", "2026-10-15", brews)
    assert hint.source == "클릭표"


def test_click_table_is_used_without_a_close_record(brews):
    hint = cal.click_hint(910, "메버릭1", "B07", "2026-10-14", brews)  # B07은 기록이 없다
    assert hint.source == "클릭표" and hint.clicks == 86  # 클릭표: 85클릭 900 µm, 90클릭 950 µm
    far = cal.click_hint(1250, "메버릭1", "B01", "2026-09-28", brews)  # 같은 봉투 기록(가장 작은 1314 µm)에서 60 µm 넘게 멀다
    assert far.source == "클릭표" and far.clicks == 120
    near = cal.click_hint(1720, "메버릭1", "B01", "2026-09-28", brews)  # 105클릭 1683 µm에서 37 µm
    assert near.source == "내 기록" and near.clicks > 105


@pytest.mark.parametrize("size, clicks", [(200, 20), (1000, 95), (1075, 102), (1500, 125)])
def test_click_table_reading(size, clicks):
    assert cal.table_clicks(size, "메버릭1") == clicks  # 표 밖은 가장 가까운 끝
    assert cal.table_clicks(size, "코만단테") is None


def test_a_logged_cup_becomes_the_next_hint(tmp_path):
    # 클릭표로 시작해 갈고 사진으로 잰 잔을 기록하면, 다음 힌트는 그 기록으로 나온다
    path = tmp_path / "brews.csv"
    shutil.copy(DATA_DIR / "brews.csv", path)
    assert cal.click_hint(910, "메버릭1", "B07", "2026-10-14", cal.load_brews(path)).source == "클릭표"
    cal.append_brew({"brew_id": "20261020-1", "date": "2026-10-20", "bean_id": "B07", "roast_date": "2026-10-14",
                     "filter": "홀츠클로츠 무표백", "grinder": "메버릭1", "grinder_clicks": 86, "grind_d50_um": 980,
                     "dose_g": 20, "water_temp_c": 87, "pours": "0:40;60:80;120:40"}, path)
    hint = cal.click_hint(950, "메버릭1", "B07", "2026-10-14", cal.load_brews(path))
    assert hint.source == "내 기록" and hint.near.brew_id == "20261020-1"


def test_size_change_per_click_comes_from_one_bag_at_several_clicks(brews):
    # B01 한 봉투를 85·95·105클릭으로 잰 기록. 인터넷 클릭표(1클릭당 약 10 µm)와 비슷하다.
    # 예전 측정은 큰 입자를 덩어리로 빼서 1클릭당 4 µm쯤으로 작게 나왔다(10/19 고침)
    assert 8 < cal.click_hint(1314, "메버릭1", "B01", "2026-09-28", brews).um_per_click < 16


def test_hint_never_points_at_a_click_that_clogged(brews):
    # B01은 85클릭에서 막혔다. 막힘 한계 바로 위 크기(1430 µm)는 내 기록으로는 85클릭(1429 µm)이지만,
    # 같은 클릭이면 또 막힐 수 있어서 막히지 않고 내린 더 굵은 클릭(95)부터 알려 준다
    hint = cal.click_hint(1430, "메버릭1", "B01", "2026-09-28", brews)
    assert hint.clicks == 95 and "85클릭에서 막힌" in hint.basis
    # 막히지 않은 더 굵은 기록이 없으면 막힌 클릭보다 5클릭 굵게 알려 준다
    only_clogged = [b for b in brews if b.brew_id == "20261016-2"]
    assert cal.click_hint(1430, "메버릭1", "B01", "2026-09-28", only_clogged).clicks == 90
    # 막힌 적 없는 원두와, 막힌 클릭보다 이미 굵은 힌트는 그대로다
    assert cal.click_hint(1150, "메버릭1", "B02", "2026-10-15", brews).clicks == 93
    assert cal.click_hint(1314, "메버릭1", "B01", "2026-09-28", brews).clicks == 95


def test_grinder_without_table_or_records_gives_no_hint(brews):
    assert cal.click_hint(1000, "코만단테", "B02", "2026-10-15", brews) is None
