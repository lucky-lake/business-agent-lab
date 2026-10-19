"""추출 기록(brews.csv) 읽기·쓰기와 입력 확인 테스트.

앱의 기록 탭은 이 함수들(dripcoffee/calibration.py)로 기록을 저장한다.
"""
import shutil

import pytest

from dripcoffee import DATA_DIR
from dripcoffee import calibration as cal


def new_record(**changes):
    record = {
        "brew_id": "20261017-1", "date": "2026-10-17", "bean_id": "B02", "bean_name": "너티 브리즈",
        "roaster": "루비아", "roast_level": "미디엄", "dripper": "V60", "filter": "홀츠클로츠 무표백", "grinder": "메버릭1",
        "grinder_clicks": "95", "grind_d50_um": "1255", "dose_g": "20", "water_temp_c": "92",
        "pours": "0:40;30:60;60:60;90:60", "drawdown_end_s": "141", "beverage_g": "180.5", "bypass_g": "80",
        "aroma": "4", "flavor": "4", "acidity": "1", "body": "3.5", "balance": "4", "aftertaste": "3.5",
        "memo": "테스트", "flag": "",
    }
    record.update(changes)
    return record


@pytest.fixture
def brews_copy(tmp_path):
    path = tmp_path / "brews.csv"
    shutil.copy(DATA_DIR / "brews.csv", path)
    return path


def test_existing_log_is_read_with_numbers_flags_and_scores():
    brews = {b.brew_id: b for b in cal.load_brews()}
    assert {"20261009-1", "20261012-1", "20261012-2", "20261016-1", "20261016-2"} <= set(brews)
    assert brews["20261009-1"].flag == "연습"
    assert brews["20261016-2"].flag == "막힘"
    b = brews["20261016-1"]
    assert (b.grinder_clicks, b.grind_d50_um, b.drawdown_end_s, b.beverage_g) == (105, 1683.0, 158.0, 118.0)
    # 만족도 점수 = Aroma, Flavor, Balance, Aftertaste의 평균 (산미·바디는 세기라서 뺀다)
    assert b.quality_score == pytest.approx((4.5 + 3.5 + 4 + 3.5) / 4)
    assert brews["20261012-2"].grind_d50_um is None  # 사진이 없는 잔은 빈칸
    # 필터: 기숙사(월~수)는 홀츠클로츠 무표백, 본가(목~일)는 하리오 화이트
    assert brews["20261012-1"].filter_paper == "홀츠클로츠 무표백"
    assert brews["20261016-1"].filter_paper == "하리오 화이트"


def test_new_record_is_appended_and_read_back(brews_copy):
    before = len(cal.load_brews(brews_copy))
    cal.append_brew(new_record(), brews_copy)
    brews = cal.load_brews(brews_copy)
    assert len(brews) == before + 1
    assert brews[-1].brew_id == "20261017-1" and brews[-1].quality_score == pytest.approx(3.875)
    header = brews_copy.read_text(encoding="utf-8").splitlines()[0]
    assert header == (DATA_DIR / "brews.csv").read_text(encoding="utf-8").splitlines()[0]  # 칸 순서 유지


@pytest.mark.parametrize("changes, message", [
    ({"brew_id": "20261016-1"}, "이미 있는 추출번호"),
    ({"pours": "0:40;abc"}, "붓기 기록"),
    ({"aroma": "0"}, "맛 점수"),  # 만족도 항목은 1~5
    ({"acidity": "6"}, "맛 점수"),  # 세기 항목은 0~5
    ({"flavor": "3.3"}, "0.5 단위"),
    ({"flag": "망함"}, "표시"),
    ({"dose_g": "0"}, "원두 무게"),
    ({"filter": ""}, "필터"),  # 필터마다 물 빠짐이 달라질 수 있어 꼭 적는다
])
def test_invalid_records_are_refused_with_a_clear_message(brews_copy, changes, message):
    before = brews_copy.read_text(encoding="utf-8")
    with pytest.raises(cal.BrewLogError, match=message):
        cal.append_brew(new_record(**changes), brews_copy)
    assert brews_copy.read_text(encoding="utf-8") == before  # 잘못된 기록은 저장하지 않는다


def test_taste_can_be_left_blank(brews_copy):
    blank = {k: "" for k in ("aroma", "flavor", "acidity", "body", "balance", "aftertaste")}
    cal.append_brew(new_record(**blank), brews_copy)
    assert cal.load_brews(brews_copy)[-1].quality_score is None


def test_empty_values_are_saved_blank(brews_copy):
    # 앱의 빈 숫자 칸은 None으로 넘어온다
    cal.append_brew(new_record(beverage_g=None, grind_d50_um=None, dose_g=20.0), brews_copy)
    b = cal.load_brews(brews_copy)[-1]
    assert b.beverage_g is None and b.grind_d50_um is None
    assert b.row["dose_g"] == "20" and b.row["brew_water_g"] == "220"  # 부은 물 양은 붓기 합으로 채운다


def test_next_brew_id_counts_cups_of_the_same_day():
    brews = cal.load_brews()
    assert cal.next_brew_id("2026-10-16", brews) == "20261016-4"
    assert cal.next_brew_id("2026-10-30", brews) == "20261030-1"


@pytest.mark.parametrize("text, seconds", [("2:42", 162), ("162", 162), ("0:58", 58), ("", None)])
def test_drawdown_time_can_be_written_as_minutes_and_seconds(text, seconds):
    assert cal.parse_clock(text) == seconds


@pytest.mark.parametrize("text", ["2:75", "abc", "-5"])
def test_unreadable_drawdown_time_is_refused(text):
    with pytest.raises(cal.BrewLogError, match="물 빠짐 종료 시각"):
        cal.parse_clock(text)


def test_every_logged_bean_is_in_the_bean_list():
    beans = cal.load_beans()
    assert {"B01", "B02", "B03", "B05", "B06", "B07", "B08", "S1"} <= set(beans)
    assert {b.bean_id for b in cal.load_brews()} <= set(beans)
    assert {row["role"] for row in beans.values()} <= set(cal.BEAN_ROLES)
