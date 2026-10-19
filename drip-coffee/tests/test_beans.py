"""원두 목록에 새 원두 더하기와, 로스팅 날짜로 본 디게싱(원두에서 가스가 빠지는 과정) 상태 테스트."""
import shutil
from datetime import date

import pytest

from dripcoffee import DATA_DIR
from dripcoffee import calibration as cal

TODAY = date(2026, 10, 19)


def new_bean(**changes):
    bean = {"name": "에티오피아 예가체프", "roaster": "어느 로스터리", "roast_date": "2026-10-12",
            "roast_level": "라이트", "origin": "에티오피아", "process": "워시드", "cup_notes": "자스민, 레몬", "amount_g": "200"}
    bean.update(changes)
    return bean


@pytest.fixture
def beans_copy(tmp_path):
    path = tmp_path / "beans.csv"
    shutil.copy(DATA_DIR / "beans.csv", path)
    return path


def test_new_bean_gets_the_next_number_and_a_normal_role(beans_copy):
    bean_id = cal.append_bean(new_bean(), beans_copy, today=TODAY)
    assert bean_id == "B09"  # B01–B08 다음 (샘플 S1은 번호 체계가 달라 세지 않는다)
    bean = cal.load_beans(beans_copy)["B09"]
    assert (bean["role"], bean["cup_notes"], bean["roast_level"]) == ("일반", "자스민, 레몬", "라이트")
    assert cal.append_bean(new_bean(name="케냐 AA"), beans_copy, today=TODAY) == "B10"


def test_roast_level_spellings_are_stored_as_one_name(beans_copy):
    bean_id = cal.append_bean(new_bean(roast_level="중강배전"), beans_copy, today=TODAY)
    assert cal.load_beans(beans_copy)[bean_id]["roast_level"] == "미디엄 다크"


@pytest.mark.parametrize("changes, message", [
    ({"name": ""}, "커피 이름"),
    ({"roast_date": "10/12"}, "로스팅 날짜"),
    ({"roast_date": "2026-10-25"}, "오늘보다 뒤"),
    ({"roast_level": "시나몬"}, "배전도"),
    ({"amount_g": "많이"}, "양"),
    ({"name": "블루문 디카페인", "roaster": "루비아", "roast_date": "2026-09-28"}, "이미 있는 원두"),
])
def test_invalid_beans_are_refused(beans_copy, changes, message):
    before = beans_copy.read_text(encoding="utf-8")
    with pytest.raises(cal.BeanError, match=message):
        cal.append_bean(new_bean(**changes), beans_copy, today=TODAY)
    assert beans_copy.read_text(encoding="utf-8") == before  # 잘못된 원두는 저장하지 않는다


@pytest.mark.parametrize("roast, roast_date, state", [
    ("미디엄", "2026-10-17", "디게싱 중"),     # 2일째
    ("미디엄", "2026-10-15", "거의 끝남"),     # 4일째 (B02 새 봉투)
    ("미디엄", "2026-10-10", "좋음"),          # 9일째
    ("미디엄", "2026-09-10", "상미 기간 지남"),  # 39일째
    ("라이트", "2026-10-13", "거의 끝남"),     # 6일째: 라이트는 7일부터
    ("강배전", "2026-10-18", "디게싱 중"),     # 1일째
    ("강배전", "2026-10-15", "거의 끝남"),     # 4일째: 다크는 5일부터
    ("강배전", "2026-10-14", "좋음"),       # 5일째
    ("미디엄", "2026-10-21", "날짜 확인"),
])
def test_rest_status_follows_the_roast_level(roast, roast_date, state):
    assert cal.rest_status(roast, roast_date, TODAY).state == state


def test_rest_message_explains_the_words_for_a_first_time_user():
    status = cal.rest_status("미디엄", "2026-10-15", TODAY)
    assert status.days == 4 and status.label == "로스팅 4일째 · 디게싱이 거의 끝나 가요"
    assert "디게싱은 로스팅한 원두에서 가스" in status.message and "미디엄은 보통 7일째부터" in status.message
    assert "다크는" in cal.rest_status("다크", "2026-10-18", TODAY).message  # 받침에 맞는 조사
    assert "상미 기간은" in cal.rest_status("미디엄", "2026-09-10", TODAY).message
    assert cal.with_object("예가체프") == "예가체프를" and cal.with_object("원두 목록") == "원두 목록을"
