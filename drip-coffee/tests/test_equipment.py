"""장비 목록(data/equipment.csv) 읽기와 '장비 추가' 테스트."""
import pytest

from dripcoffee import DATA_DIR
from dripcoffee import calibration as cal


def test_equipment_list_has_my_gear():
    equipment = cal.load_equipment()
    assert [e["name"] for e in equipment["드리퍼"]] == ["하리오 V60 02"]
    assert equipment["드리퍼"][0]["detail"] == "원뿔형"
    assert [e["name"] for e in equipment["필터"]] == ["하리오 화이트", "홀츠클로츠 무표백"]
    assert [e["name"] for e in equipment["그라인더"]] == ["메버릭1"]


def test_added_equipment_comes_back_in_order(tmp_path):
    path = tmp_path / "equipment.csv"
    path.write_bytes((DATA_DIR / "equipment.csv").read_bytes())
    cal.append_equipment("드리퍼", "  칼리타 웨이브 185 ", "바닥이 평평한 형", path)
    cal.append_equipment("그라인더", "코만단테 C40", path=path)
    equipment = cal.load_equipment(path)
    assert equipment["드리퍼"][-1] == {"name": "칼리타 웨이브 185", "detail": "바닥이 평평한 형"}
    assert [e["name"] for e in equipment["그라인더"]] == ["메버릭1", "코만단테 C40"]


@pytest.mark.parametrize("kind, name, detail, message", [
    ("드리퍼", "", "원뿔형", "이름을 적어 주세요"),
    ("드리퍼", "오리가미", "", "모양을 골라 주세요"),
    ("필터", "하리오 화이트", "", "이미 있는 필터예요"),
    ("필터", "홀츠클로츠 무표백".upper(), "", "이미 있는 필터예요"),
    ("저울", "아카이아", "", "장비 종류는"),
])
def test_bad_equipment_is_refused_and_nothing_is_written(tmp_path, kind, name, detail, message):
    path = tmp_path / "equipment.csv"
    path.write_bytes((DATA_DIR / "equipment.csv").read_bytes())
    before = path.read_bytes()
    with pytest.raises(cal.EquipmentError, match=message):
        cal.append_equipment(kind, name, detail, path)
    assert path.read_bytes() == before


def test_first_time_user_starts_with_no_equipment_and_can_add(tmp_path):
    path = tmp_path / "equipment.csv"
    assert all(not items for items in cal.load_equipment(path).values())
    cal.append_equipment("필터", "하리오 화이트", path=path)
    assert cal.load_equipment(path)["필터"] == [{"name": "하리오 화이트", "detail": ""}]
