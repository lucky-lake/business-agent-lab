"""배포한 앱의 기기 저장: 이름마다 기록 파일을 JSON 하나로 묶고 다시 푼다. 다른 기기로 옮길 때는 zip 하나로."""
import pytest

from dripcoffee import DATA_DIR
from dripcoffee import device_store as ds


def make_folder(tmp_path, name, brews):
    folder = tmp_path / name
    folder.mkdir()
    (folder / "brews.csv").write_text(brews, encoding="utf-8")
    for file in ("beans.csv", "equipment.csv"):
        (folder / file).write_text((DATA_DIR / file).read_text(encoding="utf-8"), encoding="utf-8")
    return folder


def test_pack_and_unpack_give_back_the_same_files(tmp_path):
    folders = {"아침커피": make_folder(tmp_path, "a", "date\n2026-10-24\n"), "주말": make_folder(tmp_path, "b", "date\n")}
    raw = ds.pack(folders, "아침커피")
    assert raw == ds.pack(folders, "아침커피")  # 같은 기록이면 같은 문자열이라, 바뀔 때만 저장할 수 있다
    profiles, last = ds.unpack(raw)
    assert last == "아침커피" and set(profiles) == {"아침커피", "주말"}
    restored = tmp_path / "restored"
    restored.mkdir()
    ds.write_files(profiles["아침커피"], restored)
    assert ds.pack({"아침커피": restored, "주말": folders["주말"]}, "아침커피") == raw


def test_unpack_ignores_empty_or_broken_storage():
    assert ds.unpack("") == ({}, None)
    assert ds.unpack("{not json") == ({}, None)
    assert ds.unpack('{"profiles": {"a": {"brews.csv": "x"}}}') == ({}, None)  # 파일이 모자라면 버린다
    assert ds.unpack('{"last": "없는 이름", "profiles": {}}') == ({}, None)


def test_zip_carries_one_name_to_another_device(tmp_path):
    folder = make_folder(tmp_path, "a", (DATA_DIR / "brews.csv").read_text(encoding="utf-8"))
    files = ds.from_zip(ds.to_zip(folder))
    assert files == {file: (folder / file).read_text(encoding="utf-8") for file in ds.FILES}


def test_zip_rejects_other_files():
    with pytest.raises(ValueError):
        ds.from_zip(b"not a zip")
