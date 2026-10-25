"""배포한 앱에서 방문자의 기록을 이 기기(브라우저 저장소)에 둘 때 쓰는 묶음.

이름마다 기록 파일(추출 기록, 원두, 장비)의 내용을 JSON 문자열 하나로 묶고, 다시 파일로 푼다.
"""
import json
from pathlib import Path

FILES = ("brews.csv", "beans.csv", "equipment.csv")  # 이름마다 저장하는 기록 파일


def pack(folders, last):
    """{이름: 기록 폴더}와 마지막으로 연 이름을 JSON 문자열 하나로 묶는다. 같은 기록이면 늘 같은 문자열이다."""
    profiles = {name: {file: (Path(folder) / file).read_text(encoding="utf-8") for file in FILES}
                for name, folder in folders.items()}
    return json.dumps({"last": last, "profiles": profiles}, ensure_ascii=False, sort_keys=True)


def unpack(raw):
    """pack의 반대: ({이름: {파일 이름: 내용}}, 마지막 이름). 비었거나 읽을 수 없으면 ({}, None)."""
    try:
        data = json.loads(raw) if raw else {}
        profiles = {str(name): {file: str(files[file]) for file in FILES} for name, files in data["profiles"].items()}
        last = data.get("last")
    except (ValueError, TypeError, KeyError, AttributeError):
        return {}, None
    return profiles, last if last in profiles else None


def write_files(files, folder):
    """unpack한 파일 내용을 기록 폴더에 쓴다."""
    for file, text in files.items():
        (Path(folder) / file).write_text(text, encoding="utf-8")
