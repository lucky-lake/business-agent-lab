"""배포한 앱에서 방문자의 기록을 이 기기(브라우저 저장소)나 파일로 옮길 때 쓰는 묶음.

이름마다 기록 파일(추출 기록, 원두, 장비)의 내용을 JSON 문자열 하나로 묶고, 다시 파일로 푼다.
다른 기기로 옮길 때는 한 이름의 기록 파일을 zip 하나로 묶는다.
"""
import io
import json
import zipfile
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


def to_zip(folder):
    """기록 폴더의 기록 파일을 zip 파일 하나(바이트)로 묶는다. 다른 기기에서 불러오거나 표 프로그램으로 열 수 있다."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in FILES:
            archive.writestr(file, (Path(folder) / file).read_text(encoding="utf-8"))
    return buffer.getvalue()


def from_zip(data):
    """to_zip의 반대: {파일 이름: 내용}. zip이 아니거나 기록 파일이 모자라면 ValueError."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            return {file: archive.read(file).decode("utf-8") for file in FILES}
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError) as error:
        raise ValueError("Drip Coffee에서 받은 기록 파일(zip)이 아니에요.") from error
