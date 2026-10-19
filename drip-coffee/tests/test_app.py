"""앱 화면 테스트 (Streamlit AppTest). 화면을 실제로 돌려 오류 없이 이어지는지 본다.

기록 파일은 환경 변수(DRIP_COFFEE_BREWS, DRIP_COFFEE_BEANS)로 임시 파일을 쓰게 해서 실제 기록을 건드리지 않는다.
"""
import re
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from dripcoffee import DATA_DIR, PROJECT_ROOT

APP = str(PROJECT_ROOT / "app.py")


def texts(at):
    return [re.sub(r"<[^>]+>", " ", e.proto.body) for e in at.get("html")]


def has_text(at, words):
    return any(words in t for t in texts(at))


def button(at, key):
    return next(b for b in at.button if b.key == key)


def start(monkeypatch, brews, beans):
    monkeypatch.setenv("DRIP_COFFEE_BREWS", str(brews))
    monkeypatch.setenv("DRIP_COFFEE_BEANS", str(beans))
    return AppTest.from_file(APP, default_timeout=60).run()


@pytest.fixture
def empty_files(tmp_path):
    """처음 쓰는 사람: 칸 이름만 있고 원두도 기록도 없는 파일."""
    paths = []
    for name in ("brews.csv", "beans.csv"):
        header = (DATA_DIR / name).read_text(encoding="utf-8").splitlines()[0]
        path = tmp_path / name
        path.write_text(header + "\n", encoding="utf-8")
        paths.append(path)
    return paths


@pytest.fixture
def my_files(tmp_path):
    """지금까지의 기록을 복사한 파일."""
    paths = []
    for name in ("brews.csv", "beans.csv"):
        shutil.copy(DATA_DIR / name, tmp_path / name)
        paths.append(tmp_path / name)
    return paths


def test_first_time_user_with_no_beans_or_records(monkeypatch, empty_files):
    at = start(monkeypatch, *empty_files)
    assert not at.exception and not has_text(at, "지난 한 잔")
    button(at, "home-start").click().run()
    assert has_text(at, "아직 원두가 없어요") and button(at, "flow-add-first-bean")
    for page in ("history", "check"):
        button(at, "flow-home" if page == "history" else f"history-home").click().run()
        button(at, f"home-{page}").click().run()
        assert not at.exception
    assert has_text(at, "모델 점검")
    for view in ("예측과 실제", "레시피 계산", "추천 자세히", "사진만 재기"):
        next(w for w in at.get("button_group") if w.key == "check-view").set_value(view).run()
        assert not at.exception, view


def test_history_without_tasted_cups_explains_the_golden_cup(monkeypatch, empty_files):
    at = start(monkeypatch, *empty_files)
    button(at, "home-history").click().run()
    assert not at.exception and has_text(at, "아직 맛을 매긴 잔이 없어서 골든컵")


def test_bean_starts_blank_and_must_be_chosen(monkeypatch, my_files):
    at = start(monkeypatch, *my_files)
    button(at, "home-start").click().run()
    assert at.selectbox(key="flow-bean").value is None  # 지난 원두가 저절로 넘어가지 않는다
    assert at.selectbox(key="flow-grinder").value == "메버릭1"  # 장비는 지난 잔 것을 채운다
    assert not has_text(at, "로스팅")  # 원두를 고르기 전에는 디게싱 안내가 없다
    button(at, "flow-next-1").click().run()
    assert has_text(at, "원두·필터를 골라 주세요")
    at.selectbox(key="flow-bean").set_value("B02").run()
    assert has_text(at, "디게싱") or has_text(at, "마시기 좋은 때")
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백")
    button(at, "flow-next-1").click().run()
    assert not at.exception and has_text(at, "2 / 6 단계") and has_text(at, "목표 분쇄")
