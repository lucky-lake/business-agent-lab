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


def test_grind_step_asks_for_a_photo_and_can_go_on_without_one(monkeypatch, my_files):
    # 3단계: 사진을 올리기 전에는 안내만 보이고, 사진 없이 추천대로 내리면 4단계로 넘어간다.
    # (AppTest는 사진 올리기를 흉내 낼 수 없어서, 사진을 잰 뒤의 화면은 브라우저에서 확인한다)
    at = start(monkeypatch, *my_files)
    button(at, "home-start").click().run()
    at.selectbox(key="flow-bean").set_value("B02").run()
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백")
    button(at, "flow-next-1").click().run()
    button(at, "flow-next-2").click().run()
    assert not at.exception and has_text(at, "3 / 6 단계") and has_text(at, "A칸(큰 칸)")
    assert at.number_input(key="flow-clicks").value is not None  # 2단계의 클릭 힌트를 채워 둔다
    button(at, "flow-skip-3").click().run()
    assert not at.exception and has_text(at, "4 / 6 단계")
    assert at.session_state["flow"]["measured_d50_um"] is None


def test_brew_step_compares_the_result_with_the_prediction(monkeypatch, my_files):
    # 4단계: 끊긴 시각과 음료 무게를 적으면 예측과 비교하고, 둘 다 있어야 맛 평가로 넘어간다.
    at = start(monkeypatch, *my_files)
    button(at, "home-start").click().run()
    at.selectbox(key="flow-bean").set_value("B02").run()
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백")
    button(at, "flow-next-1").click().run()
    button(at, "flow-next-2").click().run()
    button(at, "flow-skip-3").click().run()
    assert has_text(at, "붓기 일정") and has_text(at, "0:00")
    button(at, "flow-next-4").click().run()
    assert has_text(at, "물줄기가 끊긴 시각과 음료 무게를 적어 주세요")
    at.text_input(key="flow-drawdown").set_value("2:3x").run()
    assert has_text(at, "2:42(분:초)")  # 잘못 적으면 적는 법을 알려 준다
    at.text_input(key="flow-drawdown").set_value("2:20").run()
    at.number_input(key="flow-beverage").set_value(176.0).run()
    assert not at.exception and has_text(at, "끊겼어요") and has_text(at, "예측(")
    button(at, "flow-next-4").click().run()
    flow = at.session_state["flow"]
    assert flow["step"] == 5 and flow["drawdown_end_s"] == 140 and flow["beverage_g"] == 176.0 and flow["flag"] == ""


def test_taste_step_needs_all_scores_and_strength_then_saves_the_cup(monkeypatch, my_files):
    # 5단계: 맛 6항목과 진하기를 모두 골라야 저장하고, 저장한 잔에는 흐름에서 모은 값과 내리기 전 예측이 남는다.
    from dripcoffee import calibration as cal
    at = start(monkeypatch, *my_files)
    button(at, "home-start").click().run()
    at.selectbox(key="flow-bean").set_value("B02").run()
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백")
    button(at, "flow-next-1").click().run()
    button(at, "flow-next-2").click().run()
    button(at, "flow-skip-3").click().run()
    at.text_input(key="flow-drawdown").set_value("2:20").run()
    at.number_input(key="flow-beverage").set_value(176.0).run()
    button(at, "flow-next-4").click().run()
    assert not at.exception and has_text(at, "5 / 6 단계") and has_text(at, "바로")
    flow = dict(at.session_state["flow"])
    flow["measured_d50_um"] = 1150.0  # 사진으로 잰 것처럼 (AppTest는 사진 올리기를 흉내 내지 못한다)
    at.session_state["flow"] = flow
    button(at, "flow-save").click().run()
    assert has_text(at, "맛 6항목을 모두 매겨 주세요")
    for name, value in (("aroma", 3.5), ("flavor", 3.0), ("acidity", 1.5), ("body", 3.0), ("balance", 3.5),
                        ("aftertaste", 3.5)):
        at.select_slider(key=f"flow-taste-{name}").set_value(value).run()
    button(at, "flow-save").click().run()
    assert has_text(at, "진하기를 골라 주세요")
    next(w for w in at.get("button_group") if w.key == "flow-strength").set_value("진했어요").run()
    button(at, "flow-save").click().run()
    assert not at.exception and at.session_state["flow"]["step"] == 6
    saved = cal.load_brews(my_files[0])[-1]
    assert saved.bean_id == "B02" and saved.strength == "진함" and saved.drawdown_end_s == 140
    assert saved.taste["aroma"] == 3.5 and saved.filter_paper == "홀츠클로츠 무표백" and saved.grind_d50_um == 1150
    assert "앱 예측(내리기 전)" in saved.memo and saved.row["followed_recommendation"].startswith("예 (한 잔 내리기")
    # 6단계: 이번 잔과 다음 추천의 변화를 보여 준다. '진했어요'라서 목표 농도가 내려가고 가수 물이 는다
    assert has_text(at, "6 / 6 단계") and has_text(at, saved.brew_id) and has_text(at, "다음 추천은 이렇게 바뀌어요")
    from dripcoffee import recommend as rec
    after = rec.preference_target(rec.brew_points(cal.load_brews(my_files[0])))
    assert after.tds_pct < at.session_state["flow"]["target_before"][1]
    assert has_text(at, f"→ {after.tds_pct:.2f}%")
    button(at, "flow-finish-home").click().run()
    assert not at.exception and has_text(at, "오늘의 한 잔")
