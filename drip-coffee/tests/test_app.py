"""앱 화면 테스트 (Streamlit AppTest). 화면을 실제로 돌려 오류 없이 이어지는지 본다.

기록 파일은 환경 변수(DRIP_COFFEE_BREWS, DRIP_COFFEE_BEANS)로 임시 파일을 쓰게 해서 실제 기록을 건드리지 않는다.
"""
import re
import shutil

import cv2
import pytest
from streamlit.testing.v1 import AppTest

from dripcoffee import DATA_DIR, PROJECT_ROOT
from dripcoffee import calibration as cal
from tests.synthetic import render_sheet_window, simulate_photo

APP = str(PROJECT_ROOT / "app.py")


def texts(at):
    return [re.sub(r"<[^>]+>", " ", e.proto.body) for e in at.get("html")]


def has_text(at, words):
    return any(words in t for t in texts(at))


def button(at, key):
    return next(b for b in at.button if b.key == key)


def hint(at, key):
    """다음 버튼 대신 놓인 흐린 안내(누르면 흔들림)의 글."""
    return button(at, f"{key}-hint").label


def start(monkeypatch, brews, beans, equipment=None):
    monkeypatch.setenv("DRIP_COFFEE_BREWS", str(brews))
    monkeypatch.setenv("DRIP_COFFEE_BEANS", str(beans))
    monkeypatch.setenv("DRIP_COFFEE_EQUIPMENT", str(equipment or DATA_DIR / "equipment.csv"))
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
    assert not at.exception and not has_text(at, "같은 원두로 내리기") and has_text(at, "아직 없어요")
    assert button(at, "home-tutorial").label.startswith("처음이세요?")
    button(at, "home-start").click().run()
    # 기록이 하나도 없으면 사용법 네 장부터 보여 주고, 마지막 장의 '시작하기'로 1단계에 간다
    assert not at.exception and has_text(at, "1 / 4") and has_text(at, "원두와 장비를 골라요")
    for page in range(2, 5):
        button(at, "tutorial-next").click().run()
        assert not at.exception and has_text(at, f"{page} / 4")
    assert button(at, "tutorial-next").label == "시작하기" and not any(b.key == "tutorial-skip" for b in at.button)
    button(at, "tutorial-next").click().run()
    assert has_text(at, "아직 원두가 없어요") and button(at, "flow-add-first-bean")
    # 같은 접속에서 다시 '시작하기'를 누르면 사용법을 또 보여 주지 않고 바로 1단계로 간다
    button(at, "flow-home").click().run()
    button(at, "home-start").click().run()
    assert has_text(at, "1 / 6 단계")
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
    assert not at.exception and has_text(at, "아직 기록한 잔이 없어요")
    next(w for w in at.get("button_group") if w.key == "history-view").set_value("내 선호 영역").run()
    assert not at.exception and has_text(at, "아직 맛을 매긴 잔이 없어서 골든컵")


def test_history_lists_cups_by_day_and_shows_the_preference_chart(monkeypatch, my_files):
    # 기록한 잔: 날짜별 목록. 표시가 있는 잔과 사진이 없는 잔은 꼬리표를 단다. 내 선호 영역: 그림과 접힌 설명
    from dripcoffee import calibration as cal
    brews = cal.load_brews(my_files[0])
    at = start(monkeypatch, *my_files)
    button(at, "home-history").click().run()
    assert not at.exception and has_text(at, f"{len(brews)}잔을 기록했어요") and has_text(at, "평균 만족도")
    assert has_text(at, "10월 19일") and has_text(at, "막힘") and has_text(at, "사진 없음")
    next(w for w in at.get("button_group") if w.key == "history-view").set_value("내 선호 영역").run()
    assert not at.exception and has_text(at, "다음 잔 목표") and not has_text(at, "골든컵 중심</span>")
    assert has_text(at, "내 선호 중심")  # 그림에서 뺀 중심 값은 '목표는 어떻게 정해져요?'에 글로 남는다
    # 맛 변화: 처음 3잔과 최근 3잔 평균
    next(w for w in at.get("button_group") if w.key == "history-view").set_value("맛 변화").run()
    assert not at.exception and has_text(at, "처음 3잔 평균") and has_text(at, "최근 3잔 평균")
    # 원두 칩: 고른 원두의 잔만, 줄마다 누르는 버튼이 있다
    next(w for w in at.get("button_group") if w.key == "history-view").set_value("기록한 잔").run()
    next(w for w in at.get("button_group") if w.key == "history-bean").set_value("B01").run()
    cup_buttons = [b for b in at.button if (b.key or "").startswith("cup-")]
    assert not at.exception and len(cup_buttons) == sum(1 for b in brews if b.bean_id == "B01")
    cup_buttons[0].click().run()  # 누르면 자세히 창이 열린다 (창 안은 브라우저에서 확인)
    assert not at.exception


def test_same_bean_row_fills_the_last_bean_but_not_the_filter(monkeypatch, my_files):
    # 시작 화면의 '같은 원두로 내리기': 지난 잔의 원두와 양만 채우고, 필터는 장소마다 달라서 비워 둔다
    from dripcoffee import calibration as cal
    brews = cal.load_brews(my_files[0])
    at = start(monkeypatch, *my_files)
    assert has_text(at, "같은 원두로 내리기") and has_text(at, f"{len(brews)}잔")
    button(at, "home-same-bean").click().run()
    assert not at.exception and has_text(at, "1 / 6 단계")
    assert at.selectbox(key="flow-bean").value == brews[-1].bean_id
    assert at.number_input(key="flow-dose").value == brews[-1].dose_g
    assert at.selectbox(key="flow-filter").value is None


def test_tutorial_can_be_opened_from_home_and_skipped(monkeypatch, my_files):
    # 기록이 있으면 '시작하기'는 바로 1단계로 가고, 사용법은 '사용법 보기'로 언제든 볼 수 있다
    at = start(monkeypatch, *my_files)
    assert button(at, "home-tutorial").label.startswith("사용법 다시 보기")  # 기록이 있으면 '처음이세요?'라고 묻지 않는다
    button(at, "home-tutorial").click().run()
    assert not at.exception and has_text(at, "1 / 4")
    assert any("dc-tool" in e.proto.body for e in at.markdown)  # 움직이는 도구 그림 (SVG라서 markdown)
    assert not any(b.key == "tutorial-prev" for b in at.button)  # 첫 장에는 '이전'이 없다
    button(at, "tutorial-next").click().run()
    assert has_text(at, "2 / 4")
    button(at, "tutorial-prev").click().run()  # 앞 장으로 돌아가 다시 볼 수 있다
    assert not at.exception and has_text(at, "1 / 4")
    button(at, "tutorial-skip").click().run()
    assert not at.exception and has_text(at, "1 / 6 단계")
    button(at, "flow-home").click().run()
    button(at, "home-start").click().run()
    assert has_text(at, "1 / 6 단계")


def test_model_check_summarizes_errors_and_opens_the_cup_you_tap(monkeypatch, my_files):
    # 예측과 실제: 요약 카드(판정 기준 안 몇 잔), 잔마다 한 줄, 누른 잔의 자세한 비교
    from dripcoffee import calibration as cal
    comparisons = cal.compare_with_log(cal.load_brews(my_files[0]))
    at = start(monkeypatch, *my_files)
    button(at, "home-check").click().run()
    assert not at.exception and has_text(at, "판정 기준 안") and has_text(at, "치우침")
    assert all(any(b.key == f"check-cup-{c.brew.brew_id}" for b in at.button) for c in comparisons)
    newest = comparisons[-1].brew  # 처음에는 가장 최근 잔이 펼쳐져 있다
    assert has_text(at, f"{int(newest.date[5:7])}월 {int(newest.date[8:])}일 · {newest.bean_id} ")
    flagged = next(c for c in comparisons if c.brew.flag == "막힘")
    button(at, f"check-cup-{flagged.brew.brew_id}").click().run()
    assert not at.exception and has_text(at, "판정에서 뺌") and has_text(at, "보정과 위 요약에서는 빼요")
    # 나머지 보기도 요약 카드로 열린다
    for view, words in (("레시피 계산", "예측 결과"), ("추천 자세히", "추천 레시피"), ("사진만 재기", "사진을 넣으면")):
        next(w for w in at.get("button_group") if w.key == "check-view").set_value(view).run()
        assert not at.exception and has_text(at, words), view


def test_bean_starts_blank_and_must_be_chosen(monkeypatch, my_files):
    at = start(monkeypatch, *my_files)
    button(at, "home-start").click().run()
    assert at.selectbox(key="flow-bean").value is None  # 지난 원두가 저절로 넘어가지 않는다
    assert at.selectbox(key="flow-grinder").value == "메버릭1"  # 장비는 지난 잔 것을 채운다
    assert at.selectbox(key="flow-dripper").value == "하리오 V60 02"  # 목록에 하나뿐인 드리퍼는 채운다
    assert at.selectbox(key="flow-filter").value is None  # 필터는 장소마다 달라서 매번 고른다
    assert at.selectbox(key="flow-filter").options == ["하리오 화이트", "홀츠클로츠 무표백"]  # 목록에서만 고른다
    assert not has_text(at, "로스팅")  # 원두를 고르기 전에는 디게싱 안내가 없다
    # 다 고르기 전에는 다음 버튼 대신 무엇을 고르면 되는지 보여 준다 (다 고르면 버튼이 아래에서 올라온다)
    assert not any(b.key == "flow-next-1" for b in at.button) and hint(at, "flow-next-1").startswith("원두·드리퍼·필터·그라인더를")
    # 그 안내를 누르면 넘어가지 않고 흔들리기만 한다 (누를 때마다 a, b로 번갈아 바뀌어 다시 흔들린다)
    button(at, "flow-next-1-hint").click().run()
    assert at.session_state["flow"]["step"] == 1 and at.session_state["flow-next-1-taps"] == 1
    at.selectbox(key="flow-bean").set_value("B02").run()
    assert has_text(at, "디게싱") or has_text(at, "마시기 좋은 때")
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백").run()
    button(at, "flow-next-1").click().run()
    assert not at.exception and has_text(at, "2 / 6 단계") and has_text(at, "목표 분쇄")
    # 2단계: 근거 칩, 베이스에서 바뀐 것, 값마다 영향 한 줄, '이렇게 바꾸면?'
    assert has_text(at, "문헌 비교 13/14 통과") and has_text(at, "베이스에서 이렇게 바꿨어요")
    assert has_text(at, "곱게 갈수록 더 우러나고")
    assert has_text(at, "추천과 같아요")
    coarsest = min(round(at.session_state["flow"]["target_d50_um"] / 50) * 50 + 300, 1800)
    next(s for s in at.select_slider if s.key == "whatif-grind-평소대로").set_value(coarsest).run()  # 가장 굵게
    assert not at.exception and has_text(at, "덜 우러나요")
    # 오늘은 어떻게?: '더 진하게'를 고르면 이번 잔의 가수가 줄고, 기록에도 남는다
    next(w for w in at.get("button_group") if w.key == "flow-intent").set_value("더 진하게").run()
    assert not at.exception and at.session_state["flow"]["intent"] == "더 진하게" and has_text(at, "오늘은 &#x27;더 진하게&#x27;")


def test_grind_step_asks_for_a_photo_and_can_go_on_without_one(monkeypatch, my_files):
    # 3단계: 사진을 올리기 전에는 안내만 보이고, 사진 없이 추천대로 내리면 4단계로 넘어간다.
    at = start(monkeypatch, *my_files)
    button(at, "home-start").click().run()
    at.selectbox(key="flow-bean").set_value("B02").run()
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백").run()
    button(at, "flow-next-1").click().run()
    button(at, "flow-next-2").click().run()
    assert not at.exception and has_text(at, "3 / 6 단계") and has_text(at, "A칸(큰 칸)")
    assert at.number_input(key="flow-clicks").value is not None  # 2단계의 클릭 힌트를 채워 둔다
    button(at, "flow-skip-3").click().run()
    assert not at.exception and has_text(at, "4 / 6 단계")
    assert at.session_state["flow"]["measured_d50_um"] is None
    # 분쇄 칸은 '–' 대신 예측에 쓴 목표 크기와 눈대중을 보여 준다 (기록에는 잰 값이 아니라 남기지 않는다)
    target = at.session_state["flow"]["target_d50_um"]
    assert has_text(at, f"{target:.0f}") and has_text(at, "사진 없음 · ") and has_text(at, "정도")


def test_grind_check_walks_through_each_method_then_fits_the_recipe(monkeypatch, my_files):
    # 3단계: 사진을 올리면 '분쇄도 확인하기'가 나오고, 누르면 단계를 다 보여 준 뒤 잰 크기와 맞춘 레시피가 나온다
    at = start(monkeypatch, *my_files)
    button(at, "home-start").click().run()
    at.selectbox(key="flow-bean").set_value("B02").run()
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백").run()
    button(at, "flow-next-1").click().run()
    button(at, "flow-next-2").click().run()
    photo = simulate_photo(render_sheet_window(15.0, (12.0, 22.0, 108.0, 118.0), [(45.0, 55.0, 2.0), (60.0, 70.0, 1.5)]))
    at.file_uploader(key="flow-photos").upload("sheet.png", cv2.imencode(".png", photo)[1].tobytes(), "image/png").run()
    assert not at.exception and not has_text(at, "잰 대표 크기")
    button(at, "flow-check-3").click().run()
    assert not at.exception and has_text(at, "잰 대표 크기") and has_text(at, "Garrido-Jurado")
    assert at.session_state["flow"]["checked"] and not at.session_state["flow"]["checking"]
    button(at, "flow-next-3").click().run()
    assert not at.exception and has_text(at, "4 / 6 단계")
    assert at.session_state["flow"]["measured_d50_um"] > 0


def test_brew_step_compares_the_result_with_the_prediction(monkeypatch, my_files):
    # 4단계: 끊긴 시각과 음료 무게를 적으면 예측과 비교하고, 둘 다 있어야 맛 평가로 넘어간다.
    at = start(monkeypatch, *my_files)
    button(at, "home-start").click().run()
    at.selectbox(key="flow-bean").set_value("B02").run()
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백").run()
    button(at, "flow-next-1").click().run()
    button(at, "flow-next-2").click().run()
    button(at, "flow-skip-3").click().run()
    assert has_text(at, "붓기 일정") and has_text(at, "0:00") and has_text(at, "끊김 예상")  # 붓기 타임라인
    assert not any(b.key == "flow-next-4" for b in at.button) and hint(at, "flow-next-4").startswith("끊긴 시각과 음료 무게를")
    at.text_input(key="flow-drawdown").set_value("2:3x").run()
    # 잘못 적으면 위에서 내려오는 알림으로 적는 법을 알려 준다
    assert any("dc-toast" in e.proto.body and "2:42(분:초)" in e.proto.body for e in at.get("html"))
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
    at.selectbox(key="flow-filter").set_value("홀츠클로츠 무표백").run()
    button(at, "flow-next-1").click().run()
    button(at, "flow-next-2").click().run()
    button(at, "flow-skip-3").click().run()
    at.text_input(key="flow-drawdown").set_value("2:20").run()
    at.number_input(key="flow-beverage").set_value(176.0).run()
    button(at, "flow-next-4").click().run()
    assert not at.exception and has_text(at, "5 / 6 단계") and has_text(at, "바로")
    # 맛보기 안내는 누를 필요 없이 위에서 내려왔다 사라지는 알림으로 한 번 뜬다
    assert any("dc-toast" in e.proto.body and "넣고 바로 맛봐요" in e.proto.body for e in at.get("html"))
    flow = dict(at.session_state["flow"])
    flow["measured_d50_um"] = 1150.0  # 사진으로 잰 것처럼 (AppTest는 사진 올리기를 흉내 내지 못한다)
    at.session_state["flow"] = flow
    assert not any(b.key == "flow-save" for b in at.button) and hint(at, "flow-save").startswith("맛 6항목을 모두")
    for name, value in (("aroma", 3.5), ("flavor", 3.0), ("acidity", 1.5), ("body", 3.0), ("balance", 3.5),
                        ("aftertaste", 3.5)):
        at.select_slider(key=f"flow-taste-{name}").set_value(value).run()
    assert not any(b.key == "flow-save" for b in at.button) and hint(at, "flow-save").startswith("진하기를")
    next(w for w in at.get("button_group") if w.key == "flow-strength").set_value("진했어요").run()
    button(at, "flow-save").click().run()
    assert not at.exception and at.session_state["flow"]["step"] == 6
    saved = cal.load_brews(my_files[0])[-1]
    assert saved.bean_id == "B02" and saved.strength == "진함" and saved.drawdown_end_s == 140
    assert saved.taste["aroma"] == 3.5 and saved.filter_paper == "홀츠클로츠 무표백" and saved.grind_d50_um == 1150
    assert "앱 예측(내리기 전)" in saved.memo and saved.row["followed_recommendation"].startswith("예 (한 잔 내리기")
    # 6단계: 이번 잔과 다음 추천의 변화를 보여 준다. '진했어요'라서 목표 농도가 내려가고 가수 물이 는다
    assert has_text(at, "6 / 6 단계") and has_text(at, "기록했어요") and has_text(at, saved.brew_id)
    # 바뀐 이유를 내가 고른 말로 ('진했어요'라고 답해서. 따옴표는 HTML에서 &#x27;로 바뀐다)
    assert has_text(at, "다음 잔은 이렇게 바뀌어요") and has_text(at, "진했어요&#x27;라고 답해서")
    from dripcoffee import recommend as rec
    after = rec.preference_target(rec.brew_points(cal.load_brews(my_files[0])))
    assert after.tds_pct < at.session_state["flow"]["target_before"][1]
    assert has_text(at, f"→ {after.tds_pct:.2f}%")
    button(at, "flow-finish-home").click().run()
    assert not at.exception and has_text(at, "오늘의 한 잔")


def test_deployed_app_keeps_each_visitor_apart(monkeypatch, my_files):
    # 배포한 앱: 누구의 기록으로 시작할지 고른다. 둘러보기는 만든 사람 기록의 복사본이고, 내 이름은 빈 기록에서
    # 시작한다. 어느 쪽도 저장소의 기록 파일을 바꾸지 않는다
    monkeypatch.setenv("DRIP_COFFEE_SANDBOX", "1")
    originals = {path: path.read_bytes() for path in (cal.BREWS_PATH, cal.BEANS_PATH, cal.EQUIPMENT_PATH)}
    at = start(monkeypatch, *my_files)
    assert not at.exception and has_text(at, "어떻게 시작할까요?") and hint(at, "start-go")
    button(at, "start-tour").click().run()
    assert not at.exception and has_text(at, "둘러보는 중") and has_text(at, f"{len(cal.load_brews())}잔")
    button(at, "tour-leave").click().run()
    button(at, "start-mine").click().run()
    button(at, "start-go").click().run()
    assert has_text(at, "어떻게 시작할까요?")  # 이름이 비면 그대로
    at.text_input(key="start-name").set_value("아침커피")
    button(at, "start-go").click().run()
    assert not at.exception and has_text(at, "아직 없어요") and button(at, "home-who").label.startswith("아침커피의 기록")
    button(at, "home-history").click().run()
    assert not at.exception and has_text(at, "아직 기록한 잔이 없어요")
    button(at, "history-home").click().run()
    button(at, "home-check").click().run()
    assert not at.exception
    button(at, "check-home").click().run()
    button(at, "home-who").click().run()  # 이름을 누르면 바꾸기 창이 열린다
    assert not at.exception and has_text(at, "만든 사람 기록 둘러보기")
    button(at, "space-tour").click().run()
    assert not at.exception and has_text(at, "둘러보는 중")
    button(at, "tour-leave").click().run()  # 시작 화면에 이 기기의 이름이 남아 있어 다시 열 수 있다
    assert has_text(at, "아침커피")
    button(at, "start-0").click().run()
    button(at, "home-who").click().run()
    assert button(at, "space-forget")  # 지우기는 한 번 더 묻는다 (AppTest는 창 안의 버튼을 누르면 창이 닫혀서 상태로 넣는다)
    at.session_state["confirm_forget"] = True
    button(at, "home-who").click().run()
    button(at, "space-forget-yes").click().run()
    assert not at.exception and has_text(at, "어떻게 시작할까요?") and not has_text(at, "아침커피")
    # 다른 기기에서 받은 기록 파일을 불러오면 파일 이름에 적힌 이름으로 연다
    button(at, "start-import").click().run()
    at.file_uploader(key="start-file").upload("drip-coffee-주말-20261025.zip", cal.records_zip(DATA_DIR), "application/zip").run()
    button(at, "start-import-go").click().run()
    assert not at.exception and button(at, "home-who").label.startswith("주말의 기록")
    assert has_text(at, f"{len(cal.load_brews())}잔")
    button(at, "home-who").click().run()
    assert at.get("download_button")  # 바꾸기 창에서 기록 파일로 받는다
    assert all(path.read_bytes() == data for path, data in originals.items())


def test_deployed_app_renames_the_current_name(monkeypatch, my_files):
    # 바꾸기 창에서 지금 이름을 바꾼다. 비었거나 이 기기의 다른 이름과 겹치면 그대로 둔다
    monkeypatch.setenv("DRIP_COFFEE_SANDBOX", "1")
    at = start(monkeypatch, *my_files)
    button(at, "start-mine").click().run()
    at.text_input(key="start-name").set_value("아침커피")
    button(at, "start-go").click().run()
    at.session_state["asking_name"] = True  # AppTest는 창 안의 버튼을 누르면 창이 닫혀서 상태로 넣는다
    button(at, "home-who").click().run()
    at.text_input(key="space-name").set_value("점심커피")
    button(at, "space-go").click().run()
    at.session_state["space_opened"] = False  # 브라우저에서는 창만 다시 그려 이 표시를 지우지만 AppTest는 앱 전체를 다시 그린다
    folder = at.session_state["spaces"]["점심커피"]

    def rename(to):
        at.session_state["renaming"] = True
        button(at, "home-who").click().run()
        assert at.text_input(key="space-rename-name").value == at.session_state["space"]  # 지금 이름이 미리 들어 있다
        at.text_input(key="space-rename-name").set_value(to)
        button(at, "space-rename-go").click().run()
        button(at, "home-who").click().run()  # 알림은 창에 뜨므로 창을 다시 연다
        assert not at.exception

    rename("아침커피")
    assert has_text(at, "이미 있는 이름") and button(at, "home-who").label.startswith("점심커피의 기록")
    rename(" ")
    assert has_text(at, "적어 주세요") and button(at, "home-who").label.startswith("점심커피의 기록")
    rename("저녁커피")
    assert button(at, "home-who").label.startswith("저녁커피의 기록") and at.session_state["last_name"] == "저녁커피"
    assert list(at.session_state["spaces"]) == ["아침커피", "저녁커피"] and at.session_state["spaces"]["저녁커피"] == folder


def test_deployed_app_rejects_a_file_that_is_not_a_record(monkeypatch, my_files):
    monkeypatch.setenv("DRIP_COFFEE_SANDBOX", "1")
    at = start(monkeypatch, *my_files)
    button(at, "start-import").click().run()
    at.file_uploader(key="start-file").upload("drip-coffee-주말-20261025.zip", b"not a zip", "application/zip").run()
    button(at, "start-import-go").click().run()
    assert not at.exception and has_text(at, "기록 파일(zip)이 아니에요") and has_text(at, "어떻게 시작할까요?")
