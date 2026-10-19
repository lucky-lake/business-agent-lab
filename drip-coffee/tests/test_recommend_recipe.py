"""레시피 전체 추천 테스트. 설계는 docs/MODEL.md 9.2.

목표(추출수율, 가수 후 농도)를 주면 물 양, 온도, 붓기 일정, 분쇄 크기, 가수량을 모두 정한다.
기록이 쌓여도 테스트가 흔들리지 않게 목표는 고정값을 쓴다.
"""
import pytest

from dripcoffee import calibration as cal
from dripcoffee import recommend as rec
from dripcoffee.extraction_model import grind_from_d50, predict

TARGET = rec.PreferenceTarget(18.5, 1.25, n_liked=0, personal_center=None)


@pytest.fixture(scope="module")
def brews():
    return cal.load_brews()


@pytest.fixture(scope="module")
def beans():
    return cal.load_beans()


def test_recommended_recipe_hits_the_target(brews, beans):
    r = rec.recommend_recipe(TARGET, beans["B02"], brews)
    again = predict(r.recipe, grind_from_d50(r.d50_um))  # 추천 레시피를 모델로 다시 계산
    assert again.ey_pct == pytest.approx(TARGET.ey_pct, abs=0.3)  # 분쇄를 10 µm 단위로 맞춘 만큼만 어긋난다
    assert again.tds_cup_pct == pytest.approx(TARGET.tds_pct, abs=0.01)


def test_recipe_is_a_whole_recipe(brews, beans):
    r = rec.recommend_recipe(TARGET, beans["B02"], brews)
    pours = r.recipe.pours
    assert 8 <= r.water_ratio <= 16 and 2 + sum(r.pattern) == r.water_ratio
    assert pours[0].start_s == 0 and pours[0].grams == 40  # 뜸은 원두의 2배
    assert [p.grams for p in pours[1:]] == [m * 20 for m in r.pattern]  # 붓기마다 원두의 정수배
    assert r.recipe.bypass_g >= 0
    assert 700 <= r.d50_um <= 1800 and r.d50_um % 10 == 0


def test_pours_are_whole_multiples_of_the_dose():
    assert rec.split_patterns(6) == [(4, 2), (3, 3), (2, 2, 2)]  # 8배 = 2-4-2, 2-3-3, 2-2-2-2
    for rest in range(6, 15):
        for pattern in rec.split_patterns(rest):
            assert sum(pattern) == rest and set(pattern) <= {2, 3, 4} and list(pattern) == sorted(pattern, reverse=True)


def test_usual_recipes_are_rebuilt_from_multiples():
    # 평소 두 레시피(BREW_PROTOCOL)를 배수와 간격으로 읽고 그대로 다시 만들 수 있어야 한다
    assert rec.pattern_of("0:40;30:60;60:60;90:60", 20) == ((3, 3, 3), 30)
    assert rec.pattern_of("0:40;60:80;120:40", 20) == ((4, 2), 60)
    assert rec.pours_text(rec.pattern_pours(20, (3, 3, 3), 30)) == "0:40;30:60;60:60;90:60"
    assert rec.pours_text(rec.pattern_pours(20, (4, 2), 60)) == "0:40;60:80;120:40"
    assert rec.pattern_text((4, 2)) == "2-4-2"


def test_pour_interval_follows_the_usual_recipe(brews, beans):
    # 붓는 시각의 효과는 판단할 수 없어서 평소 간격을 따른다: B01은 60초, B02는 30초
    assert rec.recommend_recipe(TARGET, beans["B01"], brews).interval_s == 60
    assert rec.recommend_recipe(TARGET, beans["B02"], brews).interval_s == 30


@pytest.mark.parametrize("roast, low, high", [("강배전", 80, 88), ("라이트", 92, 96), ("미디엄", 88, 94),
                                              ("미디움 다크", 84, 90), ("미디엄 라이트", 90, 95)])
def test_temperature_range_follows_the_roast_level(roast, low, high):
    assert rec.roast_range(roast) == (low, high)


def test_recommended_temperature_stays_in_the_roast_range(brews, beans):
    for bean_id in ("B01", "B08"):
        r = rec.recommend_recipe(TARGET, beans[bean_id], brews)
        low, high = rec.roast_range(beans[bean_id]["roast_level"])
        assert low <= r.recipe.water_temp_c <= high


def test_grind_stays_coarser_than_where_the_bean_clogged(brews, beans):
    clogged = next(b for b in brews if b.brew_id == "20261016-2")  # 85클릭에서 막힘
    assert rec.clog_limit("B01", brews) == pytest.approx(clogged.grind_d50_um)
    assert rec.clog_limit("B02", brews) is None
    r = rec.recommend_recipe(TARGET, beans["B01"], brews)
    assert r.d50_um > clogged.grind_d50_um


def test_bean_without_records_gets_a_recipe_for_its_own_dose(brews, beans):
    r = rec.recommend_recipe(TARGET, beans["S1"], brews, dose_g=17.8)  # 샘플은 18 g뿐이라 17.8 g으로 내린다
    assert r.recipe.dose_g == 17.8 and r.recipe.pours[0].grams == 36
    assert [p.grams for p in r.recipe.pours[1:]] == [round(m * 17.8) for m in r.pattern]
    assert "기록이 없어" in r.usual.source


@pytest.mark.parametrize("bean_id", ["B01", "B02", "B07"])
def test_drawdown_never_goes_past_3_30(brews, beans, bean_id):
    # 오래 우리면 쓴맛이 나기 쉬운데 모델은 이를 계산하지 못해서, 끊김 2:00–3:30을 넘는 레시피는 고르지 않는다
    r = rec.recommend_recipe(TARGET, beans[bean_id], brews)
    assert 120 <= r.prediction.drawdown_end_s <= 210


def test_when_the_target_needs_more_time_grind_and_temperature_get_as_close_as_they_can(brews, beans):
    # B01은 막혔던 기록 때문에 더 곱게 갈 수 없다. 18.5%는 3:30 안에 닿지 못하므로, 허용되는 가장 고운 분쇄로
    # 범위 안에서 가장 많이 우러나는 레시피를 고르고 모자란 만큼을 알려 준다
    r = rec.recommend_recipe(TARGET, beans["B01"], brews)
    limit = rec.clog_limit("B01", brews)
    assert r.d50_um == (limit // 10 + 1) * 10  # 막힘 한계 바로 위
    assert r.ey_gap_pct == pytest.approx(r.prediction.ey_pct - TARGET.ey_pct)
    assert r.ey_gap_pct < -rec.REFIT_EY_TOLERANCE
    assert r.prediction.tds_cup_pct == pytest.approx(TARGET.tds_pct, abs=0.01)  # 진하기는 가수로 그대로 맞춘다


def test_a_target_reached_without_the_time_limit_reports_no_gap(brews, beans):
    r = rec.recommend_recipe(TARGET, beans["B02"], brews)
    assert r.ey_gap_pct == 0


def test_nothing_inside_the_time_window_is_refused_with_reasons(brews, beans, monkeypatch):
    # 범위 안 레시피가 하나도 없으면 범위 밖 레시피를 고르지 않고, 왜 못 찾았는지 알려 준다
    monkeypatch.setattr(rec, "GOOD_DRAWDOWN_S", (60.0, 61.0))
    with pytest.raises(rec.NoRecipeError, match="끊김이 2:00–3:30 밖"):
        rec.recommend_recipe(TARGET, beans["B02"], brews)


def test_unreachable_target_is_refused_with_reasons(brews, beans):
    with pytest.raises(rec.NoRecipeError, match="찾지 못했어요"):
        rec.recommend_recipe(rec.PreferenceTarget(28.0, 1.25, 0, None), beans["B02"], brews)


def test_new_bean_starts_from_the_recipe_of_its_roast_group(brews):
    # 기록이 없는 원두는 같은 배전도 묶음의 마지막 잔에서 출발한다: 중강배전 이상은 B01 방식(2-4-2 / 60초),
    # 라이트~미디엄은 B02 방식(2-3-3-3 / 30초)
    dark = rec.usual_recipe({"bean_id": "새 강배전", "roast_level": "강배전"}, brews)
    medium_dark = rec.usual_recipe({"bean_id": "새 중강배전", "roast_level": "미디움 다크"}, brews)
    medium = rec.usual_recipe({"bean_id": "새 미디엄", "roast_level": "미디엄"}, brews)
    assert (dark.pattern, dark.interval_s, dark.water_ratio) == ((4, 2), 60, 8)
    assert (medium_dark.pattern, medium_dark.interval_s, medium_dark.water_ratio) == ((4, 2), 60, 8)
    assert (medium.pattern, medium.interval_s, medium.water_ratio) == ((3, 3, 3), 30, 11)
    assert dark.temp_c == 84 and medium_dark.temp_c == 87  # 온도는 그 원두 배전도 범위의 가운데


def test_base_recipes_are_used_when_a_roast_group_has_no_records():
    usual = rec.usual_recipe({"bean_id": "새 강배전", "roast_level": "강배전"}, [])
    assert (usual.pattern, usual.interval_s, usual.water_ratio) == ((4, 2), 60, 8)
    assert rec.roast_group("미디움 다크") == rec.roast_group("강배전") == "중강배전 이상"
    assert rec.roast_group("미디엄") == rec.roast_group("미디엄 라이트") == "그 밖"


def test_refit_keeps_the_recipe_when_the_grind_came_out_as_recommended(brews, beans):
    r = rec.recommend_recipe(TARGET, beans["B02"], brews)
    fit = rec.refit_to_grind(r, r.d50_um, TARGET)
    assert fit.reached and fit.interval_change_s == 0
    assert fit.recipe.bypass_g == pytest.approx(r.recipe.bypass_g, abs=1)


def test_refit_lengthens_the_interval_for_coarser_grind_and_shortens_it_for_finer(brews, beans):
    # 사진으로 재 보니 추천보다 굵으면 덜 추출되므로 붓기 간격을 늘리고, 고우면 줄인다.
    r = rec.recommend_recipe(TARGET, beans["B02"], brews)
    coarse = rec.refit_to_grind(r, r.d50_um + 80, TARGET)
    fine = rec.refit_to_grind(r, r.d50_um - 80, TARGET)
    assert 0 < coarse.interval_change_s <= 15 and -15 <= fine.interval_change_s < 0
    for fit in (coarse, fine):
        assert fit.reached and fit.prediction.tds_cup_pct == pytest.approx(TARGET.tds_pct, abs=0.01)
        as_is = predict(r.recipe, grind_from_d50(fit.d50_um))  # 같은 가루로 추천 레시피를 그대로 내렸다면
        assert abs(fit.ey_gap_pct) < abs(as_is.ey_pct - TARGET.ey_pct)
        # 물 양, 붓기 배수, 온도는 그대로다
        assert fit.recipe.water_g == r.recipe.water_g and fit.recipe.water_temp_c == r.recipe.water_temp_c
        assert [p.grams for p in fit.recipe.pours] == [p.grams for p in r.recipe.pours]


def test_refit_does_not_choose_an_interval_that_ends_past_the_time_window(brews, beans, monkeypatch):
    # 굵게 나와 간격을 늘려야 해도, 끊김이 범위를 넘는 간격은 고르지 않는다. 범위의 끝을 제한 없이 고른 간격의
    # 끊김보다 1초 앞으로 당겨서 확인한다
    r = rec.recommend_recipe(TARGET, beans["B02"], brews)
    free = rec.refit_to_grind(r, r.d50_um + 80, TARGET)
    end = free.prediction.drawdown_end_s - 1
    monkeypatch.setattr(rec, "GOOD_DRAWDOWN_S", (end - 60, end))
    held = rec.refit_to_grind(r, r.d50_um + 80, TARGET)
    assert held.prediction.drawdown_end_s <= end and held.interval_change_s < free.interval_change_s


def test_refit_reports_the_gap_when_the_grind_is_too_far_off(brews, beans):
    r = rec.recommend_recipe(TARGET, beans["B02"], brews)
    far = rec.refit_to_grind(r, r.d50_um + 300, TARGET)
    assert not far.reached and far.interval_change_s == 15 and far.ey_gap_pct < -rec.REFIT_EY_TOLERANCE

