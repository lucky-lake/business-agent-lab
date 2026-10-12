"""추출 모델이 지켜야 할 성질 테스트.

정확한 농도·추출수율은 굴절계가 없어 알 수 없으므로, 숫자 대신 언제나 맞아야 하는 관계를 확인한다.
기준 레시피(원두 B02 기준: 20 g, 92 °C, 물 220 g, 대표 크기 1200 µm)에서 한 가지만 바꿔 비교한다.
근거는 docs/MODEL.md와 docs/research/ 조사 노트에 있다.

추출 모델(dripcoffee/extraction_model.py)은 다음 커밋에서 구현하므로,
그 전까지는 실패가 예상된 테스트(xfail)로 둔다.
"""
import pytest

try:
    from dripcoffee import extraction_model as em
except ImportError:  # 구현 전
    em = None

pytestmark = pytest.mark.xfail(em is None, reason="추출 모델은 다음 커밋에서 구현", strict=True)

BASE_POURS = "0:40;30:60;60:60;90:60"


def predict(d50_um=1200.0, temp_c=92.0, pours=BASE_POURS, dose_g=20.0, bypass_g=0.0):
    """기준 레시피에서 원하는 값만 바꿔 예측한다."""
    recipe = em.Recipe(dose_g=dose_g, water_temp_c=temp_c, pours=em.parse_pours(pours), bypass_g=bypass_g)
    return em.predict(recipe, em.grind_from_d50(d50_um))


def test_1_finer_grind_drains_slower_and_extracts_more():
    # 입자가 작으면 물길이 좁아지고(투과도 ∝ 지름²), 입자 속 성분이 빨리 나온다(속도 ∝ 1/크기²).
    # 문헌 실험(Váradý 2022 V60, Batali 2020)에서도 곱게 갈수록 추출수율이 올랐다.
    coarse, fine = predict(d50_um=1200), predict(d50_um=900)
    assert fine.drawdown_end_s > coarse.drawdown_end_s
    assert fine.ey_pct > coarse.ey_pct


def test_2_hotter_water_extracts_more_and_drains_a_bit_faster():
    # 뜨거울수록 녹는 속도가 빨라지고, 물이 덜 끈적해져(점도 96 °C가 88 °C보다 약 9% 낮음) 빨리 빠진다.
    cool, hot = predict(temp_c=88), predict(temp_c=96)
    assert hot.ey_pct > cool.ey_pct
    assert hot.drawdown_end_s < cool.drawdown_end_s


def test_3_more_water_dilutes_but_extracts_more():
    # 물이 많으면 묽어지지만, 더 오래 닿고 더 많이 씻어 내서 추출수율은 오른다.
    base, more = predict(), predict(pours="0:40;30:60;60:60;90:100")
    assert more.tds_pct < base.tds_pct
    assert more.ey_pct > base.ey_pct


def test_4_more_coffee_drains_slower():
    # 가루층이 두꺼워져 물이 지나는 길이 길어지고(다르시 법칙: 저항은 층 두께에 비례), 머금는 물도 늘어난다.
    base, more = predict(dose_g=20.0), predict(dose_g=24.0)
    assert more.drawdown_end_s > base.drawdown_end_s


def test_5_bypass_changes_strength_not_extraction():
    # 가수는 녹은 성분의 양을 바꾸지 않으므로 추출수율은 같고, 농도만 음료 무게 비율만큼 내려간다.
    plain, diluted = predict(bypass_g=0.0), predict(bypass_g=80.0)
    assert diluted.ey_pct == pytest.approx(plain.ey_pct)
    assert diluted.tds_pct == pytest.approx(plain.tds_pct)
    expected = plain.tds_pct * plain.beverage_g / (plain.beverage_g + 80.0)
    assert diluted.tds_cup_pct == pytest.approx(expected)


def test_6_mass_is_conserved():
    # 음료 무게 = 부은 물 − 가루가 머금은 액체 + 녹아 나온 성분. 물이 사라지거나 생기면 안 된다.
    p = predict()
    assert p.beverage_g == pytest.approx(220.0 - p.held_liquid_g + p.released_g, abs=0.5)
    # 음료로 나간 성분은 녹아 나온 성분보다 많을 수 없다 (일부는 가루가 머금은 액체에 남는다).
    assert 0 < p.dissolved_g <= p.released_g


@pytest.mark.parametrize("d50_um", [400.0, 1200.0, 2000.0])
@pytest.mark.parametrize("temp_c", [80.0, 98.0])
def test_7_results_stay_within_physical_limits(d50_um, temp_c):
    # 원두에서 녹아 나올 수 있는 성분은 약 30%뿐이고(Liang 2021, Moroney), 물은 다 부은 뒤에야 다 빠질 수 있다.
    p = predict(d50_um=d50_um, temp_c=temp_c)
    assert 0 < p.ey_pct <= 30.0
    assert p.drawdown_end_s > 90  # 마지막 붓기는 90초에 시작한다


def test_8_baseline_recipe_gives_realistic_numbers():
    # 문헌 드립 데이터(추출수율 16–24%)와 첫 B02 기록(물 빠짐 2:20, 음료 177.3 g)을 넉넉히 감싸는 범위.
    # 보정 전이라도 이 범위를 벗어나면 상수나 단위가 틀렸다는 뜻이다.
    p = predict()
    assert 15 <= p.ey_pct <= 26
    assert 60 <= p.drawdown_end_s <= 240
    assert 160 <= p.beverage_g <= 200


def test_9_pour_schedule_is_read_and_checked():
    pours = em.parse_pours(" 0:40 ; 30:60;60:60;90:60 ")
    assert [p.start_s for p in pours] == [0, 30, 60, 90]
    assert [p.grams for p in pours] == [40, 60, 60, 60]
    assert sum(p.grams for p in pours) == 220
    for bad in ["", "0:40;abc", "30:60;0:40", "0:-5"]:
        with pytest.raises(em.RecipeError, match="붓기 기록"):
            em.parse_pours(bad)
