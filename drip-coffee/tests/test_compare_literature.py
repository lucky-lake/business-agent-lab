"""문헌 비교 스크립트의 데이터 읽기와 판정 계산 테스트."""
import pytest

from dripcoffee import extraction_model as em
from tools import compare_literature as cl


def test_literature_data_loads_with_numbers():
    rows = cl.load_rows()
    assert len(rows) == 14
    assert {r["dataset"] for r in rows} == {"caa_v60", "batali_drip", "varady_v60"}
    assert all(r["status"] == "미확인" for r in rows)
    caa = [r for r in rows if r["dataset"] == "caa_v60"]
    assert all(isinstance(r["drawdown_s"], float) for r in caa)


def test_effect_passes_needs_same_direction_and_half_to_double_size():
    assert cl.effect_passes(2.0, 3.5)  # 0.57배
    assert not cl.effect_passes(1.5, 3.5)  # 0.43배
    assert not cl.effect_passes(8.0, 3.5)  # 2.3배
    assert not cl.effect_passes(-2.0, 3.5)  # 방향이 반대


def test_fitted_grind_reproduces_the_target_drawdown():
    recipe = em.Recipe(22, 100, em.parse_pours("0:77;45:123;105:174"))
    d50 = cl.fit_d50_to_drawdown(recipe, 187.0)
    assert em.predict(recipe, em.grind_from_d50(d50)).drawdown_end_s == pytest.approx(187.0, abs=1.0)


def test_extraction_only_rises_with_time_and_finer_grind_and_stays_below_limit():
    coarse, fine = em.grind_from_d50(1100), em.grind_from_d50(770)
    short = em.extraction_only(coarse, 176.1, 3075, 90, 400)[0]
    longer = em.extraction_only(coarse, 176.1, 3075, 90, 660)[0]
    finer = em.extraction_only(fine, 176.1, 3075, 90, 400)[0]
    assert short < longer <= 30
    assert short < finer <= 30


def test_comparisons_report_every_criterion():
    rows = cl.load_rows()
    caa = cl.compare_caa(rows)
    assert len(caa) == 3 and all({"ey_ok", "tds_ok", "bev_ok"} <= set(r) for r in caa)
    _, effects = cl.compare_batali(rows)
    assert len(effects) == 3
    varady = cl.compare_varady(rows)
    assert {"tds_ok", "time_ok"} <= set(varady)
