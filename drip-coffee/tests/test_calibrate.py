"""보정 스크립트(tools/calibrate.py) 테스트. 전체 보정과 교차 확인은 오래 걸려서 핵심 부분만 본다."""
import dataclasses
import sys

import pytest

from dripcoffee import PROJECT_ROOT
from dripcoffee import calibration as cal
from dripcoffee import extraction_model as em

sys.path.insert(0, str(PROJECT_ROOT / "tools"))
import calibrate as cb  # noqa: E402


def test_golden_section_finds_the_lowest_point():
    assert cb.golden_min(lambda x: (x - 2.3) ** 2, 0, 10) == pytest.approx(2.3, abs=1e-3)


def test_only_unflagged_cups_with_results_and_grind_are_used():
    ids = cb.usable_ids(cal.load_brews())
    assert "20261009-1" not in ids and "20261016-2" not in ids  # 연습, 막힘
    assert {"20261012-1", "20261012-2", "20261016-1", "20261016-3", "20261018-1"} <= set(ids)


def test_fitting_lowers_the_squared_errors():
    brews = cal.load_brews()
    ids = cb.usable_ids(brews)
    start = em.DEFAULT_PARAMS
    fitted = cb.fit(start, brews, ids, rounds=1)
    assert cb.squared(fitted, brews, ids, "end") <= cb.squared(start, brews, ids, "end")
    assert cb.squared(fitted, brews, ids, "bev") <= cb.squared(start, brews, ids, "bev")


def test_exponent_rule_keeps_literature_verdicts():
    # n은 교차 확인 오차가 n = 2보다 2초 넘게 줄고, n = 2에서 통과하는 문헌 항목을 모두 계속 통과할 때만 바꾼다.
    scores = {0.0: 6.0, 0.5: 10.0, 1.0: 13.0, 1.5: 18.0, 2.0: 24.0}
    same = {n: {"①", "③ 시간"} for n in scores}
    assert cb.pick_exponent(scores, same) == 0.0
    loses_time = {**same, 0.0: {"①"}, 0.5: {"①"}}  # n이 작으면 '곱게 갈수록 늦게 빠짐'을 잃는다
    assert cb.pick_exponent(scores, loses_time) == 1.0
    assert cb.pick_exponent({**scores, 1.0: 23.0, 1.5: 22.5}, loses_time) == 2.0  # 2초 넘게 줄지 않으면 그대로


def test_literature_verdicts_follow_the_parameters():
    # 문헌 판정은 보정할 계수로 다시 낸다. n = 0이면 크기가 물 빠짐에 영향이 없어 ③ 방향을 잃는다.
    base = cb.literature_passes(em.DEFAULT_PARAMS)
    flat = cb.literature_passes(dataclasses.replace(em.DEFAULT_PARAMS, resistance_size_exponent=0.0))
    assert "③ 곱게 갈수록 물 빠짐 늦어짐" in base - flat


def test_grind_size_exponent_changes_how_much_size_matters():
    recipe = em.Recipe(20, 92, em.parse_pours("0:40;30:60;60:60;90:60"), 80)
    times = {}
    for n in (0.0, 2.0):
        params = dataclasses.replace(em.DEFAULT_PARAMS, resistance_size_exponent=n)
        times[n] = [em.predict(recipe, em.grind_from_d50(d), params).drawdown_end_s for d in (900, 1300)]
    assert times[0.0][0] == pytest.approx(times[0.0][1], abs=1)  # n = 0이면 크기와 상관없다
    assert times[2.0][0] > times[2.0][1] + 20  # n = 2면 고울수록 크게 늦어진다
