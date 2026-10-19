"""우리 추출 기록으로 모델의 두 계수를 맞춘다 (보정, docs/MODEL.md 5절).

- R_ref(흐름 저항): 표시 없는 잔들의 물 빠짐 종료 시각 오차 제곱합이 가장 작아지는 값
- R_abs(가루가 머금는 액체): 음료 무게 오차 제곱합이 가장 작아지는 값
- 두 계수는 서로 조금씩 영향을 주므로 번갈아 세 번 맞춘다.
- 교차 확인: 한 잔씩 빼고 나머지로 맞춘 뒤 뺀 잔을 예측한다(기록이 적어서).
- 분쇄 크기 지수 n(흐름 저항 ∝ 1/크기ⁿ): 0, 0.5, 1, 1.5, 2를 비교한다. 결과를 보기 전에 정한 규칙(10/19):
  교차 확인의 끊김 오차(크기 평균)가 이론값 n = 2보다 2초 넘게 줄어드는 n이 있을 때만 그중 가장 작은 쪽으로 바꾼다.
- 필터별 차이와 온도 의존(Ea)을 키웠을 때의 영향은 숫자로만 보고한다(맞추지 않는다).

사용법 (drip-coffee 폴더에서):
    python tools/calibrate.py
결과를 화면에 보여주고 out/calibration.txt로도 저장한다.
"""
import dataclasses
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dripcoffee import OUT_DIR  # noqa: E402
from dripcoffee import calibration as cal  # noqa: E402
from dripcoffee import extraction_model as em  # noqa: E402

EXPONENTS = (0.0, 0.5, 1.0, 1.5, 2.0)
EXPONENT_RULE_S = 2.0  # 이보다 많이 줄어야 n = 2에서 바꾼다
R_REF_RANGE = (1e10, 1e14)
R_ABS_RANGE = (1.5, 3.0)


def golden_min(f, low, high, steps=40):
    """[low, high]에서 한 봉우리 함수 f가 가장 작아지는 x (황금 분할 탐색)."""
    g = (math.sqrt(5) - 1) / 2
    a, b = low, high
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(steps):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def usable_ids(brews):
    """보정에 쓰는 잔: 표시가 없고, 끊김 시각·음료 무게·분쇄 크기(빌린 값 포함)가 모두 있는 잔."""
    sizes = cal.grind_sizes(brews)
    return [b.brew_id for b in brews
            if not b.flag and b.drawdown_end_s is not None and b.beverage_g is not None and b.brew_id in sizes]


def comparisons(params, brews, ids):
    return [c for c in cal.compare_with_log(brews, params) if c.brew.brew_id in ids]


def squared(params, brews, ids, which):
    errors = [c.end_error_s if which == "end" else c.beverage_error_g for c in comparisons(params, brews, ids)]
    return sum(e * e for e in errors)


def fit(params, brews, ids, rounds=3):
    """R_ref(끊김 시각)와 R_abs(음료 무게)를 번갈아 맞춘다."""
    for _ in range(rounds):
        log_r = golden_min(lambda x: squared(dataclasses.replace(params, flow_resistance_ref=math.exp(x)), brews, ids, "end"),
                           *map(math.log, R_REF_RANGE))
        params = dataclasses.replace(params, flow_resistance_ref=math.exp(log_r))
        held = golden_min(lambda x: squared(dataclasses.replace(params, held_liquid_ratio=x), brews, ids, "bev"),
                          *R_ABS_RANGE)
        params = dataclasses.replace(params, held_liquid_ratio=held)
    return params


def summary(params, brews, ids):
    """(끊김 평균, 끊김 크기 평균, 음료 평균, 음료 크기 평균). 차이는 예측 − 실제."""
    cs = comparisons(params, brews, ids)
    end = [c.end_error_s for c in cs]
    bev = [c.beverage_error_g for c in cs]
    mean = lambda xs: sum(xs) / len(xs)
    return mean(end), mean([abs(e) for e in end]), mean(bev), mean([abs(e) for e in bev])


def leave_one_out(params, brews, ids):
    """한 잔씩 빼고 맞춘 뒤 뺀 잔의 (끊김 오차, 음료 오차). {추출번호: (초, g)}"""
    out = {}
    for left in ids:
        fitted = fit(params, brews, [i for i in ids if i != left])
        c = comparisons(fitted, brews, [left])[0]
        out[left] = (c.end_error_s, c.beverage_error_g)
    return out


def choose_exponent(params, brews, ids):
    """n별 교차 확인 끊김 크기 평균. 미리 정한 규칙대로 n을 고른다. (고른 n, {n: 초})"""
    scores = {}
    for n in EXPONENTS:
        loo = leave_one_out(dataclasses.replace(params, resistance_size_exponent=n), brews, ids)
        scores[n] = sum(abs(e) for e, _ in loo.values()) / len(loo)
    better = [n for n in EXPONENTS if scores[n] < scores[2.0] - EXPONENT_RULE_S]
    return (min(better, key=scores.get) if better else 2.0), scores


def ea_effect(params, ea_values=(17.3e3, 25e3, 35e3)):
    """온도 의존(Ea)을 키웠을 때, B02 기준 레시피(1255 µm)에서 88 → 96 °C 추출수율 변화."""
    out = {}
    for ea in ea_values:
        p = dataclasses.replace(params, activation_energy_j_per_mol=ea)
        ey = [em.predict(em.Recipe(20, t, em.parse_pours("0:40;30:60;60:60;90:60"), 80), em.grind_from_d50(1255), p).ey_pct
              for t in (88, 96)]
        out[ea] = (ey[0], ey[1])
    return out


def main():
    brews = cal.load_brews()
    ids = usable_ids(brews)
    start = em.DEFAULT_PARAMS
    lines = [f"보정에 쓰는 잔 {len(ids)}개: {', '.join(ids)}", ""]

    def report(title, params):
        e, ea, b, ba = summary(params, brews, ids)
        lines.append(f"{title}: R_ref {params.flow_resistance_ref:.3e}, R_abs {params.held_liquid_ratio:.2f}, "
                     f"n {params.resistance_size_exponent:g} → 끊김 평균 {e:+.1f}초·크기 평균 {ea:.1f}초, "
                     f"음료 평균 {b:+.1f} g·크기 평균 {ba:.1f} g")

    report("보정 전", start)
    n, scores = choose_exponent(start, brews, ids)
    lines.append("분쇄 크기 지수 n별 교차 확인(끊김 크기 평균): "
                 + ", ".join(f"n={k:g} {v:.1f}초" for k, v in scores.items()) + f" → n = {n:g}")
    fitted = fit(dataclasses.replace(start, resistance_size_exponent=n), brews, ids)
    report("보정 후", fitted)

    lines += ["", "잔별 (예측 − 실제): 보정 전 → 보정 후, 교차 확인(그 잔을 빼고 맞췄을 때)"]
    loo = leave_one_out(dataclasses.replace(start, resistance_size_exponent=n), brews, ids)
    before = {c.brew.brew_id: c for c in comparisons(start, brews, ids)}
    after = {c.brew.brew_id: c for c in comparisons(fitted, brews, ids)}
    for i in ids:
        b, a = before[i], after[i]
        lines.append(f"  {i} ({a.brew.bean_id}, {a.brew.filter_paper}, {a.grind_d50_um:.0f} µm): "
                     f"끊김 {b.end_error_s:+.1f} → {a.end_error_s:+.1f}초 (교차 {loo[i][0]:+.1f}), "
                     f"음료 {b.beverage_error_g:+.1f} → {a.beverage_error_g:+.1f} g (교차 {loo[i][1]:+.1f})")
    lines.append(f"교차 확인 크기 평균: 끊김 {sum(abs(e) for e, _ in loo.values()) / len(loo):.1f}초, "
                 f"음료 {sum(abs(g) for _, g in loo.values()) / len(loo):.1f} g")

    lines += ["", "필터별 보정 후 끊김 평균 (예측 − 실제)"]
    for paper in sorted({after[i].brew.filter_paper for i in ids}):
        errs = [after[i].end_error_s for i in ids if after[i].brew.filter_paper == paper]
        lines.append(f"  {paper}: {sum(errs) / len(errs):+.1f}초 ({len(errs)}잔)")

    lines += ["", "온도 의존 Ea를 키웠을 때 B02 기준 레시피 추출수율 (88 → 96 °C)"]
    for ea, (low, high) in ea_effect(fitted).items():
        lines.append(f"  Ea {ea / 1000:.1f} kJ/mol: {low:.1f}% → {high:.1f}% ({high - low:+.1f}%p)")

    text = "\n".join(lines)
    print(text)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "calibration.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\n저장: {OUT_DIR / 'calibration.txt'}")
    return fitted


if __name__ == "__main__":
    main()
