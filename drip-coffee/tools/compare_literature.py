"""문헌 실험값과 추출 모델의 예측을 비교한다.

판정 기준은 docs/MODEL.md 6절에 있고, 결과를 보기 전에 정해 커밋했다(5d1570e).
데이터는 data/reference/literature_brews.csv(조사 노트 4에서 옮김, 상태 미확인)를 쓴다.

사용법 (drip-coffee 폴더에서):
    python tools/compare_literature.py
결과 표를 화면에 보여주고 out/literature_comparison.csv로도 저장한다.
"""
import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dripcoffee import DATA_DIR, OUT_DIR  # noqa: E402
from dripcoffee import extraction_model as em  # noqa: E402

LITERATURE = DATA_DIR / "reference" / "literature_brews.csv"
NUMERIC = ("dose_g", "water_g", "temp_c", "grind_um", "contact_s", "drawdown_s", "beverage_g", "tds_pct", "ey_pct")


def load_rows(path=LITERATURE):
    """문헌 데이터를 읽어 숫자 칸은 float로 바꾼다 (빈칸은 None)."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in NUMERIC:
            row[key] = float(row[key]) if row[key] else None
    return rows


def effect_passes(predicted_delta, measured_delta, low=0.5, high=2.0):
    """바꾼 조건의 효과(추출수율 변화)가 실험과 같은 방향이고, 크기가 실험의 low~high배인지."""
    if measured_delta == 0:
        return predicted_delta == 0
    ratio = predicted_delta / measured_delta
    return low <= ratio <= high


def fit_d50_to_drawdown(recipe, target_s, lo_um=300.0, hi_um=3000.0):
    """예측 물 빠짐 시각이 target_s와 같아지는 대표 크기를 찾는다 (굵을수록 빨리 빠지므로 하나로 정해진다)."""
    for _ in range(50):
        mid = math.sqrt(lo_um * hi_um)
        if em.predict(recipe, em.grind_from_d50(mid)).drawdown_end_s > target_s:
            lo_um = mid  # 너무 늦게 빠짐 → 더 굵게
        else:
            hi_um = mid
    return math.sqrt(lo_um * hi_um)


def compare_caa(rows):
    """① Coffee ad Astra V60: 물 빠짐을 실험과 맞춘 뒤 추출수율·농도·음료 무게를 비교한다."""
    results = []
    for row in (r for r in rows if r["dataset"] == "caa_v60"):
        recipe = em.Recipe(row["dose_g"], row["temp_c"], em.parse_pours(row["pours"]))
        d50 = fit_d50_to_drawdown(recipe, row["drawdown_s"])
        p = em.predict(recipe, em.grind_from_d50(d50))
        results.append({
            "run": row["run"], "d50_um": d50, "drawdown_s": p.drawdown_end_s,
            "ey_pred": p.ey_pct, "ey_meas": row["ey_pct"],
            "tds_pred": p.tds_pct, "tds_meas": row["tds_pct"],
            "bev_pred": p.beverage_g, "bev_meas": row["beverage_g"],
            "ey_ok": 18 <= p.ey_pct <= 24,
            "tds_ok": abs(p.tds_pct - row["tds_pct"]) <= 0.3,
            "bev_ok": abs(p.beverage_g - row["beverage_g"]) <= 15,
        })
    return results


def compare_batali(rows):
    """② Batali 2020 배치 브루어: 추출 계산만으로 각 회차의 추출수율을 내고, 분쇄·시간 효과의 크기를 비교한다."""
    runs = {}
    for row in (r for r in rows if r["dataset"] == "batali_drip"):
        ey, tds = em.extraction_only(
            em.grind_from_d50(row["grind_um"]), row["dose_g"], row["water_g"], row["temp_c"], row["contact_s"]
        )
        runs[row["run"]] = {"row": row, "ey_pred": ey, "tds_pred": tds}
    effects = []
    for name, a, b in [
        ("분쇄 1115 → 770 µm (90 °C 6:44 → 87 °C 7:48)", "1", "2"),
        ("분쇄 1115 → 770 µm (90 °C 6:44 → 93 °C 7:00)", "1", "3"),
        ("시간 7:00 → 11:00 (770 µm, 93 → 90 °C)", "3", "4"),
    ]:
        measured = runs[b]["row"]["ey_pct"] - runs[a]["row"]["ey_pct"]
        predicted = runs[b]["ey_pred"] - runs[a]["ey_pred"]
        effects.append({"effect": name, "measured": measured, "predicted": predicted,
                        "ok": effect_passes(predicted, measured)})
    return runs, effects


def compare_varady(rows, d50_sweep_um=(700.0, 900.0, 1100.0, 1300.0)):
    """③ Váradý 2022 V60: 클릭에 µm가 없으므로, 곱게 갈수록 농도가 오르고 물 빠짐이 늦어지는 방향만 비교한다."""
    measured = sorted((r for r in rows if r["dataset"] == "varady_v60"), key=lambda r: float(r["run"]))
    first = measured[0]
    recipe = em.Recipe(first["dose_g"], first["temp_c"], em.parse_pours(first["pours"]))
    model = [em.predict(recipe, em.grind_from_d50(d)) for d in d50_sweep_um]
    fine, coarse = measured[0], measured[-1]  # 클릭이 작을수록 곱다
    return {
        "measured": [(r["run"], r["tds_pct"], r["drawdown_s"]) for r in measured],
        "model": [(d, p.tds_pct, p.drawdown_end_s) for d, p in zip(d50_sweep_um, model)],
        "tds_ok": (fine["tds_pct"] > coarse["tds_pct"]) == (model[0].tds_pct > model[-1].tds_pct),
        "time_ok": (fine["drawdown_s"] > coarse["drawdown_s"]) == (model[0].drawdown_end_s > model[-1].drawdown_end_s),
    }


def _mark(ok):
    return "통과" if ok else "불통과"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = load_rows()
    table = []

    print("① Coffee ad Astra V60 (22 g, 374 g, 100 °C): 물 빠짐을 맞춘 뒤 비교")
    print("회차  맞춘 크기  추출수율 예측/실험     농도 예측/실험     음료 예측/실험")
    caa = compare_caa(rows)
    for r in caa:
        print(f"{r['run']:>3}  {r['d50_um']:7.0f} µm  {r['ey_pred']:5.1f} / {r['ey_meas']:4.1f} {_mark(r['ey_ok'])}"
              f"  {r['tds_pred']:4.2f} / {r['tds_meas']:4.2f} {_mark(r['tds_ok'])}"
              f"  {r['bev_pred']:5.1f} / {r['bev_meas']:5.1f} {_mark(r['bev_ok'])}")
        for item, pred, meas, ok in [("추출수율 %", r["ey_pred"], r["ey_meas"], r["ey_ok"]),
                                     ("농도 %", r["tds_pred"], r["tds_meas"], r["tds_ok"]),
                                     ("음료 g", r["bev_pred"], r["bev_meas"], r["bev_ok"])]:
            table.append(["caa_v60", f"{r['run']}회 {item}", meas, round(pred, 2), _mark(ok)])

    print("\n② Batali 2020 배치 브루어: 추출 계산만")
    runs, effects = compare_batali(rows)
    print("회차  분쇄 µm  온도  시간   추출수율 예측/실험")
    for run, r in runs.items():
        row = r["row"]
        print(f"{run:>3}  {row['grind_um']:6.0f}  {row['temp_c']:4.0f}  {row['contact_s']:5.0f}s  "
              f"{r['ey_pred']:5.1f} / {row['ey_pct']:4.1f}")
        table.append(["batali_drip", f"{run}회 추출수율 %", row["ey_pct"], round(r["ey_pred"], 2), "참고"])
    for e in effects:
        print(f"  {e['effect']}: 실험 {e['measured']:+.1f}%p, 예측 {e['predicted']:+.1f}%p → {_mark(e['ok'])}")
        table.append(["batali_drip", e["effect"], round(e["measured"], 2), round(e["predicted"], 2), _mark(e["ok"])])

    print("\n③ Váradý 2022 V60 (12 g, 210 g, 94 °C): 방향만")
    v = compare_varady(rows)
    print("  실험 (클릭, 농도, 시간): " + ", ".join(f"{c}클릭 {t:.2f}% {s:.0f}s" for c, t, s in v["measured"]))
    print("  모델 (크기, 농도, 시간): " + ", ".join(f"{d:.0f}µm {t:.2f}% {s:.0f}s" for d, t, s in v["model"]))
    print(f"  곱게 갈수록 농도가 오름: {_mark(v['tds_ok'])}, 물 빠짐이 늦어짐: {_mark(v['time_ok'])}")
    table.append(["varady_v60", "곱게 갈수록 농도 상승 (방향)", "", "", _mark(v["tds_ok"])])
    table.append(["varady_v60", "곱게 갈수록 물 빠짐 늦어짐 (방향)", "", "", _mark(v["time_ok"])])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "literature_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "item", "measured", "predicted", "result"])
        writer.writerows(table)
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()
