"""클릭 설정별 분쇄 사진을 분석해 메버릭 클릭표와 비교하는 검증 스크립트.

사용법 (drip-coffee 폴더에서):
    python tools/verify_grind.py <사진 폴더>
사진 폴더 안에 클릭 수 이름의 하위 폴더(예: 85, 95, 105)를 두고 그 안에 사진을 넣는다.
결과 표를 화면에 보여주고 out/grind_verification.csv 로도 저장한다.

메버릭 클릭표(data/reference/maverick_click_table.csv)는 인터넷 기준 값으로 비교용이다.
절대값보다 클릭이 커질 때 측정값도 같은 순서·비슷한 비율로 커지는지를 본다.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dripcoffee import DATA_DIR, OUT_DIR  # noqa: E402
from dripcoffee import grind_analysis as ga  # noqa: E402

PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CLICK_TABLE = DATA_DIR / "reference" / "maverick_click_table.csv"


def load_click_table(path=CLICK_TABLE):
    with open(path, encoding="utf-8") as f:
        return {int(row["clicks"]): float(row["size_um"]) for row in csv.DictReader(f)}


def measure_folder(folder):
    """폴더 안 사진마다 보이는 칸(A/B)을 모두 재서 측정 결과 목록과 건너뛴 사진 메모를 돌려준다."""
    measurements, notes = [], []
    for photo in sorted(p for p in folder.iterdir() if p.suffix.lower() in PHOTO_SUFFIXES):
        image = ga.load_image_path(photo)
        markers = ga.detect_markers(image)
        measured_any = False
        for zone in ("A", "B"):
            try:
                rectified = ga.rectify_zone(image, markers, zone)
            except ga.GrindPhotoError:
                continue
            m = ga.measure_particles(rectified.image, rectified.um_per_px)
            if len(m.diameters_um):
                measurements.append(m)
                measured_any = True
        if not measured_any:
            notes.append(f"{photo.name}: 가루가 담긴 칸을 찾지 못해 건너뜀")
    return measurements, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo_root", type=Path, help="클릭 수 이름의 하위 폴더가 있는 사진 폴더")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    table = load_click_table()
    settings = sorted((int(p.name), p) for p in args.photo_root.iterdir() if p.is_dir() and p.name.isdigit())
    if not settings:
        sys.exit("클릭 수 이름의 하위 폴더(예: 85, 95, 105)가 없어요.")

    rows = []
    for clicks, folder in settings:
        measurements, notes = measure_folder(folder)
        for note in notes:
            print("  " + note)
        if not measurements:
            print(f"{clicks}클릭: 잰 사진이 없어 건너뜀")
            continue
        profile = ga.combine_measurements(measurements)
        rows.append((clicks, profile))
        for w in profile.warnings:
            print(f"  {clicks}클릭 경고: {w}")

    base_clicks, base = rows[0]
    print()
    print("클릭  사진  입자   D10    D50    D90   (사진별 D50 범위)   D50 비율  클릭표(µm)  클릭표 비율")
    out_rows = []
    for clicks, p in rows:
        ratio = p.d50_um / base.d50_um
        ref = table.get(clicks)
        ref_ratio = ref / table[base_clicks] if ref and base_clicks in table else float("nan")
        spread = f"{min(p.photo_d50_um):.0f}~{max(p.photo_d50_um):.0f}"
        print(
            f"{clicks:>4}  {p.n_photos:>4}  {p.n_particles:>4}  {p.d10_um:>5.0f}  {p.d50_um:>5.0f}  {p.d90_um:>5.0f}"
            f"   ({spread:>11})      {ratio:>5.2f}     {ref or float('nan'):>6.0f}      {ref_ratio:>5.2f}"
        )
        out_rows.append([clicks, p.n_photos, p.n_particles, round(p.d10_um), round(p.d50_um), round(p.d90_um),
                         round(min(p.photo_d50_um)), round(max(p.photo_d50_um)), ref, p.excluded_clumps])

    d50s = [p.d50_um for _, p in rows]
    in_order = all(a < b for a, b in zip(d50s, d50s[1:]))
    print()
    print("클릭 순서대로 D50이 커지나요?", "예" if in_order else "아니요")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "grind_verification.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clicks", "photos", "particles", "d10_um", "d50_um", "d90_um",
                         "photo_d50_min_um", "photo_d50_max_um", "click_table_um", "excluded_clumps"])
        writer.writerows(out_rows)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
