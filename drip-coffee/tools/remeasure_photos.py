"""측정 규칙을 바꾼 뒤, 기록해 둔 분쇄 사진을 모두 다시 재서 기록 파일을 고친다.

- data/grind_photos.csv: 사진마다 대표 크기(measured_d50_um)와 메모(해상도, 잰 입자 수, 뺀 덩어리·은피 수)를 새로 쓴다.
- data/brews.csv: 같은 가루로 찍은 사진(sample_no가 추출 번호와 같은 사진)을 합친 대표 크기로 grind_d50_um을 고친다.
원본 사진은 저장소에 올리지 않으므로 사진 폴더를 따로 받는다. 바뀐 값은 화면에 보여 준다.

사용법 (drip-coffee 폴더에서):
    python tools/remeasure_photos.py <사진 폴더>
사진 폴더는 grind_photos.csv의 file 칸이 가리키는 곳이다(추출 번호별 하위 폴더가 있는 폴더).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dripcoffee import DATA_DIR  # noqa: E402
from dripcoffee import grind_analysis as ga  # noqa: E402

PHOTOS_PATH = DATA_DIR / "grind_photos.csv"
BREWS_PATH = DATA_DIR / "brews.csv"


def read_rows(path):
    """(칸 이름, 줄 목록, 줄 끝 문자). 다시 쓸 때 원래 모양을 지키려고 줄 끝 문자도 돌려준다."""
    newline = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader), newline


def write_rows(path, fieldnames, rows, newline):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames, lineterminator=newline)
        writer.writeheader()
        writer.writerows(rows)


def remeasure(photo_root):
    """사진마다 다시 재서 grind_photos.csv를 고치고, {추출 번호: [측정 결과]}를 돌려준다."""
    fieldnames, rows, newline = read_rows(PHOTOS_PATH)
    by_sample = defaultdict(list)
    for row in rows:
        path = photo_root / row["file"]
        if not path.exists():
            print(f"  {row['photo_id']}: 사진이 없어 그대로 둠 ({path})")
            continue
        rectified = ga.rectify_zone(ga.load_image_path(path), zone_name=row["zone"])
        m = ga.measure_particles(rectified.image, rectified.um_per_px)
        device = row["notes"].split(",")[0].strip() or "폰"
        old = row["measured_d50_um"]
        row["measured_d50_um"] = f"{m.d50_volume_um:.0f}" if len(m.diameters_um) else ""
        row["notes"] = (f"{device}, {rectified.um_per_px:.0f}µm/px, 측정 {len(m.diameters_um)}개 "
                        f"(붙어 있던 부피 {m.touching_volume_fraction:.0%}), "
                        f"덩어리 {m.excluded_clumps}개·은피 {m.excluded_chaff}개 뺌")
        print(f"  {row['photo_id']}: {old or '-'} → {row['measured_d50_um'] or '-'} µm ({len(m.diameters_um)}개)")
        by_sample[row["sample_no"]].append(m)
    write_rows(PHOTOS_PATH, fieldnames, rows, newline)
    return by_sample


def update_brews(by_sample):
    """같은 가루의 사진을 합친 대표 크기로 brews.csv의 grind_d50_um을 고친다."""
    fieldnames, rows, newline = read_rows(BREWS_PATH)
    for row in rows:
        measurements = by_sample.get(row["brew_id"])
        if not measurements:
            continue
        profile = ga.combine_measurements(measurements)
        low, high = profile.d50_range_um
        old = row["grind_d50_um"]
        row["grind_d50_um"] = f"{profile.d50_um:.0f}"
        print(f"  {row['brew_id']} ({row['bean_id']} {row['grinder_clicks']}클릭): {old or '-'} → "
              f"{row['grind_d50_um']} µm (다시 찍으면 {low:.0f}~{high:.0f}, 입자 {profile.n_particles}개, "
              f"붙어 있던 부피 {profile.touching_volume_fraction:.0%})")
        for warning in profile.warnings:
            print(f"    주의: {warning}")
    write_rows(BREWS_PATH, fieldnames, rows, newline)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo_root", type=Path, help="grind_photos.csv의 file 칸이 가리키는 사진 폴더")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    print("사진별 대표 크기")
    by_sample = remeasure(args.photo_root)
    print("잔별 대표 크기 (같은 가루의 사진을 합친 값)")
    update_brews(by_sample)


if __name__ == "__main__":
    main()
