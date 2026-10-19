"""입자 검출과 지름 계산의 정확도 테스트.

크기를 정확히 아는 불규칙한 입자를 그린 합성 이미지(펴진 가루 칸을 흉내 냄)로,
검출한 입자의 개수와 지름이 정답과 얼마나 맞는지 확인한다.
"""
import cv2
import numpy as np

from dripcoffee import grind_analysis as ga
from tests.synthetic import COFFEE_BGR, DARK_CHAFF_BGR, PAPER_BGR, render_particles, scattered_particles

PX_PER_MM = 40.0  # 펴진 B칸 사진과 비슷한 해상도 (25 µm/px)
UM_PER_PX = 1000.0 / PX_PER_MM
SQUARE_MM = 25.0
SIZE_PX = int(SQUARE_MM * PX_PER_MM)


def volume_median_um(diameters_um):
    """부피(질량) 기준 중앙값: 큰 입자일수록 무게가 크므로 지름의 세제곱으로 가중한다."""
    d = np.sort(np.asarray(diameters_um, dtype=float))
    cumulative = np.cumsum(d**3) / np.sum(d**3)
    return float(np.interp(0.5, cumulative, d))


def test_generator_draws_exact_area():
    # 테스트 이미지 쪽이 틀리면 안 되므로, 생성기가 그린 면적부터 정답과 맞는지 확인한다.
    # 잡티가 있으면 덮인 비율 추정이 치우치므로 잡티 없이 그려서 확인한다.
    parts = scattered_particles(60, SQUARE_MM, 0.9, 0.35, seed=1)
    img, true_d = render_particles(SIZE_PX, PX_PER_MM, parts, noise_sigma=0)
    paper, coffee = np.mean(PAPER_BGR), np.mean(COFFEE_BGR)
    covered = (paper - img.mean(axis=2)) / (paper - coffee)
    drawn_mm2 = covered.sum() / PX_PER_MM**2
    true_mm2 = sum(np.pi * (d / 2) ** 2 for d in true_d)
    assert abs(drawn_mm2 - true_mm2) / true_mm2 < 0.005


def test_isolated_particle_sizes_match():
    parts = scattered_particles(100, SQUARE_MM, 0.9, 0.35, seed=2)
    img, true_d = render_particles(SIZE_PX, PX_PER_MM, parts)
    m = ga.measure_particles(img, UM_PER_PX)

    assert len(m.diameters_um) == len(true_d)
    measured = np.sort(m.diameters_um)
    truth = np.sort(np.array(true_d) * 1000)
    rel = np.abs(measured - truth) / truth
    assert np.median(rel) < 0.02
    assert rel.max() < 0.06


def test_volume_median_matches():
    parts = scattered_particles(100, SQUARE_MM, 0.9, 0.35, seed=3)
    img, true_d = render_particles(SIZE_PX, PX_PER_MM, parts)
    m = ga.measure_particles(img, UM_PER_PX)
    truth = volume_median_um(np.array(true_d) * 1000)
    assert abs(m.d50_volume_um - truth) / truth < 0.03


def test_chaff_is_not_counted_as_coffee():
    parts = scattered_particles(49, SQUARE_MM, 0.8, 0.3, seed=4)
    coffee, chaff = parts[:-6], parts[-6:]
    img, _ = render_particles(SIZE_PX, PX_PER_MM, coffee, chaff=chaff)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == len(coffee)
    assert m.excluded_chaff == len(chaff)


def test_particles_cut_by_square_edge_are_excluded():
    parts = scattered_particles(36, SQUARE_MM, 0.8, 0.3, seed=5)
    on_edge = [(0.0, SQUARE_MM / 2, 1.0)]  # 칸 경계에 걸쳐 일부만 보이는 입자
    img, _ = render_particles(SIZE_PX, PX_PER_MM, parts + on_edge)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == len(parts)
    assert m.excluded_border >= 1


def test_uneven_lighting_does_not_change_sizes():
    parts = scattered_particles(64, SQUARE_MM, 0.9, 0.35, seed=6)
    img, true_d = render_particles(SIZE_PX, PX_PER_MM, parts, shading=0.3)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == len(true_d)
    rel = np.abs(np.sort(m.diameters_um) - np.sort(np.array(true_d) * 1000)) / np.sort(np.array(true_d) * 1000)
    assert np.median(rel) < 0.03


def arc_cluster(cx, cy, n=13, radius=1.2, d=0.8):
    """입자 여러 개가 넓게 맞닿아 휘어진 덩어리 (촘촘히 편 사진의 엉킨 덩어리 흉내). 이웃 간격은 지름의 0.6배."""
    angles = np.linspace(0, 1.5 * np.pi, n)
    return [(cx + radius * np.cos(a), cy + radius * np.sin(a), d) for a in angles]


def test_touching_pairs_are_split_and_measured():
    # 붙은 짝을 큰 입자 하나로 재면 부피 중앙값이 커지고, 통째로 빼면 큰 입자가 빠져 작아진다.
    # 잘록한 곳에서 나눠 두 입자로 재야 정답에 가깝다.
    singles = scattered_particles(36, SQUARE_MM, 0.7, 0.2, seed=9)
    pairs = scattered_particles(36, SQUARE_MM, 0.7, 0.2, seed=10)[::3]
    touching = [(x + d, y, d) for x, y, d in pairs]  # 바로 옆에 맞닿은 짝
    singles = [p for i, p in enumerate(singles) if i % 3 != 0]  # 짝을 둔 자리는 비운다
    img, true_d = render_particles(SIZE_PX, PX_PER_MM, singles + pairs + touching)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) >= len(singles) + 1.75 * len(pairs)  # 짝의 75% 이상을 두 입자로 나눴다
    assert m.excluded_clumps <= 0.2 * len(pairs)
    truth = volume_median_um(np.array(true_d) * 1000)
    assert abs(m.d50_volume_um - truth) / truth < 0.05


def test_single_particle_with_a_waist_is_not_split_or_left_out():
    # 실제 사진에서 혼자 있는 울퉁불퉁한 큰 입자를 덩어리로 보고 빼서 대표 크기가 작게 나왔다(10/19).
    # 허리가 굵게 잘록한 입자(지름의 0.6배 거리로 겹친 두 조각, 잘록한 곳이 반지름의 0.8배)는 한 입자로 재야 한다.
    base = scattered_particles(36, SQUARE_MM, 0.8, 0.15, seed=12)[::2]
    img, _ = render_particles(SIZE_PX, PX_PER_MM, base + [(x + 0.6 * d, y, d) for x, y, d in base])
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == len(base)
    assert m.excluded_clumps == 0


def test_dark_chaff_piece_is_left_out_by_color():
    # 큰 은피 조각은 커피 입자만큼 어둡게 찍혀 커피로 잡혔다(10/19 실제 사진). 노란 정도로 가려 뺀다.
    parts = scattered_particles(49, SQUARE_MM, 0.8, 0.3, seed=4)
    coffee, chaff = parts[:-6], [(x, y, 1.6 * d) for x, y, d in parts[-6:]]
    img, _ = render_particles(SIZE_PX, PX_PER_MM, coffee, chaff=chaff, chaff_bgr=DARK_CHAFF_BGR)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == len(coffee)
    assert m.excluded_chaff == len(m.outlines["chaff"]) == len(chaff)


def test_tangled_cluster_that_cannot_be_split_is_left_out():
    # 여러 입자가 넓게 맞닿아 나눌 잘록한 곳이 없는 덩어리는 큰 입자 하나로 재지 않고 뺀다.
    parts = scattered_particles(16, SQUARE_MM, 0.7, 0.2, seed=13)
    parts = [(x, y, d) for x, y, d in parts if np.hypot(x - 12.5, y - 12.5) > 4.0]  # 가운데를 비운다
    img, _ = render_particles(SIZE_PX, PX_PER_MM, parts + arc_cluster(12.5, 12.5))
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == len(parts)
    assert m.excluded_clumps == 1


def test_outlines_match_counts_by_kind():
    # 화면에서 잰 입자·덩어리·테두리 입자를 색으로 나눠 그리려면 개수가 결과와 맞아야 한다.
    singles = scattered_particles(36, SQUARE_MM, 0.7, 0.2, seed=9)
    pairs = scattered_particles(36, SQUARE_MM, 0.7, 0.2, seed=10)[::3]
    touching = [(x + 0.95 * d, y, d) for x, y, d in pairs]
    singles = [p for i, p in enumerate(singles) if i % 3 != 0 and np.hypot(p[0] - 12.5, p[1] - 12.5) > 4.0]
    on_edge = [(0.0, SQUARE_MM / 2, 2.0)]  # 반쯤 칸 안으로 들어온 큰 입자
    img, _ = render_particles(SIZE_PX, PX_PER_MM, singles + pairs + touching + on_edge + arc_cluster(12.5, 12.5))
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.outlines["measured"]) == len(m.diameters_um)
    assert len(m.outlines["clump"]) == m.excluded_clumps > 0
    assert len(m.outlines["border"]) == m.excluded_border > 0


def test_outlines_sit_on_their_particles():
    # 윤곽선이 펴진 이미지 좌표(픽셀)로 제자리에 있어야 사진 위에 겹쳐 그릴 수 있다.
    parts = scattered_particles(25, SQUARE_MM, 0.9, 0.3, seed=11)
    img, _ = render_particles(SIZE_PX, PX_PER_MM, parts)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.outlines["measured"]) == len(parts)
    for outline, (cx_mm, cy_mm), d_um in zip(m.outlines["measured"], m.centroids_mm, m.diameters_um):
        moments = cv2.moments(outline)
        center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        assert np.hypot(center[0] - cx_mm * PX_PER_MM, center[1] - cy_mm * PX_PER_MM) < 1.5
        # 윤곽선이 감싼 넓이는 같은 지름의 원과 비슷해야 한다 (경계 픽셀 처리 차이로 조금 작다).
        area = moments["m00"]
        assert 0.8 < area / (np.pi * (d_um / UM_PER_PX / 2) ** 2) < 1.05


def test_nothing_measurable_still_reports_what_was_left_out():
    # 입자가 모두 칸 테두리에 걸렸어도, 왜 못 쟀는지 보여줄 수 있게 뺀 개수와 윤곽선을 돌려준다.
    on_edges = [(0.0, SQUARE_MM / 2, 2.0), (SQUARE_MM, SQUARE_MM / 2, 2.0)]
    img, _ = render_particles(SIZE_PX, PX_PER_MM, on_edges)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == 0
    assert m.excluded_border == 2
    assert len(m.outlines["border"]) == 2
    assert m.warnings


def test_specks_below_detection_limit_are_ignored_and_reported():
    parts = scattered_particles(36, SQUARE_MM, 0.9, 0.3, seed=7)
    cell = (SQUARE_MM - 2.0) / 6
    specks = [(1.0 + i * cell, 1.0 + j * cell, 0.06) for i in range(1, 6) for j in range(1, 6)]  # 지름 2.4 px
    img, _ = render_particles(SIZE_PX, PX_PER_MM, parts + specks)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == len(parts)
    assert m.min_diameter_um > 60  # 검출 하한을 함께 알려준다


def test_thin_diagonal_particle_does_not_stop_the_clump_check():
    # 10/18 B02 사진(20261018-1 A2)에서 측정을 멈추게 한 입자 모양. 1픽셀 폭으로 비스듬히 이어져서
    # 윤곽선이 제자리를 되밟고, 볼록 껍질의 점 순서가 뒤집혀 OpenCV가 오류를 냈다.
    rows = [".......##", "......#..", ".....###.", ".....#...", "...##....",
            "..###....", "#####....", "####.....", "###......", ".#......."]
    mask = np.array([[c == "#" for c in row] for row in rows])
    assert ga._solidity(mask) > 0  # 픽셀 수 ÷ 윤곽선 넓이라 아주 작은 입자는 1을 조금 넘을 수 있다
    pieces, k = ga._split_touching(mask)
    assert k >= 1 and np.array_equal(pieces > 0, mask)
