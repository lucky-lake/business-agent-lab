"""입자 검출과 지름 계산의 정확도 테스트.

크기를 정확히 아는 불규칙한 입자를 그린 합성 이미지(펴진 가루 칸을 흉내 냄)로,
검출한 입자의 개수와 지름이 정답과 얼마나 맞는지 확인한다.
하한 미만 미분 집계와 입자 붙음 경고는 아직 넣지 않기로 해서 보류(skip)로 둔다.
"""
import numpy as np
import pytest

from dripcoffee import grind_analysis as ga
from tests.synthetic import COFFEE_BGR, PAPER_BGR, render_particles, scattered_particles

PX_PER_MM = 40.0  # 펴진 B칸 사진과 비슷한 해상도 (25 µm/px)
UM_PER_PX = 1000.0 / PX_PER_MM
SQUARE_MM = 25.0
SIZE_PX = int(SQUARE_MM * PX_PER_MM)

HELD = pytest.mark.skip(reason="보류: 하한 미만 미분 집계와 입자 붙음 경고는 나중에 정함")


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


def test_touching_pairs_are_left_out_as_clumps():
    # 실제 사진에서 붙은 입자들이 큰 입자 하나로 재져 부피 중앙값이 3~10배 커진 경우를 막는다.
    singles = scattered_particles(36, SQUARE_MM, 0.7, 0.2, seed=9)
    pairs = scattered_particles(36, SQUARE_MM, 0.7, 0.2, seed=10)[::3]
    touching = [(x + 0.95 * d, y, d) for x, y, d in pairs]  # 바로 옆에 붙은 짝
    singles = [p for i, p in enumerate(singles) if i % 3 != 0]  # 짝을 둔 자리는 비운다
    img, true_d = render_particles(SIZE_PX, PX_PER_MM, singles + pairs + touching)
    m = ga.measure_particles(img, UM_PER_PX)
    assert m.excluded_clumps >= 0.8 * len(pairs)
    assert len(m.diameters_um) <= len(singles) + 0.2 * len(pairs)


@HELD
def test_specks_below_detection_limit_are_ignored_and_reported():
    parts = scattered_particles(36, SQUARE_MM, 0.9, 0.3, seed=7)
    cell = (SQUARE_MM - 2.0) / 6
    specks = [(1.0 + i * cell, 1.0 + j * cell, 0.06) for i in range(1, 6) for j in range(1, 6)]  # 지름 2.4 px
    img, _ = render_particles(SIZE_PX, PX_PER_MM, parts + specks)
    m = ga.measure_particles(img, UM_PER_PX)
    assert len(m.diameters_um) == len(parts)
    assert m.min_diameter_um > 60  # 검출 하한을 함께 알려준다


@HELD
def test_crowded_sample_gives_warning():
    base = scattered_particles(36, SQUARE_MM, 0.9, 0.2, seed=8)
    touching = base + [(x + 0.6 * d, y, d) for x, y, d in base]  # 입자마다 옆에 하나씩 붙여 놓는다
    img, _ = render_particles(SIZE_PX, PX_PER_MM, touching)
    m = ga.measure_particles(img, UM_PER_PX)
    assert any("붙어" in w for w in m.warnings)
