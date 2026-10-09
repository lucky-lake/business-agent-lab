"""여러 장을 합친 분쇄 프로필과 크기 구간 계산 테스트."""
import numpy as np
import pytest

from dripcoffee import grind_analysis as ga
from tests.synthetic import render_sheet_window, simulate_photo
from tools.verify_grind import load_click_table


def make_measurement(diameters, clumps=0):
    d = np.asarray(diameters, dtype=float)
    return ga.ParticleMeasurement(d, np.zeros((d.size, 2)), 25.0, ga.volume_median(d), 125.0, 0, 0, clumps)


def test_volume_percentiles_are_ordered_and_match_median():
    d = np.random.default_rng(0).lognormal(np.log(800), 0.4, 500)
    d10, d50, d90 = (ga.volume_percentile(d, q) for q in (10, 50, 90))
    assert d10 < d50 < d90
    assert d50 == pytest.approx(ga.volume_median(d))


def test_volume_fractions_weight_by_cube_and_sum_to_one():
    # 100 µm 입자 하나와 1000 µm 입자 하나: 부피는 1 : 1000
    fractions = ga.volume_fractions_by_bin([100, 1000])
    assert fractions.sum() == pytest.approx(1.0)
    assert fractions[0] == pytest.approx(1 / 1001)
    assert fractions[np.searchsorted(ga.SIZE_BIN_EDGES_UM, 1000, side="right") - 1] == pytest.approx(1000 / 1001)


def test_sizes_outside_bins_go_to_end_bins():
    fractions = ga.volume_fractions_by_bin([50, 5000])
    assert fractions[0] > 0 and fractions[-1] > 0
    assert fractions.sum() == pytest.approx(1.0)


def test_combine_pools_all_photos():
    a = make_measurement([600, 700, 800] * 60)
    b = make_measurement([900, 1000, 1100] * 60)
    profile = ga.combine_measurements([a, b])
    assert profile.n_photos == 2
    assert profile.n_particles == 360
    assert profile.d50_um == pytest.approx(ga.volume_median(np.concatenate([a.diameters_um, b.diameters_um])))
    assert profile.photo_d50_um == (a.d50_volume_um, b.d50_volume_um)
    assert profile.warnings == ()


def test_combine_warns_when_few_particles_or_many_clumps():
    profile = ga.combine_measurements([make_measurement([800] * 50, clumps=40)])
    assert any("흔들릴 수 있어요" in w for w in profile.warnings)
    assert any("더 얇게" in w for w in profile.warnings)


def test_combine_without_particles_gives_clear_error():
    with pytest.raises(ga.GrindPhotoError, match="입자를 잰 사진이 없어요"):
        ga.combine_measurements([make_measurement([])])


def test_analyze_photo_runs_rectify_and_measure():
    window = (12.0, 22.0, 108.0, 118.0)
    dots = [(45.0, 55.0, 2.0), (60.0, 70.0, 2.0)]
    photo = simulate_photo(render_sheet_window(15.0, window, dots), tilt=0.05, angle_deg=6.0)
    rectified, m = ga.analyze_photo(photo)
    assert rectified.zone.name == "A"
    assert len(m.diameters_um) == 2
    assert np.all(np.abs(m.diameters_um - 2000) / 2000 < 0.03)


def test_click_table_is_monotonic():
    table = load_click_table()
    clicks = sorted(table)
    sizes = [table[c] for c in clicks]
    assert all(a < b for a, b in zip(sizes, sizes[1:]))
    assert table[95] == 1000
