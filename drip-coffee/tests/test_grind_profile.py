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


def test_combine_reports_spread_and_keeps_particle_sizes():
    a = make_measurement(np.random.default_rng(1).lognormal(np.log(800), 0.4, 400))
    profile = ga.combine_measurements([a])
    assert profile.span == pytest.approx((profile.d90_um - profile.d10_um) / profile.d50_um)
    assert np.array_equal(profile.diameters_um, a.diameters_um)
    low, high = profile.d50_range_um
    assert low < profile.d50_um < high
    # 대부분의 범위는 부피 기준 16~84% 지점이고, 대표 크기와 가장 많은 크기가 그 안에 있다.
    spread_low, spread_high = profile.spread_range_um
    assert spread_low == pytest.approx(ga.volume_percentile(a.diameters_um, 16))
    assert spread_high == pytest.approx(ga.volume_percentile(a.diameters_um, 84))
    assert spread_low < profile.d50_um < spread_high
    assert spread_low < profile.mode_um < spread_high


def test_most_common_size_is_the_curve_peak():
    d = np.random.default_rng(4).lognormal(np.log(700), 0.35, 3000)
    sizes, density = ga.size_distribution_curve(d)
    assert ga.most_common_size(d) == sizes[np.argmax(density)]
    assert np.isnan(ga.most_common_size([800, 900]))


def test_d50_range_narrows_with_more_particles():
    # 같은 가루라도 입자를 많이 잴수록 D50이 덜 흔들려야 한다.
    rng = np.random.default_rng(2)
    few = rng.lognormal(np.log(800), 0.4, 100)
    many = rng.lognormal(np.log(800), 0.4, 1600)
    low_few, high_few = ga.d50_range(few)
    low_many, high_many = ga.d50_range(many)
    assert high_many - low_many < 0.5 * (high_few - low_few)
    # 매번 같은 결과가 나와야 화면이 새로 그려질 때 숫자가 바뀌지 않는다.
    assert ga.d50_range(few) == (low_few, high_few)


def test_distribution_curve_peaks_where_the_mass_is():
    # 로그 정규 분포는 부피로 가중해도 로그 정규라서, 부피 곡선의 봉우리가 부피 D50 근처에 있어야 한다.
    d = np.random.default_rng(3).lognormal(np.log(700), 0.35, 3000)
    sizes, density = ga.size_distribution_curve(d, by_volume=True)
    assert sizes[np.argmax(density)] == pytest.approx(ga.volume_median(d), rel=0.1)
    sizes_n, density_n = ga.size_distribution_curve(d, by_volume=False)
    assert sizes_n[np.argmax(density_n)] == pytest.approx(np.median(d), rel=0.1)
    # 곡선 아래 넓이(로그 지름 기준)는 1이다.
    assert np.trapezoid(density, np.log10(sizes)) == pytest.approx(1.0, abs=0.02)


def test_distribution_curve_needs_a_few_particles():
    assert ga.size_distribution_curve([800, 900]) is None


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
