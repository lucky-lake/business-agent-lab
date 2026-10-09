"""분쇄 사진 분석: 마커로 사진을 펴고 축척을 구하는 단계 테스트.

합성 이미지에 크기와 위치를 아는 점을 그려 두고, 비스듬히 찍은 것처럼 비튼 뒤
다시 폈을 때 점의 크기와 위치가 원래대로 나오는지 확인한다.
"""
import cv2
import numpy as np
import pytest

from dripcoffee import grind_analysis as ga
from tests.synthetic import cover_marker, render_sheet_window, simulate_photo, speckle_marker

WINDOW_A = (12.0, 22.0, 108.0, 118.0)  # A칸과 마커 네 개가 들어가는 영역 (mm)
WINDOW_B = (128.0, 33.0, 177.0, 82.0)  # B칸 영역
DOTS_A = [(45.0, 55.0, 2.0), (60.0, 70.0, 2.0), (75.0, 85.0, 1.5)]  # (x, y, 지름) mm
DOTS_B = [(147.0, 52.0, 1.0), (157.0, 62.0, 0.6)]


def measure_dots(rectified):
    """펴진 이미지에서 검은 점을 찾아 (중심 x mm, 중심 y mm, 지름 mm) 목록으로 돌려준다.
    좌표는 가루 칸 왼쪽 위 기준이다."""
    _, mask = cv2.threshold(rectified.image, 128, 255, cv2.THRESH_BINARY_INV)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    mm_per_px = rectified.um_per_px / 1000.0
    found = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20:
            continue
        diameter = 2 * np.sqrt(area / np.pi) * mm_per_px
        found.append((centroids[i][0] * mm_per_px, centroids[i][1] * mm_per_px, diameter))
    return sorted(found)


def check_dots(rectified, dots, zone_origin, size_tol=0.03, pos_tol_mm=0.2):
    measured = measure_dots(rectified)
    assert len(measured) == len(dots)
    expected = sorted((x - zone_origin[0], y - zone_origin[1], d) for x, y, d in dots)
    for (mx, my, md), (ex, ey, ed) in zip(measured, expected):
        assert abs(md - ed) / ed < size_tol, f"지름 {md:.3f} mm (정답 {ed} mm)"
        assert np.hypot(mx - ex, my - ey) < pos_tol_mm, f"위치 ({mx:.2f}, {my:.2f}) (정답 ({ex}, {ey}))"


@pytest.mark.parametrize("tilt, angle", [(0.0, 0.0), (0.08, 12.0), (0.12, -25.0)])
def test_zone_a_tilted_photo_restores_size_and_position(tilt, angle):
    flat = render_sheet_window(15.0, WINDOW_A, DOTS_A)
    photo = simulate_photo(flat, tilt=tilt, angle_deg=angle)
    rectified = ga.rectify_zone(photo)

    assert rectified.zone.name == "A"
    assert rectified.fit_error_mm < 0.2
    check_dots(rectified, DOTS_A, (rectified.zone.square_x, rectified.zone.square_y))


def test_zone_b_close_up():
    flat = render_sheet_window(40.0, WINDOW_B, DOTS_B)
    photo = simulate_photo(flat, tilt=0.06, angle_deg=-8.0, blur_sigma=1.5)
    rectified = ga.rectify_zone(photo)

    assert rectified.zone.name == "B"
    assert 20.0 < rectified.um_per_px < 30.0  # 40 px/mm 근처 = 25 µm/px 근처
    check_dots(rectified, DOTS_B, (rectified.zone.square_x, rectified.zone.square_y))


def test_both_zones_visible_can_pick_zone_by_name():
    flat = render_sheet_window(12.0, (10.0, 20.0, 190.0, 120.0))
    photo = simulate_photo(flat, tilt=0.05, angle_deg=5.0)
    assert ga.rectify_zone(photo, zone_name="A").zone.name == "A"
    assert ga.rectify_zone(photo, zone_name="B").zone.name == "B"


def test_marker_touching_photo_edge_is_still_found():
    # 실제 사진에서 오른쪽 아래 마커가 사진 가장자리에 붙어 검출되지 않은 경우를 재현한다.
    flat = render_sheet_window(15.0, WINDOW_A, DOTS_A)
    right_edge_px = int(round((98.0 - WINDOW_A[0]) * 15.0))  # A칸 오른쪽 마커의 바깥 변
    cropped = flat[:, : right_edge_px + 1]
    rectified = ga.rectify_zone(cropped)
    assert rectified.zone.name == "A"
    check_dots(rectified, DOTS_A, (rectified.zone.square_x, rectified.zone.square_y))


def test_speckled_marker_is_found_by_retrying_on_smaller_image():
    # 실제 카메라 사진에서 토너가 얼룩진 마커가 원래 해상도로는 검출되지 않은 경우를 재현한다.
    flat = render_sheet_window(40.0, WINDOW_A, DOTS_A)
    speckled = speckle_marker(flat, 40.0, WINDOW_A, marker_id=0)
    photo = simulate_photo(speckled, tilt=0.03, angle_deg=4.0, blur_sigma=0.6)

    # 원래 해상도로만 찾으면 실패하는지 먼저 확인한다 (실제 사진과 같은 상황인지).
    padded = cv2.copyMakeBorder(photo, 60, 60, 60, 60, cv2.BORDER_REPLICATE)
    _, ids, _ = cv2.aruco.ArucoDetector(ga.sheet.aruco_dictionary(), cv2.aruco.DetectorParameters()).detectMarkers(padded)
    assert ids is None or 0 not in ids.flatten()

    markers = ga.detect_markers(photo)
    assert 0 in markers
    rectified = ga.rectify_zone(photo, markers)
    assert rectified.missing_marker_ids == ()
    check_dots(rectified, DOTS_A, (rectified.zone.square_x, rectified.zone.square_y))


def test_one_hidden_marker_uses_remaining_three_with_warning():
    # 실제 사진에서 은피가 마커 테두리에 걸려 마커 하나가 검출되지 않은 경우를 재현한다.
    flat = render_sheet_window(15.0, WINDOW_A, DOTS_A)
    photo = simulate_photo(cover_marker(flat, 15.0, WINDOW_A, marker_id=2))
    rectified = ga.rectify_zone(photo)
    assert rectified.missing_marker_ids == (2,)
    assert "2번이 보이지 않아" in rectified.warnings[0]
    check_dots(rectified, DOTS_A, (rectified.zone.square_x, rectified.zone.square_y))


def test_marker_with_shifted_corner_is_left_out():
    # 실제 사진에서 마커 모서리에 가루가 붙어 꼭짓점 하나가 0.4~1.0 mm 밀려 잡힌 경우를 재현한다.
    flat = render_sheet_window(15.0, WINDOW_A, DOTS_A)
    photo = simulate_photo(flat, tilt=0.05, angle_deg=8.0)
    markers = ga.detect_markers(photo)
    px_per_mm = ga.photo_px_per_mm(markers, ga.choose_zone(markers))
    markers[1] = markers[1].copy()
    markers[1][3] += (-0.8 * px_per_mm, 0.8 * px_per_mm)  # 1번 마커의 왼쪽 아래 꼭짓점을 밀어 둔다

    rectified = ga.rectify_zone(photo, markers=markers)
    assert any("1번 위치가 다른 마커들과 맞지 않아" in w for w in rectified.warnings)
    assert rectified.fit_error_mm < 0.2
    check_dots(rectified, DOTS_A, (rectified.zone.square_x, rectified.zone.square_y))


def test_two_hidden_markers_give_clear_error():
    flat = render_sheet_window(15.0, WINDOW_A, DOTS_A)
    covered = cover_marker(cover_marker(flat, 15.0, WINDOW_A, marker_id=1), 15.0, WINDOW_A, marker_id=2)
    with pytest.raises(ga.GrindPhotoError, match="안 보이는 마커 번호: 1, 2"):
        ga.rectify_zone(simulate_photo(covered))


def test_photo_without_markers_gives_clear_error():
    blank = np.full((800, 600), 255, dtype=np.uint8)
    with pytest.raises(ga.GrindPhotoError, match="마커를 찾지 못했어요"):
        ga.rectify_zone(blank)


def test_load_image_from_bytes_and_korean_path(tmp_path):
    img = render_sheet_window(5.0, WINDOW_A)
    ok, encoded = cv2.imencode(".png", img)
    assert ok
    from_bytes = ga.load_image_bytes(encoded.tobytes())
    assert from_bytes.shape[:2] == img.shape

    path = tmp_path / "분쇄_사진.png"
    encoded.tofile(str(path))
    assert ga.load_image_path(path).shape[:2] == img.shape


def test_unreadable_file_gives_clear_error():
    with pytest.raises(ga.GrindPhotoError, match="읽을 수 없어요"):
        ga.load_image_bytes(b"not an image")
