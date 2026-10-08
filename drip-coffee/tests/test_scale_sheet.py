"""축척 용지를 렌더링한 이미지에서 마커가 설계한 위치·크기대로 읽히는지 확인한다."""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pytest

from dripcoffee import sheet
from tools.make_scale_sheet import build_figure

DPI = 400
PX_PER_MM = DPI / 25.4


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    path = tmp_path_factory.mktemp("sheet") / "sheet.png"
    fig = build_figure()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    # 경로에 한글이 있어도 읽히도록 바이트로 읽는다.
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)


def test_all_markers_detected_at_designed_positions(rendered):
    detector = cv2.aruco.ArucoDetector(sheet.aruco_dictionary(), cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(rendered)
    assert ids is not None
    found = {int(i): c.reshape(4, 2) for i, c in zip(ids.flatten(), corners)}

    expected_ids = {mid for zone in sheet.ZONES for mid in zone.marker_ids}
    assert set(found) == expected_ids

    for zone in sheet.ZONES:
        for mid in zone.marker_ids:
            expected_px = np.array(zone.marker_corners(mid)) * PX_PER_MM
            error_mm = np.abs(found[mid] - expected_px).max() / PX_PER_MM
            assert error_mm < 0.15, f"마커 {mid} 위치 오차 {error_mm:.3f} mm"


def test_zone_for_marker():
    assert sheet.zone_for_marker(0).name == "A"
    assert sheet.zone_for_marker(6).name == "B"
    assert sheet.zone_for_marker(40) is None
