"""원두 가루 사진에서 분쇄도(입자 크기 분포)를 재는 계산 모듈.

처리 순서
1. 사진을 읽는다. 한글 경로나 업로드한 파일에서도 되도록 바이트로 읽는다.
2. 축척 용지의 모서리 마커를 찾고, 네 마커가 모두 보이는 가루 칸을 고른다.
3. 마커 꼭짓점의 실제 위치(mm)를 이용해 사진을 정면에서 본 모습으로 펴고,
   픽셀 하나가 실제로 몇 µm인지 구한다.
(입자 검출과 크기 분포 계산은 다음 단계에서 추가한다.)
"""
from dataclasses import dataclass

import cv2
import numpy as np

from dripcoffee import sheet

# 마커 꼭짓점들을 한 번에 맞췄을 때 허용하는 최대 어긋남(mm).
# 이보다 크면 용지가 휘었거나, 인쇄 배율이 다르거나, 다른 용지일 가능성이 크다.
# 실제 사진으로 확인하면서 조정할 수 있다.
MAX_FIT_ERROR_MM = 0.6

# 마커 하나가 은피나 가루에 가려지는 일이 실제 사진에서 자주 생긴다.
# 실제 사진 5장으로 비교했을 때 마커 3개로 편 결과는 4개로 편 결과와
# 가루 칸 안에서 최대 0.4 mm(A칸), 0.13 mm(B칸) 차이였으므로 3개까지 허용한다.
MIN_MARKERS = 3

# 마커 모서리에 가루가 붙으면 그 꼭짓점이 가루 쪽으로 밀려 잡힌다(실제 사진에서 확인).
# 마커 하나만 다른 마커들과 이만큼(mm) 이상 어긋나면 그 마커를 빼고 다시 계산한다.
MARKER_OUTLIER_MM = 0.25


class GrindPhotoError(ValueError):
    """사진을 분석할 수 없을 때, 사용자에게 보여줄 이유를 담는 예외."""


@dataclass
class RectifiedPhoto:
    """정면으로 편 가루 칸 이미지와 축척 정보."""

    image: np.ndarray  # 가루 칸만 잘라낸 정면 이미지 (원본과 같은 채널 수)
    zone: sheet.Zone  # 어느 칸(A/B)인지
    um_per_px: float  # 펴진 이미지에서 픽셀 하나의 실제 길이 (µm)
    homography: np.ndarray  # 원본 사진 좌표 → 펴진 이미지 좌표 변환 행렬
    fit_error_mm: float  # 마커 꼭짓점이 설계 위치에서 벗어난 최대 거리 (mm)
    missing_marker_ids: tuple = ()  # 보이지 않아 계산에서 뺀 마커 번호
    warnings: tuple = ()  # 결과는 냈지만 사용자에게 알려야 할 점


def load_image_bytes(data):
    """파일 내용(바이트)에서 이미지를 읽는다. 앱에서 업로드한 사진도 이 함수로 읽는다."""
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise GrindPhotoError("이미지 파일을 읽을 수 없어요. JPG나 PNG 사진인지 확인해주세요.")
    return image


def load_image_path(path):
    """파일 경로에서 이미지를 읽는다. cv2.imread는 한글 경로에서 실패하므로 바이트로 읽는다."""
    return load_image_bytes(np.fromfile(str(path), dtype=np.uint8).tobytes())


def detect_markers(image):
    """사진에서 용지 마커를 찾아 {마커 번호: 꼭짓점 4개(px)} 로 돌려준다.
    꼭짓점 순서는 마커의 왼쪽 위부터 시계 방향이다."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 사진 가장자리에 딱 붙은 마커는 OpenCV가 후보에서 빼 버린다 (실제 사진에서 확인).
    # 가장자리 픽셀을 늘려 여백을 붙인 뒤 찾고, 좌표는 원래 사진 기준으로 되돌린다.
    pad = max(10, int(0.02 * max(gray.shape)))
    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX  # 꼭짓점을 픽셀보다 정밀하게
    detector = cv2.aruco.ArucoDetector(sheet.aruco_dictionary(), params)

    found = _run_detector(detector, padded, pad, 1.0)
    expected = {m for zone in sheet.ZONES for m in zone.marker_ids}
    for scale in (0.5, 0.25):
        if expected <= found.keys():
            break
        # 토너가 얼룩지게 인쇄된 마커는 고해상도 사진에서 흰 반점이 보여 무늬를 못 읽는다
        # (실제 카메라 사진에서 확인). 사진을 줄이면 이웃 픽셀이 평균나서 반점이 사라지므로,
        # 줄인 사진에서 한 번 더 찾고 좌표만 원래 크기로 되돌린다.
        small = cv2.resize(padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for marker_id, c in _run_detector(detector, small, pad, scale).items():
            found.setdefault(marker_id, c)
    return found


def _run_detector(detector, padded, pad, scale):
    corners, ids, _ = detector.detectMarkers(padded)
    if ids is None:
        return {}
    # 줄인 사진의 픽셀 중심 좌표를 원래 사진 좌표로 되돌린다: (c + 0.5) / scale - 0.5
    return {
        int(i): (c.reshape(4, 2).astype(np.float64) + 0.5) / scale - 0.5 - pad
        for i, c in zip(ids.flatten(), corners)
    }


def visible_ids(markers, zone):
    return [m for m in zone.marker_ids if m in markers]


def photo_px_per_mm(markers, zone):
    """사진 속 마커 변 길이로, 그 칸 근처의 대략적인 해상도(px/mm)를 구한다."""
    sides = []
    for marker_id in visible_ids(markers, zone):
        c = markers[marker_id]
        sides.extend(np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1))
    return float(np.mean(sides)) / zone.marker_size


def choose_zone(markers, zone_name=None):
    """마커가 3개 이상 보이는 칸을 고른다. 여러 칸이 가능하면 마커가 더 많이 보이고,
    더 크게 찍힌 칸을 고른다. zone_name("A" 또는 "B")을 주면 그 칸만 본다."""
    candidates = [z for z in sheet.ZONES if zone_name in (None, z.name)]
    usable = [z for z in candidates if len(visible_ids(markers, z)) >= MIN_MARKERS]
    if usable:
        return max(usable, key=lambda z: (len(visible_ids(markers, z)), photo_px_per_mm(markers, z)))

    if not any(visible_ids(markers, z) for z in candidates):
        raise GrindPhotoError(
            "용지의 마커를 찾지 못했어요. 가루 칸 네 모서리의 검은 네모 무늬가 보이게, "
            "초점을 맞춰 다시 찍어주세요."
        )
    best = max(candidates, key=lambda z: len(visible_ids(markers, z)))
    missing = [m for m in best.marker_ids if m not in markers]
    raise GrindPhotoError(
        f"{best.name}칸 마커 4개 중 {4 - len(missing)}개만 보여요 "
        f"(안 보이는 마커 번호: {', '.join(map(str, missing))}). "
        f"최소 {MIN_MARKERS}개가 보여야 해요. 마커를 가리는 가루나 은피를 치우고 다시 찍어주세요."
    )


def rectify_zone(image, markers=None, zone_name=None):
    """마커를 이용해 가루 칸을 정면에서 본 이미지로 펴고 축척(µm/px)을 구한다.

    보이는 마커들의 꼭짓점(마커 4개면 16개)이 실제로 어디 있어야 하는지(mm) 알고 있으므로,
    사진 좌표를 그 위치로 옮기는 원근 변환을 구해 사진 전체에 적용한다.
    펴진 이미지의 해상도는 원래 사진의 해상도와 비슷하게 맞춰서 정보를 잃지 않게 한다.
    """
    if markers is None:
        markers = detect_markers(image)
    zone = choose_zone(markers, zone_name)
    px_per_mm = photo_px_per_mm(markers, zone)
    ids = visible_ids(markers, zone)
    missing = tuple(m for m in zone.marker_ids if m not in markers)
    notes = []

    homography, residual_mm = _fit_homography(markers, zone, ids, px_per_mm)
    worst = max(residual_mm, key=residual_mm.get)
    if residual_mm[worst] > MARKER_OUTLIER_MM and len(ids) > MIN_MARKERS:
        # 마커 하나만 크게 어긋나면 그 마커를 빼고 나머지로 다시 맞춘다.
        ids = [m for m in ids if m != worst]
        homography, residual_mm = _fit_homography(markers, zone, ids, px_per_mm)
        notes.append(
            f"마커 {worst}번 위치가 다른 마커들과 맞지 않아(모서리에 가루가 붙었을 수 있어요) 빼고 계산했어요."
        )

    fit_error_mm = max(residual_mm.values())
    if fit_error_mm > MAX_FIT_ERROR_MM:
        raise GrindPhotoError(
            f"마커 위치가 서로 {fit_error_mm:.2f} mm 어긋나요. 용지가 휘었거나 인쇄 배율이 "
            "달라졌을 수 있어요. 용지를 평평하게 두고 다시 찍어주세요."
        )

    size = int(round(zone.square_size * px_per_mm))
    white = 255 if image.ndim == 2 else (255, 255, 255)
    warped = cv2.warpPerspective(image, homography, (size, size), flags=cv2.INTER_LINEAR, borderValue=white)

    if missing:
        notes.insert(0, f"마커 {', '.join(map(str, missing))}번이 보이지 않아 빼고 계산했어요.")
    if len(ids) < len(zone.marker_ids):
        notes.append(f"마커 {len(ids)}개로 계산해서 크기 기준이 조금 덜 정확할 수 있어요.")
    return RectifiedPhoto(warped, zone, 1000.0 / px_per_mm, homography, fit_error_mm, missing, tuple(notes))


def _fit_homography(markers, zone, ids, px_per_mm):
    """주어진 마커들로 원근 변환을 구하고, 마커마다 설계 위치에서 벗어난 정도(mm)를 함께 돌려준다."""
    src = np.vstack([markers[m] for m in ids])
    dst_mm = np.vstack([zone.marker_corners(m) for m in ids])
    dst = (dst_mm - (zone.square_x, zone.square_y)) * px_per_mm  # 가루 칸 왼쪽 위를 원점으로

    homography, _ = cv2.findHomography(src, dst, 0)
    if homography is None:
        raise GrindPhotoError("마커 위치로 사진을 펴지 못했어요. 다시 찍어주세요.")
    mapped = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    corner_error = np.linalg.norm(mapped - dst, axis=1) / px_per_mm
    residual_mm = {m: float(corner_error[4 * k : 4 * k + 4].max()) for k, m in enumerate(ids)}
    return homography, residual_mm
