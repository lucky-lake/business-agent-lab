"""원두 가루 사진에서 분쇄도(입자 크기 분포)를 재는 계산 모듈.

처리 순서
1. 사진을 읽는다. 한글 경로나 업로드한 파일에서도 되도록 바이트로 읽는다.
2. 축척 용지의 모서리 마커를 찾고, 네 마커가 모두 보이는 가루 칸을 고른다.
3. 마커 꼭짓점의 실제 위치(mm)를 이용해 사진을 정면에서 본 모습으로 펴고,
   픽셀 하나가 실제로 몇 µm인지 구한다.
4. 펴진 가루 칸에서 입자를 찾아 입자마다 지름(µm)을 재고, 부피 기준 중앙값을 낸다.
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


# ---------------------------------------------------------------------------
# 입자 검출과 지름 계산
# ---------------------------------------------------------------------------

# 이보다 작은(지름, 픽셀) 점은 크기를 믿을 수 없어 측정에서 뺀다.
# 하한 아래 점을 미분으로 따로 셀지는 아직 정하지 않았다.
MIN_DIAMETER_PX = 5.0

# 가루 칸 테두리 선과 펴기 오차를 피하려고, 칸 가장자리에서 이만큼(mm)은 보지 않는다.
# 이 띠에 걸친 입자는 칸 경계에 걸려 잘린 입자로 보고 뺀다.
BORDER_BAND_MM = 0.5

# 붙은 입자 덩어리를 가려내는 두 기준.
# - 넓이 ÷ 볼록 껍질 넓이가 이보다 작으면 여러 입자가 엉킨 덩어리로 본다.
#   실제 사진에서 이 값이 0.85보다 작은 덩어리를 빼자 클릭별 부피 중앙값 비율이 메버릭 클릭표 비율과 비슷해졌다.
# - 두 입자가 딱 붙은 짝은 넓이 비율이 0.88쯤이라 위 기준에 걸리지 않는다. 대신 붙은 자리가 잘록하게
#   들어가므로, 가장 깊게 들어간 곳의 깊이가 같은 넓이 원 반지름의 이 비율보다 크면 덩어리로 본다.
#   합성 입자에서 단일 입자는 0.18 이하였고, 0.35로 두면 붙은 짝의 약 90%를 잡는다.
#   실제 사진의 단일 입자는 가장자리가 더 거칠어서 0.25로는 너무 많이 뺐다.
#   메버릭 클릭표는 출처를 확인하지 못해 이 값을 그 표에 맞추지는 않았다.
MIN_SOLIDITY = 0.85
MAX_NECK_DEPTH_RATIO = 0.35

# 배경(종이) 밝기를 추정할 때 지우는 가장 큰 입자 크기(mm). 이보다 큰 입자는 배경으로 섞일 수 있다.
MAX_PARTICLE_MM = 3.0


@dataclass
class ParticleMeasurement:
    """펴진 가루 칸 하나에서 잰 입자들."""

    diameters_um: np.ndarray  # 입자마다 면적이 같은 원의 지름 (µm)
    centroids_mm: np.ndarray  # 입자 중심 (가루 칸 왼쪽 위 기준, mm)
    um_per_px: float
    d50_volume_um: float  # 부피(질량) 기준 중앙값 지름
    min_diameter_um: float  # 이보다 작은 점은 재지 않았다 (검출 하한)
    excluded_chaff: int  # 은피로 보고 뺀 조각 수
    excluded_border: int  # 칸 경계에 걸려 뺀 입자 수
    excluded_clumps: int = 0  # 여러 입자가 붙은 덩어리로 보여 뺀 수
    warnings: tuple = ()


def volume_median(diameters):
    """부피 기준 중앙값. 큰 입자일수록 무게가 크므로 지름의 세제곱으로 가중한다."""
    d = np.sort(np.asarray(diameters, dtype=float))
    if d.size == 0:
        return float("nan")
    cumulative = np.cumsum(d**3) / np.sum(d**3)
    return float(np.interp(0.5, cumulative, d))


def flatten_lighting(image, px_per_mm):
    """종이 배경의 밝기를 1로 맞춘 밝기 지도를 만든다 (조명이 고르지 않아도 같게 보이도록).

    입자보다 큰 범위로 '닫기' 연산을 하면 어두운 입자가 지워지고 종이 밝기만 남는다.
    원래 밝기를 이 배경 밝기로 나누면 종이는 1 근처, 커피 입자는 0.1~0.3 근처가 된다.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)
    # 큰 이미지는 줄여서 배경을 추정하고 다시 키운다 (배경은 천천히 변하므로 충분하다).
    scale = min(1.0, 400.0 / max(gray.shape))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    k = max(3, int(MAX_PARTICLE_MM * px_per_mm * scale) | 1)
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), k / 2)
    background = cv2.resize(background, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return gray / np.maximum(background, 1.0)


def measure_particles(image, um_per_px):
    """펴진 가루 칸 이미지에서 커피 입자를 찾아 지름을 잰다.

    1. 조명을 고르게 맞춘 밝기 지도를 만든다.
    2. 종이와 커피 입자 밝기의 가운데를 기준으로 입자를 나눈다.
       경계 픽셀은 반쯤 덮였으므로, 가운데 기준이면 면적이 치우치지 않는다.
    3. 커피보다 밝고 노란 조각은 은피로 보고 센 뒤 뺀다.
    4. 칸 가장자리 띠에 걸친 입자는 잘린 입자로 보고 뺀다.
    5. 오목하게 들어간 곳이 많은 덩어리(붙은 입자)는 큰 입자 하나로 잘못 재지 않도록 뺀다.
    6. 입자마다 덮인 비율을 더해 면적을 구하고, 같은 면적의 원 지름으로 바꾼다.
    """
    px_per_mm = 1000.0 / um_per_px
    level = flatten_lighting(image, px_per_mm)
    h, w = level.shape
    band = max(1, int(round(BORDER_BAND_MM * px_per_mm)))
    inner = np.zeros((h, w), dtype=bool)
    inner[band : h - band, band : w - band] = True

    empty = ParticleMeasurement(
        np.array([]), np.zeros((0, 2)), um_per_px, float("nan"), MIN_DIAMETER_PX * um_per_px, 0, 0, 0,
        ("가루 칸에서 입자를 찾지 못했어요.",),
    )
    dark = level[inner & (level < 0.6)]
    if dark.size == 0:
        return empty
    coffee_level = float(np.percentile(dark, 10))  # 커피 입자 안쪽의 밝기
    threshold = float(np.clip((1.0 + coffee_level) / 2, 0.45, 0.75))
    coffee = (level < threshold) & inner

    excluded_chaff = _count_chaff(image, level, coffee, inner, threshold)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(coffee.astype(np.uint8), connectivity=8)
    coverage = np.clip((1.0 - level) / (1.0 - coffee_level), 0.0, 1.0)
    diameters, centroids, excluded_border, excluded_clumps = [], [], 0, 0
    for i in range(1, n):
        x, y, bw, bh, pixel_count = stats[i]
        if x <= band or y <= band or x + bw >= w - band or y + bh >= h - band:
            excluded_border += 1
            continue
        if 2.0 * np.sqrt(pixel_count / np.pi) >= MIN_DIAMETER_PX and _is_clump(labels[y : y + bh, x : x + bw] == i):
            excluded_clumps += 1
            continue
        # 경계의 반쯤 덮인 픽셀까지 넣으려고 한 픽셀 넓힌 영역에서 덮인 비율을 더한다.
        y0, y1, x0, x1 = max(y - 1, 0), min(y + bh + 1, h), max(x - 1, 0), min(x + bw + 1, w)
        region = cv2.dilate((labels[y0:y1, x0:x1] == i).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        area_px = float(coverage[y0:y1, x0:x1][region].sum())
        d_px = 2.0 * np.sqrt(area_px / np.pi)
        if d_px < MIN_DIAMETER_PX:
            continue
        ys, xs = np.nonzero(region)
        weights = coverage[y0:y1, x0:x1][region]
        cx = (np.sum(xs * weights) / weights.sum() + x0) / px_per_mm
        cy = (np.sum(ys * weights) / weights.sum() + y0) / px_per_mm
        diameters.append(d_px * um_per_px)
        centroids.append((cx, cy))

    if not diameters:
        return empty
    d = np.array(diameters)
    return ParticleMeasurement(
        d, np.array(centroids), um_per_px, volume_median(d), MIN_DIAMETER_PX * um_per_px,
        excluded_chaff, excluded_border, excluded_clumps,
    )


def _is_clump(mask):
    """여러 입자가 붙은 덩어리인지 판단한다 (위 MIN_SOLIDITY, MAX_NECK_DEPTH_RATIO 설명 참고)."""
    padded = cv2.copyMakeBorder(mask.astype(np.uint8), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    contours, _ = cv2.findContours(padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    area = float(np.count_nonzero(mask))
    if area / max(cv2.contourArea(cv2.convexHull(contour)), 1.0) < MIN_SOLIDITY:
        return True
    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if len(hull_idx) <= 3:
        return False
    defects = cv2.convexityDefects(contour, hull_idx)
    if defects is None:
        return False
    deepest = defects[:, 0, 3].max() / 256.0  # OpenCV는 깊이를 256배 한 정수로 준다
    return deepest / np.sqrt(area / np.pi) > MAX_NECK_DEPTH_RATIO


def _count_chaff(image, level, coffee, inner, threshold):
    """은피 조각 수를 센다. 은피는 커피 입자보다 밝고(종이보다는 어둡고) 노란빛이 강하다.

    색은 종이 색과 비교한다. 따뜻한 조명에서는 종이 자체가 노랗게 찍혀서,
    색의 진하기(채도)만으로 판단하면 종이 결과 입자 그림자를 은피로 잘못 센다(실제 사진에서 확인).
    """
    if image.ndim == 2:
        return 0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    paper_pixels = inner & (level > 0.95)
    if not paper_pixels.any():
        return 0
    paper = np.median(lab[paper_pixels], axis=0)
    color_diff = np.hypot(lab[..., 1] - paper[1], lab[..., 2] - paper[2])
    near_coffee = cv2.dilate(coffee.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    chaff = inner & (level >= threshold) & (level < 0.9) & (color_diff > 14) & ~near_coffee
    chaff = cv2.morphologyEx(chaff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(chaff, connectivity=8)
    min_area = np.pi * (MIN_DIAMETER_PX / 2) ** 2
    return int(np.sum(stats[1:, cv2.CC_STAT_AREA] >= min_area))
