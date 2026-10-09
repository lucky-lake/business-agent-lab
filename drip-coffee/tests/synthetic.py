"""테스트용 합성 이미지 만들기.

축척 용지의 일부를 원하는 해상도로 그리고, 크기를 정확히 아는 점(가짜 입자)을 올린 뒤,
폰으로 비스듬히 찍은 것처럼 비틀고 흐리게 한다. 정답을 알고 있으므로
분석 결과가 맞는지 숫자로 확인할 수 있다.
"""
import cv2
import numpy as np

from dripcoffee import sheet


def render_sheet_window(px_per_mm, window_mm, dots=()):
    """용지에서 window_mm=(x0, y0, x1, y1) 영역을 흰 바탕 회색조 이미지로 그린다.

    dots: (중심 x mm, 중심 y mm, 지름 mm) 목록. 용지 좌표 기준.
    """
    x0, y0, x1, y1 = window_mm
    width = int(round((x1 - x0) * px_per_mm))
    height = int(round((y1 - y0) * px_per_mm))
    img = np.full((height, width), 255, dtype=np.uint8)

    for zone in sheet.ZONES:
        size_px = int(round(zone.marker_size * px_per_mm))
        for marker_id, (mx, my) in zip(zone.marker_ids, zone.marker_origins()):
            px = int(round((mx - x0) * px_per_mm))
            py = int(round((my - y0) * px_per_mm))
            if px < 0 or py < 0 or px + size_px > width or py + size_px > height:
                continue  # 창 밖에 있는 마커는 그리지 않는다
            marker = cv2.aruco.generateImageMarker(sheet.aruco_dictionary(), marker_id, size_px, borderBits=1)
            img[py : py + size_px, px : px + size_px] = marker

    for cx, cy, d in dots:
        draw_exact_disc(img, (cx - x0) * px_per_mm, (cy - y0) * px_per_mm, d / 2 * px_per_mm)
    return img


def draw_exact_disc(img, cx, cy, r, supersample=8):
    """면적이 정확한 검은 원을 그린다. 좌표는 픽셀 단위이고 픽셀 i는 [i, i+1) 구간이다.

    cv2.circle은 원을 반 픽셀 정도 크게 그려서 정답 크기가 틀어지므로,
    픽셀을 잘게 나눠 원 안에 들어가는 비율만큼 어둡게 칠한다."""
    x_lo, x_hi = int(np.floor(cx - r)), int(np.ceil(cx + r))
    y_lo, y_hi = int(np.floor(cy - r)), int(np.ceil(cy + r))
    offsets = (np.arange(supersample) + 0.5) / supersample
    xs = (np.arange(x_lo, x_hi)[:, None] + offsets[None, :]).ravel()
    ys = (np.arange(y_lo, y_hi)[:, None] + offsets[None, :]).ravel()
    inside = (xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2 <= r * r
    coverage = inside.reshape(y_hi - y_lo, supersample, x_hi - x_lo, supersample).mean(axis=(1, 3))
    patch = img[y_lo:y_hi, x_lo:x_hi]
    np.minimum(patch, (255 * (1 - coverage)).astype(np.uint8), out=patch)


def simulate_photo(img, tilt=0.08, angle_deg=12.0, blur_sigma=1.0, noise_sigma=4.0, seed=0):
    """평평한 이미지를 폰으로 비스듬히 찍은 것처럼 만든다.
    위쪽을 좁혀 기울어진 효과를 주고, 회전시키고, 흐림과 잡티를 더한다."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dx = tilt * w
    dst = np.float32([[dx, 0], [w - dx, 0], [w, h], [0, h]])

    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    dst = (dst - dst.mean(axis=0)) @ rot.T
    dst -= dst.min(axis=0) - 20  # 가장자리에 20px 여백
    out_w, out_h = (dst.max(axis=0) + 20).astype(int)

    matrix = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
    photo = cv2.warpPerspective(img, matrix, (int(out_w), int(out_h)), flags=cv2.INTER_LINEAR, borderValue=255)
    if blur_sigma > 0:
        photo = cv2.GaussianBlur(photo, (0, 0), blur_sigma)
    if noise_sigma > 0:
        rng = np.random.default_rng(seed)
        photo = np.clip(photo + rng.normal(0, noise_sigma, photo.shape), 0, 255).astype(np.uint8)
    return photo


def cover_marker(img, px_per_mm, window_mm, marker_id):
    """마커 하나를 흰색으로 덮는다 (마커가 가려진 사진 흉내)."""
    zone = sheet.zone_for_marker(marker_id)
    mx, my = zone.marker_origins()[zone.marker_ids.index(marker_id)]
    x0, y0 = window_mm[0], window_mm[1]
    pad = 1.0  # mm
    p0 = (int((mx - pad - x0) * px_per_mm), int((my - pad - y0) * px_per_mm))
    p1 = (int((mx + zone.marker_size + pad - x0) * px_per_mm), int((my + zone.marker_size + pad - y0) * px_per_mm))
    out = img.copy()
    cv2.rectangle(out, p0, p1, 255, -1)
    return out



def speckle_marker(img, px_per_mm, window_mm, marker_id, fraction=0.5, blob_px=3, gray=190, seed=0):
    """마커 하나의 검은 부분에 흰 반점을 뿌린다 (토너가 얼룩지게 인쇄된 마커 흉내).
    실제 카메라 사진에서 이런 마커가 원래 해상도에서는 검출되지 않았다.
    반점은 blob_px 크기의 덩어리로 뿌려야 실제처럼 원래 해상도에서 검출이 실패한다."""
    zone = sheet.zone_for_marker(marker_id)
    mx, my = zone.marker_origins()[zone.marker_ids.index(marker_id)]
    x0 = int(round((mx - window_mm[0]) * px_per_mm))
    y0 = int(round((my - window_mm[1]) * px_per_mm))
    size = int(round(zone.marker_size * px_per_mm))
    out = img.copy()
    region = out[y0 : y0 + size, x0 : x0 + size]
    rng = np.random.default_rng(seed)
    n = int(np.ceil(size / blob_px))
    blobs = (rng.random((n, n)) < fraction).astype(np.uint8)
    speck = cv2.resize(blobs, (n * blob_px, n * blob_px), interpolation=cv2.INTER_NEAREST)[:size, :size].astype(bool)
    region[speck & (region < 128)] = gray
    return out
