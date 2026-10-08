"""분쇄도 측정용 축척 용지의 배치 정의.

용지를 만드는 스크립트와 사진을 분석하는 코드가 같은 치수를 쓰도록,
마커와 칸의 위치는 모두 여기서만 정한다.
단위는 mm이고, 원점은 용지 왼쪽 위, y는 아래쪽으로 커진다.

가루 칸마다 네 모서리에 ArUco 마커를 둔다. 마커 번호로 어느 칸인지 알 수 있고,
네 마커의 꼭짓점 위치를 알고 있으므로 비스듬히 찍은 사진도 정면으로 펴서
픽셀당 실제 길이를 계산할 수 있다.
용지 버전도 마커 번호로 구분한다. v1은 0~7번을 쓰고, 배치를 바꾸면 다른 번호를 쓴다.
"""
from dataclasses import dataclass

import cv2

SHEET_VERSION = 1

PAGE_W_MM = 210.0
PAGE_H_MM = 297.0

ARUCO_DICT_ID = cv2.aruco.DICT_4X4_50
ARUCO_GRID_CELLS = 6  # 4x4 비트 + 테두리 1칸씩


@dataclass(frozen=True)
class Zone:
    """가루를 펼치는 칸 하나와 그 둘레의 마커 네 개."""

    name: str
    label: str
    square_x: float  # 가루 칸 왼쪽 위 x (mm)
    square_y: float  # 가루 칸 왼쪽 위 y (mm)
    square_size: float  # 가루 칸 한 변 (mm)
    marker_size: float  # 마커 한 변 (mm)
    gap: float  # 가루 칸과 마커 사이 간격 (mm)
    marker_ids: tuple  # (왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래)

    def marker_origins(self):
        """마커 네 개의 왼쪽 위 꼭짓점 위치(mm)를 marker_ids 순서대로 돌려준다."""
        near = -self.gap - self.marker_size
        far = self.square_size + self.gap
        x0, y0 = self.square_x, self.square_y
        return [
            (x0 + near, y0 + near),
            (x0 + far, y0 + near),
            (x0 + far, y0 + far),
            (x0 + near, y0 + far),
        ]

    def marker_corners(self, marker_id):
        """마커 꼭짓점 네 개(mm)를 OpenCV ArUco 검출 결과와 같은 순서
        (왼쪽 위부터 시계 방향)로 돌려준다."""
        x, y = self.marker_origins()[self.marker_ids.index(marker_id)]
        m = self.marker_size
        return [(x, y), (x + m, y), (x + m, y + m), (x, y + m)]


# A는 보통 거리에서 찍는 칸, B는 가까이(2배 줌·매크로) 찍어 해상도를 높이는 칸.
ZONES = (
    Zone("A", "A 표준: 가루 칸 50 mm", 35.0, 45.0, 50.0, 10.0, 3.0, (0, 1, 2, 3)),
    Zone("B", "B 고배율: 가루 칸 25 mm", 140.0, 45.0, 25.0, 6.0, 2.0, (4, 5, 6, 7)),
)

# 인쇄 배율 확인용 기준 막대. 인쇄 후 자로 100 mm인지 잰다.
REF_BAR_X_MM, REF_BAR_Y_MM, REF_BAR_LEN_MM = 25.0, 130.0, 100.0


def aruco_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)


def zone_for_marker(marker_id):
    """마커 번호가 속한 칸을 찾는다. 없으면 None."""
    for zone in ZONES:
        if marker_id in zone.marker_ids:
            return zone
    return None
