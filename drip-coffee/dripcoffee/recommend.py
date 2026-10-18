"""추천 목표: 맛 기록에서 '내 선호 영역'을 찾는다. 설계는 docs/MODEL.md 9.1.

커피는 본질적으로 기호식품이라 좋은 한 잔의 기준이 사람마다 다르다. 그래서 업계 평균인 골든컵에서
시작해, 맛 점수가 좋았던 잔의 위치(추출수율, 가수 후 농도) 쪽으로 목표를 옮겨 간다.
잔의 위치는 굴절계로 잰 값이 아니라 모델 추정값이다.
"""
from collections import defaultdict
from dataclasses import dataclass

from dripcoffee.extraction_model import DEFAULT_PARAMS, grind_from_d50, predict

GOLDEN_CUP_CENTER = (20.0, 1.25)  # 추출수율 %, 농도 % (골든컵 범위 18–22%, 1.15–1.35%의 가운데)
GOLDEN_CUP_WEIGHT = 5  # 골든컵을 '좋았던 잔 5잔 분량'으로 친다 (가정)
LIKED_ABOVE = 2.5  # 이 점수 이하인 잔은 중심을 당기지 않는다 (가정)


@dataclass(frozen=True)
class BrewPoint:
    """기록한 한 잔의 위치와 맛 점수."""

    brew_id: str
    bean_id: str
    ey_pct: float  # 추출수율 (모델 추정)
    tds_cup_pct: float  # 가수 후 농도 (모델 추정)
    score: float  # 만족도 점수 (1~5)

    @property
    def weight(self):
        """좋았던 잔일수록 크다. 보통(3점)이면 0.5, 2.5점 이하는 0."""
        return max(self.score - LIKED_ABOVE, 0.0)


@dataclass(frozen=True)
class PreferenceTarget:
    ey_pct: float  # 추천 목표 추출수율
    tds_pct: float  # 추천 목표 농도 (가수 후)
    n_liked: int  # 중심을 당긴 잔 수 (가중치가 0보다 큰 잔)
    personal_center: tuple | None  # 내 선호 중심 (추출수율, 농도). 좋았던 잔이 없으면 None


def preference_target(points):
    """목표 = (n × 내 선호 중심 + 5 × 골든컵 중심) ÷ (n + 5). 기록이 쌓일수록 내 선호 쪽으로 옮겨 간다."""
    liked = [p for p in points if p.weight > 0]
    if not liked:
        return PreferenceTarget(*GOLDEN_CUP_CENTER, n_liked=0, personal_center=None)
    total = sum(p.weight for p in liked)
    center = (sum(p.weight * p.ey_pct for p in liked) / total, sum(p.weight * p.tds_cup_pct for p in liked) / total)
    n = len(liked)
    ey, tds = ((n * c + GOLDEN_CUP_WEIGHT * g) / (n + GOLDEN_CUP_WEIGHT) for c, g in zip(center, GOLDEN_CUP_CENTER))
    return PreferenceTarget(ey, tds, n_liked=n, personal_center=center)


def brew_points(brews, params=DEFAULT_PARAMS):
    """기록한 잔마다 모델로 위치를 계산한다. 표시가 있는 잔과 맛 점수가 없는 잔은 뺀다.

    위치는 기록한 레시피와 분쇄 대표 크기로 모델이 예측한 값이다. 실제로 잰 물 빠짐 시각이나 음료 무게를
    섞지 않아야, 추천이 후보 레시피를 같은 모델로 계산해 목표와 비교할 수 있다.
    사진이 없는 잔은 같은 원두·같은 그라인더·같은 클릭으로 간 다른 잔의 측정값 평균을 빌려 쓰고,
    빌릴 값도 없으면 뺀다.
    """
    measured = defaultdict(list)
    for b in brews:  # 표시가 있는 잔이어도 분쇄 측정은 쓸 수 있다
        if b.grind_d50_um is not None:
            measured[(b.bean_id, b.grinder, b.grinder_clicks)].append(b.grind_d50_um)

    points = []
    for b in brews:
        score = b.quality_score
        if b.flag or score is None:
            continue
        d50 = b.grind_d50_um
        if d50 is None:
            same = measured.get((b.bean_id, b.grinder, b.grinder_clicks))
            if not same:
                continue
            d50 = sum(same) / len(same)
        prediction = predict(b.recipe(), grind_from_d50(d50), params)
        points.append(BrewPoint(b.brew_id, b.bean_id, prediction.ey_pct, prediction.tds_cup_pct, score))
    return points
