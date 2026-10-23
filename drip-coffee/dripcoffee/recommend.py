"""추천: 맛 기록에서 '내 선호 영역'을 찾고(MODEL.md 9.1), 그 목표에 맞는 레시피 전체를 고른다(9.2).
갈고 나서 사진으로 잰 크기가 추천과 다르면 그 가루에 맞게 레시피를 다시 맞춘다(9.3).
목표는 골든컵에서 시작해 맛 점수가 좋았던 잔의 위치(추출수율, 가수 후 농도의 모델 추정값) 쪽으로 옮겨 간다.
"""
import math
import re
from dataclasses import dataclass, field, replace

from dripcoffee.calibration import grind_sizes, roast_name
from dripcoffee.extraction_model import (DEFAULT_PARAMS, Pour, Recipe, RecipeError, grind_from_d50, parse_pours,
                                         predict)

GOLDEN_CUP_CENTER = (20.0, 1.25)  # 추출수율 %, 농도 % (골든컵 범위 18–22%, 1.15–1.35%의 가운데)
GOLDEN_CUP_WEIGHT = 5  # 골든컵을 '좋았던 잔 5잔 분량'으로 친다 (가정)
LIKED_ABOVE = 2.5  # 이 점수 이하인 잔은 중심을 당기지 않는다 (가정)
# 진하기 답이 알려 주는 '좋았을 농도' = 그 잔의 농도 × 이 값. '연함'이면 10% 진하게, '진함'이면 10% 묽게 (가정).
# 한 잔 250 g쯤에서 가수 물 20~25 g 차이로, 19-1 메모 '물을 조금 더 타도 좋을 듯'과 비슷한 크기다.
STRENGTH_SHIFT = {"연함": 1.10, "좋음": 1.0, "진함": 0.90}
STRENGTH_RECENT = 3  # 목표 농도는 진하기 답이 있는 최근 이만큼의 잔으로 정한다. 원두나 입맛이 바뀌면 옛 답이 빠지게 (가정)


@dataclass(frozen=True)
class BrewPoint:
    """기록한 한 잔의 위치와 맛 점수."""

    brew_id: str
    bean_id: str
    ey_pct: float  # 추출수율 (모델 추정)
    tds_cup_pct: float  # 가수 후 농도 (모델 추정)
    score: float  # 만족도 점수 (1~5)
    strength: str = ""  # 진하기 답 ('연함', '좋음', '진함', 모르면 '')

    @property
    def preferred_tds_pct(self):
        """이 잔이 알려 주는 '좋았을 농도'. 진했다면 조금 묽게, 연했다면 조금 진하게 옮긴다."""
        return self.tds_cup_pct * STRENGTH_SHIFT.get(self.strength, 1.0)

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
    n_strength: int = 0  # 목표 농도를 정한 진하기 답 수. 0이면 목표 농도도 맛 점수 중심과 골든컵으로 정했다


def preference_target(points):
    """추천 목표 (MODEL.md 9.1). 얼마나 우러나게(추출수율)는 맛 점수로, 얼마나 진하게(농도)는 진하기 답으로 배운다.

    - 목표 추출수율 = (n × 내 선호 중심 + 5 × 골든컵 중심) ÷ (n + 5). 기록이 쌓일수록 내 선호 쪽으로 옮겨 간다.
    - 목표 농도 = 진하기 답이 있는 최근 3잔의 '좋았을 농도' 평균. 진하기는 내 입맛을 직접 물은 답이라 골든컵을
      섞지 않고, 맛없었던 잔의 답도 쓴다. 한 번 답하면 다음 잔이 바로 바뀌고, '딱 좋았어요'가 나오면 멈춘다.
      진하기 답이 하나도 없으면 추출수율과 같은 방식(맛 점수 중심과 골든컵)으로 정한다.
    """
    liked = [p for p in points if p.weight > 0]
    if liked:
        total = sum(p.weight for p in liked)
        center = (sum(p.weight * p.ey_pct for p in liked) / total, sum(p.weight * p.tds_cup_pct for p in liked) / total)
        n = len(liked)
        ey, tds = ((n * c + GOLDEN_CUP_WEIGHT * g) / (n + GOLDEN_CUP_WEIGHT) for c, g in zip(center, GOLDEN_CUP_CENTER))
    else:
        center, n = None, 0
        ey, tds = GOLDEN_CUP_CENTER
    answered = [p for p in points if p.strength in STRENGTH_SHIFT][-STRENGTH_RECENT:]
    if answered:
        tds = sum(p.preferred_tds_pct for p in answered) / len(answered)
    return PreferenceTarget(ey, tds, n_liked=n, personal_center=center, n_strength=len(answered))


def brew_points(brews, params=DEFAULT_PARAMS):
    """기록한 잔마다 모델로 위치를 계산한다. 표시가 있는 잔과 맛 점수가 없는 잔은 뺀다.

    위치는 기록한 레시피와 분쇄 대표 크기로 모델이 예측한 값이다. 실제로 잰 물 빠짐 시각이나 음료 무게를
    섞지 않아야, 추천이 후보 레시피를 같은 모델로 계산해 목표와 비교할 수 있다.
    사진이 없는 잔은 같은 원두·클릭의 측정값을 빌려 쓰고(calibration.grind_sizes), 빌릴 값도 없으면 뺀다.
    """
    sizes = grind_sizes(brews)
    points = []
    for b in brews:
        score = b.quality_score
        if b.flag or score is None or b.brew_id not in sizes:
            continue
        prediction = predict(b.recipe(), grind_from_d50(sizes[b.brew_id][0]), params)
        points.append(BrewPoint(b.brew_id, b.bean_id, prediction.ey_pct, prediction.tds_cup_pct, score, b.strength))
    return points


# ---------------------------------------------------------------------------
# 레시피 전체 추천 (MODEL.md 9.2)
# ---------------------------------------------------------------------------

WATER_RATIOS = range(8, 17)  # 전체 물 후보: 원두의 8–16배, 1배 간격 (가정)
# 배전도별 물 온도 범위(°C, 1 °C 간격). 라이트·미디엄·다크는 일반적으로 권하는 범위(가정),
# 미디엄 라이트·미디엄 다크는 두 이웃 범위 사이로 정한 값(가정).
ROAST_TEMPS = {"라이트": (92, 96), "미디엄 라이트": (90, 95), "미디엄": (88, 94), "미디엄 다크": (84, 90), "다크": (80, 88)}
# 기록이 없는 원두의 출발점: 같은 배전도 묶음의 마지막 잔, 그것도 없으면 기본 레시피 두 가지(BREW_PROTOCOL).
# 중강배전(미디엄 다크) 이상은 B01 방식 1:8, 2-4-2배 / 60초, 그 밖(라이트~미디엄)은 B02 방식 1:11, 2-3-3-3배 / 30초.
BASE_RECIPES = {"중강배전 이상": (8.0, (4, 2), 60.0), "그 밖": (11.0, (3, 3, 3), 30.0)}
BLOOM_MULTIPLE = 2  # 뜸 물 = 원두의 2배, 0초에 붓는다
POUR_MULTIPLES = (4, 3, 2)  # 뜸 뒤 한 번에 붓는 양 = 원두의 2·3·4배 중 하나
GRIND_RANGE_UM = (700.0, 1800.0)  # 핸드드립으로 보는 대표 크기 범위
GOOD_DRAWDOWN_S = (120.0, 210.0)  # 끊김 시각 범위 (2:00–3:30). 이 밖의 레시피는 추천하지 않는다
OUT_OF_WINDOW_LIMIT = 30  # 끊김이 범위 밖인 후보가 이만큼 쌓이면 목표에 딱 맞추기를 그만두고 가장 가까운 레시피를 찾는다
GRIND_STEP_UM = 10  # 추천 크기는 10 µm 단위로 보여준다
REFIT_INTERVAL_S = 15  # 잰 분쇄에 맞춰 붓기 간격을 바꿀 수 있는 폭 (±)
REFIT_STEP_S = 5  # 간격은 5초 단위로 바꾼다 (타이머를 보며 따라 하기 쉽게)
REFIT_EY_TOLERANCE = 0.3  # 추출수율이 목표와 이만큼(%p) 안이면 맞았다고 본다 (간격 5초에 약 0.6%p라 그 절반)


class NoRecipeError(ValueError):
    """조건을 만족하는 레시피가 없을 때, 사용자에게 보여줄 이유를 담는 예외."""


@dataclass
class Usual:
    """'평소 레시피'. 후보가 이것과 덜 다를수록 먼저 고른다."""

    water_ratio: float
    temp_c: float
    pattern: tuple  # 뜸 뒤 붓기마다 원두의 몇 배인지. 예: (3, 3, 3)
    interval_s: float  # 붓기 간격
    source: str  # 어디서 가져왔는지 (화면 설명용)
    brew_id: str = ""  # 가져온 잔. 기본 레시피면 빈 값


@dataclass
class Recommendation:
    recipe: Recipe  # 원두 무게, 물 온도, 붓기 일정, 가수량
    d50_um: float  # 추천 분쇄 대표 크기
    prediction: object  # 추천 레시피를 모델로 계산한 결과 (끊김 시각, 음료, 수율, 농도)
    water_ratio: int  # 전체 물 = 원두의 몇 배
    pattern: tuple  # 뜸 뒤 붓기마다 원두의 몇 배인지
    interval_s: float
    usual: Usual
    temp_range: tuple
    clog_limit_um: float | None  # 이보다 고운 분쇄는 뺐다 (같은 원두가 막혔던 가장 굵은 크기)
    checked: int  # 계산해 본 후보 수
    skipped: dict = field(default_factory=dict)  # 뺀 이유별 후보 수
    # 예측 추출수율 − 목표. 끊김 제한 안에서는 목표에 닿지 못해 가장 가까운 레시피를 골랐을 때만 0이 아니다 (음수 = 덜 우러남)
    ey_gap_pct: float = 0.0


@dataclass
class Refit:
    """갈고 나서 사진으로 잰 크기에 맞춰 다시 맞춘 레시피 (한 잔 내리기 3단계)."""

    recipe: Recipe  # 붓기 간격과 가수량만 바꾼 레시피
    d50_um: float  # 사진으로 잰 대표 크기
    interval_s: float
    interval_change_s: float  # 추천보다 몇 초 바꿨는지 (+면 늘림)
    prediction: object  # 다시 맞춘 레시피를 잰 크기로 계산한 결과
    reached: bool  # 간격 ±15초 안에서 목표 추출수율에 닿았는지
    ey_gap_pct: float  # 예측 추출수율 − 목표 (닿지 못했을 때 얼마나 모자라거나 넘치는지)


def roast_range(roast_level):
    """배전도 이름(원두 목록의 표기)에 맞는 물 온도 범위. 모르는 이름이면 미디엄 범위를 쓴다."""
    return ROAST_TEMPS.get(roast_name(roast_level), ROAST_TEMPS["미디엄"])


def roast_group(roast_level):
    """레시피 출발점을 고르는 배전도 묶음: 미디엄 다크(중강배전)와 다크(강배전)는 '중강배전 이상', 나머지는 '그 밖'."""
    return "중강배전 이상" if roast_name(roast_level) in ("미디엄 다크", "다크") else "그 밖"


def split_patterns(rest, largest=max(POUR_MULTIPLES)):
    """뜸 뒤 남은 물(원두의 rest배)을 2·3·4배 붓기로 나누는 모든 방법. 큰 붓기를 먼저 붓는다.
    예: 6 → (4, 2), (3, 3), (2, 2, 2)"""
    if rest == 0:
        return [()]
    return [(m, *tail) for m in POUR_MULTIPLES if m <= min(rest, largest)
            for tail in split_patterns(rest - m, m)]


def pattern_pours(dose_g, pattern, interval_s):
    """뜸(원두의 2배)을 0초에 붓고, 간격마다 원두의 pattern배씩 붓는다. 양은 1 g 단위."""
    pours = [Pour(0.0, float(round(BLOOM_MULTIPLE * dose_g)))]
    pours += [Pour(float(interval_s * (i + 1)), float(round(m * dose_g))) for i, m in enumerate(pattern)]
    return pours


def pattern_text(pattern):
    """'2-4-2'처럼 뜸까지 넣은 배수 표기."""
    return "-".join(str(m) for m in (BLOOM_MULTIPLE, *pattern))


def pours_text(pours):
    """Pour 목록을 기록 형식 '시작초:g;...'로."""
    return ";".join(f"{p.start_s:g}:{p.grams:g}" for p in pours)


def pattern_of(pours_value, dose_g):
    """기록한 붓기에서 뜸 뒤 붓기 배수와 간격을 읽는다. 예: '0:40;60:80;120:40', 20 g → ((4, 2), 60)"""
    pours = parse_pours(pours_value)
    pattern = tuple(max(1, round(p.grams / dose_g)) for p in pours[1:])
    interval = pours[1].start_s if len(pours) > 1 else 30.0
    return pattern, interval


def pattern_gap(a, b):
    """두 붓기 배수의 차이. 짧은 쪽은 0배 붓기로 채워 같은 자리끼리 비교한다."""
    n = max(len(a), len(b))
    return sum(abs(x - y) for x, y in zip(a + (0,) * (n - len(a)), b + (0,) * (n - len(b))))


# 3단계에서 사진에 맞춰 붓기 간격을 바꾼 잔의 기록(followed_recommendation)에 남는 표시. 예: '간격 30 → 45초'
REFIT_NOTE = re.compile(r"간격 (\d+(?:\.\d+)?) → (\d+(?:\.\d+)?)초")


def usual_interval(brew):
    """그 잔의 평소 붓기 간격. 3단계에서 사진에 맞춰 간격을 바꾼 잔은 바꾸기 전(추천받은) 간격을 쓴다.
    바꾼 간격은 그 잔의 가루에 맞춘 것이라 다음 잔에 물려주지 않는다."""
    found = REFIT_NOTE.search(brew.row.get("followed_recommendation", "") or "")
    return float(found.group(1)) if found else pattern_of(brew.pours, brew.dose_g)[1]


def usual_recipe(bean, brews):
    """평소 레시피: 같은 원두의 마지막 정상 잔. 기록이 없는 원두는 같은 배전도 묶음(중강배전 이상 / 그 밖)의 마지막
    정상 잔의 물 양·붓기를 쓰고, 그것도 없으면 그 묶음의 기본 레시피(BASE_RECIPES)를 쓴다. 온도는 그 원두
    배전도 범위의 가운데로 둔다(다른 원두의 온도는 배전도가 달라 그대로 쓰지 않는다). 붓기 간격은 사진에 맞춰
    바꾸기 전 간격을 쓴다(usual_interval)."""
    normal = [b for b in brews if not b.flag]
    same = [b for b in normal if b.bean_id == bean["bean_id"]]
    low, high = roast_range(bean.get("roast_level", ""))
    if same:
        b = same[-1]
        return Usual(b.recipe().water_g / b.dose_g, b.water_temp_c, pattern_of(b.pours, b.dose_g)[0], usual_interval(b),
                     f"같은 원두의 마지막 잔 {b.brew_id}", b.brew_id)
    group = roast_group(bean.get("roast_level", ""))
    alike = [b for b in normal if roast_group(b.row.get("roast_level", "")) == group]
    if alike:
        b = alike[-1]
        return Usual(b.recipe().water_g / b.dose_g, (low + high) / 2, pattern_of(b.pours, b.dose_g)[0], usual_interval(b),
                     f"이 원두는 기록이 없어 같은 배전도 묶음({group})의 마지막 잔 {b.brew_id}의 물 양·붓기와 "
                     "배전도 범위 가운데 온도", b.brew_id)
    ratio, pattern, interval = BASE_RECIPES[group]
    return Usual(ratio, (low + high) / 2, pattern, interval, f"기록이 없어 {group} 기본 레시피")


# 이번 잔만의 요청 → (목표 추출수율에 더할 %p, 목표 농도에 곱할 배수). 농도 배수는 진하기 답(STRENGTH_SHIFT)과 같은 크기,
# 추출수율 1%p는 분쇄 약 80 µm만큼이다. '평소대로'면 목표를 바꾸지 않는다
INTENTS = {"평소대로": (0.0, 1.0), "더 진하게": (0.0, 1.10), "더 연하게": (0.0, 0.90), "더 우러나게": (1.0, 1.0),
           "더 산뜻하게": (-1.0, 1.0)}


def with_intent(target, intent):
    """이번 잔의 목표에 오늘의 요청을 더한다. 기록에서 배우는 목표(preference_target)는 바꾸지 않는다."""
    ey_shift, tds_scale = INTENTS[intent]
    return replace(target, ey_pct=target.ey_pct + ey_shift, tds_pct=target.tds_pct * tds_scale)


def clog_limit(bean_id, brews):
    """같은 원두에서 막혔던 잔의 대표 크기 중 가장 굵은 값. 이보다 고운 분쇄는 추천하지 않는다 (9.2)."""
    sizes = grind_sizes(brews)
    clogged = [sizes[b.brew_id][0] for b in brews
               if b.bean_id == bean_id and b.flag == "막힘" and b.brew_id in sizes]
    return max(clogged) if clogged else None


def recommend_recipe(target, bean, brews, dose_g=20.0, params=DEFAULT_PARAMS):
    """목표(추출수율, 가수 후 농도)에 맞는 레시피 전체를 고른다 (MODEL.md 9.2).

    1. 후보: 전체 물(원두의 8–16배) × 붓기 나누기(뜸 2배 + 2·3·4배 붓기의 모든 조합) × 온도(배전도 범위, 1 °C 간격).
       붓기 간격은 평소 레시피를 따른다. 붓는 시각이 맛에 주는 효과는 모델도 기록도 판단할 수 없어서다.
    2. 후보마다 추출수율이 목표와 같아지는 분쇄 크기를 찾는다(곱게 갈수록 수율이 오르므로 하나로 정해짐).
       700–1800 µm 밖이거나 같은 원두가 막혔던 크기보다 고우면 뺀다.
    3. 가수량 = 음료로 나간 성분 ÷ 목표 농도 − 음료 무게. 가수 없이도 목표보다 묽으면 뺀다.
    4. 끊김은 2:00–3:30 안이어야 한다(모델이 시간 효과를 계산하지 못해서, 6절). 남은 후보는 모두 목표에 맞으므로
       평소 레시피와 덜 다른 것을 고른다. 계산을 줄이려고 평소 레시피와 가까운 순서로 계산하고, 범위 안인 첫 후보에서 멈춘다.
    5. 범위 안에서 목표에 딱 맞는 후보가 없으면(범위 밖 후보가 30개 쌓이면 그만 찾는다), 분쇄와 온도로 최대한
       맞춘다: 후보마다 허용되는 가장 고운 분쇄(막힘 한계 바로 위)로 한 번씩 계산해, 끊김이 범위 안인 것 중
       추출수율이 목표에 가장 가까운 것을 고른다. 목표와 0.3%p 안이면 맞은 것으로 보고(3단계 다시 맞추기와 같은
       기준) 그중 가장 평소다운 것을 고른다. 모자란 만큼은 ey_gap_pct로 알려 화면에서 설명한다.
    """
    temp_low, temp_high = roast_range(bean.get("roast_level", ""))
    usual = usual_recipe(bean, brews)
    limit = clog_limit(bean["bean_id"], brews)
    lowest = max(GRIND_RANGE_UM[0], limit) if limit else GRIND_RANGE_UM[0]

    def distance(c):
        ratio, pattern, temp = c
        return abs(ratio - usual.water_ratio) + abs(temp - usual.temp_c) / 2 + pattern_gap(pattern, usual.pattern) / 2

    def in_window(seconds):
        return GOOD_DRAWDOWN_S[0] <= seconds <= GOOD_DRAWDOWN_S[1]

    def plain(pattern, temp):
        return Recipe(dose_g, float(temp), pattern_pours(dose_g, pattern, usual.interval_s), 0.0)

    def allowed(d50):
        """10 µm 단위로 맞추되 막힘 한계 안쪽으로 들어가지 않게."""
        d50 = round(d50 / GRIND_STEP_UM) * GRIND_STEP_UM
        while d50 <= (limit or 0):
            d50 += GRIND_STEP_UM
        return d50

    def build(c, d50, ey_gap=0.0):
        """분쇄 크기가 정해진 후보에 가수량을 붙여 추천으로 만든다. 가수 없이도 목표보다 묽으면 None."""
        ratio, pattern, temp = c
        result = predict(plain(pattern, temp), grind_from_d50(d50), params)
        bypass = round(result.dissolved_g / (target.tds_pct / 100) - result.beverage_g)
        if bypass < 0:
            skipped["가수 없이도 목표 농도보다 묽음"] += 1
            return None
        recipe = Recipe(dose_g, float(temp), plain(pattern, temp).pours, float(bypass))
        return Recommendation(recipe, float(d50), predict(recipe, grind_from_d50(d50), params), ratio, pattern,
                              usual.interval_s, usual, (temp_low, temp_high), limit, checked, skipped, ey_gap)

    def solve(c):
        """목표 추출수율에 딱 맞는 분쇄 크기로 추천을 만든다. 맞출 수 없으면 None."""
        recipe = plain(c[1], c[2])

        def ey(d50):
            try:
                return predict(recipe, grind_from_d50(d50), params).ey_pct
            except RecipeError:  # 너무 오래 걸리는 레시피 등
                return math.inf

        if ey(lowest) < target.ey_pct:
            skipped["가장 곱게 갈아도 목표 수율보다 낮음"] += 1
            return None
        if ey(GRIND_RANGE_UM[1]) > target.ey_pct:
            skipped["가장 굵게 갈아도 목표 수율보다 높음"] += 1
            return None
        fine, coarse = math.log(lowest), math.log(GRIND_RANGE_UM[1])
        for _ in range(9):  # 로그 크기에서 반씩 줄여 약 2 µm까지 좁힌다
            middle = (fine + coarse) / 2
            fine, coarse = (middle, coarse) if ey(math.exp(middle)) > target.ey_pct else (fine, middle)
        return build(c, allowed(math.exp((fine + coarse) / 2)))

    candidates = sorted(((ratio, pattern, temp) for ratio in WATER_RATIOS
                         for pattern in split_patterns(ratio - BLOOM_MULTIPLE)
                         for temp in range(temp_low, temp_high + 1)), key=distance)
    skipped = {"가장 곱게 갈아도 목표 수율보다 낮음": 0, "가장 굵게 갈아도 목표 수율보다 높음": 0,
               "가수 없이도 목표 농도보다 묽음": 0, "끊김이 2:00–3:30 밖": 0}
    checked = 0
    for c in candidates:  # 목표에 딱 맞는 레시피 (4)
        checked += 1
        found = solve(c)
        if found is None:
            continue
        if in_window(found.prediction.drawdown_end_s):
            found.checked = checked
            return found
        skipped["끊김이 2:00–3:30 밖"] += 1
        if skipped["끊김이 2:00–3:30 밖"] >= OUT_OF_WINDOW_LIMIT:
            break

    finest = allowed(lowest)  # 목표에 가장 가까운 레시피 (5): 허용되는 가장 고운 분쇄로 한 번씩
    close, nearest = [], None
    for c in candidates:
        checked += 1
        try:
            result = predict(plain(c[1], c[2]), grind_from_d50(finest), params)
        except RecipeError:
            continue
        if not in_window(result.drawdown_end_s):
            continue
        if result.ey_pct > target.ey_pct + REFIT_EY_TOLERANCE:  # 더 굵게 갈면 목표에 딱 맞출 수 있다
            found = solve(c)
            if found is not None and in_window(found.prediction.drawdown_end_s):
                found.checked = checked
                return found
            continue
        gap = result.ey_pct - target.ey_pct
        if gap >= -REFIT_EY_TOLERANCE:
            close.append((c, gap))
        elif nearest is None or gap > nearest[1]:
            nearest = (c, gap)
    for c, gap in close + ([nearest] if nearest else []):  # 맞은 것은 평소다운 순서로, 없으면 가장 가까운 것
        found = build(c, finest, 0.0 if gap >= -REFIT_EY_TOLERANCE else gap)
        if found is not None:
            found.checked = checked
            return found
    reasons = ", ".join(f"{k} {v}개" for k, v in skipped.items() if v)
    raise NoRecipeError(f"목표(추출수율 {target.ey_pct:.1f}%, 농도 {target.tds_pct:.2f}%)에 맞는 레시피를 찾지 못했어요. "
                        f"후보 {checked}개 중 {reasons}.")


def refit_to_grind(recommendation, measured_d50_um, target, params=DEFAULT_PARAMS):
    """갈고 나서 사진으로 잰 크기에 맞춰 붓기 간격과 가수량만 다시 정한다 (MODEL.md 9.3).

    - 붓기 간격을 추천 ±15초 안에서 5초 단위로 바꿔, 추출수율이 목표에 가장 가까운 간격을 고른다. 목표와 0.3%p 안인
      간격이 여럿이면 가장 덜 바꾼 것을 고른다(간격 15초 ≈ 추출수율 2%p ≈ 분쇄 120 µm).
    - 끊김이 2:00–3:30 밖으로 나가는 간격은 고르지 않는다. 모든 간격이 밖이면 범위에 가장 가까운 간격을 쓴다.
    - 물 양, 붓기 배수, 온도는 그대로 둔다.
    - 가수량은 바뀐 레시피의 음료로 목표 농도를 다시 맞춘다. 가수 없이도 묽으면 0으로 둔다.
    """
    base = recommendation.recipe
    grind = grind_from_d50(measured_d50_um)
    options = []
    for change in range(-REFIT_INTERVAL_S, REFIT_INTERVAL_S + 1, REFIT_STEP_S):
        interval = recommendation.interval_s + change
        if interval <= 0:
            continue
        recipe = Recipe(base.dose_g, base.water_temp_c, pattern_pours(base.dose_g, recommendation.pattern, interval), 0.0)
        try:
            options.append((change, recipe, predict(recipe, grind, params)))
        except RecipeError:
            continue
    if not options:
        raise NoRecipeError("잰 분쇄 크기로는 이 레시피를 계산할 수 없어요. 분쇄를 다시 확인해 주세요.")

    def outside(seconds):  # 끊김 범위에서 몇 초 벗어났는지 (안이면 0)
        return max(GOOD_DRAWDOWN_S[0] - seconds, seconds - GOOD_DRAWDOWN_S[1], 0.0)

    options = ([o for o in options if outside(o[2].drawdown_end_s) == 0]
               or [min(options, key=lambda o: outside(o[2].drawdown_end_s))])
    close = [o for o in options if abs(o[2].ey_pct - target.ey_pct) <= REFIT_EY_TOLERANCE]
    change, recipe, result = (min(close, key=lambda o: abs(o[0])) if close
                              else min(options, key=lambda o: abs(o[2].ey_pct - target.ey_pct)))
    bypass = max(round(result.dissolved_g / (target.tds_pct / 100) - result.beverage_g), 0)
    recipe = Recipe(recipe.dose_g, recipe.water_temp_c, recipe.pours, float(bypass))
    final = predict(recipe, grind, params)
    return Refit(recipe, float(measured_d50_um), recommendation.interval_s + change, float(change), final,
                 bool(close), final.ey_pct - target.ey_pct)
