"""추출 기록(data/brews.csv) 읽기·쓰기, 입력 확인, 기록과 모델 예측의 비교.

앱의 기록 탭은 append_brew로 새 잔을 저장하고, 보정(MODEL.md 5절)과 내 선호 영역(9.1절)은
load_brews로 읽은 기록을 쓴다. 측정 방법과 맛 기준은 docs/BREW_PROTOCOL.md를 따른다.
"""
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

from dripcoffee import DATA_DIR
from dripcoffee.extraction_model import (DEFAULT_PARAMS, Prediction, Recipe, RecipeError, grind_from_d50, parse_pours,
                                         predict)

BREWS_PATH = DATA_DIR / "brews.csv"
BEANS_PATH = DATA_DIR / "beans.csv"

# 배전도: 앱에서 고르는 이름과, 원두 목록에 섞여 있는 다른 표기
ROAST_LEVELS = ("라이트", "미디엄 라이트", "미디엄", "미디엄 다크", "다크")
ROAST_ALIASES = {"강배전": "다크", "약배전": "라이트", "중배전": "미디엄", "중강배전": "미디엄 다크", "미디움": "미디엄",
                 "미디움 다크": "미디엄 다크", "미디움 라이트": "미디엄 라이트"}
BEAN_ROLES = ("학습", "검증", "시험", "여유", "일반")  # 앱에서 더한 원두는 '일반'
# 로스팅 뒤 디게싱(원두에서 가스가 빠지는 과정): 배전도 → (이보다 이르면 '디게싱 중', 이날부터 '마시기 좋음',
# 이날을 넘으면 '상미 기간 지남').
# 사용자가 모은 로스터·바리스타 자료(블로그·영상, 조사 노트 5)에 맞춘 가정. '오래됨'은 자료에 숫자가 없어
# '강배전일수록 상미 기간이 짧다'는 설명에 맞춰 정했다. 그 사이('이르면' 기준과 '좋음' 사이)는 '거의 끝남'.
REST_DAYS = {"라이트": (3, 7, 45), "미디엄 라이트": (3, 7, 40), "미디엄": (3, 7, 35), "미디엄 다크": (3, 7, 30),
             "다크": (2, 5, 25)}

# 필터 종이. 종이마다 물 빠지는 속도가 다를 수 있어 잔마다 적는다 (10/19). 목록 밖의 필터도 적을 수 있다.
FILTERS = ("하리오 화이트", "홀츠클로츠 무표백")

# 표시(flag): 정상이 아닌 잔. 표시한 잔은 보정과 선호 영역에서 뺀다.
FLAGS = ("연습", "흘림", "막힘", "측정실수", "기타")

# 맛 6항목: (칸 이름, 화면 이름, 최솟값). 산미와 바디는 세기라서 0점부터, 나머지는 만족도라서 1점부터.
TASTE_ITEMS = (
    ("aroma", "Aroma", 1),
    ("flavor", "Flavor", 1),
    ("acidity", "산미", 0),
    ("body", "Body", 0),
    ("balance", "Balance", 1),
    ("aftertaste", "Aftertaste", 1),
)
TASTE_MAX = 5
QUALITY_ITEMS = ("aroma", "flavor", "balance", "aftertaste")  # 만족도 점수에 넣는 항목

# 숫자 칸: (칸 이름, 화면 이름, 0을 허용하는지)
NUMBER_FIELDS = (
    ("grinder_clicks", "그라인더 클릭", True),
    ("grind_d50_um", "분쇄 대표 크기", False),
    ("dose_g", "원두 무게", False),
    ("water_temp_c", "물 온도", False),
    ("brew_water_g", "부은 물 양", False),
    ("drawdown_end_s", "물 빠짐 종료 시각", False),
    ("beverage_g", "음료 무게", False),
    ("bypass_g", "가수량", True),
)
REQUIRED_FIELDS = (("brew_id", "추출번호"), ("date", "날짜"), ("bean_id", "원두 번호"), ("filter", "필터"),
                   ("dose_g", "원두 무게"), ("water_temp_c", "물 온도"), ("pours", "붓기 기록"))


class BrewLogError(ValueError):
    """기록을 저장할 수 없을 때, 사용자에게 보여줄 이유를 담는 예외."""


@dataclass
class Brew:
    """brews.csv의 한 줄. 계산에 쓰는 칸은 숫자로 바꿔 두고, 나머지는 row에 그대로 둔다."""

    brew_id: str
    date: str
    bean_id: str
    filter_paper: str  # 필터 종이 (예: 하리오 화이트)
    grinder: str
    grinder_clicks: int | None
    grind_d50_um: float | None  # 이 잔의 가루로 찍은 사진 측정값. 사진이 없으면 None
    dose_g: float
    water_temp_c: float
    pours: str  # '시작초:g;...'
    bypass_g: float
    drawdown_end_s: float | None
    beverage_g: float | None  # 가수 전 음료 무게
    taste: dict  # 맛 6항목, 비어 있으면 None
    flag: str
    memo: str
    row: dict = field(default_factory=dict, repr=False)

    @property
    def quality_score(self):
        """만족도 점수 = Aroma, Flavor, Balance, Aftertaste의 평균. 산미·바디는 세기라서 뺀다."""
        values = [self.taste.get(k) for k in QUALITY_ITEMS]
        if any(v is None for v in values):
            return None
        return sum(values) / len(values)

    def recipe(self):
        return Recipe(self.dose_g, self.water_temp_c, parse_pours(self.pours), self.bypass_g)


def load_brews(path=BREWS_PATH):
    """추출 기록을 기록한 순서대로 읽는다."""
    with open(path, encoding="utf-8", newline="") as f:
        return [_brew_from_row(row) for row in csv.DictReader(f)]


def grind_sizes(brews, pooled=False):
    """잔마다 계산에 쓸 분쇄 대표 크기. {추출번호: (크기 µm, 빌렸는지)}.

    사진이 없는 잔은 같은 봉투(원두와 로스팅 날짜)·같은 그라인더·같은 클릭으로 간 다른 잔의 측정값 평균을
    빌려 쓴다. 같은 원두라도 봉투가 바뀌면 같은 클릭에서 크기가 달라질 수 있어서다(10/18 B02: 1406 → 1176 µm).
    표시가 있는 잔이어도 분쇄 측정은 맞으므로 빌려 줄 수 있다. 빌릴 값도 없는 잔은 넣지 않는다.

    pooled이면 사진이 있는 잔도 같은 봉투·그라인더·클릭의 평균을 쓴다(보정용). 사진 한 장의 흔들림은 대부분
    그 사진에 큰 입자가 몇 개 들어갔는지에서 오고, 가루를 다시 펴서 생기는 차이는 작았으므로(10/19, 사진 47장)
    같은 가루로 볼 수 있는 잔들을 합치면 덜 흔들린다. 이때 표시가 있는 잔(연습 등)의 사진은 평균에 넣지 않는다.
    10/9 연습 잔은 찍는 방법을 정하기 전이라 가루가 촘촘해, 붙은 입자 때문에 크게 나왔다.
    """
    def same_bag(b):
        return b.bean_id, b.row.get("roast_date", ""), b.grinder, b.grinder_clicks

    measured = defaultdict(list)
    for b in brews:
        if b.grind_d50_um is not None and not (pooled and b.flag):
            measured[same_bag(b)].append(b.grind_d50_um)
    sizes = {}
    for b in brews:
        same = measured.get(same_bag(b))
        if b.grind_d50_um is not None and (not pooled or not same):
            sizes[b.brew_id] = (b.grind_d50_um, False)
        elif same:
            sizes[b.brew_id] = (sum(same) / len(same), b.grind_d50_um is None)
    return sizes


@dataclass
class LogComparison:
    """기록한 한 잔의 모델 예측과 실제. 차이는 '예측 − 실제'라서, 음수면 예측이 짧거나 적다."""

    brew: Brew
    grind_d50_um: float  # 예측에 쓴 대표 크기
    borrowed_grind: bool  # 같은 원두·클릭의 다른 잔에서 빌렸는지
    prediction: Prediction

    @property
    def end_error_s(self):
        actual = self.brew.drawdown_end_s
        return None if actual is None else self.prediction.drawdown_end_s - actual

    @property
    def beverage_error_g(self):
        actual = self.brew.beverage_g
        return None if actual is None else self.prediction.beverage_g - actual


def compare_with_log(brews, params=DEFAULT_PARAMS, pooled=False):
    """기록한 잔마다 같은 레시피·분쇄로 모델을 돌려 실제(물 빠짐 종료 시각, 음료 무게)와 나란히 둔다.

    보정(MODEL.md 5절) 전후의 오차를 재는 데 쓴다. 표시가 있는 잔도 넣고, 뺄지는 쓰는 쪽이 brew.flag로 정한다.
    분쇄 크기를 알 수 없는 잔은 뺀다. pooled는 grind_sizes와 같다.
    """
    sizes = grind_sizes(brews, pooled)
    comparisons = []
    for b in brews:
        if b.brew_id not in sizes:
            continue
        d50, borrowed = sizes[b.brew_id]
        comparisons.append(LogComparison(b, d50, borrowed, predict(b.recipe(), grind_from_d50(d50), params)))
    return comparisons


CLICK_HINT_REACH_UM = 60  # 목표가 같은 봉투 기록과 이만큼(µm) 안이면 그 기록으로 알려 준다 (사진 측정 흔들림 ±5~9%의 작은 쪽)
# 그라인더 이름에 들어간 낱말 → 인터넷 기준 클릭표(비교용, 대략). 내 기록이 없을 때 시작점으로 쓴다.
CLICK_TABLES = {"메버릭": DATA_DIR / "reference" / "maverick_click_table.csv"}


@dataclass
class ClickHint:
    """목표 분쇄 크기에 맞는 그라인더 클릭의 시작점 힌트. 갈고 나서 사진으로 다시 확인한다."""

    clicks: int | None  # 시작할 클릭. 근거가 멀어 숫자를 줄 수 없으면 None (방향만 알려 줌)
    direction: str  # 숫자 대신 방향을 줄 때 '더 곱게' 또는 '더 굵게', 아니면 ''
    source: str  # '내 기록' 또는 '클릭표'
    basis: str  # 화면에 보여줄 근거
    near: Brew | None = None  # 근거가 된 잔 (내 기록일 때)
    um_per_click: float | None = None  # 내 기록에서 구한 1클릭당 크기 변화


def table_clicks(target_d50_um, grinder):
    """인터넷 클릭표에서 목표 크기에 해당하는 클릭. 표가 없는 그라인더면 None. 표 밖이면 가장 가까운 끝."""
    path = next((path for name, path in CLICK_TABLES.items() if name in str(grinder)), None)
    if path is None:
        return None
    with open(path, encoding="utf-8", newline="") as f:
        table = sorted((float(r["size_um"]), float(r["clicks"])) for r in csv.DictReader(f))
    if target_d50_um <= table[0][0]:
        return round(table[0][1])
    for (s0, c0), (s1, c1) in zip(table, table[1:]):
        if target_d50_um <= s1:
            return round(c0 + (c1 - c0) * (target_d50_um - s0) / (s1 - s0))
    return round(table[-1][1])


def click_hint(target_d50_um, grinder, bean_id, roast_date, brews):
    """목표 크기에 맞는 클릭의 시작점. 같은 봉투 기록 → 클릭표 → 다른 기록(방향만) 순서로 정한다. 근거가 없으면 None.

    - 같은 봉투를 이 그라인더로 잰 기록이 있고 목표와 60 µm 안이면, 그 잔을 기준으로 1클릭당 크기 변화만큼 옮긴다. 같은 원두라도 봉투가 바뀌면 같은 클릭에서 크기가 달라서(95클릭: 1406 → 1176 µm)
      다른 봉투 기록은 쓰지 않는다.
    - 같은 봉투 기록이 없거나 멀면 인터넷 클릭표로 알려 준다. 사진 측정과 기준이 달라 대략이지만, 갈고 나서
      사진으로 실제 크기를 재고 그에 맞춰 레시피를 바꾸므로(한 잔 내리기 3단계) 시작점으로는 충분하다.
    - 그 잔을 기록하면 클릭과 사진 크기가 함께 남아, 다음에 같은 봉투를 내릴 때는 내 기록으로 알려 준다.
    - 클릭표가 없는 그라인더는 가장 가까운 내 기록에서 방향만 알려 준다.
    """
    measured = [b for b in brews if b.grinder == grinder and b.grind_d50_um is not None and b.grinder_clicks is not None]
    bags = defaultdict(list)
    for b in measured:
        bags[(b.bean_id, b.row.get("roast_date", ""))].append(b)
    num = den = 0.0
    for cups in bags.values():  # 1클릭당 크기 변화는 한 봉투를 여러 클릭으로 잰 기록에서만 구한다
        mean_c = sum(b.grinder_clicks for b in cups) / len(cups)
        mean_d = sum(b.grind_d50_um for b in cups) / len(cups)
        num += sum((b.grinder_clicks - mean_c) * (b.grind_d50_um - mean_d) for b in cups)
        den += sum((b.grinder_clicks - mean_c) ** 2 for b in cups)
    slope = num / den if den and num > 0 else None

    same_bag = bags.get((bean_id, roast_date))
    if same_bag:
        near = min(same_bag, key=lambda b: abs(b.grind_d50_um - target_d50_um))
        if abs(target_d50_um - near.grind_d50_um) <= CLICK_HINT_REACH_UM:
            gap = (target_d50_um - near.grind_d50_um) / slope if slope else 0
            clicks = min(max(round(near.grinder_clicks + gap), 0), 180)  # 메버릭 클릭 범위 0–180
            return ClickHint(clicks, "", "내 기록", f"같은 봉투를 {near.grinder_clicks}클릭으로 갈았을 때 "
                             f"{near.grind_d50_um:.0f} µm였어요 ({near.brew_id})", near, slope)
    table = table_clicks(target_d50_um, grinder)
    if table is not None:
        return ClickHint(table, "", "클릭표", "인터넷 클릭표 기준이라 대략이에요. 갈고 나서 사진으로 확인해요.",
                         None, slope)
    if not measured:
        return None
    near = min(measured, key=lambda b: abs(b.grind_d50_um - target_d50_um))
    direction = "더 곱게" if target_d50_um < near.grind_d50_um else "더 굵게"
    return ClickHint(None, direction, "내 기록", f"가장 가까운 기록은 {near.grinder_clicks}클릭 "
                     f"{near.grind_d50_um:.0f} µm예요 ({near.brew_id}, {near.bean_id})", near, slope)


def roast_name(roast_level):
    """원두 목록의 배전도 표기를 앱의 다섯 이름 중 하나로. 모르는 표기는 그대로 돌려준다."""
    name = str(roast_level).strip()
    return ROAST_ALIASES.get(name, name)


class BeanError(ValueError):
    """원두를 추가할 수 없을 때, 사용자에게 보여줄 이유를 담는 예외."""


@dataclass
class RestStatus:
    days: int  # 로스팅 뒤 며칠째
    state: str  # '디게싱 중', '거의 끝남', '좋음', '상미 기간 지남', '날짜 확인'
    label: str  # 화면에 굵게 보일 한 줄
    message: str  # 처음 쓰는 사람도 알 수 있게 풀어 쓴 설명


def _has_final(word):
    last = word[-1]
    return "가" <= last <= "힣" and (ord(last) - ord("가")) % 28 != 0


def _topic(word):
    """'미디엄은', '다크는'처럼 받침에 맞는 조사."""
    return word + ("은" if _has_final(word) else "는")


def with_object(word):
    """'예가체프를', '원두을'이 아니라 받침에 맞는 목적격 조사를 붙인다."""
    return word + ("을" if _has_final(word) else "를")


DEGASSING_MEANING = "디게싱은 로스팅한 원두에서 가스(이산화탄소)가 빠져나가는 과정이에요."
SHELF_MEANING = "상미 기간은 원두를 맛있게 마실 수 있는 기간이에요."


def rest_status(roast_level, roast_date, today):
    """로스팅 날짜로 디게싱(가스 빠짐)이 얼마나 됐는지, 상미 기간이 지났는지 알려 준다. 기준은 REST_DAYS."""
    level = roast_name(roast_level)
    early, ready, stale = REST_DAYS.get(level, REST_DAYS["미디엄"])
    days = (today - date.fromisoformat(str(roast_date))).days
    who = _topic(level)
    if days < 0:
        return RestStatus(days, "날짜 확인", "로스팅 날짜를 확인해 주세요", "로스팅 날짜가 오늘보다 뒤예요.")
    if days < early:
        extra = " 강배전은 이때 가스 맛이나 쓴맛이 강할 수 있어요." if level == "다크" else ""
        return RestStatus(days, "디게싱 중", f"로스팅 {days}일째 · 디게싱 중이에요",
                          f"{DEGASSING_MEANING} {who} 보통 {ready}일째부터 맛이 안정돼요. 지금 내리면 뜸(처음에 물을 조금 "
                          f"부어 가루를 적시는 단계)에서 거품이 많이 올라오고, 물이 빠지는 속도와 맛이 평소와 다를 수 있어요.{extra}")
    if days < ready:
        extra = " 라이트는 10일 넘게 걸리기도 해요." if level == "라이트" else ""
        return RestStatus(days, "거의 끝남", f"로스팅 {days}일째 · 디게싱이 거의 끝나 가요",
                          f"{DEGASSING_MEANING} 아직 가스가 조금 남았을 수 있어요. {who} 보통 {ready}일째부터 맛이 "
                          f"안정돼요.{extra}")
    if days <= stale:
        return RestStatus(days, "좋음", f"로스팅 {days}일째 · 마시기 좋은 때예요",
                          "가스가 충분히 빠져 맛이 안정된 때예요.")
    return RestStatus(days, "상미 기간 지남", f"로스팅 {days}일째 · 상미 기간이 지났어요",
                      f"{SHELF_MEANING} {who} 보통 {stale}일쯤 지나면 향이 약해져요. 마셔도 괜찮지만 향이 덜할 수 있어요.")


def append_bean(record, path=BEANS_PATH, today=None):
    """새 원두를 확인한 뒤 원두 목록 끝에 더하고 원두 번호를 돌려준다. 번호는 B 다음 숫자를 하나씩 올린다.
    문제가 있으면 BeanError를 내고 아무것도 쓰지 않는다."""
    today = today or date.today()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        beans = list(reader)
    row = {name: _text(record.get(name, "")) for name in header}
    for name, label in (("name", "커피 이름"), ("roaster", "로스터리"), ("roast_date", "로스팅 날짜"),
                        ("roast_level", "배전도")):
        if not row[name]:
            raise BeanError(f"{label}을(를) 적어 주세요.")
    try:
        roasted = date.fromisoformat(row["roast_date"])
    except ValueError:
        raise BeanError(f"로스팅 날짜는 2026-10-15처럼 적어 주세요: '{row['roast_date']}'") from None
    if roasted > today:
        raise BeanError("로스팅 날짜가 오늘보다 뒤예요.")
    row["roast_level"] = roast_name(row["roast_level"])
    if row["roast_level"] not in ROAST_LEVELS:
        raise BeanError(f"배전도는 {', '.join(ROAST_LEVELS)} 중 하나로 골라 주세요.")
    if row.get("amount_g") and (_number(row["amount_g"]) is None or _number(row["amount_g"]) < 0):
        raise BeanError(f"양은 g 숫자로 적어 주세요: '{row['amount_g']}'")
    same = (row["name"], row["roaster"], row["roast_date"])
    if any((b["name"], b["roaster"], b["roast_date"]) == same for b in beans):
        raise BeanError("이미 있는 원두예요 (같은 이름·로스터리·로스팅 날짜).")
    numbers = [int(b["bean_id"][1:]) for b in beans if re.fullmatch(r"B\d+", b["bean_id"])]
    row["bean_id"] = f"B{max(numbers, default=0) + 1:02d}"
    row["role"] = row["role"] or "일반"
    with open(path, "rb") as f:
        ends_with_newline = f.read().endswith(b"\n")
    with open(path, "a", encoding="utf-8", newline="") as f:
        if not ends_with_newline:
            f.write("\n")
        csv.writer(f, lineterminator="\n").writerow([row[name] for name in header])
    return row["bean_id"]


def load_beans(path=BEANS_PATH):
    """원두 목록을 {원두 번호: 칸 dict}로 읽는다. 순서는 파일에 적은 순서."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row["bean_id"]: row for row in csv.DictReader(f)}


def next_brew_id(date, brews):
    """그날의 다음 추출번호. 예: 2026-10-16에 이미 세 잔이 있으면 '20261016-4'."""
    day = str(date).replace("-", "")
    taken = [int(b.brew_id.rsplit("-", 1)[1]) for b in brews if re.fullmatch(rf"{day}-\d+", b.brew_id)]
    return f"{day}-{max(taken, default=0) + 1}"


def parse_clock(text):
    """'2:42'(분:초)나 '162'(초)를 초로 읽는다. 비어 있으면 None."""
    text = str(text).strip()
    if not text:
        return None
    match = re.fullmatch(r"(\d+):([0-5]\d)", text)
    if match:
        return int(match.group(1)) * 60 + int(match.group(2))
    seconds = _number(text)
    if seconds is None or seconds <= 0:
        raise BrewLogError(f"물 빠짐 종료 시각은 2:42(분:초)나 162(초)처럼 적어 주세요: '{text}'")
    return seconds


def append_brew(record, path=BREWS_PATH):
    """새 잔을 확인한 뒤 기록 끝에 더한다. 문제가 있으면 BrewLogError를 내고 아무것도 쓰지 않는다."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        existing = {row["brew_id"] for row in reader}

    unknown = set(record) - set(header)
    if unknown:
        raise BrewLogError(f"기록에 없는 칸이에요: {', '.join(sorted(unknown))}")
    row = {name: _text(record.get(name, "")) for name in header}
    _check(row, existing)
    if not row["brew_water_g"]:
        row["brew_water_g"] = _text(sum(p.grams for p in parse_pours(row["pours"])))

    with open(path, "rb") as f:
        ends_with_newline = f.read().endswith(b"\n")
    with open(path, "a", encoding="utf-8", newline="") as f:
        if not ends_with_newline:
            f.write("\n")
        csv.writer(f, lineterminator="\n").writerow([row[name] for name in header])


def _check(row, existing):
    for name, label in REQUIRED_FIELDS:
        if not row[name]:
            raise BrewLogError(f"{label}을(를) 적어 주세요.")
    if row["brew_id"] in existing:
        raise BrewLogError(f"이미 있는 추출번호예요: {row['brew_id']}. 같은 날 두 번째 잔은 -2처럼 번호를 올려 주세요.")

    numbers = {}
    for name, label, zero_ok in NUMBER_FIELDS:
        if not row[name]:
            continue
        value = _number(row[name])
        if value is None:
            raise BrewLogError(f"{label}은(는) 숫자로 적어 주세요: '{row[name]}'")
        if value < 0 or (value == 0 and not zero_ok):
            raise BrewLogError(f"{label}은(는) 0보다 커야 해요: '{row[name]}'")
        numbers[name] = value
    if numbers["water_temp_c"] > 100:
        raise BrewLogError(f"물 온도는 100 °C 이하로 적어 주세요: '{row['water_temp_c']}'")

    try:
        pours = parse_pours(row["pours"])
    except RecipeError as error:
        raise BrewLogError(str(error)) from None
    poured = sum(p.grams for p in pours)
    if "brew_water_g" in numbers and abs(numbers["brew_water_g"] - poured) > 0.5:
        raise BrewLogError(f"부은 물 양({row['brew_water_g']} g)이 붓기 기록의 합({poured:g} g)과 달라요.")

    filled = [name for name, _, _ in TASTE_ITEMS if row[name]]
    if filled and len(filled) < len(TASTE_ITEMS):
        raise BrewLogError("맛 점수는 6항목을 모두 적거나 모두 비워 주세요.")
    for name, label, low in TASTE_ITEMS:
        if not row[name]:
            continue
        score = _number(row[name])
        if score is None or not low <= score <= TASTE_MAX:
            raise BrewLogError(f"맛 점수 {label}은(는) {low}~{TASTE_MAX} 사이로 적어 주세요: '{row[name]}'")
        if score * 2 != round(score * 2):
            raise BrewLogError(f"맛 점수는 0.5 단위로 적어 주세요: {label} '{row[name]}'")

    if row["flag"] and row["flag"] not in FLAGS:
        raise BrewLogError(f"표시는 {', '.join(FLAGS)} 중 하나로 적거나 비워 주세요: '{row['flag']}'")


def _brew_from_row(row):
    clicks = _number(row.get("grinder_clicks", ""))
    return Brew(
        brew_id=row["brew_id"],
        date=row["date"],
        bean_id=row["bean_id"],
        filter_paper=row.get("filter", ""),
        grinder=row.get("grinder", ""),
        grinder_clicks=None if clicks is None else int(clicks),
        grind_d50_um=_number(row.get("grind_d50_um", "")),
        dose_g=_number(row["dose_g"]),
        water_temp_c=_number(row["water_temp_c"]),
        pours=row["pours"],
        bypass_g=_number(row.get("bypass_g", "")) or 0.0,
        drawdown_end_s=_number(row.get("drawdown_end_s", "")),
        beverage_g=_number(row.get("beverage_g", "")),
        taste={name: _number(row.get(name, "")) for name, _, _ in TASTE_ITEMS},
        flag=row.get("flag", ""),
        memo=row.get("memo", ""),
        row=row,
    )


def _number(text):
    """칸의 글자를 숫자로 바꾼다. 비었거나 숫자가 아니면 None."""
    try:
        value = float(str(text).strip())
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _text(value):
    """저장할 칸의 글자. 빈 값(None)은 빈칸으로, 숫자는 20.0이 아니라 20처럼 짧게 적는다."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value).strip()
