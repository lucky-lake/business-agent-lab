"""추출 기록(data/brews.csv) 읽기·쓰기와 입력 확인.

앱의 기록 탭은 append_brew로 새 잔을 저장하고, 보정(MODEL.md 5절)과 내 선호 영역(9.1절)은
load_brews로 읽은 기록을 쓴다. 측정 방법과 맛 기준은 docs/BREW_PROTOCOL.md를 따른다.
"""
import csv
import math
import re
from dataclasses import dataclass, field

from dripcoffee import DATA_DIR
from dripcoffee.extraction_model import Recipe, RecipeError, parse_pours

BREWS_PATH = DATA_DIR / "brews.csv"
BEANS_PATH = DATA_DIR / "beans.csv"

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
REQUIRED_FIELDS = (("brew_id", "추출번호"), ("date", "날짜"), ("bean_id", "원두 번호"),
                   ("dose_g", "원두 무게"), ("water_temp_c", "물 온도"), ("pours", "붓기 기록"))


class BrewLogError(ValueError):
    """기록을 저장할 수 없을 때, 사용자에게 보여줄 이유를 담는 예외."""


@dataclass
class Brew:
    """brews.csv의 한 줄. 계산에 쓰는 칸은 숫자로 바꿔 두고, 나머지는 row에 그대로 둔다."""

    brew_id: str
    date: str
    bean_id: str
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
