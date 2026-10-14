"""레시피와 분쇄도로 물 빠짐 시간, 음료 무게, 추출수율, 농도를 예측하는 추출 모델.

식과 상수의 근거는 docs/MODEL.md와 docs/research/ 조사 노트에 있다. 식 번호 (1)~(12)는 MODEL.md와 같다.

계산 순서 (0.5초 간격)
1. 붓기 일정대로 드리퍼에 물이 들어온다.
2. 드리퍼 안 액체 중 가루가 머금는 양을 뺀 나머지가 가루층 위에 고이고, 고인 높이에 따라 흘러나간다 (다르시 법칙).
3. 크기 구간마다 '표면에서 바로 나오는 성분 + 입자 속에서 천천히 나오는 성분'이 녹아 나와 드리퍼 안 액체에 섞인다.
4. 흘러나간 액체는 그때의 농도만큼 녹은 성분을 가지고 나간다.
5. 마지막 붓기가 끝난 뒤 고인 물이 0이 되면(물줄기가 끊기는 순간) 드리퍼를 뺀 것으로 보고 끝낸다.
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np

from dripcoffee.grind_analysis import volume_percentile

GRAVITY = 9.81  # m/s²
GAS_CONSTANT = 8.314  # J/(mol·K)

# 물 점도(Pa·s)와 밀도(kg/m³). NIST WebBook(IAPWS 식) 값 (노트1 F7). 표 밖의 온도는 가장 가까운 두 점으로 늘려 쓴다.
VISCOSITY_TABLE = ((80, 3.5405e-4), (82, 3.4539e-4), (86, 3.2914e-4), (90, 3.1418e-4), (94, 3.0037e-4), (96, 2.9386e-4))
DENSITY_TABLE = ((82, 970.53), (90, 965.31), (96, 961.19))


class RecipeError(ValueError):
    """레시피를 계산할 수 없을 때, 사용자에게 보여줄 이유를 담는 예외."""


@dataclass(frozen=True)
class Pour:
    start_s: float  # 이 붓기를 시작한 시각 (첫 물이 닿은 때가 0초)
    grams: float  # 이 회차에 부은 물 (g)


@dataclass
class Recipe:
    dose_g: float  # 원두 무게
    water_temp_c: float  # 주전자 물 온도
    pours: list  # Pour 목록
    bypass_g: float = 0.0  # 추출 뒤 가수량

    @property
    def water_g(self):
        return sum(p.grams for p in self.pours)


@dataclass
class Grind:
    """크기 구간별 부피 비율로 나타낸 분쇄도. 사진에 안 잡히는 미분은 모델에서 따로 더한다."""

    diameters_um: np.ndarray  # 구간 대표 지름
    volume_fractions: np.ndarray  # 구간 부피 비율 (합이 1)
    d50_um: float  # 부피 기준 대표 크기


@dataclass(frozen=True)
class ModelParams:
    """모델 상수. 값과 근거, 상태(미확인/가정/보정)는 docs/MODEL.md 4절 표와 같다."""

    # 물 흐름 (노트1)
    flow_resistance_ref: float = 1.57e12  # R_ref (m⁻³): 대표 크기 1000 µm, 원두 20 g 기준. 보정 계수 1. 시작값은 첫 B02 기록(96 °C, 2:20)에 맞춤
    resistance_d_ref_um: float = 1000.0
    resistance_dose_ref_g: float = 20.0
    held_liquid_ratio: float = 2.1  # R_abs (g/g): 가루가 머금는 액체. 보정 계수 2 (노트3 M2)
    cone_half_angle_deg: float = 30.0  # V60 원뿔 반각 (가정 → 측정 예정)
    bulk_density_g_per_ml: float = 0.35  # 마른 가루 부피 밀도 (가정 → 측정 예정)
    pour_rate_g_per_s: float = 5.0  # 붓는 속도 (가정 → 측정 예정)
    # 추출 (노트2)
    max_extractable: float = 0.30  # E_max: 녹아 나올 수 있는 성분 비율 (K8, K5)
    slow_rate_ref_per_s: float = 2.67e-3  # k_ref: 입자 속 성분의 추출 속도 (K2 카페인, 반지름 925 µm, 80 °C)
    slow_rate_r_ref_um: float = 925.0
    slow_rate_t_ref_c: float = 80.0
    activation_energy_j_per_mol: float = 17.3e3  # Ea (K3, 물속 카페인 확산. 커피 전체 성분 값이 없어 가정)
    surface_shell_um: float = 30.0  # δ: 바로 나오는 표면층 두께 (K4 세포 크기)
    fines_fraction: float = 0.15  # 사진에 안 잡히는 100 µm 미만 미분의 부피 비율 (K6, 다른 그라인더 값)
    fines_diameter_um: float = 50.0
    slurry_temp_drop_c: float = 8.0  # 주전자와 드리퍼 안 물의 온도 차이 (노트4 Batali 표, 가정)
    # 계산
    time_step_s: float = 0.5
    max_time_s: float = 3600.0  # 계산을 멈추는 안전장치 (물리적 한계가 아님)


DEFAULT_PARAMS = ModelParams()


@dataclass
class Prediction:
    drawdown_end_s: float  # 물줄기가 끊기는 시각 (드리퍼를 뺌)
    beverage_g: float  # 가수 전 음료 무게
    held_liquid_g: float  # 드리퍼를 뺄 때 가루가 머금고 있는 액체
    released_g: float  # 원두에서 녹아 나온 성분 전체
    dissolved_g: float  # 그중 음료로 나간 성분
    ey_pct: float  # 추출수율 = 음료로 나간 성분 ÷ 원두 무게
    tds_pct: float  # 가수 전 농도
    tds_cup_pct: float  # 가수 후 농도
    timeline: dict = field(default_factory=dict)  # 1초 간격 기록: 시각, 고인 물, 나온 음료 (그래프용)


def parse_pours(text):
    """붓기 기록 '시작초:g;시작초:g;...'를 Pour 목록으로 읽는다. 예: '0:40;30:60;60:60;90:60'"""
    parts = [p.strip() for p in str(text).split(";") if p.strip()]
    if not parts:
        raise RecipeError("붓기 기록이 비어 있어요. '시작초:g'를 ;로 이어 적어 주세요. 예: 0:40;30:60")
    pours = []
    for part in parts:
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", part)
        if not match:
            raise RecipeError(f"붓기 기록을 읽을 수 없어요: '{part}'. '시작초:g' 형식으로 적어 주세요. 예: 0:40;30:60")
        start, grams = float(match.group(1)), float(match.group(2))
        if grams <= 0:
            raise RecipeError(f"붓기 기록의 물 양은 0보다 커야 해요: '{part}'")
        if pours and start <= pours[-1].start_s:
            raise RecipeError(f"붓기 기록의 시작 시각은 앞 붓기보다 늦어야 해요: '{part}'")
        pours.append(Pour(start, grams))
    return pours


def grind_from_d50(d50_um, spread=1.5, n_bins=25):
    """대표 크기만 알 때 쓰는 분쇄도: 부피 기준 로그 정규 분포.

    spread는 '대부분의 범위(D16~D84)'의 기하 표준편차다. 실제 사진 측정에서 1.46(95클릭 카메라),
    1.55(95클릭 폰)가 나와서 1.5로 둔다.
    """
    log_sigma = math.log(spread)
    edges = np.exp(np.linspace(math.log(d50_um) - 3 * log_sigma, math.log(d50_um) + 3 * log_sigma, n_bins + 1))
    centers = np.sqrt(edges[:-1] * edges[1:])
    weights = np.exp(-0.5 * ((np.log(centers) - math.log(d50_um)) / log_sigma) ** 2)
    return Grind(centers, weights / weights.sum(), float(d50_um))


def grind_from_diameters(diameters_um, n_bins=30, lo_um=100.0, hi_um=4000.0):
    """사진으로 잰 입자 지름들(분쇄 측정 저장값)로 만드는 분쇄도."""
    d = np.asarray(diameters_um, dtype=float)
    if d.size < 3:
        raise RecipeError("분쇄 측정에 입자가 너무 적어요. 사진을 더 넣어 측정해 주세요.")
    edges = np.exp(np.linspace(math.log(lo_um), math.log(hi_um), n_bins + 1))
    idx = np.clip(np.searchsorted(edges, d, side="right") - 1, 0, n_bins - 1)
    volume = np.bincount(idx, weights=d**3, minlength=n_bins)
    keep = volume > 0
    centers = np.sqrt(edges[:-1] * edges[1:])
    return Grind(centers[keep], volume[keep] / volume.sum(), volume_percentile(d, 50))


def water_viscosity(temp_c):
    return _log_interp(temp_c, VISCOSITY_TABLE)


def water_density(temp_c):
    temps, values = zip(*DENSITY_TABLE)
    if temp_c <= temps[0] or temp_c >= temps[-1]:  # 표 밖은 양 끝 두 점의 직선으로
        i = 0 if temp_c <= temps[0] else len(temps) - 2
        slope = (values[i + 1] - values[i]) / (temps[i + 1] - temps[i])
        return values[i] + slope * (temp_c - temps[i])
    return float(np.interp(temp_c, temps, values))


def _log_interp(temp_c, table):
    """점도는 온도에 대해 로그가 거의 직선이라 로그 값으로 보간한다. 표 밖은 가장 가까운 두 점으로 늘린다."""
    temps = [t for t, _ in table]
    logs = [math.log(v) for _, v in table]
    i = int(np.clip(np.searchsorted(temps, temp_c) - 1, 0, len(temps) - 2))
    slope = (logs[i + 1] - logs[i]) / (temps[i + 1] - temps[i])
    return math.exp(logs[i] + slope * (temp_c - temps[i]))


def _release_curve(grind, dose_g, slurry_temp_c, p):
    """첫 물이 닿은 뒤 t초 동안 원두에서 녹아 나온 성분(g)을 돌려주는 함수를 만든다 (식 4~7).

    크기 구간마다 '표면에서 바로 나오는 몫 + 입자 속에서 천천히 나오는 몫'을 더하고,
    사진에 안 잡히는 미분은 따로 한 구간으로 넣는다.
    """
    diameters = np.append(grind.diameters_um, p.fines_diameter_um)
    fractions = np.append(grind.volume_fractions * (1 - p.fines_fraction), p.fines_fraction)
    shell = np.clip(1 - (1 - 2 * p.surface_shell_um / diameters) ** 3, 0, 1)
    temp_factor = math.exp(
        -p.activation_energy_j_per_mol / GAS_CONSTANT
        * (1 / (slurry_temp_c + 273.15) - 1 / (p.slow_rate_t_ref_c + 273.15))
    )
    slow_rates = p.slow_rate_ref_per_s * (p.slow_rate_r_ref_um / (diameters / 2)) ** 2 * temp_factor

    def released_at(t):
        released_fraction = shell + (1 - shell) * (1 - np.exp(-slow_rates * t))
        return p.max_extractable * dose_g * float(np.sum(fractions * released_fraction))

    return released_at


def extraction_only(grind, dose_g, water_g, water_temp_c, contact_s, params=DEFAULT_PARAMS):
    """물 흐름 없이 추출만 계산한다. 드리퍼가 아닌 기구(배치 브루어 등)의 문헌 데이터와 비교할 때 쓴다.

    contact_s 동안 녹아 나온 성분이 액체 전체에 고루 섞여 있다고 보고, 가루가 머금은 액체에 남는 몫을 뺀
    나머지가 음료로 나간다고 계산한다. (추출수율 %, 가수 전 농도 %)를 돌려준다.
    """
    p = params
    released = _release_curve(grind, dose_g, water_temp_c - p.slurry_temp_drop_c, p)(contact_s)
    liquid = water_g + released
    beverage = liquid - p.held_liquid_ratio * dose_g
    if beverage <= 0:
        raise RecipeError("부은 물이 가루가 머금는 양보다 적어서 음료가 나오지 않아요.")
    dissolved = released * beverage / liquid
    return 100 * dissolved / dose_g, 100 * dissolved / beverage


def predict(recipe, grind, params=DEFAULT_PARAMS):
    """레시피와 분쇄도로 추출 결과를 예측한다."""
    if recipe.dose_g <= 0:
        raise RecipeError("원두 무게는 0보다 커야 해요.")
    p = params
    dose = recipe.dose_g
    held = p.held_liquid_ratio * dose
    if recipe.water_g <= held:
        raise RecipeError(f"부은 물({recipe.water_g:.0f} g)이 가루가 머금는 양(약 {held:.0f} g)보다 적어서 음료가 나오지 않아요.")

    temp_c = recipe.water_temp_c - p.slurry_temp_drop_c
    rho = water_density(temp_c)  # kg/m³
    mu = water_viscosity(temp_c)
    rho_g_per_ml = rho / 1000.0

    # 물 흐름: 원뿔 모양 드리퍼 (식 1~3)
    tan2 = math.tan(math.radians(p.cone_half_angle_deg)) ** 2
    bed_ml = dose / p.bulk_density_g_per_ml
    resistance = (
        p.flow_resistance_ref
        * (p.resistance_d_ref_um / grind.d50_um) ** 2
        * (dose / p.resistance_dose_ref_g)  # 가루층이 길수록 저항이 크다 (다르시: 저항 ∝ 층 길이)
    )

    def outflow_g_per_s(free_g):
        volume_ml = bed_ml + free_g / rho_g_per_ml
        height_m = (3 * volume_ml / (math.pi * tan2)) ** (1 / 3) / 100  # 물 표면 높이 (식 1)
        return rho * GRAVITY * height_m / (mu * resistance) * 1e6 * rho_g_per_ml  # 식 2, m³/s → g/s

    released_at = _release_curve(grind, dose, temp_c, p)

    pours = recipe.pours
    last_pour_end = max(pour.start_s + pour.grams / p.pour_rate_g_per_s for pour in pours)
    dt = p.time_step_s
    liquid = solute = beverage = dissolved = released = 0.0
    timeline = {"time_s": [], "free_g": [], "beverage_g": []}
    t = 0.0
    while True:
        # 1. 붓기
        poured = sum(
            max(0.0, min(t + dt, pour.start_s + pour.grams / p.pour_rate_g_per_s) - max(t, pour.start_s))
            * p.pour_rate_g_per_s
            for pour in pours
        )
        liquid += poured
        # 3. 추출 (첫 물이 닿은 뒤부터)
        new_released = released_at(t + dt)
        solute += new_released - released
        liquid += new_released - released
        released = new_released
        # 2, 4. 흘러나감
        free = liquid - held
        if free > 0:
            out = min(free, outflow_g_per_s(free) * dt)
            out_solute = out * solute / liquid
            liquid -= out
            solute -= out_solute
            beverage += out
            dissolved += out_solute
        t += dt
        if abs(t - round(t)) < 1e-9:
            timeline["time_s"].append(round(t))
            timeline["free_g"].append(max(0.0, liquid - held))
            timeline["beverage_g"].append(beverage)
        # 5. 마지막 붓기 뒤 고인 물이 다 빠지면 끝 (물줄기가 끊기는 순간)
        if t >= last_pour_end and liquid - held <= 1e-6:
            break
        if t >= p.max_time_s:
            raise RecipeError(f"{p.max_time_s / 60:.0f}분이 지나도 물이 다 빠지지 않아요. 분쇄가 너무 곱거나 물이 너무 많아요.")

    return Prediction(
        drawdown_end_s=t,
        beverage_g=beverage,
        held_liquid_g=liquid,
        released_g=released,
        dissolved_g=dissolved,
        ey_pct=100 * dissolved / dose,
        tds_pct=100 * dissolved / beverage,
        tds_cup_pct=100 * dissolved / (beverage + recipe.bypass_g),
        timeline=timeline,
    )
