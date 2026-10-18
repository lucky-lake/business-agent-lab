"""핸드드립 추출 분석·추천 도구의 화면 (Streamlit).

화면은 입력과 표시만 맡고, 계산은 dripcoffee 패키지에서 한다.
디자인은 토스(Toss) 스타일을 따른다: 흰 배경, 포인트 컬러 하나(파랑)와 무채색, 은은한 그림자,
큰 숫자와 작은 단위·설명, 테두리와 구분선 대신 넓은 여백. 색과 글꼴은 .streamlit/config.toml에 있다.
실행: streamlit run app.py
"""
import math
from datetime import date
from html import escape

import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from dripcoffee import DATA_DIR, LOCAL_DATA_DIR
from dripcoffee import calibration as cal
from dripcoffee import grind_analysis as ga

PROFILE_DIR = LOCAL_DATA_DIR / "grind_profiles"
CLICK_TABLE = DATA_DIR / "reference" / "maverick_click_table.csv"
BREWS_PATH = cal.BREWS_PATH

BRAND = "#3182F6"  # 포인트 컬러는 이것 하나만 쓴다
TEXT = "#2A2A2A"
GRAY_DARK = "#4E5968"
GRAY = "#8B95A1"
GRAY_LIGHT = "#B0B8C1"
FONT = "Pretendard Variable, sans-serif"

# 사진 위 윤곽선 (RGB): 잰 입자만 포인트 컬러로 그리고, 뺀 입자(덩어리, 칸 테두리)는 회색으로 낮춘다.
OUTLINE_COLORS = {"measured": (49, 130, 246), "clump": (139, 149, 161), "border": (139, 149, 161)}
SIZE_TICKS = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]

# 맛 6항목의 설명 (docs/BREW_PROTOCOL.md '맛 기준'과 같다). 무엇을 매기는지, 점수 기준.
TASTE_GUIDE = {
    "aroma": ("마시기 전에 맡는 향의 만족도", "1 향이 거의 없거나 불쾌함 · 3 무난한 커피 향 · 5 뚜렷하고 좋은 향"),
    "flavor": ("머금었을 때 맛과 향의 만족도", "1 밍밍하거나 쓴맛·탄 맛이 덮음 · 3 커피다운 맛, 특징은 약함 · 5 선명하고 풍부함"),
    "acidity": ("산미의 세기", "0 전혀 없음 · 1 거의 안 느껴짐 · 3 적당히 느껴짐 · 5 아주 강함"),
    "body": ("입안의 무게감·질감의 세기", "0 전혀 없음 · 1 물처럼 가벼움 · 3 저지방 우유 정도 · 5 진한 우유처럼 묵직함"),
    "balance": ("단맛·산미·쓴맛·바디가 어울리는 정도", "1 한 가지가 튀어 거슬림 · 3 대체로 괜찮지만 한쪽으로 치우침 · 5 잘 어우러짐"),
    "aftertaste": ("삼키고 5초쯤 뒤 남는 맛의 만족도", "1 금방 사라지거나 떫고 쓴 맛이 남음 · 3 무난하게 남다 사라짐 · 5 좋은 맛이 길게 남음"),
}
UNRATED = "–"  # 아직 매기지 않은 맛 항목
# 원두 목록(beans.csv)의 칸 → 추출 기록(brews.csv)의 칸. 원두를 고르면 이 칸들을 채운다.
BEAN_TO_BREW = {"name": "bean_name", "roaster": "roaster", "origin": "origin", "process": "process",
                "roast_level": "roast_level"}

STYLE = """
<style>
.block-container { max-width: 980px; padding-top: 3.5rem; padding-bottom: 6rem; }
.stTabs [data-baseweb="tab-list"] { gap: 28px; }
.stTabs [data-baseweb="tab"] { font-size: 17px; font-weight: 600; padding: 10px 0; }
.stTabs [data-baseweb="tab-border"] { background: #F2F4F6; }
[data-testid="stExpander"] details { border: none; background: #F9FAFB; border-radius: 20px; }
[data-testid="stExpander"] summary { padding: 18px 22px; }
[data-testid="stExpander"] summary p { font-size: 16px; font-weight: 600; }

.dc-eyebrow { font-size: 15px; font-weight: 600; color: #3182F6; margin-bottom: 6px; }
.dc-title { font-size: 32px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; line-height: 1.3; }
.dc-lead { font-size: 16px; color: #4E5968; line-height: 1.6; margin: 4px 0 28px; }
.dc-section { font-size: 22px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; margin: 48px 0 4px; }
.dc-caption { font-size: 14px; color: #8B95A1; margin-bottom: 16px; }

.dc-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0 8px; }
.dc-card { background: #FFFFFF; border-radius: 24px; padding: 28px 28px 24px;
           box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04); animation: dc-rise .5s cubic-bezier(.2, .8, .2, 1) both; }
.dc-card.hero { grid-column: 1 / -1; }
.dc-card:nth-child(2) { animation-delay: .07s; }
.dc-card:nth-child(3) { animation-delay: .14s; }
.dc-label { font-size: 15px; font-weight: 500; color: #6B7684; margin-bottom: 6px; }
.dc-value { font-size: 48px; font-weight: 700; color: #2A2A2A; line-height: 1.2;
            letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
.dc-card.hero .dc-value { color: #3182F6; }
.dc-unit { font-size: 24px; font-weight: 600; color: #4E5968; margin-left: 6px; letter-spacing: 0; }
.dc-sub { font-size: 14px; color: #8B95A1; margin-top: 10px; line-height: 1.5; }

.dc-notice { background: #F2F4F6; border-radius: 16px; padding: 16px 20px;
             font-size: 15px; color: #4E5968; line-height: 1.6; animation: dc-rise .4s ease-out both; }
.dc-notice.error { background: #FFF1F1; color: #2A2A2A; }

.dc-rows { margin-top: 4px; }
.dc-row { display: flex; justify-content: space-between; gap: 12px; padding: 9px 0; font-size: 15px; }
.dc-row span { color: #6B7684; }
.dc-row b { color: #2A2A2A; font-weight: 600; text-align: right; }

.st-key-taste-title h3 { font-size: 22px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em;
                         margin-top: 48px; padding-bottom: 0; }
.dc-hexagon { background: #F9FAFB; border-radius: 24px; padding: 16px 12px 20px; text-align: center; }
.dc-hexagon svg { width: 100%; max-width: 380px; height: auto; }
.dc-hexagon-score { font-size: 15px; color: #6B7684; margin-top: 4px; }
.dc-hexagon-score b { font-size: 24px; font-weight: 700; color: #3182F6; margin-left: 4px; font-variant-numeric: tabular-nums; }
.dc-hexagon .dc-caption { margin: 6px 16px 0; }

@keyframes dc-rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .dc-card, .dc-notice { animation: none; } }
@media (max-width: 640px) { .dc-cards { grid-template-columns: 1fr; } .dc-value { font-size: 40px; } }
</style>
"""


def html(markup):
    st.html(markup)


def notice(text, error=False):
    """경고·안내 문구. 노란 상자 대신 연한 회색 상자로 조용하게 보여준다."""
    html(f'<div class="dc-notice{" error" if error else ""}">{escape(text)}</div>')


def section(title, caption=""):
    html(f'<div class="dc-section">{escape(title)}</div>' + (f'<div class="dc-caption">{escape(caption)}</div>' if caption else ""))


def card(label, value, unit, sub="", hero=False):
    return (
        f'<div class="dc-card{" hero" if hero else ""}"><div class="dc-label">{escape(label)}</div>'
        f'<div class="dc-value">{escape(value)}<span class="dc-unit">{escape(unit)}</span></div>'
        + (f'<div class="dc-sub">{escape(sub)}</div>' if sub else "")
        + "</div>"
    )


def rows(items):
    """'이름 — 값' 목록. 표 테두리 없이 왼쪽 회색 이름, 오른쪽 진한 값으로 보여준다."""
    body = "".join(f"<div class='dc-row'><span>{escape(k)}</span><b>{escape(v)}</b></div>" for k, v in items)
    return f'<div class="dc-rows">{body}</div>'


@st.cache_data(show_spinner=False, max_entries=30)
def analyze_upload(data):
    """사진 바이트를 분석한다. (펴진 사진, 측정 결과, 오류 메시지) 중 성공하면 오류가 None이다."""
    try:
        rectified, measurement = ga.analyze_photo(ga.load_image_bytes(data))
    except ga.GrindPhotoError as e:
        return None, None, str(e)
    return rectified, measurement, None


def outline_image(rectified, measurement):
    """펴진 가루 칸 위에 입자를 색으로 칠한 RGB 이미지.

    화면에서 사진이 작게 줄어도 보이도록 선만 긋지 않고 입자 안을 반투명하게 칠한다.
    잰 입자는 파랗게, 뺀 입자는 회색으로 옅게 만들어 잰 입자에 눈이 가게 한다.
    """
    image = rectified.image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    thickness = max(1, round(max(rgb.shape[:2]) / 500))
    for kind in ("clump", "border", "measured"):  # 잰 입자를 마지막에 칠해 위에 보이게 한다
        outlines = measurement.outlines.get(kind, [])
        if not outlines:
            continue
        mask = np.zeros(rgb.shape[:2], np.uint8)
        cv2.drawContours(mask, outlines, -1, 255, cv2.FILLED)
        color = np.array(OUTLINE_COLORS[kind], dtype=np.float32)
        filled = mask > 0
        rgb[filled] = (0.4 * rgb[filled] + 0.6 * color).astype(np.uint8)
        cv2.drawContours(rgb, outlines, -1, OUTLINE_COLORS[kind], thickness)
    return rgb


def distribution_chart(profile, show_count):
    """종 모양 분포 곡선: 대부분의 범위는 옅게 칠하고, 대표 크기는 점선, 가장 많은 크기는 점으로 표시한다."""
    sizes, density = ga.size_distribution_curve(profile.diameters_um, by_volume=True)
    curve = pd.DataFrame({"size": sizes, "density": density / density.max()})
    low, high = profile.spread_range_um
    x = alt.X(
        "size:Q",
        title="입자 지름 (µm)",
        scale=alt.Scale(type="log", domain=list(ga.CURVE_RANGE_UM)),
        axis=alt.Axis(values=SIZE_TICKS, format="d", grid=False, ticks=False, domainColor="#E5E8EB",
                      labelColor=GRAY, labelFontSize=12, titleColor=GRAY, titleFontSize=13, titleFontWeight=500),
    )
    y = alt.Y("density:Q", axis=None, scale=alt.Scale(domain=[0, 1.2]))
    band = alt.Chart(curve[(curve["size"] >= low) & (curve["size"] <= high)]).mark_area(
        color=BRAND, opacity=0.12
    ).encode(x=x, y=y)
    line = alt.Chart(curve).mark_line(color=BRAND, strokeWidth=3).encode(
        x=x, y=y, tooltip=[alt.Tooltip("size:Q", title="지름 (µm)", format=".0f")]
    )
    # 대표 크기 점선은 바닥에서 곡선까지만 그어 봉우리 이름표와 겹치지 않게 한다.
    d50_height = float(np.interp(np.log10(profile.d50_um), np.log10(curve["size"]), curve["density"]))
    d50 = alt.Chart(pd.DataFrame({"size": [profile.d50_um], "base": [0.0], "top": [d50_height]})).mark_rule(
        color=GRAY_DARK, strokeDash=[4, 4], strokeWidth=1
    ).encode(x=x, y=alt.Y("base:Q", axis=None, scale=alt.Scale(domain=[0, 1.2])), y2="top:Q")
    peak = pd.DataFrame({"size": [profile.mode_um], "density": [1.0], "label": [f"가장 많은 크기 {profile.mode_um:.0f}"]})
    peak_point = alt.Chart(peak).mark_point(color=BRAND, filled=True, size=90, opacity=1).encode(x=x, y=y)
    # 봉우리 이름표는 점 바로 위 가운데에 둔다 (좁은 화면에서도 오른쪽이 잘리지 않게).
    peak_text = alt.Chart(peak).mark_text(
        align="center", dy=-18, fontSize=13, color=GRAY_DARK, font=FONT
    ).encode(x=x, y=y, text="label:N")
    layers = band + line + d50 + peak_point + peak_text
    if show_count:
        count_curve = ga.size_distribution_curve(profile.diameters_um, by_volume=False)
        if count_curve is not None:
            count = pd.DataFrame({"size": count_curve[0], "density": count_curve[1] / count_curve[1].max()})
            layers += alt.Chart(count).mark_line(color=GRAY_LIGHT, strokeDash=[3, 3], strokeWidth=2).encode(x=x, y=y)
    return layers.properties(height=300).configure_view(stroke=None).configure(font=FONT)


def click_table_size(clicks):
    """메버릭 클릭표(인터넷 기준, 비교용)에서 클릭 수에 해당하는 크기. 표 범위 밖이면 None."""
    table = pd.read_csv(CLICK_TABLE).sort_values("clicks")
    if not table["clicks"].min() <= clicks <= table["clicks"].max():
        return None
    return float(np.interp(clicks, table["clicks"], table["size_um"]))


def show_photo_result(rectified, measurement):
    """사진 한 장의 결과: 윤곽선을 그린 가루 칸 사진과 숫자, 경고."""
    left, right = st.columns([3, 2], gap="large")
    left.image(outline_image(rectified, measurement), width="stretch")
    with right:
        measured = len(measurement.diameters_um)
        html(rows([
            ("이 사진의 대표 크기", f"{measurement.d50_volume_um:.0f} µm" if measured else "없음"),
            ("잰 입자", f"{measured}개"),
            ("붙어 있어서 뺀 덩어리", f"{measurement.excluded_clumps}개"),
            ("칸 테두리에 걸려 뺀 입자", f"{measurement.excluded_border}개"),
            ("은피로 보고 뺀 조각", f"{measurement.excluded_chaff}개"),
            ("사용한 칸", f"{rectified.zone.name}칸, 픽셀당 {rectified.um_per_px:.0f} µm"),
        ]))
        for warning in rectified.warnings + measurement.warnings:
            notice(warning)


def show_profile(profile, grinder, clicks):
    """이번 측정(같은 가루로 찍은 사진을 합친 결과)의 요약 카드, 분포 곡선, 자세한 값."""
    low, high = profile.d50_range_um
    spread_low, spread_high = profile.spread_range_um
    html(
        '<div class="dc-cards">'
        + card("대표 크기", f"{profile.d50_um:.0f}", "µm",
               f"가루 무게의 절반이 이보다 작아요. 같은 가루를 다시 찍으면 {low:.0f}~{high:.0f} µm 사이로 나올 수 있어요.",
               hero=True)
        + card("가장 많은 크기", f"{profile.mode_um:.0f}", "µm", "분포 곡선에서 가장 높은 곳이에요.")
        + card("대부분의 범위", f"{spread_low:.0f}~{spread_high:.0f}", "µm",
               "가루 무게의 68%가 여기에 있어요. 평균 ± 표준편차 범위와 같아요.")
        + "</div>"
    )

    st.altair_chart(distribution_chart(profile, st.session_state.get("show_count", False)), width="stretch")
    st.checkbox("개수 기준 곡선도 함께 보기 (회색 점선, 참고용)", key="show_count")
    html('<div class="dc-caption">옅게 칠한 구간이 대부분의 범위, 점선이 대표 크기예요. '
         "무게 기준이라 작은 가루가 많아도 곡선에서는 낮게 보여요.</div>")

    for warning in profile.warnings:
        notice(warning)
    if "메버릭" in grinder:
        reference = click_table_size(clicks)
        if reference is not None:
            notice(f"참고: 메버릭 클릭표(인터넷 기준, 비교용)에서 {clicks}클릭은 약 {reference:.0f} µm예요.")

    with st.expander("자세히 보기"):
        html(rows([
            ("D10 · D90", f"{profile.d10_um:.0f} · {profile.d90_um:.0f} µm"),
            ("분포 폭 (D90 − D10) ÷ D50", f"{profile.span:.2f}"),
            ("사진별 대표 크기", ", ".join(f"{v:.0f}" for v in profile.photo_d50_um) + " µm"),
            ("검출 하한 (이보다 작은 가루는 재지 않음)", f"{profile.min_diameter_um:.0f} µm"),
            ("잰 입자 · 사진", f"{profile.n_particles}개 · {profile.n_photos}장"),
            ("뺀 것", f"덩어리 {profile.excluded_clumps}, 테두리 {profile.excluded_border}, 은피 {profile.excluded_chaff}"),
        ]))
        html('<div class="dc-caption">D10은 가루 무게의 10%가 이보다 작은 크기, D90은 10%가 이보다 큰 크기예요.</div>')


def show_saved_profiles():
    records = ga.load_profiles(PROFILE_DIR)
    with st.expander(f"저장한 측정 {len(records)}개"):
        if not records:
            html('<div class="dc-caption">아직 저장한 측정이 없어요.</div>')
            return
        table = pd.DataFrame(
            {
                "저장 시각": r["saved_at"].replace("T", " ")[:16],
                "원두": r["bean_id"],
                "그라인더": r["grinder"],
                "클릭": r["grinder_clicks"],
                "추출번호": r["brew_id"],
                "대표 크기": round(r["d50_um"]),
                "가장 많은 크기": round(r["mode_um"]),
                "대부분의 범위": f"{r['spread_range_um'][0]:.0f}~{r['spread_range_um'][1]:.0f}",
                "입자 수": r["n_particles"],
            }
            for r in reversed(records)
        )
        st.dataframe(table, hide_index=True)


def grind_tab():
    html(
        '<div class="dc-lead">축척 용지의 가루 칸에 원두 가루를 듬성듬성 펴고, 네 모서리 마커가 모두 보이게 찍어 주세요.<br>'
        "같은 가루로 찍은 사진을 여러 장 넣으면 하나로 합쳐져서 결과가 더 안정돼요.</div>"
    )
    c1, c2, c3, c4 = st.columns(4)
    bean_id = c1.text_input("원두 번호", "B01")
    grinder = c2.text_input("그라인더", "메버릭1")
    clicks = c3.number_input("클릭 수", min_value=0, max_value=180, value=95, step=1)
    brew_id = c4.text_input("추출번호 (선택)", "", placeholder="예: 20261012-1")

    source = st.segmented_control("사진 넣는 방법", ["사진 올리기", "카메라로 찍기"], default="사진 올리기")
    if source == "카메라로 찍기":
        notice("브라우저 카메라는 해상도가 낮아 작은 입자를 놓칠 수 있어요. 폰이나 카메라로 찍은 원본을 올리면 더 정확해요.")
        shot = st.camera_input("가루 칸 찍기", resolution="1080p")
        uploads = [shot] if shot else []
    else:
        uploads = st.file_uploader(
            "사진 (여러 장 가능)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
        ) or []

    if not uploads:
        notice("사진을 넣으면 입자를 하나하나 찾아서, 이 가루의 대표 크기와 분포를 보여줘요.")
        show_saved_profiles()
        return

    section("사진별 결과", "파란 선은 잰 입자, 회색 선은 뺀 입자(서로 붙은 덩어리, 칸 테두리에 걸린 입자)예요.")
    measurements, names = [], []
    for upload in uploads:
        with st.spinner(f"{upload.name} 분석 중..."):
            rectified, measurement, error = analyze_upload(upload.getvalue())
        if error:
            notice(f"{upload.name}: {error}", error=True)
            continue
        measured = len(measurement.diameters_um)
        label = f"{upload.name} · " + (f"대표 크기 {measurement.d50_volume_um:.0f} µm" if measured else "잰 입자 없음")
        with st.expander(label, expanded=len(uploads) == 1):
            show_photo_result(rectified, measurement)
        if measured:
            measurements.append(measurement)
            names.append(upload.name)

    if not measurements:
        notice("입자를 잰 사진이 없어요. 가루를 칸 안쪽에 듬성듬성 펴고 다시 찍어 주세요.", error=True)
        show_saved_profiles()
        return

    profile = ga.combine_measurements(measurements)
    section("이번 측정 결과", f"사진 {profile.n_photos}장 · 잰 입자 {profile.n_particles}개")
    if ga.size_distribution_curve(profile.diameters_um) is None:
        notice("잰 입자가 너무 적어 분포를 계산할 수 없어요. 사진을 더 넣어 주세요.", error=True)
        show_saved_profiles()
        return
    show_profile(profile, grinder, int(clicks))

    if st.button("이 측정 저장하기", type="primary", width="stretch"):
        record = ga.profile_record(profile, bean_id, grinder, int(clicks), brew_id, names)
        path = ga.save_profile(record, PROFILE_DIR)
        st.toast(f"저장했어요: {path.name}")
    show_saved_profiles()


def clock(seconds):
    """초를 '2:42'처럼 보여준다."""
    if seconds is None:
        return ""
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def saved_grind(brew_id):
    """분쇄도 탭에서 이 추출번호로 저장한 측정 중 가장 최근 것. 없으면 None."""
    matches = [r for r in ga.load_profiles(PROFILE_DIR) if brew_id and r["brew_id"] == brew_id]
    return matches[-1] if matches else None


def mark_touched(flag):
    st.session_state[flag] = True


def taste_inputs(key):
    """맛 6항목 슬라이더. 맨 왼쪽(산미·바디 0점, 나머지 1점)에서 시작하지만, 한 번 움직이기 전까지는 '–'(안 매김)로
    보이고 점수로 치지 않는다. 그래서 매기지 않은 항목이 맨 왼쪽 점수로 저장되는 일이 없다.
    일부만 매기면 저장할 때 기록 함수가 거절한다."""
    scores = {}
    for name, label, low in cal.TASTE_ITEMS:  # 기록지와 같은 순서
        flag = key(f"{name}-touched")  # 움직이면 콜백이 켠다. 저장하면 key가 바뀌어 다시 꺼진 상태로 시작한다
        rated = st.session_state.get(flag, False)
        options = [low + 0.5 * j for j in range(int((cal.TASTE_MAX - low) * 2) + 1)]
        value = st.select_slider(
            f"{label} · {TASTE_GUIDE[name][0]}", options, value=low,
            format_func=lambda v, rated=rated, low=low: UNRATED if not rated and v == low else f"{v:g}",
            key=key(name), on_change=mark_touched, args=(flag,),
        )
        scores[name] = value if rated else None
    return scores


def taste_hexagon(scores):
    """맛 점수 육각형(SVG). 기록지 순서로 맨 위 Aroma부터 시계 방향으로 돈다.

    매긴 항목은 점으로 찍고, 6항목을 다 매기면 면을 칠한다. 산미·바디는 좋고 나쁨이 아니라 세기라서
    이름을 회색으로 낮추고 '세기'를 붙인다. 그래야 육각형이 클수록 좋은 잔이라고 읽지 않는다.
    """
    width, height, cx, cy, radius = 360, 300, 180, 150, 90

    def at(i, value):
        angle = math.radians(-90 + 60 * i)
        r = radius * value / cal.TASTE_MAX
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    def hexagon(value):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (at(i, value) for i in range(6)))

    parts = [f'<polygon points="{hexagon(5)}" fill="#FFFFFF"/>']
    parts += [f'<polygon points="{hexagon(level)}" fill="none" stroke="#E5E8EB" stroke-width="1"/>' for level in range(1, 6)]
    parts += [f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#E5E8EB" stroke-width="1"/>'
              for x, y in (at(i, 5) for i in range(6))]

    values = [scores.get(name) for name, _, _ in cal.TASTE_ITEMS]
    if all(v is not None for v in values):
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in (at(i, v) for i, v in enumerate(values)))
        parts.append(f'<polygon points="{points}" fill="{BRAND}" fill-opacity="0.15" stroke="{BRAND}" '
                     'stroke-width="2" stroke-linejoin="round"/>')
    parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{BRAND}"/>'
              for x, y in (at(i, v) for i, v in enumerate(values) if v is not None)]

    for i, ((name, label, low), value) in enumerate(zip(cal.TASTE_ITEMS, values)):
        x, y = at(i, 5 * (radius + 22) / radius)
        dx = x - cx
        anchor = "start" if dx > 1 else "end" if dx < -1 else "middle"
        name_y = y - 14 if y < cy - radius else y + 10 if y > cy + radius else y - 2  # 위·아래·옆 자리
        intensity = low == 0  # 산미·바디
        name_text = f"{label} · 세기" if intensity else label
        parts.append(f'<text x="{x:.1f}" y="{name_y:.1f}" text-anchor="{anchor}" font-size="14" font-weight="600" '
                     f'fill="{GRAY if intensity else GRAY_DARK}">{escape(name_text)}</text>')
        shown, color = (UNRATED, GRAY_LIGHT) if value is None else (f"{value:g}", BRAND)
        parts.append(f'<text x="{x:.1f}" y="{name_y + 19:.1f}" text-anchor="{anchor}" font-size="17" '
                     f'font-weight="700" fill="{color}">{shown}</text>')

    quality = [scores.get(name) for name in cal.QUALITY_ITEMS]
    score = f"{sum(quality) / len(quality):.2f}" if all(v is not None for v in quality) else UNRATED
    svg = (f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
           f'font-family="{FONT}" role="img" aria-label="맛 점수 육각형">{"".join(parts)}</svg>')
    return (f'<div class="dc-hexagon">{svg}<div class="dc-hexagon-score">만족도<b>{score}</b></div>'
            '<div class="dc-caption">Aroma, Flavor, Balance, Aftertaste의 평균이에요. '
            "산미·바디는 세기라서 클수록 좋은 게 아니에요.</div></div>")


def taste_section(key):
    """맛 입력: 왼쪽 슬라이더로 매기면 오른쪽 육각형이 바로 따라 그려진다. 점수 기준은 제목 옆 '?' 하나에 모은다."""
    guide = "\n".join(f"- **{label}** ({TASTE_GUIDE[name][0]}): {TASTE_GUIDE[name][1]}"
                      for name, label, _ in cal.TASTE_ITEMS)
    st.container(key="taste-title").subheader("맛", help=guide, anchor=False)
    html('<div class="dc-caption">가수한 잔 기준, 0.5 단위예요. 3점을 보통으로 두고 올리거나 내려요. '
         "움직이기 전에는 '–'(안 매김)로 보여요. 맨 왼쪽 점수(0점이나 1점)를 주려면 한 번 움직였다가 돌려놓으세요. "
         "6항목을 모두 매기거나 모두 '–'로 두세요.</div>")
    left, right = st.columns(2, gap="large")
    with left:
        scores = taste_inputs(key)
    with right:
        st.markdown(taste_hexagon(scores), unsafe_allow_html=True)  # st.html은 SVG를 지워서 markdown으로 넣는다
    return scores


def recent_brews_table(brews, limit=8):
    table = pd.DataFrame(
        {
            "추출번호": b.brew_id,
            "원두": f"{b.bean_id} {b.row.get('bean_name', '')}".strip(),
            "클릭": b.grinder_clicks,
            "대표 크기 (µm)": None if b.grind_d50_um is None else round(b.grind_d50_um),
            "온도 (°C)": f"{b.water_temp_c:g}",
            "물 빠짐": clock(b.drawdown_end_s),
            "음료 (g)": b.beverage_g,
            "만족도": None if b.quality_score is None else round(b.quality_score, 2),
            "표시": b.flag,
        }
        for b in reversed(brews[-limit:])
    )
    st.dataframe(table, hide_index=True)


def log_tab():
    html(
        '<div class="dc-lead">한 잔을 내린 뒤 결과를 적어 주세요. 저장하면 추출 기록 끝에 더해지고, '
        "모델 보정과 내 선호 영역 계산에 쓰여요.<br>측정 방법과 맛 기준은 docs/BREW_PROTOCOL.md와 같아요.</div>"
    )
    saved = st.session_state.pop("log_saved", None)
    if saved:
        notice(saved)

    brews = cal.load_brews(BREWS_PATH)
    beans = cal.load_beans()
    form = st.session_state.setdefault("log_form", 0)  # 저장할 때마다 올려서 입력칸을 새로 만든다

    def key(name):
        return f"log-{form}-{name}"

    section("잔 정보")
    c1, c2, c3 = st.columns(3)
    brew_date = c1.date_input("날짜", value=date.today(), key=key("date"))
    brew_id = c2.text_input("추출번호", cal.next_brew_id(brew_date.isoformat(), brews), key=key(f"id-{brew_date}"))
    bean_ids = list(beans)
    last_bean = brews[-1].bean_id if brews and brews[-1].bean_id in beans else bean_ids[0]
    bean_id = c3.selectbox(
        "원두", bean_ids, index=bean_ids.index(last_bean),
        format_func=lambda b: f"{b} {beans[b]['name']} ({beans[b]['role']})", key=key("bean"),
    )
    bean = beans[bean_id]
    same_bean = [b for b in brews if b.bean_id == bean_id]
    previous = same_bean[-1] if same_bean else (brews[-1] if brews else None)  # 입력칸 기본값은 지난 잔
    if bean["role"] == "시험":
        notice("시험용 원두예요. 모델을 동결한 뒤 최종 시험에서만 써요 (MODEL.md 6절).")

    c1, c2, c3, c4 = st.columns(4)
    roast_date = c1.text_input("로스팅 날짜", bean["roast_date"], key=key(f"roast-{bean_id}"))
    dripper = c2.text_input("드리퍼", previous.row.get("dripper", "V60") if previous else "V60", key=key("dripper"))
    grinder = c3.text_input("그라인더", previous.grinder if previous else "메버릭1", key=key("grinder"))
    clicks = c4.number_input("클릭 수", min_value=0, max_value=180, step=1,
                             value=previous.grinder_clicks if previous and previous.grinder_clicks is not None else 95,
                             key=key(f"clicks-{bean_id}"))

    grind = saved_grind(brew_id)
    d50 = st.number_input("분쇄 대표 크기 (µm, 선택)", min_value=0.0, step=1.0, format="%.0f",
                          value=float(round(grind["d50_um"])) if grind else None, key=key(f"d50-{brew_id}"))
    if grind:
        html(f'<div class="dc-caption">분쇄도 탭에서 이 추출번호로 저장한 측정을 불러왔어요 '
             f"(사진 {grind['n_photos']}장, {grind['grinder_clicks']}클릭).</div>")
    else:
        html('<div class="dc-caption">이 추출번호로 저장한 분쇄 측정이 없어요. 비워 두면 같은 원두·같은 클릭으로 간 '
             "다른 잔의 값을 빌려 써요.</div>")

    section("레시피")
    c1, c2, c3 = st.columns(3)
    dose = c1.number_input("원두 무게 (g)", min_value=0.0, step=0.1, format="%.1f",
                           value=previous.dose_g if previous else 20.0, key=key(f"dose-{bean_id}"))
    temp = c2.number_input("물 온도 (°C)", min_value=0.0, max_value=100.0, step=1.0, format="%.0f",
                           value=previous.water_temp_c if previous else 92.0, key=key(f"temp-{bean_id}"))
    bypass = c3.number_input("가수량 (g)", min_value=0.0, step=1.0, format="%.0f",
                             value=previous.bypass_g if previous else 0.0, key=key(f"bypass-{bean_id}"))
    c1, c2 = st.columns([2, 1])
    pours = c1.text_input("붓기 기록 (시작초:g)", previous.pours if previous else "0:40;30:60;60:60;90:60",
                          help="각 붓기를 시작한 시각(초)과 그 회차에 부은 양(g)을 ;로 이어 적어요. 예: 0:40;30:60;60:60;90:60",
                          key=key(f"pours-{bean_id}"))
    recipe_name = c2.text_input("레시피 이름 (선택)", previous.row.get("recipe", "") if previous else "",
                                key=key(f"recipe-{bean_id}"))

    section("결과", "드리퍼를 뺀 시각과, 가수하기 전에 잰 음료 무게예요.")
    c1, c2 = st.columns(2)
    drawdown = c1.text_input("물 빠짐 종료 시각", placeholder="예: 2:42", key=key("drawdown"),
                             help="물줄기가 끊기고 방울로 바뀌어 드리퍼를 뺀 순간이에요. 분:초나 초로 적어요.")
    beverage = c2.number_input("음료 무게 (g, 가수 전)", min_value=0.0, step=0.1, format="%.1f", value=None,
                               key=key("beverage"))

    taste = taste_section(key)
    memo = st.text_area("메모", key=key("memo"),
                        placeholder="가수 전/후 차이, 앞 잔과 비교한 느낌, 물 고임. "
                                    "쓰는 말: 시큼함, 밍밍함, 떫음, 쓴맛 강함, 탄 맛, 단맛 좋음")
    flag = st.selectbox("표시", ["", *cal.FLAGS], format_func=lambda f: f or "없음 (정상)", key=key("flag"),
                        help="정상이 아닌 잔에 표시하면 보정과 내 선호 영역에서 빠져요.")

    if st.button("이 잔 저장하기", type="primary", width="stretch"):
        try:
            record = {
                "brew_id": brew_id, "date": brew_date.isoformat(), "bean_id": bean_id,
                **{brew_col: bean[bean_col] for bean_col, brew_col in BEAN_TO_BREW.items()},
                "roast_date": roast_date, "dripper": dripper, "grinder": grinder, "grinder_clicks": int(clicks),
                "grind_d50_um": d50 or None, "recipe": recipe_name, "dose_g": dose, "water_temp_c": temp,
                "pours": pours, "drawdown_end_s": cal.parse_clock(drawdown), "beverage_g": beverage,
                "bypass_g": bypass, **taste, "memo": memo, "flag": flag,
            }
            cal.append_brew(record, BREWS_PATH)
        except cal.BrewLogError as error:
            notice(str(error), error=True)
        else:
            score = cal.load_brews(BREWS_PATH)[-1].quality_score
            st.session_state["log_saved"] = f"저장했어요: {brew_id}" + (f" · 만족도 {score:.2f}" if score else "")
            st.session_state["log_form"] = form + 1
            st.rerun()

    section("최근 기록", f"모두 {len(brews)}잔")
    recent_brews_table(brews)


st.set_page_config(page_title="Drip Coffee", layout="wide")
html(STYLE)
html('<div class="dc-eyebrow">Drip Coffee</div><div class="dc-title">핸드드립 추출 분석·추천</div>')

tab_grind, tab_predict, tab_recommend, tab_log = st.tabs(["분쇄도 측정", "추출 예측", "레시피 추천", "기록·보정"])

with tab_grind:
    grind_tab()

with tab_predict:
    notice("레시피와 분쇄도로 농도, 추출수율, 물 빠짐 시간을 예측하는 화면이에요. (준비 중)")

with tab_recommend:
    notice("목표 농도와 추출수율에 맞는 레시피를 추천하는 화면이에요. (준비 중)")

with tab_log:
    log_tab()
