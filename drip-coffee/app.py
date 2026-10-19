"""핸드드립 추출 분석·추천 도구의 화면 (Streamlit).

화면은 입력과 표시만 맡고, 계산은 dripcoffee 패키지에서 한다.
구조: 첫 화면의 '시작하기'로 한 잔 내리기 흐름(원두와 장비 → 기초 추천 → 분쇄 확인 → 내리기와 기록 → 맛 평가 →
마무리)을 한 화면씩 따라간다. '내 기록 보기'와 '모델 점검'은 첫 화면의 작은 링크로 들어간다.
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
from dripcoffee import extraction_model as em
from dripcoffee import grind_analysis as ga
from dripcoffee import recommend as rec

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
.dc-home { margin: 8vh 0 28px; }
[data-testid="stBaseButton-primary"] { min-height: 54px; border-radius: 16px; }  /* 토스처럼 크게 누르는 버튼 */
[data-testid="stBaseButton-primary"] p { font-size: 17px; font-weight: 700; }
.dc-hero { font-size: 40px; font-weight: 800; color: #2A2A2A; letter-spacing: -0.03em; line-height: 1.25; margin: 4px 0 8px; }
.dc-title { font-size: 32px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; line-height: 1.3; }
.dc-lead { font-size: 16px; color: #4E5968; line-height: 1.6; margin: 4px 0 28px; }
.dc-section { font-size: 22px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; margin: 48px 0 4px; }
.dc-caption { font-size: 14px; color: #8B95A1; margin-bottom: 16px; }

.dc-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0 8px; }
.dc-card { background: #FFFFFF; border-radius: 24px; padding: 28px 28px 24px;
           box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04); animation: dc-rise .5s cubic-bezier(.2, .8, .2, 1) both; }
.dc-cards.three { grid-template-columns: repeat(3, 1fr); }  /* 큰 카드 하나 + 작은 카드 셋 */
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
.dc-row span { color: #6B7684; flex-shrink: 0; white-space: nowrap; }  /* 이름은 줄이 바뀌지 않게 */
.dc-row b { color: #2A2A2A; font-weight: 600; text-align: right; }

.st-key-taste-title h3 { font-size: 22px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em;
                         margin-top: 48px; padding-bottom: 0; }
.dc-legend { font-size: 13px; color: #4E5968; line-height: 2; margin: -4px 0 8px; }
.dc-legend span { white-space: nowrap; margin-right: 14px; }  /* 항목 중간에서 줄이 바뀌지 않게 */
.dc-legend i { font-style: normal; margin-right: 4px; }
.dc-hexagon { background: #F9FAFB; border-radius: 24px; padding: 16px 12px 20px; text-align: center; }
.dc-hexagon svg { width: 100%; max-width: 380px; height: auto; }
.dc-hexagon-score { font-size: 15px; color: #6B7684; margin-top: 4px; }
.dc-hexagon-score b { font-size: 24px; font-weight: 700; color: #3182F6; margin-left: 4px; font-variant-numeric: tabular-nums; }
.dc-hexagon .dc-caption { margin: 6px 16px 0; }

@keyframes dc-rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .dc-card, .dc-notice { animation: none; } }
@media (max-width: 640px) { .dc-cards, .dc-cards.three { grid-template-columns: 1fr; } .dc-value { font-size: 40px; } }
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
    """초를 '2:42'처럼 보여준다. 빈 값은 '–'."""
    if seconds is None:
        return "–"
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def cell(value, form="{:g}"):
    """표의 한 칸. 빈 값은 'None' 대신 '–'로 보여준다."""
    return "–" if value is None else form.format(value)


def saved_grind(brew_id):
    """'사진만 재기'에서 이 추출번호로 저장한 측정 중 가장 최근 것. 없으면 None."""
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
            "클릭": cell(b.grinder_clicks),
            "대표 크기 (µm)": cell(b.grind_d50_um, "{:.0f}"),
            "온도 (°C)": cell(b.water_temp_c),
            "물 빠짐": clock(b.drawdown_end_s),
            "음료 (g)": cell(b.beverage_g),
            "만족도": cell(b.quality_score, "{:.2f}"),
            "필터": b.filter_paper or "–",
            "추천": "예" if b.row.get("followed_recommendation") else "–",
            "표시": b.flag or "–",
        }
        for b in reversed(brews[-limit:])
    )
    st.dataframe(table, hide_index=True)


GOLDEN_CUP_BOX = ((18.0, 22.0), (1.15, 1.35))  # 골든컵 범위: 추출수율 %, 농도 % (MODEL.md 9.1)
# 선호 영역 그림의 표시: 이름 → (모양, 색, 크기, 속 채움)
PREFERENCE_MARKS = {"골든컵 중심": ("cross", GRAY, 80, False), "내 선호 중심": ("circle", BRAND, 260, False),
                    "추천 목표": ("diamond", BRAND, 220, True)}
# 그림 아래 기호 설명. 기호 모양과 색은 그림과 같게 맞춘다.
PREFERENCE_LEGEND = '<div class="dc-legend">' + "".join(
    f'<span><i style="color:{color}">{symbol}</i>{name}</span>'
    for symbol, color, name in [("●", BRAND, "좋았던 잔"), ("●", GRAY_LIGHT, "중심을 당기지 않은 잔"),
                                ("◆", BRAND, "추천 목표"), ("○", BRAND, "내 선호 중심"), ("✚", GRAY, "골든컵 중심")]
) + "</div>"


def axis(title):
    return alt.Axis(title=title, grid=False, ticks=False, domainColor="#E5E8EB", labelColor=GRAY, labelFontSize=12,
                    titleColor=GRAY, titleFontSize=13, titleFontWeight=500, tickCount=6)


def preference_chart(points, target):
    """잔마다 (추출수율, 가수 후 농도)를 점으로 찍고, 골든컵 범위·내 선호 중심·추천 목표를 함께 그린다.

    좋았던 잔(가중치 > 0)은 파랗게, 점수가 높을수록 진하게 칠하고, 중심을 당기지 않은 잔은 회색으로 둔다.
    골든컵 중심에서 추천 목표로 점선을 그어, 기록이 목표를 어느 쪽으로 옮겼는지 보이게 한다.
    """
    (ey_low, ey_high), (tds_low, tds_high) = GOLDEN_CUP_BOX
    cups = pd.DataFrame(
        {
            "brew_id": p.brew_id, "bean": p.bean_id, "ey": p.ey_pct, "tds": p.tds_cup_pct,
            "score": round(p.score, 2), "weight": p.weight, "liked": "좋았던 잔" if p.weight > 0 else "중심을 당기지 않은 잔",
        }
        for p in points
    )
    marks = [(rec.GOLDEN_CUP_CENTER[0], rec.GOLDEN_CUP_CENTER[1], "골든컵 중심"), (target.ey_pct, target.tds_pct, "추천 목표")]
    if target.personal_center:
        marks.append((*target.personal_center, "내 선호 중심"))
    all_ey = [m[0] for m in marks] + list(cups["ey"]) + [ey_low, ey_high]
    all_tds = [m[1] for m in marks] + list(cups["tds"]) + [tds_low, tds_high]
    x = alt.X("ey:Q", scale=alt.Scale(domain=[math.floor(min(all_ey)) - 1, math.ceil(max(all_ey)) + 1], nice=False),
              axis=axis("추출수율 (%) · 모델 추정"))
    y = alt.Y("tds:Q", scale=alt.Scale(domain=[round(min(all_tds) - 0.05, 2), round(max(all_tds) + 0.05, 2)], nice=False),
              axis=axis("가수 후 농도 (%) · 모델 추정"))

    box = pd.DataFrame({"ey": [ey_low], "ey2": [ey_high], "tds": [tds_low], "tds2": [tds_high], "label": ["골든컵 범위"]})
    layers = alt.Chart(box).mark_rect(color="#F2F4F6", cornerRadius=12).encode(x=x, x2="ey2:Q", y=y, y2="tds2:Q")
    layers += alt.Chart(box).mark_text(align="left", baseline="top", dx=10, dy=10, fontSize=12, color=GRAY, font=FONT).encode(
        x="ey:Q", y="tds2:Q", text="label:N")
    move = pd.DataFrame({"ey": [marks[0][0], marks[1][0]], "tds": [marks[0][1], marks[1][1]], "order": [0, 1]})
    layers += alt.Chart(move).mark_line(color=GRAY_LIGHT, strokeDash=[4, 4], strokeWidth=1.5).encode(x=x, y=y, order="order:Q")

    if not cups.empty:
        layers += alt.Chart(cups).mark_circle(size=150, stroke="#FFFFFF", strokeWidth=1.5).encode(
            x=x, y=y,
            color=alt.Color("liked:N", scale=alt.Scale(domain=["좋았던 잔", "중심을 당기지 않은 잔"], range=[BRAND, GRAY_LIGHT]),
                            legend=None),
            opacity=alt.Opacity("weight:Q", scale=alt.Scale(domain=[0, 2.5], range=[0.45, 1]), legend=None),
            tooltip=[alt.Tooltip("brew_id:N", title="추출번호"), alt.Tooltip("bean:N", title="원두"),
                     alt.Tooltip("score:Q", title="만족도"), alt.Tooltip("ey:Q", title="추출수율 (%)", format=".1f"),
                     alt.Tooltip("tds:Q", title="가수 후 농도 (%)", format=".2f")],
        )

    marks_df = pd.DataFrame(marks, columns=["ey", "tds", "label"])
    # 이름은 그림 안에 쓰지 않고 아래 기호 설명(PREFERENCE_LEGEND)에 둔다. 폰 화면에서 서로 겹쳐서.
    for label, (shape, color, size, filled) in PREFERENCE_MARKS.items():
        one = marks_df[marks_df["label"] == label]
        if one.empty:
            continue
        layers += alt.Chart(one).mark_point(shape=shape, color=color, size=size, filled=filled, strokeWidth=2,
                                            opacity=1).encode(
            x=x, y=y, tooltip=[alt.Tooltip("label:N", title="표시"), alt.Tooltip("ey:Q", title="추출수율 (%)", format=".1f"),
                               alt.Tooltip("tds:Q", title="가수 후 농도 (%)", format=".2f")])
    return layers.properties(height=360).configure_view(stroke=None).configure(font=FONT)


def preference_section(brews):
    """기록에서 찾은 내 선호 영역과 추천 목표 (MODEL.md 9.1)."""
    points = rec.brew_points(brews)
    target = rec.preference_target(points)
    section("내 선호 영역", "기록한 잔마다 모델로 계산한 위치예요. 굴절계로 잰 값이 아니라 모델 추정값이에요. "
                          "만족도 2.5를 넘는 잔이 내 선호 중심을 당기고, 추천 목표는 골든컵 중심과 섞어서 정해요. "
                          "좋았던 잔이 쌓일수록 목표가 내 쪽으로 옮겨 가요.")
    center = target.personal_center
    html(
        '<div class="dc-cards">'
        + card("추천 목표 추출수율", f"{target.ey_pct:.1f}", "%", "추천은 이 값에 맞는 레시피를 찾아요.", hero=True)
        + card("추천 목표 농도", f"{target.tds_pct:.2f}", "%", "가수한 뒤의 농도예요.")
        + card("중심을 당긴 잔", f"{target.n_liked}", "잔",
               f"내 선호 중심은 {center[0]:.1f}%, {center[1]:.2f}%예요." if center else "아직 없어서 골든컵 중심을 목표로 써요.")
        + "</div>"
    )
    st.altair_chart(preference_chart(points, target), width="stretch")
    html(PREFERENCE_LEGEND)

    used = {p.brew_id for p in points}
    skipped = [f"{b.brew_id} {b.flag or ('맛 점수 없음' if b.quality_score is None else '분쇄 크기 없음')}"
               for b in brews if b.brew_id not in used]
    if skipped:
        html(f'<div class="dc-caption">계산에서 뺀 잔: {escape(", ".join(skipped))}</div>')


def log_form():
    """한 잔 기록 입력 양식. '내 기록' 화면의 '직접 기록하기'에 들어간다."""
    html(
        '<div class="dc-caption">흐름 밖에서 내린 잔을 적어요. 저장하면 추출 기록 끝에 더해지고, '
        "모델 보정과 내 선호 영역 계산에 쓰여요. 측정 방법과 맛 기준은 docs/BREW_PROTOCOL.md와 같아요.</div>"
    )
    saved = st.session_state.pop("log_saved", None)
    if saved:
        notice(saved)

    brews = cal.load_brews(BREWS_PATH)
    beans = cal.load_beans()
    form = st.session_state.setdefault("log_form", 0)  # 저장할 때마다 올려서 입력칸을 새로 만든다

    def key(name):
        return f"log-{form}-{name}"

    prefill = st.session_state.get("log_prefill")  # 추천 화면의 '이 레시피로 기록하기'
    if prefill:
        notice(f"추천 레시피를 채워 뒀어요 (추천 분쇄 {prefill['d50_um']:.0f} µm). "
               "다르게 내렸다면 고쳐 주세요. 분쇄 대표 크기 칸에는 추천값이 아니라 사진으로 잰 값을 넣어요.")
        if st.button("채운 레시피 지우기", key=key("clear-prefill")):
            st.session_state.pop("log_prefill")
            st.session_state["log_form"] = form + 1
            st.rerun()

    section("잔 정보")
    c1, c2, c3 = st.columns(3)
    brew_date = c1.date_input("날짜", value=date.today(), key=key("date"))
    brew_id = c2.text_input("추출번호", cal.next_brew_id(brew_date.isoformat(), brews), key=key(f"id-{brew_date}"))
    bean_ids = list(beans)
    last_bean = brews[-1].bean_id if brews and brews[-1].bean_id in beans else bean_ids[0]
    if prefill and prefill["bean_id"] in beans:
        last_bean = prefill["bean_id"]
    bean_id = c3.selectbox(
        "원두", bean_ids, index=bean_ids.index(last_bean),
        format_func=lambda b: f"{b} {beans[b]['name']} ({beans[b]['role']})", key=key("bean"),
    )
    bean = beans[bean_id]
    same_bean = [b for b in brews if b.bean_id == bean_id]
    previous = same_bean[-1] if same_bean else (brews[-1] if brews else None)  # 입력칸 기본값은 지난 잔
    if bean["role"] == "시험":
        notice("시험용 원두예요. 모델을 동결한 뒤 최종 시험에서만 써요 (MODEL.md 6절).")

    c1, c2, c3 = st.columns(3)
    roast_date = c1.text_input("로스팅 날짜", bean["roast_date"], key=key(f"roast-{bean_id}"))
    dripper = c2.text_input("드리퍼", previous.row.get("dripper", "V60") if previous else "V60", key=key("dripper"))
    # 필터는 장소마다 달라서 지난 잔 값을 미리 채우지 않고 매번 고르게 한다
    filters = list(dict.fromkeys([*cal.FILTERS, *(b.filter_paper for b in brews if b.filter_paper)]))
    filter_paper = c3.selectbox("필터", filters, index=None, placeholder="필터를 골라 주세요", accept_new_options=True,
                                help="종이마다 물 빠지는 속도가 달라질 수 있어요. 목록에 없으면 이름을 적어요.",
                                key=key("filter"))
    c1, c2 = st.columns(2)
    grinder = c1.text_input("그라인더", previous.grinder if previous else "메버릭1", key=key("grinder"))
    clicks = c2.number_input("클릭 수", min_value=0, max_value=180, step=1,
                             value=previous.grinder_clicks if previous and previous.grinder_clicks is not None else 95,
                             key=key(f"clicks-{bean_id}"))

    grind = saved_grind(brew_id)
    d50 = st.number_input("분쇄 대표 크기 (µm, 선택)", min_value=0.0, step=1.0, format="%.0f",
                          value=float(round(grind["d50_um"])) if grind else None, key=key(f"d50-{brew_id}"))
    if grind:
        html(f'<div class="dc-caption">“사진만 재기”에서 이 추출번호로 저장한 측정을 불러왔어요 '
             f"(사진 {grind['n_photos']}장, {grind['grinder_clicks']}클릭).</div>")
    else:
        html('<div class="dc-caption">이 추출번호로 저장한 분쇄 측정이 없어요. 비워 두면 같은 원두·같은 클릭으로 간 '
             "다른 잔의 값을 빌려 써요.</div>")

    section("레시피")
    # 기본값: 추천에서 넘어온 레시피(같은 원두일 때) → 같은 원두의 지난 잔 → 마지막 잔
    use_prefill = bool(prefill) and prefill["bean_id"] == bean_id
    default = {
        "dose": prefill["dose_g"] if use_prefill else (previous.dose_g if previous else 20.0),
        "temp": prefill["water_temp_c"] if use_prefill else (previous.water_temp_c if previous else 92.0),
        "bypass": prefill["bypass_g"] if use_prefill else (previous.bypass_g if previous else 0.0),
        "pours": prefill["pours"] if use_prefill else (previous.pours if previous else "0:40;30:60;60:60;90:60"),
        "recipe": "추천 레시피" if use_prefill else (previous.row.get("recipe", "") if previous else ""),
    }
    c1, c2, c3 = st.columns(3)
    dose = c1.number_input("원두 무게 (g)", min_value=0.0, step=0.1, format="%.1f",
                           value=default["dose"], key=key(f"dose-{bean_id}"))
    temp = c2.number_input("물 온도 (°C)", min_value=0.0, max_value=100.0, step=1.0, format="%.0f",
                           value=default["temp"], key=key(f"temp-{bean_id}"))
    bypass = c3.number_input("가수량 (g)", min_value=0.0, step=1.0, format="%.0f",
                             value=default["bypass"], key=key(f"bypass-{bean_id}"))
    c1, c2 = st.columns([2, 1])
    pours = c1.text_input("붓기 기록 (시작초:g)", default["pours"],
                          help="각 붓기를 시작한 시각(초)과 그 회차에 부은 양(g)을 ;로 이어 적어요. 예: 0:40;30:60;60:60;90:60",
                          key=key(f"pours-{bean_id}"))
    recipe_name = c2.text_input("레시피 이름 (선택)", default["recipe"], key=key(f"recipe-{bean_id}"))
    followed = use_prefill and st.checkbox("추천 레시피를 따라 내렸어요", value=True, key=key("followed"),
                                           help="10/25 점검(추천 레시피로 내린 잔 수)과 추천이 맞았는지 분석에 써요.")

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
                "roast_date": roast_date, "dripper": dripper, "filter": filter_paper or "",
                "grinder": grinder, "grinder_clicks": int(clicks),
                "grind_d50_um": d50 or None, "recipe": recipe_name, "dose_g": dose, "water_temp_c": temp,
                "pours": pours, "drawdown_end_s": cal.parse_clock(drawdown), "beverage_g": beverage,
                "bypass_g": bypass, **taste, "memo": memo, "flag": flag,
                "followed_recommendation": prefill["followed"] if followed else "",
            }
            cal.append_brew(record, BREWS_PATH)
        except cal.BrewLogError as error:
            notice(str(error), error=True)
        else:
            score = cal.load_brews(BREWS_PATH)[-1].quality_score
            st.session_state["log_saved"] = f"저장했어요: {brew_id}" + (f" · 만족도 {score:.2f}" if score else "")
            st.session_state["log_form"] = form + 1
            st.session_state.pop("log_prefill", None)
            st.rerun()



def profile_label(record):
    """저장한 분쇄 측정을 고르는 목록의 한 줄."""
    brew = record["brew_id"] or "추출번호 없음"
    saved = record["saved_at"].replace("T", " ")[5:16]
    return f"{brew} · {record['bean_id']} {record['grinder_clicks']}클릭 · {record['d50_um']:.0f} µm ({saved} 저장)"


def grind_input():
    """예측에 쓸 분쇄도. 저장한 측정은 사진에서 잰 크기 분포를 그대로 쓰고, 대표 크기만 넣으면 분포를 가정한다."""
    records = ga.load_profiles(PROFILE_DIR)
    ways = ["저장한 측정", "대표 크기 입력"]
    way = st.segmented_control("분쇄 넣는 방법", ways, default=ways[0] if records else ways[1], key="predict-grind-way")
    if way == ways[0] and records:
        newest_first = list(reversed(records))
        i = st.selectbox("측정", range(len(newest_first)), format_func=lambda i: profile_label(newest_first[i]),
                         key="predict-profile")
        record = newest_first[i]
        html(f'<div class="dc-caption">사진 {record["n_photos"]}장, 입자 {record["n_particles"]}개에서 잰 크기 분포를 '
             "그대로 써요.</div>")
        return em.grind_from_diameters(record["diameters_um"])
    if way == ways[0]:
        notice("저장한 분쇄 측정이 없어요. '사진만 재기'에서 측정을 저장하거나, 대표 크기를 직접 넣어 주세요.")
    d50 = st.number_input("분쇄 대표 크기 (µm)", min_value=300.0, max_value=3000.0, value=1200.0, step=10.0,
                          format="%.0f", key="predict-d50")
    html('<div class="dc-caption">대표 크기만 알 때는 실제 사진 측정과 비슷한 폭의 분포(대부분의 범위가 대표 크기의 '
         "약 0.67~1.5배)로 가정해요.</div>")
    return em.grind_from_d50(d50)


def timeline_chart(prediction, recipe, actual_end=None):
    """시간에 따라 나온 음료와 드리퍼에 고인 물. 붓기 시점은 회색 점선, 물줄기가 끊기는 시각은 파란 점선.
    기록한 잔과 비교할 때는 실제로 끊긴 시각(actual_end)을 진한 회색 점선으로 함께 그린다."""
    t = prediction.timeline
    data = pd.concat([
        pd.DataFrame({"time": t["time_s"], "grams": t["beverage_g"], "what": "나온 음료 (가수 전)"}),
        pd.DataFrame({"time": t["time_s"], "grams": t["free_g"], "what": "드리퍼에 고인 물"}),
    ])
    end = prediction.drawdown_end_s
    ticks = list(range(0, int(max(end, actual_end or 0)) + 31, 30))
    x = alt.X("time:Q", scale=alt.Scale(domain=[0, ticks[-1]], nice=False),
              axis=alt.Axis(title="시간 (분:초)", values=ticks, grid=False, ticks=False, domainColor="#E5E8EB",
                            labelColor=GRAY, labelFontSize=12, titleColor=GRAY, titleFontSize=13, titleFontWeight=500,
                            labelExpr="floor(datum.value / 60) + ':' + (datum.value % 60 < 10 ? '0' : '') + datum.value % 60"))
    y = alt.Y("grams:Q", axis=axis("g"))
    color = alt.Color("what:N", scale=alt.Scale(domain=["나온 음료 (가수 전)", "드리퍼에 고인 물"], range=[BRAND, GRAY_LIGHT]),
                      legend=alt.Legend(title=None, orient="bottom", labelColor=GRAY_DARK, labelFontSize=12))
    lines = alt.Chart(data).mark_line(strokeWidth=3).encode(
        x=x, y=y, color=color,
        tooltip=[alt.Tooltip("time:Q", title="시간 (초)"), alt.Tooltip("what:N", title="무엇"),
                 alt.Tooltip("grams:Q", title="g", format=".0f")])
    pours = pd.DataFrame({"time": [p.start_s for p in recipe.pours], "label": [f"+{p.grams:g} g" for p in recipe.pours]})
    pour_rules = alt.Chart(pours).mark_rule(color=GRAY_LIGHT, strokeDash=[3, 3]).encode(x=x)
    pour_text = alt.Chart(pours).mark_text(align="left", baseline="top", dx=4, y=4, fontSize=11, color=GRAY, font=FONT).encode(
        x=x, text="label:N")
    stop = pd.DataFrame({"time": [end], "label": [f"{'예측' if actual_end is not None else '끊김'} {clock(end)}"]})
    stop_rule = alt.Chart(stop).mark_rule(color=BRAND, strokeDash=[4, 4]).encode(x=x)
    # 끊김 이름표는 붓기 이름표(맨 위 줄)보다 한 줄 아래에 둔다. 좁은 화면에서 마지막 붓기 이름표와 겹쳐서.
    stop_text = alt.Chart(stop).mark_text(align="right", baseline="top", dx=-4, y=22, fontSize=12, fontWeight=700,
                                          color=BRAND, font=FONT).encode(x=x, text="label:N")
    layers = pour_rules + pour_text + stop_rule + stop_text + lines
    if actual_end is not None:
        real = pd.DataFrame({"time": [actual_end], "label": [f"실제 {clock(actual_end)}"]})
        layers += alt.Chart(real).mark_rule(color=GRAY_DARK, strokeDash=[2, 3]).encode(x=x)
        layers += alt.Chart(real).mark_text(align="left", baseline="top", dx=4, y=40, fontSize=12, fontWeight=700,
                                            color=GRAY_DARK, font=FONT).encode(x=x, text="label:N")
    return layers.properties(height=300).configure_view(stroke=None).configure(font=FONT)


def gap(error, actual_text, unit, fewer, more):
    """예측과 실제의 차이를 말로. error는 예측 − 실제."""
    if error is None:
        return "실제 값은 기록하지 않았어요."
    if abs(error) < 0.5:
        return f"실제 {actual_text} · 실제와 거의 같아요."
    return f"실제 {actual_text} · 예측이 {abs(error):.1f}{unit} {fewer if error < 0 else more}.".replace(".0" + unit, unit)


def comparison_table(comparisons):
    table = pd.DataFrame(
        {
            "추출번호": c.brew.brew_id,
            "원두": c.brew.bean_id,
            "클릭": cell(c.brew.grinder_clicks),
            "대표 크기 (µm)": f"{c.grind_d50_um:.0f}",
            "끊김 예측": clock(c.prediction.drawdown_end_s),
            "끊김 실제": clock(c.brew.drawdown_end_s),
            "차이 (초)": cell(c.end_error_s, "{:+.0f}"),
            "음료 예측 (g)": f"{c.prediction.beverage_g:.1f}",
            "음료 실제 (g)": cell(c.brew.beverage_g),
            "차이 (g)": cell(c.beverage_error_g, "{:+.1f}"),
            "필터": c.brew.filter_paper or "–",
            "표시": c.brew.flag or "–",
        }
        for c in reversed(comparisons)
    )
    st.dataframe(table, hide_index=True)


def error_summary(comparisons):
    """표시가 없는 잔의 평균 차이. 보정(MODEL.md 5절) 전후를 이 값으로 비교한다."""
    normal = [c for c in comparisons if not c.brew.flag]
    lines = []
    for label, errors, unit in [("끊김 시각", [c.end_error_s for c in normal], "초"),
                                ("음료 무게", [c.beverage_error_g for c in normal], " g")]:
        errors = [e for e in errors if e is not None]
        if errors:
            mean = sum(errors) / len(errors)
            size = sum(abs(e) for e in errors) / len(errors)
            lines.append((f"{label} ({len(errors)}잔)", f"평균 {mean:+.1f}{unit} · 크기 평균 {size:.1f}{unit}"))
    if lines:
        html(rows(lines))
        html('<div class="dc-caption">표시가 없는 잔만 셌어요. \'평균\'은 예측이 한쪽으로 치우친 정도, '
             "'크기 평균'은 부호를 빼고 얼마나 빗나가는지예요.</div>")


def compare_view(brews):
    """기록한 잔을 같은 레시피·분쇄로 다시 계산해 실제와 나란히 보여준다."""
    comparisons = cal.compare_with_log(brews)
    if not comparisons:
        notice("비교할 기록이 없어요. 분쇄 대표 크기가 있는 잔을 기록하면 여기서 예측과 비교할 수 있어요.")
        return
    newest_first = list(reversed(comparisons))

    def label(i):
        b = newest_first[i].brew
        return f"{b.brew_id} · {b.bean_id} {cell(b.grinder_clicks)}클릭 · 실제 끊김 {clock(b.drawdown_end_s)}"

    i = st.selectbox("기록한 잔", range(len(newest_first)), format_func=label, key="predict-compare-cup")
    c = newest_first[i]
    b, prediction = c.brew, c.prediction
    html(rows([
        ("원두", f"{b.bean_id} {b.row.get('bean_name', '')}".strip()),
        ("분쇄", f"{b.grinder} {cell(b.grinder_clicks)}클릭 · {c.grind_d50_um:.0f} µm"
                 + (" (같은 원두·클릭 잔의 측정)" if c.borrowed_grind else "")),
        ("레시피", f"원두 {b.dose_g:g} g · {b.water_temp_c:g} °C · 붓기 {b.pours} · 가수 {b.bypass_g:g} g"),
        ("드리퍼·필터", f"{b.row.get('dripper', '')} · {b.filter_paper or '–'}"),
    ]))
    if b.flag:
        notice(f"'{b.flag}' 표시가 있는 잔이에요. 보정과 아래 평균에서는 빼요.")

    section("예측과 실제", "같은 레시피·분쇄로 모델을 다시 돌린 값이에요. 추출수율과 농도는 굴절계가 없어 실제와 비교할 수 없어요.")
    html(
        '<div class="dc-cards">'
        + card("물줄기가 끊기는 시각 (예측)", clock(prediction.drawdown_end_s), "",
               gap(c.end_error_s, clock(b.drawdown_end_s), "초", "짧아요", "길어요"))
        + card("음료 (가수 전, 예측)", f"{prediction.beverage_g:.0f}", "g",
               gap(c.beverage_error_g, f"{cell(b.beverage_g)} g", " g", "적어요", "많아요"))
        + card("추출수율 (예측)", f"{prediction.ey_pct:.1f}", "%", "원두 무게 중 음료로 녹아 나온 비율이에요.")
        + card("농도 (가수 후, 예측)", f"{prediction.tds_cup_pct:.2f}", "%", f"가수 전 {prediction.tds_pct:.2f}%예요.")
        + "</div>"
    )
    st.altair_chart(timeline_chart(prediction, b.recipe(), actual_end=b.drawdown_end_s), width="stretch")

    section("모든 기록과 비교", "차이는 예측 − 실제예요. 음수면 예측이 짧거나 적고, 양수면 길거나 많아요.")
    comparison_table(comparisons)
    error_summary(comparisons)


def new_recipe_view(brews):
    """새 레시피를 넣어 예측한다. 입력칸 기본값은 마지막으로 기록한 잔의 레시피."""
    previous = brews[-1] if brews else None

    section("분쇄")
    grind = grind_input()

    section("레시피")
    c1, c2, c3 = st.columns(3)
    dose = c1.number_input("원두 무게 (g)", min_value=0.0, step=0.1, format="%.1f",
                           value=previous.dose_g if previous else 20.0, key="predict-dose")
    temp = c2.number_input("물 온도 (°C)", min_value=70.0, max_value=100.0, step=1.0, format="%.0f",
                           value=previous.water_temp_c if previous else 92.0, key="predict-temp")
    bypass = c3.number_input("가수량 (g)", min_value=0.0, step=1.0, format="%.0f",
                             value=previous.bypass_g if previous else 0.0, key="predict-bypass")
    pours_text = st.text_input("붓기 기록 (시작초:g)", previous.pours if previous else "0:40;30:60;60:60;90:60",
                               help="각 붓기를 시작한 시각(초)과 그 회차에 부은 양(g)을 ;로 이어 적어요. 예: 0:40;30:60;60:60;90:60",
                               key="predict-pours")

    try:
        recipe = em.Recipe(dose, temp, em.parse_pours(pours_text), bypass)
        prediction = em.predict(recipe, grind)
    except em.RecipeError as error:
        notice(str(error), error=True)
        return

    last = recipe.pours[-1]
    after_last = prediction.drawdown_end_s - (last.start_s + last.grams / em.DEFAULT_PARAMS.pour_rate_g_per_s)
    section("예측 결과", "보정 전 모델이라 실제와 차이가 날 수 있어요. 기록이 쌓이면 흐름 저항과 머금는 물을 기록에 맞춰요.")
    html(
        '<div class="dc-cards three">'
        + card("물줄기가 끊기는 시각", clock(prediction.drawdown_end_s), "",
               f"이때 드리퍼를 빼요. 마지막 붓기를 마치고 약 {after_last:.0f}초 뒤예요.", hero=True)
        + card("음료 (가수 전)", f"{prediction.beverage_g:.0f}", "g",
               f"부은 물 {recipe.water_g:g} g 중 가루가 약 {prediction.held_liquid_g:.0f} g을 머금고 남아요.")
        + card("추출수율", f"{prediction.ey_pct:.1f}", "%", "원두 무게 중 음료로 녹아 나온 비율이에요.")
        + card("농도 (가수 후)", f"{prediction.tds_cup_pct:.2f}", "%",
               f"가수 전 {prediction.tds_pct:.2f}%에 물 {bypass:g} g을 더한 값이에요.")
        + "</div>"
    )
    st.altair_chart(timeline_chart(prediction, recipe), width="stretch")
    html('<div class="dc-caption">점선은 붓기를 시작한 시각이에요. 고인 물이 0이 되는 순간을 물줄기가 끊기는 순간으로 봐요.</div>')


@st.cache_data(show_spinner=False, max_entries=20)
def cached_recommendation(bean_id, dose_g, log_version):
    """추천은 0.1–2초 걸려서, 기록 파일이 바뀌지 않았으면(log_version) 지난 결과를 다시 쓴다."""
    brews = cal.load_brews(BREWS_PATH)
    target = rec.preference_target(rec.brew_points(brews))
    return target, rec.recommend_recipe(target, cal.load_beans()[bean_id], brews, dose_g)


def pour_rows(recommendation):
    """붓기 일정: 시각, 이번에 붓는 양(원두의 몇 배), 누적."""
    items, total = [], 0.0
    multiples = (rec.BLOOM_MULTIPLE, *recommendation.pattern)
    for i, (pour, multiple) in enumerate(zip(recommendation.recipe.pours, multiples)):
        total += pour.grams
        what = "뜸" if i == 0 else f"{i}번째 붓기"
        items.append((f"{clock(pour.start_s)} · {what}", f"{pour.grams:g} g (원두의 {multiple}배) · 누적 {total:g} g"))
    return items


def recommend_tab():
    html(
        '<div class="dc-lead">원두를 고르면 내 선호 영역의 목표에 맞는 레시피 전체를 추천해요.<br>'
        "분쇄는 그라인더 클릭이 아니라 대표 크기(µm)로 알려 줘요. 갈고 나서 사진으로 확인해요.</div>"
    )
    brews = cal.load_brews(BREWS_PATH)
    beans = cal.load_beans()
    bean_ids = list(beans)
    last_bean = brews[-1].bean_id if brews and brews[-1].bean_id in beans else bean_ids[0]
    c1, c2 = st.columns([2, 1])
    bean_id = c1.selectbox("원두", bean_ids, index=bean_ids.index(last_bean),
                           format_func=lambda b: f"{b} {beans[b]['name']} ({beans[b]['roast_level']})", key="rec-bean")
    dose = c2.number_input("원두 무게 (g)", min_value=5.0, max_value=40.0, value=20.0, step=0.1, format="%.1f",
                           key="rec-dose")
    bean = beans[bean_id]
    if bean["role"] == "시험":
        notice("시험용 원두예요. 모델을 동결한 뒤 최종 시험에서만 내려요 (MODEL.md 6절).")

    try:
        with st.spinner("레시피 후보를 계산하는 중..."):
            target, r = cached_recommendation(bean_id, float(dose), BREWS_PATH.stat().st_mtime)
    except rec.NoRecipeError as error:
        notice(str(error), error=True)
        return

    center = target.personal_center
    section("목표", "내 선호 영역에서 정한 값이에요 ('내 기록 보기'의 그림).")
    html(rows([
        ("추출수율 · 농도 (가수 후)", f"{target.ey_pct:.1f}% · {target.tds_pct:.2f}%"),
        ("근거", f"좋았던 잔 {target.n_liked}잔의 중심({center[0]:.1f}%, {center[1]:.2f}%)과 골든컵(20%, 1.25%)을 섞음"
         if center else "아직 좋았던 잔이 없어 골든컵 중심을 써요"),
    ]))

    recipe, prediction = r.recipe, r.prediction
    low, high = r.temp_range
    section("추천 레시피", f"원두 {recipe.dose_g:g} g · 물 {recipe.water_g:g} g (원두의 {r.water_ratio}배, "
                         f"{rec.pattern_text(r.pattern)}배) · {r.interval_s:g}초 간격")
    html(
        '<div class="dc-cards three">'
        + card("분쇄 대표 크기", f"{r.d50_um:.0f}", "µm", "갈고 나서 사진으로 재 보세요.", hero=True)
        + card("물 온도", f"{recipe.water_temp_c:g}", "°C", f"{bean['roast_level']} 범위 {low}–{high} °C 안에서 골랐어요.")
        + card("물", f"{recipe.water_g:g}", "g", f"원두의 {r.water_ratio}배를 아래 일정대로 나눠 부어요.")
        + card("가수", f"{recipe.bypass_g:g}", "g", "물줄기가 끊겨 드리퍼를 뺀 뒤 더하는 물이에요.")
        + "</div>"
    )
    section("붓기 일정", "붓는 방식(센터 푸어, 원형 푸어 등)은 평소대로 해요. 모델은 붓는 시각과 양만 계산해요.")
    html(rows(pour_rows(r)))

    section("예상", "보정 전 모델이라 시간은 실제와 다를 수 있어요. 끊김은 예상 시각이 아니라 물줄기가 끊기는 순간에 드리퍼를 빼요.")
    html(
        '<div class="dc-cards three">'
        + card("물줄기가 끊기는 시각", clock(prediction.drawdown_end_s), "", "이때쯤 드리퍼를 빼게 돼요.", hero=True)
        + card("음료 (가수 전)", f"{prediction.beverage_g:.0f}", "g", "가수하기 전에 잰 무게예요.")
        + card("추출수율", f"{prediction.ey_pct:.1f}", "%", f"목표 {target.ey_pct:.1f}%")
        + card("농도 (가수 후)", f"{prediction.tds_cup_pct:.2f}", "%", f"목표 {target.tds_pct:.2f}%")
        + "</div>"
    )
    st.altair_chart(timeline_chart(prediction, recipe), width="stretch")

    if st.button("이 레시피로 기록하기", type="primary", width="stretch", key="rec-to-log"):
        st.session_state["log_prefill"] = {
            "bean_id": bean_id, "dose_g": recipe.dose_g, "water_temp_c": recipe.water_temp_c,
            "pours": rec.pours_text(recipe.pours), "bypass_g": recipe.bypass_g, "d50_um": r.d50_um,
            # 10/25 점검(추천 레시피 3잔 이상)과 나중 분석을 위해 무엇을 추천했는지 함께 남긴다
            "followed": f"예 (추천 {r.d50_um:.0f} µm, 목표 {target.ey_pct:.1f}% · {target.tds_pct:.2f}%)",
        }
        st.session_state["log_form"] = st.session_state.get("log_form", 0) + 1  # 입력칸을 새로 만들어 채운다
        st.session_state["page"] = "history"  # '내 기록'의 '직접 기록하기'로 바로 간다
        st.rerun()

    with st.expander("왜 이 레시피인가요?"):
        usual = r.usual
        html(rows([
            ("출발점", usual.source),
            ("출발점 레시피", f"원두의 {usual.water_ratio:g}배, {rec.pattern_text(usual.pattern)}배, "
                          f"{usual.interval_s:g}초 간격, {usual.temp_c:g} °C"),
            ("막힘 한계", f"{r.clog_limit_um:.0f} µm보다 고운 분쇄는 뺐어요 (같은 원두가 막혔던 크기)"
             if r.clog_limit_um else "이 원두는 막힌 기록이 없어요"),
            ("계산한 후보", f"{r.checked}개"),
        ]))
        html('<div class="dc-caption">후보마다 목표 추출수율에 맞는 분쇄 크기를 찾고, 가수량으로 농도를 맞춰요. '
             "그중 끊김이 2:00–3:30 안이고 출발점과 가장 덜 다른 레시피를 골라요 (MODEL.md 9.2).</div>")


# 한 잔 내리기: 원두·장비 → 기초 추천 → 분쇄 확인 → 내리기와 기록 → 맛 평가 → 마무리
FLOW_STEPS = ("원두와 장비", "기초 추천", "분쇄 확인", "내리기와 기록", "맛 평가", "마무리")
BASE_NAMES = {((4, 2), 60.0): "484 레시피 (중강배전 이상)", ((3, 3, 3), 30.0): "4666 레시피"}


def flow_header(step):
    html(f'<div class="dc-eyebrow">{step} / {len(FLOW_STEPS)} 단계</div>'
         f'<div class="dc-title" style="font-size:26px">{escape(FLOW_STEPS[step - 1])}</div>')
    st.progress(step / len(FLOW_STEPS))


def go_to(step):
    st.session_state["flow"]["step"] = step
    st.rerun()


def base_name(usual):
    """출발점 레시피의 이름. 평소 두 레시피면 그 이름, 아니면 배수 표기."""
    return BASE_NAMES.get((tuple(usual.pattern), float(usual.interval_s)),
                          f"{rec.pattern_text(usual.pattern)}배 / {usual.interval_s:g}초 레시피")


def hint_text(hint, grinder):
    """클릭 힌트를 한 문장으로."""
    if hint is None:
        return f"{grinder}는 클릭 기록이 없어요. 평소 감대로 갈고 다음 단계에서 사진으로 확인해요."
    if hint.clicks is None:
        return f"{grinder}로 {hint.near.grinder_clicks}클릭보다 {hint.direction} 갈아 보세요. {hint.basis}."
    return f"{grinder}로 약 {hint.clicks}클릭에서 시작해 보세요. {hint.basis}"


def flow_equipment(flow):
    """1단계: 원두와 장비."""
    html('<div class="dc-lead">원두와 장비를 알려 주면 오늘 한 잔을 처음부터 끝까지 안내해요.<br>'
         "레시피를 추천하고, 간 가루를 사진으로 확인해 맞추고, 내린 결과와 맛을 기록해 다음 추천에 반영해요.</div>")
    brews = cal.load_brews(BREWS_PATH)
    beans = cal.load_beans()
    bean_ids = list(beans)
    last = brews[-1] if brews else None
    default_bean = flow.get("bean_id") or (last.bean_id if last and last.bean_id in beans else bean_ids[0])
    c1, c2 = st.columns([2, 1])
    bean_id = c1.selectbox("원두", bean_ids, index=bean_ids.index(default_bean),
                           format_func=lambda b: f"{b} {beans[b]['name']} ({beans[b]['roast_level']})", key="flow-bean")
    dose = c2.number_input("원두 무게 (g)", min_value=5.0, max_value=40.0, step=0.1, format="%.1f",
                           value=float(flow.get("dose_g", 20.0)), key="flow-dose")
    if beans[bean_id]["role"] == "시험":
        notice("시험용 원두예요. 모델을 동결한 뒤 최종 시험에서만 내려요 (MODEL.md 6절).")
    c1, c2, c3 = st.columns(3)
    dripper = c1.text_input("드리퍼", flow.get("dripper") or (last.row.get("dripper", "V60") if last else "V60"),
                            key="flow-dripper")
    filters = list(dict.fromkeys([*cal.FILTERS, *(b.filter_paper for b in brews if b.filter_paper)]))
    chosen = flow.get("filter")
    filter_paper = c2.selectbox("필터", filters, index=filters.index(chosen) if chosen in filters else None,
                                placeholder="필터를 골라 주세요", accept_new_options=True, key="flow-filter",
                                help="종이마다 물 빠지는 속도가 달라질 수 있어요. 목록에 없으면 이름을 적어요.")
    grinders = list(dict.fromkeys([*(b.grinder for b in reversed(brews) if b.grinder), "메버릭1"]))
    grinder = c3.selectbox("그라인더", grinders, index=grinders.index(flow["grinder"]) if flow.get("grinder") in grinders
                           else 0, accept_new_options=True, key="flow-grinder")
    if st.button("다음: 기초 추천 받기", type="primary", width="stretch", key="flow-next-1"):
        if not filter_paper:
            notice("필터를 골라 주세요. 필터마다 물 빠지는 속도가 달라질 수 있어 기록에 함께 남겨요.", error=True)
            return
        flow.update(bean_id=bean_id, dose_g=float(dose), dripper=dripper, filter=filter_paper, grinder=grinder)
        go_to(2)


def flow_base_recipe(flow):
    """2단계: 기초 추천. 온도, 베이스 레시피, 목표 분쇄 크기와 클릭 힌트."""
    brews = cal.load_brews(BREWS_PATH)
    beans = cal.load_beans()
    bean = beans[flow["bean_id"]]
    try:
        with st.spinner("레시피 후보를 계산하는 중..."):
            target, r = cached_recommendation(flow["bean_id"], flow["dose_g"], BREWS_PATH.stat().st_mtime)
    except rec.NoRecipeError as error:
        notice(str(error), error=True)
        if st.button("이전", key="flow-back-2"):
            go_to(1)
        return
    hint = cal.click_hint(r.d50_um, flow["grinder"], bean["bean_id"], bean["roast_date"], brews)
    flow.update(target_d50_um=r.d50_um, hint_clicks=hint.clicks if hint else None)
    recipe = r.recipe
    changed = (r.water_ratio != round(r.usual.water_ratio) or tuple(r.pattern) != tuple(r.usual.pattern))

    html(f'<div class="dc-lead">{escape(bean["bean_id"])} {escape(bean["name"])} · {escape(bean["roast_level"])} · '
         f"원두 {recipe.dose_g:g} g<br>베이스는 <b>{escape(base_name(r.usual))}</b>"
         + ("이고, 목표에 맞추려고 물 양이나 붓기를 조금 바꿨어요." if changed else "예요.") + "</div>")
    html(
        '<div class="dc-cards three">'
        + card("목표 분쇄", f"{r.d50_um:.0f}", "µm", hint_text(hint, flow["grinder"]), hero=True)
        + card("물 온도", f"{recipe.water_temp_c:g}", "°C", f"{bean['roast_level']} 범위 안에서 골랐어요.")
        + card("물", f"{recipe.water_g:g}", "g", f"원두의 {r.water_ratio}배 · {rec.pattern_text(r.pattern)}배")
        + card("가수", f"{recipe.bypass_g:g}", "g", "드리퍼를 뺀 뒤 더하는 물이에요.")
        + "</div>"
    )
    section("붓기 일정", f"{r.interval_s:g}초 간격이에요. 붓는 방식(센터·원형 푸어)은 평소대로 해요.")
    html(rows(pour_rows(r)))
    html(f'<div class="dc-caption">목표: 추출수율 {target.ey_pct:.1f}%, 농도 {target.tds_pct:.2f}% (내 선호 영역). '
         f"예상 끊김 {clock(r.prediction.drawdown_end_s)} (보정 전 모델이라 실제와 다를 수 있어요).</div>")
    notice("이제 위 클릭으로 원두를 갈고, 사진용으로 조금 덜어 축척 용지에 얇게 펴 주세요. "
           "다음 단계에서 사진으로 실제 크기를 재고, 목표와 다르면 그 가루에 맞게 레시피를 고쳐요.")
    c1, c2 = st.columns([1, 2])
    if c1.button("이전", width="stretch", key="flow-back-2"):
        go_to(1)
    if c2.button("다음: 갈고 사진 찍기", type="primary", width="stretch", key="flow-next-2"):
        go_to(3)


def flow_later(flow):
    """3단계부터는 준비 중. 지금은 '모델 점검'과 '내 기록 보기'로 이어서 한다."""
    notice("3단계(사진으로 분쇄 확인 → 레시피 맞추기)부터는 준비 중이에요. 지금은 첫 화면의 '모델 점검 → 사진만 재기'로 "
           "사진을 재고, '내 기록 보기 → 직접 기록하기'로 기록해 주세요.")
    c1, c2 = st.columns([1, 2])
    if c1.button("이전", width="stretch", key="flow-back-3"):
        go_to(2)
    c2.button("처음으로", width="stretch", key="flow-restart", on_click=open_page, args=("home",))


def flow_page():
    back_link("flow")
    flow = st.session_state.setdefault("flow", {"step": 1})
    step = flow["step"]
    flow_header(min(step, 3))
    {1: flow_equipment, 2: flow_base_recipe}.get(step, flow_later)(flow)


def open_page(page):
    """화면을 바꾼다. 한 잔 내리기는 늘 1단계부터 시작한다."""
    st.session_state["page"] = page
    if page == "flow":
        st.session_state["flow"] = {"step": 1}


def back_link(name):
    st.button("← 처음으로", type="tertiary", key=f"{name}-home", on_click=open_page, args=("home",))


def page_header(name, title, lead):
    back_link(name)
    html(f'<div class="dc-eyebrow">Drip Coffee</div><div class="dc-title">{escape(title)}</div>'
         f'<div class="dc-lead">{lead}</div>')


def home_page():
    """첫 화면: '시작하기' 하나를 크게 두고, 나머지는 작은 링크로."""
    html('<div class="dc-home"><div class="dc-eyebrow">Drip Coffee</div><div class="dc-hero">오늘의 한 잔</div>'
         '<div class="dc-lead">원두와 장비만 알려 주면 레시피부터 기록까지 안내해요.<br>'
         "사진으로 잰 분쇄와 내 맛 기록으로, 나에게 맞는 한 잔을 찾아가요.</div></div>")
    st.button("시작하기", type="primary", width="stretch", key="home-start", on_click=open_page, args=("flow",))
    brews = cal.load_brews(BREWS_PATH)
    if brews:
        last = brews[-1]
        score = f"만족도 {last.quality_score:.2f}" if last.quality_score is not None else "맛 점수 없음"
        html(f'<div class="dc-caption" style="text-align:center; margin-top:16px">지난 한 잔 · {escape(last.brew_id)} · '
             f"{escape(last.bean_id)} {escape(last.row.get('bean_name', ''))} · {score}</div>")
    c1, c2 = st.columns(2)
    c1.button("내 기록 보기", type="tertiary", width="stretch", key="home-history", on_click=open_page, args=("history",))
    c2.button("모델 점검", type="tertiary", width="stretch", key="home-check", on_click=open_page, args=("check",))


def history_page():
    """내 기록: 내 선호 영역, 기록한 잔들, 흐름 밖에서 내린 잔을 적는 '직접 기록하기'."""
    page_header("history", "내 기록", "기록한 잔과 내 선호 영역이에요. 흐름 밖에서 내린 잔은 아래 '직접 기록하기'로 적어요.")
    brews = cal.load_brews(BREWS_PATH)
    preference_section(brews)
    section("기록한 잔", f"모두 {len(brews)}잔, 최근 순서")
    recent_brews_table(brews, limit=len(brews))
    writing = "log_prefill" in st.session_state or "log_saved" in st.session_state
    with st.expander("직접 기록하기", expanded=writing):
        log_form()


def check_page():
    """모델 점검: 예측과 실제 비교, 레시피 직접 계산, 추천 계산 자세히, 커피 없이 분쇄 사진만 재기."""
    page_header("check", "모델 점검", "모델이 실제 잔을 얼마나 맞히는지 보고, 레시피를 직접 계산해 봐요. "
                "굴절계로 잰 값이 아니라 모델로 계산한 값이에요. 식과 근거는 docs/MODEL.md에 있어요.")
    views = ["예측과 실제", "레시피 계산", "추천 자세히", "사진만 재기"]
    view = st.segmented_control("보기", views, default=views[0], key="check-view") or views[0]
    brews = cal.load_brews(BREWS_PATH)
    if view == "예측과 실제":
        compare_view(brews)
    elif view == "레시피 계산":
        new_recipe_view(brews)
    elif view == "추천 자세히":
        recommend_tab()
    else:
        grind_tab()


st.set_page_config(page_title="Drip Coffee", layout="wide")
html(STYLE)
PAGES = {"home": home_page, "flow": flow_page, "history": history_page, "check": check_page}
PAGES[st.session_state.setdefault("page", "home")]()
