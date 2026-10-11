"""핸드드립 추출 분석·추천 도구의 화면 (Streamlit).

화면은 입력과 표시만 맡고, 계산은 dripcoffee 패키지에서 한다.
실행: streamlit run app.py
"""
import altair as alt
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from dripcoffee import DATA_DIR, LOCAL_DATA_DIR
from dripcoffee import grind_analysis as ga

PROFILE_DIR = LOCAL_DATA_DIR / "grind_profiles"
CLICK_TABLE = DATA_DIR / "reference" / "maverick_click_table.csv"

# 사진 위에 그리는 윤곽선 색 (RGB): 잰 입자, 붙어서 뺀 덩어리, 칸 테두리에 걸려 뺀 입자
OUTLINE_COLORS = {"measured": (34, 160, 70), "clump": (220, 40, 40), "border": (160, 70, 230)}
CURVE_COLOR = "#2f855a"
SIZE_TICKS = [100, 200, 300, 500, 700, 1000, 1500, 2000, 3000]


@st.cache_data(show_spinner=False, max_entries=30)
def analyze_upload(data):
    """사진 바이트를 분석한다. (펴진 사진, 측정 결과, 오류 메시지) 중 성공하면 오류가 None이다."""
    try:
        rectified, measurement = ga.analyze_photo(ga.load_image_bytes(data))
    except ga.GrindPhotoError as e:
        return None, None, str(e)
    return rectified, measurement, None


def outline_image(rectified, measurement):
    """펴진 가루 칸 위에 입자 윤곽선을 종류별 색으로 그린 RGB 이미지."""
    image = rectified.image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    thickness = max(1, round(max(rgb.shape[:2]) / 600))
    for kind, color in OUTLINE_COLORS.items():
        cv2.drawContours(rgb, measurement.outlines.get(kind, []), -1, color, thickness)
    return rgb


def distribution_chart(profile, show_count):
    """종 모양 분포 곡선: 대부분의 범위는 색칠, 대표 크기는 점선, 가장 많은 크기는 점으로 표시한다."""
    sizes, density = ga.size_distribution_curve(profile.diameters_um, by_volume=True)
    curve = pd.DataFrame({"size": sizes, "density": density / density.max()})
    low, high = profile.spread_range_um
    x = alt.X(
        "size:Q",
        title="입자 지름 (µm, 로그 눈금)",
        scale=alt.Scale(type="log", domain=list(ga.CURVE_RANGE_UM)),
        axis=alt.Axis(values=SIZE_TICKS, format="d"),
    )
    y = alt.Y("density:Q", axis=None, scale=alt.Scale(domain=[0, 1.15]))
    band = alt.Chart(curve[(curve["size"] >= low) & (curve["size"] <= high)]).mark_area(
        color=CURVE_COLOR, opacity=0.18
    ).encode(x=x, y=y)
    line = alt.Chart(curve).mark_line(color=CURVE_COLOR, strokeWidth=2.5).encode(
        x=x, y=y, tooltip=[alt.Tooltip("size:Q", title="지름 (µm)", format=".0f")]
    )
    d50 = alt.Chart(pd.DataFrame({"size": [profile.d50_um]})).mark_rule(
        color=CURVE_COLOR, strokeDash=[6, 4]
    ).encode(x=x)
    peak = pd.DataFrame({"size": [profile.mode_um], "density": [1.0], "label": [f"가장 많은 크기 {profile.mode_um:.0f}"]})
    peak_point = alt.Chart(peak).mark_point(color=CURVE_COLOR, filled=True, size=70).encode(x=x, y=y)
    peak_text = alt.Chart(peak).mark_text(align="left", dx=8, dy=-10, fontSize=12).encode(x=x, y=y, text="label:N")
    layers = band + line + d50 + peak_point + peak_text
    if show_count:
        count_curve = ga.size_distribution_curve(profile.diameters_um, by_volume=False)
        if count_curve is not None:
            count = pd.DataFrame({"size": count_curve[0], "density": count_curve[1] / count_curve[1].max()})
            layers += alt.Chart(count).mark_line(color="#888888", strokeDash=[3, 3]).encode(x=x, y=y)
    return layers.properties(height=300)


def click_table_size(clicks):
    """메버릭 클릭표(인터넷 기준, 비교용)에서 클릭 수에 해당하는 크기. 표 범위 밖이면 None."""
    table = pd.read_csv(CLICK_TABLE).sort_values("clicks")
    if not table["clicks"].min() <= clicks <= table["clicks"].max():
        return None
    return float(np.interp(clicks, table["clicks"], table["size_um"]))


def show_photo_result(name, rectified, measurement):
    """사진 한 장의 결과: 윤곽선을 그린 가루 칸 사진과 숫자, 경고."""
    left, right = st.columns([3, 2])
    left.image(outline_image(rectified, measurement), width="stretch")
    right.markdown(f"**{name}**")
    right.markdown(
        f"- 칸: {rectified.zone.name}칸, 픽셀 하나 {rectified.um_per_px:.0f} µm\n"
        f"- 잰 입자: {len(measurement.diameters_um)}개 (초록)\n"
        f"- 붙어서 뺀 덩어리: {measurement.excluded_clumps}개 (빨강)\n"
        f"- 칸 테두리에 걸려 뺀 입자: {measurement.excluded_border}개 (보라)\n"
        f"- 은피로 보고 뺀 조각: {measurement.excluded_chaff}개\n"
        f"- 이 사진의 대표 크기: "
        + (f"{measurement.d50_volume_um:.0f} µm" if len(measurement.diameters_um) else "없음")
    )
    for warning in rectified.warnings + measurement.warnings:
        right.warning(warning)


def show_profile(profile, grinder, clicks):
    """이번 측정(같은 가루로 찍은 사진을 합친 결과)의 요약, 분포 곡선, 자세한 값."""
    low, high = profile.d50_range_um
    spread_low, spread_high = profile.spread_range_um
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "대표 크기 (D50)", f"{profile.d50_um:.0f} µm",
        help="가루 무게의 절반이 이보다 작은 입자예요. 아래 범위는 같은 가루를 다시 찍었을 때 달라질 수 있는 정도예요.",
    )
    c1.caption(f"흔들림 범위 {low:.0f}~{high:.0f} µm")
    c2.metric("가장 많은 크기", f"{profile.mode_um:.0f} µm", help="무게 기준 분포 곡선에서 가장 높은 곳이에요.")
    c3.metric(
        "대부분의 범위", f"{spread_low:.0f}~{spread_high:.0f} µm",
        help="가루 무게의 68%가 이 범위에 있어요. 정규분포의 '평균 ± 표준편차' 범위에 해당해요.",
    )
    c3.caption(f"잰 입자 {profile.n_particles}개, 사진 {profile.n_photos}장")

    show_count = st.checkbox("개수 기준 곡선도 보기 (회색 점선, 참고용)")
    st.altair_chart(distribution_chart(profile, show_count), width="stretch")
    st.caption("색칠한 구간은 대부분의 범위, 점선은 대표 크기예요. 곡선은 무게 기준이라 작은 가루가 많아도 낮게 보여요.")

    for warning in profile.warnings:
        st.warning(warning)
    if "메버릭" in grinder:
        reference = click_table_size(clicks)
        if reference is not None:
            st.caption(f"참고: 메버릭 클릭표(인터넷 기준, 비교용)에서 {clicks}클릭은 약 {reference:.0f} µm예요.")

    with st.expander("자세히 보기"):
        photo_d50 = ", ".join(f"{v:.0f}" for v in profile.photo_d50_um)
        st.markdown(
            f"- D10 / D90: {profile.d10_um:.0f} / {profile.d90_um:.0f} µm "
            "(가루 무게의 10%가 D10보다 작고, 10%가 D90보다 커요)\n"
            f"- 분포 폭 (D90 − D10) / D50: {profile.span:.2f} (작을수록 고르게 갈렸어요)\n"
            f"- 사진별 대표 크기: {photo_d50} µm\n"
            f"- 검출 하한: {profile.min_diameter_um:.0f} µm (이보다 작은 가루는 재지 않았어요)\n"
            f"- 뺀 것: 덩어리 {profile.excluded_clumps}개, 테두리 입자 {profile.excluded_border}개, "
            f"은피 {profile.excluded_chaff}개"
        )


def show_saved_profiles():
    records = ga.load_profiles(PROFILE_DIR)
    with st.expander(f"저장한 측정 ({len(records)}개)"):
        if not records:
            st.caption("아직 저장한 측정이 없어요.")
            return
        table = pd.DataFrame(
            {
                "저장 시각": r["saved_at"].replace("T", " "),
                "원두": r["bean_id"],
                "그라인더": r["grinder"],
                "클릭": r["grinder_clicks"],
                "추출번호": r["brew_id"],
                "대표 크기 (µm)": round(r["d50_um"]),
                "가장 많은 크기 (µm)": round(r["mode_um"]),
                "대부분의 범위 (µm)": f"{r['spread_range_um'][0]:.0f}~{r['spread_range_um'][1]:.0f}",
                "입자 수": r["n_particles"],
                "사진 수": r["n_photos"],
            }
            for r in reversed(records)
        )
        st.dataframe(table, hide_index=True)


def grind_tab():
    st.markdown(
        "축척 용지의 가루 칸에 원두 가루를 **듬성듬성** 펴고, 네 모서리 마커가 모두 보이게 찍은 사진을 넣어 주세요. "
        "**같은 가루로 찍은 사진**은 여러 장 넣으면 하나의 측정으로 합쳐져서 결과가 더 안정돼요."
    )
    c1, c2, c3, c4 = st.columns(4)
    bean_id = c1.text_input("원두 번호", "B01")
    grinder = c2.text_input("그라인더", "메버릭1")
    clicks = c3.number_input("클릭 수", min_value=0, max_value=180, value=95, step=1)
    brew_id = c4.text_input("추출번호 (선택)", "", placeholder="예: 20261012-1")

    source = st.radio("사진 넣는 방법", ["사진 파일 올리기", "카메라로 찍기"], horizontal=True)
    if source == "사진 파일 올리기":
        uploads = st.file_uploader(
            "사진 (여러 장 가능)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True
        ) or []
    else:
        st.caption("브라우저 카메라는 해상도가 낮아 작은 입자를 놓칠 수 있어요. 폰이나 카메라로 찍은 원본을 올리면 더 정확해요.")
        shot = st.camera_input("가루 칸 찍기", resolution="1080p")
        uploads = [shot] if shot else []

    if not uploads:
        st.info("사진을 넣으면 사진마다 입자를 표시하고, 대표 크기와 분포를 보여줘요.")
        show_saved_profiles()
        return

    st.subheader("사진별 결과")
    st.caption("초록: 잰 입자, 빨강: 서로 붙어서 뺀 덩어리, 보라: 칸 테두리에 걸려 뺀 입자")
    measurements, names = [], []
    for upload in uploads:
        with st.spinner(f"{upload.name} 분석 중..."):
            rectified, measurement, error = analyze_upload(upload.getvalue())
        with st.expander(upload.name, expanded=len(uploads) == 1 or error is not None):
            if error:
                st.error(error)
                continue
            show_photo_result(upload.name, rectified, measurement)
        if len(measurement.diameters_um):
            measurements.append(measurement)
            names.append(upload.name)

    if not measurements:
        st.error("입자를 잰 사진이 없어요. 가루를 칸 안쪽에 듬성듬성 펴고 다시 찍어 주세요.")
        show_saved_profiles()
        return

    st.subheader(f"이번 측정 결과 (사진 {len(measurements)}장)")
    profile = ga.combine_measurements(measurements)
    if ga.size_distribution_curve(profile.diameters_um) is None:
        st.warning("잰 입자가 너무 적어 분포를 계산할 수 없어요. 사진을 더 넣어 주세요.")
        show_saved_profiles()
        return
    show_profile(profile, grinder, int(clicks))

    if st.button("이 측정 저장", type="primary"):
        record = ga.profile_record(profile, bean_id, grinder, int(clicks), brew_id, names)
        path = ga.save_profile(record, PROFILE_DIR)
        st.success(f"저장했어요: {path.name} (data/local/grind_profiles, 커밋되지 않는 폴더)")
    show_saved_profiles()


st.set_page_config(page_title="Drip Coffee", layout="wide")
st.title("Drip Coffee: 핸드드립 추출 분석·추천")

tab_grind, tab_predict, tab_recommend, tab_log = st.tabs(
    ["① 분쇄도 측정", "② 추출 예측", "③ 레시피 추천", "④ 기록·보정"]
)

with tab_grind:
    grind_tab()

with tab_predict:
    st.info("레시피와 분쇄도로 농도, 추출수율, 물 빠짐 시간을 예측하는 화면입니다. (준비 중)")

with tab_recommend:
    st.info("목표 농도와 추출수율에 맞는 레시피를 추천하는 화면입니다. (준비 중)")

with tab_log:
    st.info("실제 추출 결과를 기록하고 예측과 비교해 모델을 보정하는 화면입니다. (준비 중)")
