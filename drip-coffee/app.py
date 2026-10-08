"""핸드드립 추출 분석·추천 도구의 화면 (Streamlit).

화면은 입력과 표시만 맡고, 계산은 dripcoffee 패키지에서 한다.
실행: streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Drip Coffee", layout="wide")
st.title("Drip Coffee: 핸드드립 추출 분석·추천")

tab_grind, tab_predict, tab_recommend, tab_log = st.tabs(
    ["① 분쇄도 측정", "② 추출 예측", "③ 레시피 추천", "④ 기록·보정"]
)

with tab_grind:
    st.info("축척 용지 위에 펼친 원두 가루 사진으로 분쇄도를 재는 화면입니다. (준비 중)")

with tab_predict:
    st.info("레시피와 분쇄도로 농도, 추출수율, 물 빠짐 시간을 예측하는 화면입니다. (준비 중)")

with tab_recommend:
    st.info("목표 농도와 추출수율에 맞는 레시피를 추천하는 화면입니다. (준비 중)")

with tab_log:
    st.info("실제 추출 결과를 기록하고 예측과 비교해 모델을 보정하는 화면입니다. (준비 중)")
