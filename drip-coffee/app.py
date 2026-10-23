"""핸드드립 추출 분석·추천 도구의 화면 (Streamlit).

화면은 입력과 표시만 맡고, 계산은 dripcoffee 패키지에서 한다.
구조: 첫 화면의 '시작하기'로 한 잔 내리기 흐름(원두와 장비 → 기초 추천 → 분쇄 확인 → 내리기와 기록 → 맛 평가 →
마무리)을 한 화면씩 따라간다. '내 기록 보기'와 '모델 점검'은 첫 화면의 작은 링크로 들어간다.
디자인은 토스(Toss) 스타일을 따른다: 흰 배경, 포인트 컬러 하나(파랑)와 무채색, 은은한 그림자,
큰 숫자와 작은 단위·설명, 테두리와 구분선 대신 넓은 여백. 색과 글꼴은 .streamlit/config.toml에 있다.
실행: streamlit run app.py
"""
import math
import os
from datetime import date
from pathlib import Path
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

PROFILE_DIR = Path(os.environ.get("DRIP_COFFEE_PROFILES", LOCAL_DATA_DIR / "grind_profiles"))  # 분쇄 측정 저장 폴더
CLICK_TABLE = DATA_DIR / "reference" / "maverick_click_table.csv"
# 추출 기록과 원두 목록. 테스트·미리보기에서는 환경 변수로 다른 파일을 쓰게 해 실제 기록을 건드리지 않는다.
BREWS_PATH = Path(os.environ.get("DRIP_COFFEE_BREWS", cal.BREWS_PATH))
BEANS_PATH = Path(os.environ.get("DRIP_COFFEE_BEANS", cal.BEANS_PATH))
EQUIPMENT_PATH = Path(os.environ.get("DRIP_COFFEE_EQUIPMENT", cal.EQUIPMENT_PATH))

BRAND = "#A85A17"  # 포인트 컬러(캐러멜). 사진 위 잰 입자 표시만 커피색과 구별되게 파랑(OUTLINE_COLORS)
TEXT = "#2A2A2A"
GRAY_DARK = "#4E5968"
GRAY = "#8B95A1"
GRAY_LIGHT = "#B0B8C1"
FONT = "Pretendard Variable, sans-serif"

# 사진 위 윤곽선 (RGB): 잰 입자만 포인트 컬러로 그리고, 뺀 입자(덩어리, 칸 테두리, 은피)는 회색으로 낮춘다.
OUTLINE_COLORS = {"measured": (49, 130, 246), "clump": (139, 149, 161), "border": (139, 149, 161),
                  "chaff": (139, 149, 161)}
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
/* 움직임 값은 토스에서 직접 뽑았다 (10/20, toss.im 코드와 TDS Mobile 문서).
   튀는 이동 0.2~0.3초 cubic-bezier(0.34, 1.56, 0.64, 1), 부드러운 이동 cubic-bezier(0.4, 0, 0.2, 1), 색 바뀜 0.1초 */
:root { --dc-spring: cubic-bezier(0.34, 1.56, 0.64, 1); --dc-ease: cubic-bezier(0.4, 0, 0.2, 1); }
.block-container { max-width: 980px; padding-top: 3.5rem; padding-bottom: 10rem; }  /* 아래 고정 버튼에 가리지 않게 */
.stTabs [data-baseweb="tab-list"] { gap: 28px; }
.stTabs [data-baseweb="tab"] { font-size: 17px; font-weight: 600; padding: 10px 0; }
.stTabs [data-baseweb="tab-border"] { background: #F2F4F6; }
[data-testid="stExpander"] details { border: none; background: #F9FAFB; border-radius: 20px; }
[data-testid="stExpander"] summary { padding: 18px 22px; }
[data-testid="stExpander"] summary p { font-size: 16px; font-weight: 600; }

.dc-eyebrow { font-size: 15px; font-weight: 600; color: #A85A17; margin-bottom: 6px; }
.dc-home { margin: 8vh 0 28px; }
[data-testid="stBaseButton-primary"] { min-height: 56px; border-radius: 16px; }  /* 토스 BottomCTA: 높이 56px, 모서리 16px */
[data-testid="stBaseButton-primary"] p { font-size: 17px; font-weight: 700; }
.dc-hero { font-size: 40px; font-weight: 800; color: #2A2A2A; letter-spacing: -0.03em; line-height: 1.25; margin: 4px 0 8px; }
.dc-title { font-size: 32px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; line-height: 1.3; }
.dc-lead { font-size: 16px; color: #4E5968; line-height: 1.6; margin: 4px 0 28px; word-break: keep-all; }  /* 낱말 중간에서 줄을 바꾸지 않는다 */
.dc-section { font-size: 22px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; margin: 48px 0 4px; }
.dc-caption { font-size: 14px; color: #8B95A1; margin-bottom: 16px; }

.dc-cards { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 20px 0 8px; }
.dc-card { background: #FFFFFF; border-radius: 24px; padding: 28px 28px 24px;
           box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04); animation: dc-rise .4s var(--dc-ease) both; }
.dc-cards.three { grid-template-columns: repeat(3, 1fr); }  /* 큰 카드 하나 + 작은 카드 셋 */
.dc-card.hero { grid-column: 1 / -1; }
.dc-card:nth-child(2) { animation-delay: .07s; }
.dc-card:nth-child(3) { animation-delay: .14s; }
.dc-label { font-size: 15px; font-weight: 500; color: #6B7684; margin-bottom: 6px; }
.dc-value { font-size: 48px; font-weight: 700; color: #2A2A2A; line-height: 1.2;
            letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }
.dc-card.hero .dc-value { color: #A85A17; }
.dc-unit { font-size: 24px; font-weight: 600; color: #4E5968; margin-left: 6px; letter-spacing: 0; }
.dc-sub { font-size: 14px; color: #8B95A1; margin-top: 10px; line-height: 1.5; }

.dc-notice { background: #F2F4F6; border-radius: 16px; padding: 16px 20px;
             font-size: 15px; color: #4E5968; line-height: 1.6; animation: dc-rise .3s var(--dc-ease) both; }
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
.dc-hexagon-score b { font-size: 24px; font-weight: 700; color: #A85A17; margin-left: 4px; font-variant-numeric: tabular-nums; }
.dc-hexagon .dc-caption { margin: 6px 16px 0; }

@keyframes dc-rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }

/* 한 잔 내리기: 단계가 바뀌면 새 화면이 살짝 올라오며 나타난다. 단계마다 상자(key)가 바뀌어도 화면 요소는 다시 쓰일 수
   있어서, 홀수·짝수 단계의 애니메이션 이름을 달리해 매번 다시 재생되게 한다 */
[class*="st-key-flow-step-"] { animation: .3s var(--dc-ease) both; }
.st-key-flow-step-1, .st-key-flow-step-3, .st-key-flow-step-5 { animation-name: dc-step-a; }
.st-key-flow-step-2, .st-key-flow-step-4, .st-key-flow-step-6 { animation-name: dc-step-b; }
@keyframes dc-step-a { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }
@keyframes dc-step-b { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }

/* 버튼을 누르면 살짝 작아졌다가 튀듯이 돌아온다 */
[data-testid^="stBaseButton"] { transition: transform .2s var(--dc-spring), background-color .1s ease-in-out,
                                            color .1s ease-in-out; }
[data-testid^="stBaseButton"]:active { transform: scale(.97); }

/* 진행 막대가 다음 단계로 부드럽게 찬다 */
[data-testid="stProgressBarTrack"] > div { transition: transform .4s var(--dc-ease); }

/* 큰 숫자가 0부터 올라가며 나타난다. 정수만. 화면 읽기 프로그램과 글자 복사는 숨긴 진짜 숫자를 쓴다 */
@property --dc-n { syntax: "\\3C integer\\3E"; initial-value: 0; inherits: false; }  /* 정수. 꺾쇠를 그대로 쓰면 HTML 정리기가 스타일을 통째로 지워서 CSS 이스케이프로 쓴다 */
.dc-count { counter-reset: dc-n var(--dc-n); animation: dc-count .7s var(--dc-ease) both; }
.dc-count::after { content: counter(dc-n); }
@keyframes dc-count { from { --dc-n: 0; } }
.dc-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }

/* 한 잔 내리기의 아래 고정 버튼 (토스 디자인 시스템 BottomCTA: 화면 아래에 붙고, 위로 36px 흰 그라데이션,
   아래 여백 20px). 다 채우기 전에는 버튼 자리에 무엇을 채우면 되는지 흐리게 보여 주고, 다 채우면 버튼이 아래에서
   튀어 올라온다 */
.st-key-flow-cta, .st-key-home-cta { position: fixed; bottom: 0; left: 50%; transform: translateX(-50%); z-index: 100;
                   width: 100%; max-width: 980px; padding: 8px 1rem 20px; background: #FFFFFF; }
.st-key-flow-cta::before, .st-key-home-cta::before { content: ""; position: absolute; left: 0; right: 0; top: -36px; height: 36px;
                           pointer-events: none; background: linear-gradient(to top, #FFFFFF 25%, rgba(255, 255, 255, 0)); }
.st-key-flow-cta [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: 8px; }  /* 좁은 화면에서도 한 줄 */
.st-key-flow-cta [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
.st-key-flow-cta [data-testid="stColumn"]:last-child { flex: 2 1 0 !important; }
.st-key-flow-cta [data-testid="stBaseButton-secondary"] { min-height: 56px; border-radius: 16px; border: none;
                                                          background: #F2F4F6; color: #4E5968; }
.st-key-flow-cta [data-testid="stBaseButton-secondary"] p,
.st-key-flow-cta [data-testid="stBaseButton-primary"] p { font-size: 17px; font-weight: 600; }
.st-key-flow-cta [class*="st-key-flow-next"]:not([class*="-hint"]), .st-key-flow-cta .st-key-flow-save,
.st-key-flow-cta .st-key-flow-finish-home, .st-key-home-start { animation: dc-cta-in .35s var(--dc-spring) both; }
@keyframes dc-cta-in { from { opacity: 0; transform: translateY(32px); } to { opacity: 1; transform: none; } }
/* 아직 갈 수 없을 때의 흐린 안내(cta_hint). 누르면 좌우로 흔들리고 글자가 잠깐 빨개진다. 누를 때마다 상자가 새로
   그려지므로, 아래에서 올라오는 움직임은 처음 나타날 때(still)만 주고 다음 버튼의 튀어 오르기에서도 뺀다
   (누를 때마다 다시 올라오면 위아래로도 흔들려 보인다) */
.st-key-flow-cta [class*="st-key-cta-hint-still-"] { animation: dc-rise .3s var(--dc-ease) both; }
.st-key-flow-cta [class*="st-key-cta-hint-"] button { min-height: 56px; padding: 0 16px; border: none;
                                                      border-radius: 16px; background: #F2F4F6; color: #8B95A1;
                                                      cursor: default; }
.st-key-flow-cta [class*="st-key-cta-hint-"] button:hover,
.st-key-flow-cta [class*="st-key-cta-hint-"] button:focus:not(:active) {
  border: none; background: #F2F4F6; color: #8B95A1; }
.st-key-flow-cta [class*="st-key-cta-hint-"] button p { font-size: 15px; font-weight: 600; line-height: 1.4; }
.st-key-flow-cta [class*="st-key-cta-hint-a-"] button { animation: dc-shake-a .42s var(--dc-ease) both; }
.st-key-flow-cta [class*="st-key-cta-hint-b-"] button { animation: dc-shake-b .42s var(--dc-ease) both; }
@keyframes dc-shake-a { 0%, 100% { transform: none; } 15% { transform: translateX(-9px); color: #F04452; }
                        35% { transform: translateX(8px); } 55% { transform: translateX(-5px); }
                        75% { transform: translateX(3px); color: #F04452; } }
@keyframes dc-shake-b { 0%, 100% { transform: none; } 15% { transform: translateX(-9px); color: #F04452; }
                        35% { transform: translateX(8px); } 55% { transform: translateX(-5px); }
                        75% { transform: translateX(3px); color: #F04452; } }

/* 아이콘 요약 카드: 숫자 여러 개를 회색 판 위 흰 칸으로 한 번에 보여 준다 (토스 자산 목록처럼). 아이콘은 Streamlit에
   들어 있는 Material Symbols 글꼴이라 인터넷 없이도 보인다. 2×2로 두고, 넓은 화면에서만 한 줄 4칸(3+1로 깨지지 않게) */
.dc-summary { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; padding: 8px;
              margin: 20px 0 8px; border-radius: 24px; background: #F2F4F6; animation: dc-rise .4s var(--dc-ease) both; }
.dc-tile { background: #FFFFFF; border-radius: 18px; padding: 16px 16px 14px; min-width: 0; }
@media (min-width: 760px) { .dc-summary { grid-template-columns: repeat(4, minmax(0, 1fr)); } }
.dc-icon { font-family: "Material Symbols Rounded"; font-size: 20px; line-height: 1; width: 36px; height: 36px;
           border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-bottom: 10px;
           background: #FBEFE3; color: #A85A17; font-variation-settings: "FILL" 1; }
.dc-tile-label { font-size: 14px; color: #6B7684; }
.dc-tile-value { font-size: 28px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; line-height: 1.25;
                 font-variant-numeric: tabular-nums; }
.dc-tile-value.text { font-size: 20px; line-height: 1.4; }
.dc-tile-value small { font-size: 16px; font-weight: 600; color: #4E5968; margin-left: 3px; letter-spacing: 0; }
.dc-tile-sub { font-size: 13px; color: #8B95A1; margin-top: 2px; line-height: 1.4; }
.dc-tile-effect { font-size: 12px; color: #874711; background: #FBEFE3; border-radius: 8px; padding: 5px 7px;
                  margin-top: 8px; line-height: 1.4; }
.dc-summary.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.dc-evidence { display: flex; flex-wrap: wrap; gap: 6px; margin: -16px 0 4px; }
.dc-evidence span { font-size: 12px; font-weight: 600; padding: 5px 10px; border-radius: 100px; background: #FBEFE3;
                    color: #874711; }
.dc-verdict { font-size: 14px; line-height: 1.5; margin: 4px 0 8px; padding: 10px 12px; border-radius: 12px;
              background: #FBEFE3; color: #874711; }

/* 붓기 타임라인: 막대 위치는 붓는 시각, 길이는 붓는 양. 위는 붓는 양, 아래는 시각과 저울에 보일 누적.
   글자는 화면 크기와 상관없이 같은 크기로 보이게 SVG가 아니라 위치를 %로 둔 상자로 그린다. 막대는 차례로 자란다 */
.dc-tl { position: relative; height: 128px; margin: 16px 10px 8px; }
.dc-tl-track { position: absolute; left: 0; right: 0; top: 52px; height: 12px; border-radius: 6px; background: #F2F4F6; }
.dc-tl-pour { position: absolute; top: 0; height: 128px; }
.dc-tl-bar { position: absolute; left: 0; right: 0; top: 42px; height: 32px; min-width: 10px; border-radius: 8px;
             background: #A85A17; transform-origin: left center; animation: dc-grow .45s var(--dc-spring) both;
             animation-delay: calc(var(--i) * 90ms); }
.dc-tl-pour.bloom .dc-tl-bar { background: #D9A877; }
.dc-tl-g, .dc-tl-t, .dc-tl-c { position: absolute; left: 50%; transform: translateX(-50%); white-space: nowrap; }
.dc-tl-g { top: 14px; font-size: 15px; font-weight: 700; color: #2A2A2A; }
.dc-tl-t { top: 82px; font-size: 13px; font-weight: 600; color: #4E5968; font-variant-numeric: tabular-nums; }
.dc-tl-c { top: 102px; font-size: 12px; color: #8B95A1; }
.dc-tl-end { position: absolute; top: 4px; bottom: 34px; border-left: 2px dashed #B0B8C1; }
.dc-tl-end span { position: absolute; top: -6px; right: 8px; white-space: nowrap; font-size: 13px; font-weight: 700;
                  color: #A85A17; }
@keyframes dc-grow { from { transform: scaleX(0); } to { transform: none; } }

/* 맛 평가: 슬라이더를 '좋았던 정도'와 '세기' 두 묶음의 회색 판에 담고, 육각형은 위에 작게 둔다 */
.st-key-taste-good, .st-key-taste-strength { background: #F9FAFB; border-radius: 20px; padding: 14px 18px 6px; }
.dc-group { font-size: 15px; font-weight: 700; color: #2A2A2A; }
.dc-group span { font-size: 13px; font-weight: 500; color: #8B95A1; margin-left: 6px; }
.dc-hexagon.compact { display: flex; align-items: center; gap: 12px; padding: 8px 16px; text-align: left; }
.dc-hexagon.compact svg { max-width: 190px; flex: 1 1 auto; }
.dc-hexagon.compact .dc-hexagon-score { margin: 0; flex: none; }
.dc-hexagon.compact .dc-hexagon-score b { display: block; font-size: 36px; margin: 0; }
.dc-chip { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px 6px 8px; border-radius: 100px;
           background: #FBEFE3; color: #874711; font-size: 14px; font-weight: 600; margin: 4px 0 8px; }
.dc-chip .dc-icon { width: 24px; height: 24px; font-size: 15px; margin: 0; background: #FFFFFF; }
/* 진하기: 세 가지를 아이콘이 있는 큰 선택 카드로 */
.st-key-flow-strength [data-testid="stButtonGroup"] > div { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
/* 2단계 '오늘은 어떻게?': 진하기 카드와 같은 모양으로 다섯 칸 */
.st-key-flow-intent [data-testid="stButtonGroup"] > div { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
.st-key-flow-intent button { min-height: 68px; padding: 6px 2px; border-radius: 14px !important;
                             border: 1px solid #E5E8EB !important; background: #FFFFFF; }
.st-key-flow-intent button p { font-size: 13px; font-weight: 600; line-height: 1.3; }
.st-key-flow-intent button [data-testid="stIconMaterial"] { font-size: 20px; }
.st-key-flow-intent button div > span { flex-direction: column; align-items: center; gap: 2px; }  /* 아이콘 위, 글자 아래 */
.st-key-flow-intent button, .st-key-flow-intent button div, .st-key-flow-intent button p {
  overflow: visible; white-space: normal; text-overflow: clip; word-break: keep-all; text-align: center; }
.st-key-flow-intent button[aria-checked="true"] { border: 2px solid #A85A17 !important; background: #FBEFE3; color: #874711; }
.st-key-flow-strength button { min-height: 76px; border-radius: 16px !important; border: 1px solid #E5E8EB !important;
                               background: #FFFFFF; }
.st-key-flow-strength button p { font-size: 15px; font-weight: 600; }
.st-key-flow-strength button [data-testid="stIconMaterial"] { font-size: 24px; }
.st-key-flow-strength button div { flex-direction: column; gap: 4px; }
.st-key-flow-strength button[aria-checked="true"] { border: 2px solid #A85A17 !important; background: #FBEFE3;
                                                   color: #874711; }
/* 6단계의 기록 완료 */
.dc-done { text-align: center; }
.dc-done .dc-icon { width: 64px; height: 64px; font-size: 34px; margin: 4px auto 14px; animation: dc-pop .45s var(--dc-spring) both; }
.dc-done-title { font-size: 22px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; }
.dc-done-sub { font-size: 14px; color: #8B95A1; margin-top: 4px; }
@keyframes dc-pop { from { transform: scale(.4); opacity: 0; } to { transform: none; opacity: 1; } }
/* 다음 잔은 이렇게 바뀌어요: 바뀐 값과 변화량 */
.dc-change { background: #FFFFFF; border-radius: 24px; padding: 6px 20px; box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04); }
.dc-change-row { display: flex; align-items: center; gap: 10px; padding: 14px 0; border-bottom: 1px solid #F2F4F6; }
.dc-change-row:last-child { border-bottom: none; }
.dc-change-row .dc-icon { width: 32px; height: 32px; font-size: 18px; margin: 0; }
.dc-change-name { flex: 1; min-width: 0; font-size: 15px; color: #4E5968; }
.dc-change-name small { display: block; font-size: 12px; color: #8B95A1; margin-top: 2px; }
.dc-change-from { font-size: 14px; color: #B0B8C1; }
.dc-change-to { font-size: 17px; font-weight: 700; color: #2A2A2A; }
.dc-pill { font-size: 12px; font-weight: 700; padding: 3px 8px; border-radius: 100px; background: #FBEFE3; color: #874711; }
.dc-pill.same { background: #F2F4F6; color: #8B95A1; }

/* 시작 화면: 커피 아이콘이 튀어나오고, 지난 잔과 내 기록은 누를 수 있는 목록으로 */
.dc-home-icon { width: 64px; height: 64px; font-size: 34px; margin-bottom: 18px; animation: dc-pop .45s var(--dc-spring) both; }
.st-key-home-list, .st-key-check-list { gap: 0; padding: 6px 0; border-radius: 24px; background: #FFFFFF;
                    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04); animation: dc-rise .4s .08s var(--dc-ease) both; }
.st-key-home-check button p { font-size: 13px; color: #8B95A1; }
/* 누를 수 있는 목록 한 줄 (토스 ListRow, list_row): 보이는 내용은 HTML, 그 위를 투명한 버튼이 통째로 덮는다 */
[class*="st-key-row-"] { position: relative; gap: 0; }
[class*="st-key-row-"] [data-testid="stElementContainer"]:has(button) { position: absolute; inset: 0; z-index: 1; }
[class*="st-key-row-"] [data-testid="stElementContainer"]:has(button) div { height: 100%; }
[class*="st-key-row-"] button { width: 100%; height: 100%; min-height: 0; padding: 0; border: none; border-radius: 16px;
                                background: transparent; color: transparent; }
[class*="st-key-row-"] button:hover, [class*="st-key-row-"] button:focus { background: transparent; color: transparent; }
[class*="st-key-row-"] button p { color: transparent; }
.dc-lrow { display: flex; align-items: center; gap: 12px; min-height: 68px; padding: 12px 20px; border-radius: 16px;
           transition: background-color .1s ease-in-out, transform .2s var(--dc-spring); }
[class*="st-key-row-"]:hover .dc-lrow { background: #F9FAFB; }
[class*="st-key-row-"]:has(button:active) .dc-lrow { transform: scale(.98); }
.dc-lrow .dc-icon, .dc-avatar { flex: none; width: 40px; height: 40px; margin: 0; }
.dc-avatar { border-radius: 50%; background: #FBEFE3; color: #874711; font-size: 13px; font-weight: 700;
             display: flex; align-items: center; justify-content: center; }
.dc-lrow-text { flex: 1; min-width: 0; }
.dc-lrow-text b { display: block; font-size: 16px; font-weight: 600; color: #2A2A2A; overflow: hidden;
                  white-space: nowrap; text-overflow: ellipsis; }
.dc-lrow-text small { display: block; font-size: 13px; color: #8B95A1; margin-top: 1px; overflow: hidden;
                      white-space: nowrap; text-overflow: ellipsis; }
.dc-lrow-go { font-size: 14px; font-weight: 600; color: #A85A17; white-space: nowrap; }
.dc-lrow-go.muted { color: #8B95A1; font-weight: 500; }
.dc-lrow-chev { font-family: "Material Symbols Rounded"; font-size: 22px; color: #B0B8C1; margin-left: -8px; }
@media (max-width: 480px) { .dc-wide { display: none; } .dc-lrow { gap: 10px; padding: 12px 16px; } }
.dc-lrow.on { background: #FBEFE3; }
/* 모델 점검 > 예측과 실제: 가운데 막대가 판정 기준(±30초), 점이 예측과 실제의 차이. 기준 밖은 빨갛게, 표시 잔은 회색 */
.dc-band { position: relative; flex: none; width: 72px; height: 16px; }
.dc-band::before { content: ""; position: absolute; left: 0; right: 0; top: 6px; height: 4px; border-radius: 2px;
                   background: #F3DCC4; }
.dc-band::after { content: ""; position: absolute; left: 50%; top: 2px; width: 1px; height: 12px; background: #B0B8C1; }
.dc-band i { position: absolute; top: 3px; z-index: 1; width: 10px; height: 10px; margin-left: -5px; border-radius: 50%;
             background: #A85A17; }
.dc-band i.out { background: #F04452; }
.dc-band i.muted { background: #B0B8C1; }
.dc-check-pill { flex: none; min-width: 52px; text-align: center; }
.dc-pill.out { background: #FFF1F1; color: #F04452; }
.dc-pill.muted { background: #F2F4F6; color: #8B95A1; }

/* 내 기록: 날짜별 잔 목록. 표시(막힘 등)가 있는 잔은 흐리게, 사진이 없는 잔은 꼬리표 */
.dc-cups { animation: dc-rise .4s var(--dc-ease) both; }
.dc-cups-date { font-size: 13px; font-weight: 600; color: #8B95A1; margin: 20px 0 2px; }
.dc-cup { display: flex; align-items: center; gap: 12px; padding: 12px 0; border-bottom: 1px solid #F2F4F6; }
.dc-cup .dc-avatar { flex: none; width: 40px; height: 40px; }
.dc-cup.muted .dc-avatar { background: #F2F4F6; color: #8B95A1; }
.dc-cup-text { flex: 1; min-width: 0; }
.dc-cup-text b { font-size: 16px; font-weight: 600; color: #2A2A2A; }
.dc-cup-text small { display: block; font-size: 13px; color: #8B95A1; margin-top: 1px; }
.dc-cup-tag { display: inline-block; margin-left: 6px; padding: 1px 7px; border-radius: 6px; background: #F2F4F6;
              color: #6B7684; font-size: 12px; font-weight: 600; vertical-align: 2px; }
.dc-cup-score { flex: none; text-align: right; }
.dc-cup-score b { font-size: 17px; font-weight: 700; color: #2A2A2A; font-variant-numeric: tabular-nums; }
.dc-cup.muted .dc-cup-score b { color: #B0B8C1; }
.dc-cup-score small { display: block; font-size: 12px; color: #8B95A1; }
.st-key-cup-list { gap: 0; }
[class*="st-key-row-cup-"] .dc-cup { margin: 0 -8px; padding: 12px 8px; border-bottom: none; border-radius: 14px;
                                     transition: background-color .1s ease-in-out, transform .2s var(--dc-spring); }
[class*="st-key-row-cup-"]:hover .dc-cup { background: #F9FAFB; }
[class*="st-key-row-cup-"]:has(button:active) .dc-cup { transform: scale(.98); }
.dc-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin: 0 4px 0 10px; }

/* 사용법(튜토리얼): 장마다 움직이는 그림. 그림은 화면에 새로 그려질 때 한 번 재생된다 */
[class*="st-key-home-guide-"] { margin: -20px 0 8px; }
.st-key-home-tutorial button p { font-size: 15px; font-weight: 500; color: #8B95A1; }
.st-key-home-guide-new .st-key-home-tutorial button p { font-weight: 600; color: #A85A17; }
.st-key-tutorial-skip button p { color: #8B95A1; }
.dc-tut-art { position: relative; height: 240px; margin: 4px 0 24px; border-radius: 24px; background: #F2F4F6;
              overflow: hidden; }
.dc-tut-title { font-size: 26px; font-weight: 700; color: #2A2A2A; letter-spacing: -0.02em; line-height: 1.3;
                animation: dc-rise .4s var(--dc-ease) both; }
.dc-tut-lead { font-size: 16px; color: #4E5968; line-height: 1.6; margin-top: 8px; word-break: keep-all;
               animation: dc-rise .4s .06s var(--dc-ease) both; }
.dc-dots { display: flex; gap: 6px; justify-content: center; margin-top: 28px; }
.dc-dots i { width: 6px; height: 6px; border-radius: 3px; background: #E5E8EB; }
.dc-dots i.on { width: 18px; background: #A85A17; animation: dc-dot .3s var(--dc-ease) both; }
@keyframes dc-dot { from { width: 6px; } }
@keyframes dc-pop-in { 0% { opacity: 0; transform: scale(.6); } 70% { opacity: 1; transform: scale(1.08); }
                       100% { opacity: 1; transform: none; } }
/* 1장: 도구 이름이 하나씩 나올 때마다 가운데 그림이 그 도구로 바뀐다(회전하며 커짐). 이름표 줄은 가운데 정렬이라
   새 이름표의 자리가 0에서 넓어지면서 앞의 것들을 왼쪽으로 민다 */
.dc-tool { position: absolute; top: 34px; left: 50%; width: 96px; height: 96px; margin-left: -48px; fill: none;
           stroke: #A85A17; stroke-width: 3.8; stroke-linecap: round; stroke-linejoin: round; opacity: 0;
           animation: dc-tool-in .45s var(--dc-spring) var(--d) both,
                      dc-tool-out .2s var(--dc-ease) calc(var(--d) + .68s) forwards; }
.dc-tool.last { animation: dc-tool-in .5s var(--dc-spring) var(--d) both; }
@keyframes dc-tool-in { from { opacity: 0; transform: scale(.3) rotate(-25deg); } to { opacity: 1; transform: none; } }
@keyframes dc-tool-out { to { opacity: 0; transform: scale(.6) translateY(-12px); } }
.dc-tool-names { position: absolute; left: 0; right: 0; bottom: 28px; display: flex; justify-content: center; }
.dc-tool-slot { display: flex; justify-content: center; max-width: 0; animation: dc-slot .45s var(--dc-ease) var(--d) both; }
.dc-tool-slot + .dc-tool-slot { animation-name: dc-slot-next; }
@keyframes dc-slot { from { max-width: 0; } to { max-width: 140px; } }
@keyframes dc-slot-next { from { max-width: 0; margin-left: 0; } to { max-width: 140px; margin-left: 6px; } }
.dc-tool-name { display: flex; align-items: center; white-space: nowrap; padding: 6px 11px; border-radius: 12px;
                font-size: 14px; font-weight: 600;
                animation: dc-pop-in .45s var(--dc-spring) calc(var(--d) + .06s) both, dc-name-on .75s linear var(--d) both; }
@keyframes dc-name-on { from, 85% { background: #A85A17; color: #FFFFFF; } to { background: #FFFFFF; color: #2A2A2A; } }
.dc-tool-check { width: 0; overflow: hidden; opacity: 0; color: #A85A17;
                 animation: dc-check-in .3s var(--dc-ease) calc(var(--d) + .7s) both; }
@keyframes dc-check-in { from { width: 0; opacity: 0; } to { width: 16px; opacity: 1; } }
/* 2장: 축척 용지 위 가루를 선이 훑고 지나가면 알갱이마다 파란 테두리(실제 분석 화면과 같은 색)가 생기고,
   대표 크기가 튀어나온다 */
.dc-tut-paper { position: absolute; inset: 22px 32px; border-radius: 10px; background: #FFFFFF;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06); }
.dc-tut-paper b { position: absolute; width: 12px; height: 12px; background: #2A2A2A; }
.dc-grain { position: absolute; border-radius: 45%; background: #4A3426; animation: dc-ring .3s ease-out both; }
@keyframes dc-ring { from { box-shadow: 0 0 0 0 rgba(49, 130, 246, 0); } to { box-shadow: 0 0 0 2px #3182F6; } }
.dc-scan { position: absolute; top: 0; bottom: 0; width: 2px; background: #3182F6; box-shadow: 0 0 12px #3182F6;
           animation: dc-scan 1.6s .3s var(--dc-ease) both; }
@keyframes dc-scan { from { left: 0; opacity: 1; } 90% { opacity: 1; } to { left: 100%; opacity: 0; } }
.dc-tut-size { position: absolute; right: 12px; bottom: 12px; padding: 4px 12px; border-radius: 100px;
               background: #A85A17; color: #FFFFFF; font-size: 14px; font-weight: 700;
               animation: dc-pop-in .45s var(--dc-spring) 2s both; }
/* 3장: 커피포트가 붓기 시간표를 4.5초에 지나간다. 파란 칸(붓는 때)에 닿으면 기울어 붓고, 칸이 끝날 때쯤 다시
   똑바로 선다. 칸은 포트가 지나가는 만큼 채워지고, 기울었을 때 주둥이 끝이 그 채워지는 끝 바로 위에 온다.
   기울기의 % 값은 TUTORIAL_POURS(4–18, 34–50, 64–80)에 맞춘 것이라 칸을 바꾸면 함께 바꾼다 */
.dc-tut-track { position: absolute; left: 80px; right: 28px; top: 124px; height: 12px; border-radius: 6px;
                background: #FFFFFF; }
.dc-tut-pour { position: absolute; top: 0; bottom: 0; overflow: hidden; border-radius: 6px; background: #F3DCC4; }
.dc-tut-pour::after { content: ""; position: absolute; inset: 0; background: #A85A17; transform-origin: left center;
                      animation: dc-fill var(--t) linear var(--d) both; }
@keyframes dc-fill { from { transform: scaleX(0); } }
.dc-tut-time { position: absolute; top: 22px; font-size: 13px; font-weight: 600; color: #8B95A1; white-space: nowrap; }
.dc-tut-pourer { position: absolute; top: 0; left: 0; animation: dc-play 4.5s .3s linear both; }
@keyframes dc-play { from { left: 0; } to { left: 100%; } }
.dc-kettle { position: absolute; left: -61px; top: -86px; width: 64px; height: 64px; fill: none; stroke: #A85A17;
             stroke-width: 3.4; stroke-linecap: round; stroke-linejoin: round; transform-origin: 50% 60%;
             animation: dc-kettle-tilt 4.5s .3s var(--dc-ease) both; }
@keyframes dc-kettle-tilt { 0%, 2% { transform: none; } 5%, 16% { transform: rotate(30deg); } 20%, 32% { transform: none; }
                            35%, 48% { transform: rotate(30deg); } 52%, 62% { transform: none; }
                            65%, 78% { transform: rotate(30deg); } 82%, 100% { transform: none; } }
.dc-tut-note { position: absolute; left: 0; right: 0; bottom: 20px; text-align: center; font-size: 14px; color: #4E5968; }
/* 4장: 맛 육각형이 커지고, 다음 잔이 어떻게 바뀌는지 한 줄이 올라온다 */
.dc-tut-hex { position: absolute; top: 12px; left: 50%; width: 170px; height: 158px; margin-left: -85px; }
.dc-tut-shape { transform-origin: 75px 70px; animation: dc-hex-grow .7s .2s var(--dc-spring) both; }
@keyframes dc-hex-grow { from { transform: scale(0); } to { transform: none; } }
.dc-tut-change { position: absolute; left: 16px; right: 16px; bottom: 14px; display: flex; align-items: center; gap: 8px;
                 padding: 10px 14px; border-radius: 16px; background: #FFFFFF; font-size: 14px; color: #4E5968;
                 animation: dc-rise .45s 1.1s var(--dc-spring) both; }
.dc-tut-change .dc-icon { width: 28px; height: 28px; font-size: 16px; margin: 0; }
.dc-tut-change b { color: #2A2A2A; }

/* 위에서 내려오는 알림 (토스 디자인 시스템 Toast: 흰 알약 모양, 옅은 그림자, 15px 굵은 글자, 왼쪽 아이콘,
   위쪽 알림은 48px 위에서 내려오고 3초 뒤 사라짐). 알림 자리는 화면 흐름 밖에 고정해 내용을 밀지 않는다 */
.st-key-dc-toasts { position: fixed; top: 0; left: 0; right: 0; height: 0; z-index: 1000100; pointer-events: none; }
.dc-toast { position: fixed; top: 16px; left: 50%; display: flex; align-items: center; gap: 8px;
            width: max-content; max-width: min(92vw, 560px); padding: 12px 16px 12px 12px; border-radius: 100px;
            background: #FFFFFF; box-shadow: 0 2px 30px rgba(0, 27, 55, 0.1); color: rgba(0, 12, 30, 0.8);
            font-size: 15px; font-weight: 600; line-height: 1.45; animation: dc-toast 3.4s both; }
.dc-toast-icon { flex: none; width: 22px; height: 22px; border-radius: 50%; display: inline-flex; align-items: center;
                 justify-content: center; color: #FFFFFF; font-size: 14px; font-weight: 800; background: #FFB800; }
.dc-toast.error .dc-toast-icon { background: #F04452; }
.dc-toast .dc-toast-icon.info { background: #A85A17; font-family: "Material Symbols Rounded"; font-weight: 400;
                                font-size: 15px; font-variation-settings: "FILL" 1; }
@keyframes dc-toast {
  0% { opacity: 0; transform: translate(-50%, -48px); animation-timing-function: var(--dc-spring); }
  9% { opacity: 1; transform: translate(-50%, 0); }
  88% { opacity: 1; transform: translate(-50%, 0); animation-timing-function: var(--dc-ease); }
  100% { opacity: 0; transform: translate(-50%, -24px); visibility: hidden; }
}
@keyframes dc-toast-still { 0%, 88% { opacity: 1; } 100% { opacity: 0; visibility: hidden; } }

/* 휴대폰에서 '동작 줄이기'를 켠 사람에게는 움직이지 않는다 (숫자는 바로 최종 값, 알림은 움직이지 않고 사라지기만) */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
  .dc-toast { transform: translateX(-50%); animation: dc-toast-still 3.4s steps(1, end) both !important; }
  .dc-tool.last, .dc-tool-check { opacity: 1; } .dc-tool-check { width: 16px; } .dc-tool-name { background: #FFFFFF; }
  .dc-tool-slot { max-width: none; } .dc-tool-slot + .dc-tool-slot { margin-left: 6px; }
}
@media (max-width: 640px) { .dc-cards, .dc-cards.three { grid-template-columns: 1fr; } .dc-value { font-size: 40px; } }
</style>
"""


def html(markup):
    st.html(markup)


def notice(text, error=False):
    """경고·안내 문구. 노란 상자 대신 연한 회색 상자로 조용하게 보여준다."""
    html(f'<div class="dc-notice{" error" if error else ""}">{escape(text)}</div>')


TOASTS = None  # 위에서 내려오는 알림 자리. 페이지를 그리기 전에 만든다 (화면 흐름 밖에 고정)


def toast(text, error=False, icon=None):
    """토스 디자인 시스템의 Toast처럼 화면 위에서 내려왔다가 3초 뒤 사라지는 알림.
    잘못 적은 값, 읽지 못한 사진, 저장 실패처럼 잠깐 알리면 되는 것에 쓴다. 노란 '!'는 고쳐 달라는 뜻, 빨간 '!'는 실패.
    icon(Material Symbols 이름)을 주면 파란 아이콘의 안내 알림이 된다(예: 맛보기 전 가수 안내).
    화면에 계속 남아야 하는 상태(추천을 찾지 못함 등)는 notice로 보여 준다."""
    mark = (f'<span class="dc-toast-icon info" aria-hidden="true">{icon}</span>' if icon
            else '<span class="dc-toast-icon" aria-hidden="true">!</span>')
    markup = (f'<div class="dc-toast{" error" if error else ""}" role="{"status" if icon else "alert"}" '
              f'aria-live="{"polite" if icon else "assertive"}">{mark}{escape(text)}</div>')
    if TOASTS is None:  # 페이지 밖(테스트 등)에서 부르면 그 자리에 그린다
        html(markup)
        return
    with TOASTS:
        html(markup)


def photo_errors_text(errors):
    """읽지 못한 사진들을 알림 한 줄로. errors는 (파일 이름, 이유) 목록."""
    name, reason = errors[0]
    return f"{name}: {reason}" if len(errors) == 1 else f"사진 {len(errors)}장을 읽지 못했어요. {name}: {reason}"


def section(title, caption=""):
    html(f'<div class="dc-section">{escape(title)}</div>' + (f'<div class="dc-caption">{escape(caption)}</div>' if caption else ""))


def counted(value):
    """정수면 0부터 올라가며 나타나는 숫자로, 아니면(2:32, 3.38 같은) 그대로. 진짜 숫자는 숨겨 두어 글자로도 읽힌다."""
    if not value.isdigit():
        return escape(value)
    return f'<span class="dc-count" style="--dc-n:{int(value)}" aria-hidden="true"></span><span class="dc-sr">{value}</span>'


def card(label, value, unit, sub="", hero=False):
    return (
        f'<div class="dc-card{" hero" if hero else ""}"><div class="dc-label">{escape(label)}</div>'
        f'<div class="dc-value">{counted(value)}<span class="dc-unit">{escape(unit)}</span></div>'
        + (f'<div class="dc-sub">{escape(sub)}</div>' if sub else "")
        + "</div>"
    )


def tile(icon, label, value, unit="", sub="", effect=""):
    """아이콘 요약 카드의 한 칸. icon은 Material Symbols 이름(예: grain, water_drop). effect는 이 값이 맛에 주는 영향."""
    return (f'<div class="dc-tile"><span class="dc-icon" aria-hidden="true">{icon}</span>'
            f'<div class="dc-tile-label">{escape(label)}</div>'
            f'<div class="dc-tile-value{"" if any(ch.isdigit() for ch in value) else " text"}">'  # 글 값은 작게
            f'{counted(value)}<small>{escape(unit)}</small></div>'
            + (f'<div class="dc-tile-sub">{escape(sub)}</div>' if sub else "")
            + (f'<div class="dc-tile-effect">{escape(effect)}</div>' if effect else "") + "</div>")


def summary(*tiles):
    """숫자 여러 개를 한 번에 보여 주는 아이콘 요약 카드."""
    html('<div class="dc-summary">' + "".join(tiles) + "</div>")


def pour_timeline(recipe, end_s=None, end_label="끊김 예상"):
    """붓기 일정을 가로 막대로 그린다. 막대 위치는 붓는 시각, 길이는 붓는 양(붓는 속도로 걸리는 시간)이다.
    위에는 붓는 양, 아래에는 시각과 저울에 보일 누적을 적고, end_s를 주면 물줄기가 끊길 것으로 보는 시각을 표시한다.
    글 목록보다 한눈에 들어오게 바꿨다 (10/20 사용자 요청)."""
    rate = em.DEFAULT_PARAMS.pour_rate_g_per_s
    span = max(end_s or 0, max(p.start_s + p.grams / rate for p in recipe.pours) + 20) * 1.04

    def pct(seconds):
        return 100 * seconds / span

    parts, total, spoken = ['<div class="dc-tl-track"></div>'], 0.0, []
    for i, p in enumerate(recipe.pours):
        total += p.grams
        bloom = i == 0
        below = "뜸" if bloom else f"{total:g} g까지"
        parts.append(f'<div class="dc-tl-pour{" bloom" if bloom else ""}" style="left:{pct(p.start_s):.2f}%;'
                     f'width:{pct(p.grams / rate):.2f}%;--i:{i}"><div class="dc-tl-g">+{p.grams:g} g</div>'
                     f'<div class="dc-tl-bar"></div><div class="dc-tl-t">{clock(p.start_s)}</div>'
                     f'<div class="dc-tl-c">{below}</div></div>')
        spoken.append(f"{clock(p.start_s)}에 {p.grams:g} g")
    if end_s:
        parts.append(f'<div class="dc-tl-end" style="left:{pct(end_s):.2f}%"><span>{end_label} {clock(end_s)}</span></div>')
    label = "붓기 일정: " + ", ".join(spoken) + (f", 물줄기는 {clock(end_s)}쯤 끊겨요" if end_s else "")
    html(f'<div class="dc-tl" role="img" aria-label="{escape(label)}">{"".join(parts)}</div>')


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
    for kind in ("clump", "border", "chaff", "measured"):  # 잰 입자를 마지막에 칠해 위에 보이게 한다
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
            ("여러 알이 엉켜서 뺀 덩어리", f"{measurement.excluded_clumps}개"),
            ("칸 테두리에 걸려 뺀 입자", f"{measurement.excluded_border}개"),
            ("은피로 보고 뺀 조각", f"{measurement.excluded_chaff}개"),
            ("다른 입자와 붙어 있어 나눠 잰 양", f"{measurement.touching_volume_fraction:.0%}"),
            ("사용한 칸", f"{rectified.zone.name}칸, 픽셀당 {rectified.um_per_px:.0f} µm"),
        ]))
        for warning in rectified.warnings + measurement.warnings:
            notice(warning)


def show_profile(profile, grinder, clicks):
    """이번 측정(같은 가루로 찍은 사진을 합친 결과)의 요약 카드, 분포 곡선, 자세한 값."""
    low, high = profile.d50_range_um
    spread_low, spread_high = profile.spread_range_um
    summary(
        tile("grain", "대표 크기", f"{profile.d50_um:.0f}", "µm", f"{grind_feel(profile.d50_um)} · 다시 찍으면 {low:.0f}~{high:.0f}"),
        tile("bar_chart", "가장 많은 크기", f"{profile.mode_um:.0f}", "µm", "분포 곡선의 꼭대기"),
        tile("straighten", "대부분의 범위", f"{spread_low:.0f}~{spread_high:.0f}", "µm", "가루 무게의 68%"),
        tile("photo_camera", "잰 입자", f"{profile.n_particles}", "개", f"사진 {profile.n_photos}장"),
    )
    html('<div class="dc-caption">대표 크기는 가루 무게의 절반이 이보다 작은 크기예요.</div>')

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

    section("사진별 결과", "파란 선은 잰 입자, 회색 선은 뺀 것(여러 알이 엉킨 덩어리, 칸 테두리에 걸린 입자, 은피)이에요. "
                        "붙은 입자는 잘록한 곳에서 나눠 따로 재요.")
    measurements, names, errors = [], [], []
    for upload in uploads:
        with st.spinner(f"{upload.name} 분석 중..."):
            rectified, measurement, error = analyze_upload(upload.getvalue())
        if error:
            errors.append((upload.name, error))
            continue
        measured = len(measurement.diameters_um)
        label = f"{upload.name} · " + (f"대표 크기 {measurement.d50_volume_um:.0f} µm" if measured else "잰 입자 없음")
        with st.expander(label, expanded=len(uploads) == 1):
            show_photo_result(rectified, measurement)
        if measured:
            measurements.append(measurement)
            names.append(upload.name)

    if errors:
        toast(photo_errors_text(errors))
    if not measurements:
        if not errors:
            toast("입자를 잰 사진이 없어요. 가루를 칸 안쪽에 듬성듬성 펴고 다시 찍어 주세요.")
        show_saved_profiles()
        return

    profile = ga.combine_measurements(measurements)
    section("이번 측정 결과", f"사진 {profile.n_photos}장 · 잰 입자 {profile.n_particles}개")
    if ga.size_distribution_curve(profile.diameters_um) is None:
        toast("잰 입자가 너무 적어 분포를 계산할 수 없어요. 사진을 더 넣어 주세요.")
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


# 한 잔 내리기 5단계에서 슬라이더 이름 앞에 붙이는 아이콘 (Material Symbols)
TASTE_ICONS = {"aroma": "air", "flavor": "restaurant", "acidity": "bolt", "body": "fitness_center",
               "balance": "balance", "aftertaste": "hourglass_bottom"}


def taste_inputs(key, names=None, short=False):
    """맛 6항목 슬라이더. 맨 왼쪽(산미·바디 0점, 나머지 1점)에서 시작하지만, 한 번 움직이기 전까지는 '–'(안 매김)로
    보이고 점수로 치지 않는다. 그래서 매기지 않은 항목이 맨 왼쪽 점수로 저장되는 일이 없다.
    일부만 매기면 저장할 때 기록 함수가 거절한다. names로 일부만 그릴 수 있고, short면 이름 앞에 아이콘만 붙인다."""
    scores = {}
    for name, label, low in cal.TASTE_ITEMS:  # 기록지와 같은 순서
        if names is not None and name not in names:
            continue
        flag = key(f"{name}-touched")  # 움직이면 콜백이 켠다. 저장하면 key가 바뀌어 다시 꺼진 상태로 시작한다
        rated = st.session_state.get(flag, False)
        options = [low + 0.5 * j for j in range(int((cal.TASTE_MAX - low) * 2) + 1)]
        value = st.select_slider(
            f":material/{TASTE_ICONS[name]}: {label}" if short else f"{label} · {TASTE_GUIDE[name][0]}", options, value=low,
            format_func=lambda v, rated=rated, low=low: UNRATED if not rated and v == low else f"{v:g}",
            key=key(name), on_change=mark_touched, args=(flag,),
        )
        scores[name] = value if rated else None
    return scores


def taste_hexagon(scores, compact=False):
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
    if compact:  # 한 잔 내리기 5단계: 육각형 옆에 만족도만 크게
        return (f'<div class="dc-hexagon compact">{svg}<div class="dc-hexagon-score">만족도<b>{score}</b></div></div>')
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
CUP_KINDS = ("좋았던 잔", "다른 잔")  # 그림의 점: 목표를 당긴 잔(만족도 2.5 초과)과 나머지
# 그림 아래 기호 설명. 기호 모양과 색은 그림과 같게 맞춘다. 내 선호 중심과 골든컵 중심 표시는 처음 보는 사람에게
# 읽기 어려워 그림에서 뺐고, 값은 '목표는 어떻게 정해져요?'에 글로 적는다 (10/20)
PREFERENCE_LEGEND = '<div class="dc-legend">' + "".join(
    f'<span><i style="color:{color}">{symbol}</i>{name}</span>'
    for symbol, color, name in [("●", BRAND, CUP_KINDS[0]), ("●", GRAY_LIGHT, CUP_KINDS[1]), ("◆", BRAND, "다음 잔 목표")]
) + "</div>"


def axis(title):
    return alt.Axis(title=title, grid=False, ticks=False, domainColor="#E5E8EB", labelColor=GRAY, labelFontSize=12,
                    titleColor=GRAY, titleFontSize=13, titleFontWeight=500, tickCount=6)


def preference_chart(points, target, pick=None):
    """잔마다 (추출수율, 가수 후 농도)를 점으로 찍고, 골든컵 범위와 다음 잔 목표를 함께 그린다.
    좋았던 잔(가중치 > 0)은 강조색으로, 점수가 높을수록 진하게 칠하고, 나머지는 회색으로 둔다.
    축 이름에 '오른쪽일수록 더 우러나요', '위일수록 진해요'를 붙여 숫자를 몰라도 읽히게 한다.
    pick(altair 선택)을 주면 점을 눌러 그 잔을 고를 수 있다."""
    (ey_low, ey_high), (tds_low, tds_high) = GOLDEN_CUP_BOX
    cups = pd.DataFrame(
        {
            "brew_id": p.brew_id, "bean": p.bean_id, "ey": p.ey_pct, "tds": p.tds_cup_pct,
            "score": round(p.score, 2), "weight": p.weight, "kind": CUP_KINDS[0] if p.weight > 0 else CUP_KINDS[1],
        }
        for p in points
    )
    all_ey = [target.ey_pct, ey_low, ey_high] + list(cups["ey"])
    all_tds = [target.tds_pct, tds_low, tds_high] + list(cups["tds"])
    x = alt.X("ey:Q", scale=alt.Scale(domain=[math.floor(min(all_ey)) - 1, math.ceil(max(all_ey)) + 1], nice=False),
              axis=axis("추출수율 (%) · 오른쪽일수록 더 우러나요"))
    y = alt.Y("tds:Q", scale=alt.Scale(domain=[round(min(all_tds) - 0.05, 2), round(max(all_tds) + 0.05, 2)], nice=False),
              axis=axis("농도 (%) · 위일수록 진해요"))

    box = pd.DataFrame({"ey": [ey_low], "ey2": [ey_high], "tds": [tds_low], "tds2": [tds_high], "label": ["골든컵 범위"]})
    layers = alt.Chart(box).mark_rect(color="#F2F4F6", cornerRadius=12).encode(x=x, x2="ey2:Q", y=y, y2="tds2:Q")
    layers += alt.Chart(box).mark_text(align="left", baseline="top", dx=10, dy=10, fontSize=12, color=GRAY, font=FONT).encode(
        x="ey:Q", y="tds2:Q", text="label:N")
    if not cups.empty:
        dots = alt.Chart(cups).mark_circle(size=150, stroke="#FFFFFF", strokeWidth=1.5).encode(
            x=x, y=y,
            color=alt.Color("kind:N", scale=alt.Scale(domain=list(CUP_KINDS), range=[BRAND, GRAY_LIGHT]), legend=None),
            opacity=alt.Opacity("weight:Q", scale=alt.Scale(domain=[0, 2.5], range=[0.45, 1]), legend=None),
            tooltip=[alt.Tooltip("brew_id:N", title="추출번호"), alt.Tooltip("bean:N", title="원두"),
                     alt.Tooltip("score:Q", title="만족도"), alt.Tooltip("ey:Q", title="추출수율 (%)", format=".1f"),
                     alt.Tooltip("tds:Q", title="가수 후 농도 (%)", format=".2f")],
        )
        if pick is not None:
            dots = dots.add_params(pick)
        layers += dots
    goal = pd.DataFrame({"ey": [target.ey_pct], "tds": [target.tds_pct], "label": ["다음 잔 목표"]})
    layers += alt.Chart(goal).mark_point(shape="diamond", color=BRAND, size=220, filled=True, opacity=1).encode(
        x=x, y=y, tooltip=[alt.Tooltip("label:N", title="표시"), alt.Tooltip("ey:Q", title="추출수율 (%)", format=".1f"),
                           alt.Tooltip("tds:Q", title="가수 후 농도 (%)", format=".2f")])
    return layers.properties(height=360).configure_view(stroke=None).configure(font=FONT)


def history_summary(brews, target):
    """내 기록 위의 요약 카드 넷: 기록한 잔, 평균 만족도, 다음 잔 목표(추출수율, 농도)."""
    scores = [b.quality_score for b in brews if b.quality_score is not None]
    beans = len({b.bean_id for b in brews})
    summary(
        tile("coffee", "기록한 잔", f"{len(brews)}", "잔", f"원두 {beans}가지" if brews else "아직 없어요"),
        tile("star", "평균 만족도", f"{sum(scores) / len(scores):.2f}" if scores else "–", "",
             f"최근 3잔 {sum(scores[-3:]) / len(scores[-3:]):.2f}" if len(scores) > 3 else "Aroma·Flavor·Balance·Aftertaste"),
        tile("science", "목표 추출수율", f"{target.ey_pct:.1f}", "%", "얼마나 우러나게"),
        tile("opacity", "목표 농도", f"{target.tds_pct:.2f}", "%", "얼마나 진하게"),
    )


def cup_markup(b):
    """잔 한 줄: 원두, 클릭·온도·물 빠짐, 오른쪽에 만족도와 진하기 답. 표시(막힘 등)가 있으면 흐리게, 사진이 없으면 꼬리표."""
    tags = ([b.flag] if b.flag else []) + ([] if b.grind_d50_um is not None else ["사진 없음"])
    sub = " · ".join(x for x in (f"{b.grinder_clicks}클릭" if b.grinder_clicks is not None else "",
                                 f"{b.water_temp_c:g}°C", clock(b.drawdown_end_s) if b.drawdown_end_s else "") if x)
    strength = STRENGTH_CHOICES.get(b.strength, "")
    return (f'<div class="dc-cup{" muted" if b.flag else ""}"><span class="dc-avatar">{escape(b.bean_id)}</span>'
            f'<span class="dc-cup-text"><b>{escape(b.row.get("bean_name", "") or b.bean_id)}</b>'
            + "".join(f'<span class="dc-cup-tag">{escape(t)}</span>' for t in tags)
            + f'<small>{escape(sub)}</small></span><span class="dc-cup-score">'
            f'<b>{cell(b.quality_score, "{:.2f}")}</b>' + (f"<small>{escape(strength)}</small>" if strength else "")
            + '</span><span class="dc-lrow-chev" aria-hidden="true">chevron_right</span></div>')


def open_cup(brew_id):
    st.session_state["history-open"] = brew_id


def cup_list(brews):
    """기록한 잔을 날짜별로, 최근 것부터. 줄을 누르면 그 잔을 자세히 본다. 모든 칸은 '표로 자세히 보기'에 둔다."""
    if not brews:
        notice("아직 기록한 잔이 없어요. 한 잔 내리고 맛을 매기면 여기에 쌓여요.")
        return
    with st.container(key="cup-list"):
        day = None
        for b in reversed(brews):
            if b.date != day:
                day = b.date
                html(f'<div class="dc-cups-date">{escape(short_date(day))}</div>')
            list_row(f"cup-{b.brew_id}", cup_markup(b), f"{short_date(b.date)} {b.bean_id} 잔 자세히 보기", open_cup, (b.brew_id,))
    with st.expander("표로 자세히 보기"):
        recent_brews_table(brews, limit=len(brews))


BEAN_COLORS = (BRAND, "#5B7FA6", "#6E8F4E", "#B5838D", "#8A6FB0", "#C9A227")  # 맛 변화 그림의 원두별 색


def opened_from_chart(event, name):
    """그래프에서 누른 잔을 연다. 같은 선택으로 다시 그릴 때 창이 또 열리지 않게 마지막으로 연 선택을 기억한다."""
    picked = [p.get("brew_id") for p in (event.selection.get(name) or [])] if event else []
    if picked and picked != st.session_state.get(f"{name}-last"):
        st.session_state[f"{name}-last"] = picked
        open_cup(picked[0])
    elif not picked:
        st.session_state.pop(f"{name}-last", None)


def taste_trend(brews):
    """잔마다 만족도를 기록한 순서대로 찍는다. 원두마다 색을 달리하고 표시 잔은 회색. 처음 3잔과 최근 3잔 평균을 함께."""
    cups = [b for b in brews if b.quality_score is not None]
    if not cups:
        notice("아직 맛을 매긴 잔이 없어요.")
        return
    beans = list(dict.fromkeys(b.bean_id for b in cups))
    data = pd.DataFrame({"n": i + 1, "brew_id": b.brew_id, "day": short_date(b.date), "bean": b.bean_id,
                         "color": "막힘·연습" if b.flag else b.bean_id, "score": round(b.quality_score, 2)}
                        for i, b in enumerate(cups))
    x_axis = axis("잔 (오래된 → 최근)")
    x_axis.tickMinStep = 1
    x = alt.X("n:Q", axis=x_axis, scale=alt.Scale(domain=[0.5, len(cups) + 0.5]))
    y = alt.Y("score:Q", axis=axis("만족도"), scale=alt.Scale(domain=[1, 5]))
    band = alt.Chart(pd.DataFrame({"low": [3], "high": [5]})).mark_rect(color="#FBEFE3", opacity=0.6).encode(
        y=alt.Y("low:Q", scale=alt.Scale(domain=[1, 5])), y2="high:Q")
    line = alt.Chart(data[data["color"] != "막힘·연습"]).mark_line(color="#D9A877", strokeWidth=2).encode(x=x, y=y)
    pick = alt.selection_point(name="trend", fields=["brew_id"], on="click")
    dots = alt.Chart(data).mark_circle(size=170, stroke="#FFFFFF", strokeWidth=2, opacity=1).encode(
        x=x, y=y,
        color=alt.Color("color:N", legend=None, scale=alt.Scale(domain=beans + ["막힘·연습"],
                                                                range=list(BEAN_COLORS[:len(beans)]) + [GRAY_LIGHT])),
        tooltip=[alt.Tooltip("day:N", title="날짜"), alt.Tooltip("bean:N", title="원두"), alt.Tooltip("score:Q", title="만족도")],
    ).add_params(pick)
    chart = (band + line + dots).properties(height=260).configure_view(stroke=None).configure(font=FONT)
    opened_from_chart(st.altair_chart(chart, width="stretch", on_select="rerun", key="trend-chart"), "trend")
    html('<div class="dc-legend">' + "".join(f'<span><i class="dc-dot" style="background:{BEAN_COLORS[i]}"></i>{escape(bean)}</span>'
                                             for i, bean in enumerate(beans))
         + f'<span><i class="dc-dot" style="background:{GRAY_LIGHT}"></i>막힘·연습</span></div>')
    normal = [b.quality_score for b in cups if not b.flag]
    if len(normal) >= 2:
        first, recent = normal[:3], normal[-3:]
        summary(tile("history", "처음 3잔 평균", f"{sum(first) / len(first):.2f}", "", "만족도"),
                tile("trending_up", "최근 3잔 평균", f"{sum(recent) / len(recent):.2f}", "", "만족도"))
    html('<div class="dc-caption">옅은 띠는 만족도 3 이상이에요. 점을 누르면 그 잔을 자세히 봐요.</div>')


@st.dialog("잔 자세히", width="medium")
def cup_detail(brew_id):
    """기록한 잔 하나: 예측과 실제, 만족도와 진하기, 맛 육각형, 붓기 일정, 메모, 같은 원두로 다음 잔."""
    brews = cal.load_brews(BREWS_PATH)
    b = next((x for x in brews if x.brew_id == brew_id), None)
    if b is None:
        notice("이 잔을 찾지 못했어요.")
        return
    c = next((x for x in cal.compare_with_log(brews) if x.brew.brew_id == brew_id), None)
    size = f"{b.grind_d50_um:.0f} µm" if b.grind_d50_um else "사진 없음"
    html(f'<div class="dc-done-title">{escape(b.bean_id)} {escape(b.row.get("bean_name", ""))}</div>'
         f'<div class="dc-done-sub">{escape(short_date(b.date))} · {escape(b.row.get("roast_level", ""))} · '
         + (f"{b.grinder_clicks}클릭 · " if b.grinder_clicks is not None else "") + escape(size)
         + (f" · {escape(b.flag)}" if b.flag else "") + "</div>")
    summary(
        tile("timer", "물줄기 끊김", clock(b.drawdown_end_s), "",
             versus_time(c.prediction.drawdown_end_s, b.drawdown_end_s) if c and b.drawdown_end_s else "예측과 비교할 수 없어요"),
        tile("local_cafe", "음료 (가수 전)", cell(b.beverage_g), "g",
             versus_grams(c.prediction.beverage_g, b.beverage_g) if c and b.beverage_g else "예측과 비교할 수 없어요"),
        tile("star", "만족도", cell(b.quality_score, "{:.2f}"), "", "맛 4항목 평균"),
        tile("coffee", "진하기", STRENGTH_CHOICES.get(b.strength, "–"), "", f"가수 {b.bypass_g:g} g"),
    )
    st.markdown(taste_hexagon(b.taste, compact=True), unsafe_allow_html=True)  # SVG라서 markdown
    section("붓기", f"{b.water_temp_c:g} °C · 물 {b.recipe().water_g:g} g")
    pour_timeline(b.recipe(), b.drawdown_end_s, "끊김")
    memo = (b.memo or "").split("앱 예측(내리기 전)")[0].strip()
    if memo:
        html(f'<div class="dc-notice">{escape(memo)}</div>')
    if st.button("이 원두로 다음 잔", type="primary", width="stretch", key="cup-detail-next"):
        brew_same_bean(b)
        st.rerun()


def preference_section(brews, points, target):
    """기록에서 찾은 내 선호 영역과 다음 잔 목표 (MODEL.md 9.1). 그림은 짧게 읽히게 두고, 정하는 방식은 접어 둔다."""
    if not points:
        notice("아직 맛을 매긴 잔이 없어서 골든컵(많은 사람이 대체로 좋아하는 범위)의 가운데를 목표로 써요. "
               "잔을 내리고 맛을 매길수록 목표가 내 입맛 쪽으로 옮겨 가요.")
        return
    html('<div class="dc-caption">점 하나가 기록한 한 잔이에요. 진한 점이 좋았던 잔, ◆가 다음 잔 목표예요. '
         "점을 누르면 그 잔을 자세히 봐요. 굴절계로 잰 값이 아니라 모델로 계산한 위치예요.</div>")
    pick = alt.selection_point(name="preference", fields=["brew_id"], on="click")
    opened_from_chart(st.altair_chart(preference_chart(points, target, pick), width="stretch", on_select="rerun",
                                      key="preference-chart"), "preference")
    html(PREFERENCE_LEGEND)
    with st.expander("목표는 어떻게 정해져요?"):
        center = target.personal_center
        html('<div class="dc-caption">얼마나 우러나게(추출수율)는 맛 점수로 배워요. 만족도 2.5를 넘는 잔이 내 선호 중심을 '
             "당기고, 골든컵 중심(많은 사람이 대체로 좋아하는 곳)과 섞어서 정해요. 좋았던 잔이 쌓일수록 목표가 내 쪽으로 "
             "옮겨 가요. 얼마나 진하게(농도)는 '진하기' 답으로 배워요. 최근 세 잔의 답으로 바로 맞춰요.</div>")
        used = {p.brew_id for p in points}
        skipped = [f"{b.brew_id} {b.flag or ('맛 점수 없음' if b.quality_score is None else '분쇄 크기 없음')}"
                   for b in brews if b.brew_id not in used]
        html(rows([
            ("다음 잔 목표", f"추출수율 {target.ey_pct:.1f}% · 농도 {target.tds_pct:.2f}%"),
            ("내 선호 중심", f"{center[0]:.1f}% · {center[1]:.2f}% (좋았던 잔 {target.n_liked}잔)" if center
             else "아직 좋았던 잔이 없어요"),
            ("골든컵 중심", f"{rec.GOLDEN_CUP_CENTER[0]:g}% · {rec.GOLDEN_CUP_CENTER[1]:.2f}%"),
            ("농도를 정한 답", f"최근 진하기 답 {target.n_strength}잔" if target.n_strength else "아직 진하기 답이 없어요"),
            *([("계산에서 뺀 잔", ", ".join(skipped))] if skipped else []),
        ]))


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
    beans = cal.load_beans(BEANS_PATH)
    if not beans:
        notice("아직 원두가 없어요. 첫 화면의 '시작하기'에서 원두를 먼저 추가해 주세요.")
        return
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

    equipment = cal.load_equipment(EQUIPMENT_PATH)
    c1, c2, c3 = st.columns(3)
    roast_date = c1.text_input("로스팅 날짜", bean["roast_date"], key=key(f"roast-{bean_id}"))
    dripper = equipment_select(c2, equipment, "드리퍼", key("dripper"), previous.row.get("dripper") if previous else None)
    # 필터는 장소마다 달라서 지난 잔 값을 미리 채우지 않고 매번 고르게 한다 (장비는 한 잔 내리기 1단계에서 더한다)
    filter_paper = equipment_select(c3, equipment, "필터", key("filter"))
    c1, c2 = st.columns(2)
    grinder = equipment_select(c1, equipment, "그라인더", key("grinder"), previous.grinder if previous else None)
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
                "roast_date": roast_date, "dripper": dripper or "", "filter": filter_paper or "",
                "grinder": grinder or "", "grinder_clicks": int(clicks),
                "grind_d50_um": d50 or None, "recipe": recipe_name, "dose_g": dose, "water_temp_c": temp,
                "pours": pours, "drawdown_end_s": cal.parse_clock(drawdown), "beverage_g": beverage,
                "bypass_g": bypass, **taste, "memo": memo, "flag": flag,
                "followed_recommendation": prefill["followed"] if followed else "",
            }
            cal.append_brew(record, BREWS_PATH)
        except cal.BrewLogError as error:
            toast(str(error), error=True)
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
    size = f"{abs(error):.0f}" if unit == "초" else f"{abs(error):.1f}".removesuffix(".0")  # 초는 줄의 '−22초'와 같게 정수로
    return f"실제 {actual_text} · 예측이 {size}{unit} {fewer if error < 0 else more}."


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


VALIDATION_LIMITS = (30.0, 6.0)  # 판정 기준: 끊김 ±30초, 음료 ±6 g (MODEL.md 6절, 검증 잔을 내리기 전에 정함)


def within_limits(c):
    """끊김과 음료 무게가 판정 기준 안인지. 음료를 적지 않은 잔은 끊김만 본다."""
    end_ok = c.end_error_s is not None and abs(c.end_error_s) <= VALIDATION_LIMITS[0]
    return end_ok and (c.beverage_error_g is None or abs(c.beverage_error_g) <= VALIDATION_LIMITS[1])


def check_summary(comparisons):
    """표시(막힘, 연습 등)가 없는 잔으로 모델이 얼마나 맞히는지: 빗나간 크기, 판정 기준 안 잔 수, 치우침."""
    normal = [c for c in comparisons if not c.brew.flag]
    ends = [c.end_error_s for c in normal if c.end_error_s is not None]
    drinks = [c.beverage_error_g for c in normal if c.beverage_error_g is not None]
    if not ends:
        return
    bias = sum(ends) / len(ends)
    summary(
        tile("timer", "끊김 시각", f"{sum(map(abs, ends)) / len(ends):.0f}", "초", "평균 이만큼 빗나가요"),
        tile("local_cafe", "음료 무게", f"{sum(map(abs, drinks)) / len(drinks):.1f}" if drinks else "–", "g" if drinks else "",
             "평균 이만큼 빗나가요"),
        tile("check_circle", "판정 기준 안", f"{sum(map(within_limits, normal))}", f"/ {len(normal)}잔",
             f"끊김 ±{VALIDATION_LIMITS[0]:g}초 · 음료 ±{VALIDATION_LIMITS[1]:g} g"),
        tile("swap_horiz", "치우침", f"{bias:+.0f}", "초", "예측이 짧은 편이에요" if bias < 0 else "예측이 긴 편이에요"),
    )
    html('<div class="dc-caption">표시(막힘, 연습 등)가 없는 잔만 셌어요. 치우침은 (예측 − 실제)의 평균이에요.</div>')


def choose_check_cup(brew_id):
    st.session_state["check-cup"] = brew_id


def check_row(c, chosen):
    """잔 한 줄: 날짜·원두·클릭, 예측과 실제 끊김, 판정 기준 막대 위의 점, 차이. 누르면 그 잔을 아래에 자세히."""
    b, error = c.brew, c.end_error_s
    state = "muted" if b.flag else ("" if within_limits(c) else "out")
    spot = 50 + max(-1.0, min(1.0, (error or 0) / VALIDATION_LIMITS[0])) * 50  # 막대 끝을 넘으면 끝에 붙인다
    title = f"{short_date(b.date)} · {b.bean_id} {cell(b.grinder_clicks)}클릭"
    sub = f"예측 {clock(c.prediction.drawdown_end_s)} · 실제 {clock(b.drawdown_end_s)}" + (f" · {b.flag}, 판정에서 뺌" if b.flag else "")
    markup = (f'<div class="dc-lrow{" on" if chosen else ""}"><span class="dc-lrow-text"><b>{escape(title)}</b>'
              f'<small>{escape(sub)}</small></span><span class="dc-band" aria-hidden="true">'
              f'<i class="{state}" style="left:{spot:.0f}%"></i></span>'
              f'<span class="dc-pill dc-check-pill {state}">{f"{error:+.0f}초" if error is not None else "–"}</span></div>')
    list_row(f"check-cup-{b.brew_id}", markup, f"{title} 자세히 보기", choose_check_cup, (b.brew_id,))


def check_detail(c):
    """누른 잔: 같은 레시피·분쇄로 다시 계산한 예측과 실제, 레시피, 고인 물 그래프."""
    b, prediction = c.brew, c.prediction
    section(f"{short_date(b.date)} · {b.bean_id} {b.row.get('bean_name', '')}".strip(),
            "같은 레시피·분쇄로 모델을 다시 돌린 값이에요. 추출수율과 농도는 굴절계가 없어 실제와 비교할 수 없어요.")
    if b.flag:
        notice(f"'{b.flag}' 표시가 있는 잔이에요. 보정과 위 요약에서는 빼요.")
    summary(
        tile("timer", "끊김 (예측)", clock(prediction.drawdown_end_s), "",
             gap(c.end_error_s, clock(b.drawdown_end_s), "초", "짧아요", "길어요")),
        tile("local_cafe", "음료 (예측)", f"{prediction.beverage_g:.0f}", "g",
             gap(c.beverage_error_g, f"{cell(b.beverage_g)} g", " g", "적어요", "많아요")),
        tile("science", "추출수율 (예측)", f"{prediction.ey_pct:.1f}", "%", "원두 중 녹아 나온 비율"),
        tile("opacity", "농도 (예측)", f"{prediction.tds_cup_pct:.2f}", "%", f"가수 전 {prediction.tds_pct:.2f}%"),
    )
    st.altair_chart(timeline_chart(prediction, b.recipe(), actual_end=b.drawdown_end_s), width="stretch")
    html(rows([
        ("분쇄", f"{b.grinder} {cell(b.grinder_clicks)}클릭 · {c.grind_d50_um:.0f} µm"
                 + (" (같은 원두·클릭 잔의 측정)" if c.borrowed_grind else "")),
        ("레시피", f"원두 {b.dose_g:g} g · {b.water_temp_c:g} °C · 붓기 {b.pours} · 가수 {b.bypass_g:g} g"),
        ("드리퍼·필터", f"{b.row.get('dripper', '')} · {b.filter_paper or '–'}"),
    ]))


def compare_view(brews):
    """기록한 잔을 같은 레시피·분쇄로 다시 계산해 실제와 비교한다. 위에 요약 카드, 잔마다 한 줄(판정 기준 막대),
    누른 잔의 자세한 비교, 모든 칸은 '표로 자세히 보기'에."""
    comparisons = cal.compare_with_log(brews)
    if not comparisons:
        notice("비교할 기록이 없어요. 분쇄 대표 크기가 있는 잔을 기록하면 여기서 예측과 비교할 수 있어요.")
        return
    check_summary(comparisons)
    newest_first = list(reversed(comparisons))
    ids = [c.brew.brew_id for c in newest_first]
    chosen = st.session_state.get("check-cup") if st.session_state.get("check-cup") in ids else ids[0]
    section("잔마다", f"가운데 막대가 판정 기준(±{VALIDATION_LIMITS[0]:g}초)이고, 점이 예측과 실제의 차이예요. "
                    "줄을 누르면 아래에 자세히 나와요.")
    with st.container(key="check-list"):
        for c in newest_first:
            check_row(c, c.brew.brew_id == chosen)
    check_detail(newest_first[ids.index(chosen)])
    with st.expander("표로 자세히 보기"):
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
    section("예측 결과", "모델 예측이라 원두에 따라 20초쯤 다를 수 있어요.")
    summary(
        tile("timer", "끊김", clock(prediction.drawdown_end_s), "", f"마지막 붓기 끝나고 약 {after_last:.0f}초 뒤"),
        tile("local_cafe", "음료 (가수 전)", f"{prediction.beverage_g:.0f}", "g",
             f"가루가 약 {prediction.held_liquid_g:.0f} g을 머금어요"),
        tile("science", "추출수율", f"{prediction.ey_pct:.1f}", "%", "원두 중 녹아 나온 비율"),
        tile("opacity", "농도 (가수 후)", f"{prediction.tds_cup_pct:.2f}", "%", f"가수 전 {prediction.tds_pct:.2f}%"),
    )
    pour_timeline(recipe, prediction.drawdown_end_s)
    st.altair_chart(timeline_chart(prediction, recipe), width="stretch")
    html('<div class="dc-caption">점선은 붓기를 시작한 시각이에요. 고인 물이 0이 되는 순간을 물줄기가 끊기는 순간으로 봐요.</div>')


@st.cache_data(show_spinner=False, max_entries=20)
def cached_recommendation(bean_id, dose_g, log_version, intent="평소대로"):
    """추천은 0.1–2초 걸려서, 기록 파일이 바뀌지 않았으면(log_version) 지난 결과를 다시 쓴다.
    intent는 이번 잔만의 요청(rec.INTENTS)이다."""
    brews = cal.load_brews(BREWS_PATH)
    target = rec.with_intent(rec.preference_target(rec.brew_points(brews)), intent)
    return target, rec.recommend_recipe(target, cal.load_beans(BEANS_PATH)[bean_id], brews, dose_g)


def recommend_tab():
    html(
        '<div class="dc-lead">원두를 고르면 내 선호 영역의 목표에 맞는 레시피 전체를 추천해요.<br>'
        "분쇄는 그라인더 클릭이 아니라 대표 크기(µm)로 알려 줘요. 갈고 나서 사진으로 확인해요.</div>"
    )
    brews = cal.load_brews(BREWS_PATH)
    beans = cal.load_beans(BEANS_PATH)
    if not beans:
        notice("아직 원두가 없어요. 첫 화면의 '시작하기'에서 원두를 먼저 추가해 주세요.")
        return
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
        *([("농도", f"최근 진하기 답 {target.n_strength}잔으로 정함")] if target.n_strength else []),
    ]))

    recipe, prediction = r.recipe, r.prediction
    low, high = r.temp_range
    time_limit_notice(r, target)
    section("추천 레시피", f"원두 {recipe.dose_g:g} g · 물 {recipe.water_g:g} g (원두의 {r.water_ratio}배, "
                         f"{rec.pattern_text(r.pattern)}배) · {r.interval_s:g}초 간격")
    summary(
        tile("grain", "분쇄", f"{r.d50_um:.0f}", "µm", grind_feel(r.d50_um)),
        tile("device_thermostat", "물 온도", f"{recipe.water_temp_c:g}", "°C", f"{bean['roast_level']} {low}–{high} °C"),
        tile("water_drop", "물", f"{recipe.water_g:g}", "g", f"원두의 {r.water_ratio}배"),
        tile("water_full", "가수", f"{recipe.bypass_g:g}", "g", "드리퍼를 뺀 뒤 더해요"),
    )
    section("붓기 일정", "붓는 방식(센터 푸어, 원형 푸어 등)은 평소대로 해요. 모델은 붓는 시각과 양만 계산해요.")
    pour_timeline(r.recipe, prediction.drawdown_end_s)

    section("예상", "끊김은 예상 시각이 아니라 물줄기가 끊기는 순간에 드리퍼를 빼요. 모델 예측이라 원두에 따라 20초쯤 "
                  "다를 수 있어요.")
    summary(
        tile("timer", "끊김", clock(prediction.drawdown_end_s), "", "이때쯤 드리퍼를 빼요"),
        tile("local_cafe", "음료 (가수 전)", f"{prediction.beverage_g:.0f}", "g", "가수하기 전 무게"),
        tile("science", "추출수율", f"{prediction.ey_pct:.1f}", "%", f"목표 {target.ey_pct:.1f}%"),
        tile("opacity", "농도 (가수 후)", f"{prediction.tds_cup_pct:.2f}", "%", f"목표 {target.tds_pct:.2f}%"),
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
             "끊김이 2:00–3:30을 넘는 레시피는 고르지 않고, 그 안에서 출발점과 가장 덜 다른 것을 골라요. 그 안에서 목표에 "
             "닿지 못하면 가장 가까운 것을 골라요 (MODEL.md 9.2).</div>")


# 한 잔 내리기: 원두·장비 → 기초 추천 → 분쇄 확인 → 내리기와 기록 → 맛 평가 → 마무리
FLOW_STEPS = ("원두와 장비", "기초 추천", "분쇄 확인", "내리기와 기록", "맛 평가", "마무리")
BASE_NAMES = {((4, 2), 60.0): "484 레시피 (중강배전 이상)", ((3, 3, 3), 30.0): "4666 레시피"}


def flow_header(step):
    """단계 머리: 몇 단계인지, 이름, 진행 막대. 마무리는 이름 대신 '기록했어요' 표시를 크게 두어 이름을 뺀다."""
    title = "" if step == len(FLOW_STEPS) else f'<div class="dc-title" style="font-size:26px">{escape(FLOW_STEPS[step - 1])}</div>'
    html(f'<div class="dc-eyebrow">{step} / {len(FLOW_STEPS)} 단계</div>{title}')
    st.progress(step / len(FLOW_STEPS))


FLOW_CTA = None  # 한 잔 내리기 화면 아래에 붙는 버튼 자리. flow_page가 실행마다 단계 화면 밖에 만든다


def bottom_cta(label, key, ready=True, hint="", back_to=None, back_key=None):
    """토스 디자인 시스템의 BottomCTA처럼 화면 아래에 붙은 버튼. 눌렀으면 True.

    ready가 아니면 버튼 대신, 무엇을 채우면 다음으로 갈 수 있는지(hint)를 같은 자리에 흐리게 보여 준다. 그 자리를
    누르면 좌우로 흔들려 아직 갈 수 없다고 알린다(cta_hint). 다 채우면 버튼이 아래에서 튀어 올라온다(STYLE).
    back_to가 있으면 왼쪽에 '이전' 버튼을 둔다.
    단계 화면은 나타날 때 움직이므로(transform), 그 안에 두면 고정 위치가 흔들려서 단계 화면 밖의 자리에 그린다.
    """
    with FLOW_CTA:
        if back_to is not None:
            left, right = st.columns([1, 2])
            if left.button("이전", width="stretch", key=back_key):
                go_to(back_to)
        else:
            right = st.container()
        if not ready:
            with right:
                cta_hint(hint, key)
            return False
        return right.button(label, type="primary", width="stretch", key=key)


def tap_hint(taps_key):
    st.session_state[taps_key] = st.session_state.get(taps_key, 0) + 1


def cta_hint(hint, key):
    """아직 갈 수 없을 때 버튼 자리에 두는 흐린 안내. 누르면 좌우로 흔들린다(iOS·토스의 '안 돼요' 흔들림).
    애니메이션은 화면 요소가 바뀌어야 다시 재생되므로, 누를 때마다 상자 이름(key)을 a, b로 번갈아 바꾼다."""
    taps = st.session_state.get(f"{key}-taps", 0)
    state = "still" if not taps else "ab"[taps % 2]
    with st.container(key=f"cta-hint-{state}-{key}"):
        st.button(hint, key=f"{key}-hint", width="stretch", on_click=tap_hint, args=(f"{key}-taps",))


def go_to(step):
    st.session_state["flow"]["step"] = step
    st.rerun()


def base_name(usual):
    """출발점 레시피의 이름. 평소 두 레시피면 그 이름, 아니면 배수 표기."""
    return BASE_NAMES.get((tuple(usual.pattern), float(usual.interval_s)),
                          f"{rec.pattern_text(usual.pattern)}배 / {usual.interval_s:g}초 레시피")


# 대표 크기를 눈으로 볼 때의 비유: (이 크기까지, 비유). 추출 안내서들이 흔히 쓰는 대략적인 비유라 눈대중일 뿐이다
GRIND_FEELS = ((500, "고운 소금 정도"), (800, "백설탕 정도"), (1000, "굵은 설탕 정도"), (1300, "굵은 소금 정도"))


def grind_feel(d50_um):
    """대표 크기(µm)를 눈대중 비유로. 예: 1090 → '굵은 소금 정도'."""
    return next((feel for upto, feel in GRIND_FEELS if d50_um <= upto), "통후추를 부순 정도")


def hint_short(hint, grinder):
    """클릭 힌트를 요약 카드 한 줄로."""
    if hint is None:
        return f"{grinder} 클릭 기록 없음"
    if hint.clicks is None:
        return f"{hint.near.grinder_clicks}클릭보다 {hint.direction}"
    return f"{grinder} 약 {hint.clicks}클릭"


def hint_text(hint, grinder):
    """클릭 힌트를 한 문장으로."""
    if hint is None:
        return f"{grinder}는 클릭 기록이 없어요. 평소 감대로 갈고 다음 단계에서 사진으로 확인해요."
    if hint.clicks is None:
        return f"{grinder}로 {hint.near.grinder_clicks}클릭보다 {hint.direction} 갈아 보세요. {hint.basis}."
    return f"{grinder}로 약 {hint.clicks}클릭에서 시작해 보세요. {hint.basis}"


@st.dialog("새 원두 추가")
def add_bean_dialog():
    """원두 목록에 없는 원두를 더한다. 저장하면 1단계 원두 칸에서 바로 고른 상태가 된다."""
    html('<div class="dc-caption">원두 봉투에 적힌 대로 적어 주세요. 로스팅 날짜를 알면 원두가 마시기 좋은 때인지 '
         "알려 드려요.</div>")
    name = st.text_input("커피 이름", key="nb-name", placeholder="예: 에티오피아 예가체프")
    roaster = st.text_input("로스터리", key="nb-roaster", help="원두를 볶아서 판 가게 이름이에요.")
    c1, c2 = st.columns(2)
    roast_date = c1.date_input("로스팅 날짜", value=date.today(), max_value=date.today(), key="nb-date",
                               help="원두를 볶은 날이에요. 봉투에 '로스팅일'이나 'Roasted on'으로 적혀 있어요.")
    roast_level = c2.selectbox("배전도", cal.ROAST_LEVELS, index=2, key="nb-level",
                               help="원두를 볶은 정도예요. 라이트(약배전)는 산뜻하고, 다크(강배전)는 쓴맛과 무게감이 강해요. "
                                    "봉투에 약배전·중배전·강배전으로 적혀 있기도 해요.")
    c1, c2 = st.columns(2)
    origin = c1.text_input("산지 (선택)", key="nb-origin", placeholder="예: 에티오피아")
    process = c2.text_input("가공 (선택)", placeholder="예: 워시드, 내추럴", key="nb-process",
                            help="커피 열매에서 씨앗(생두)을 꺼내는 방식이에요. 봉투에 적혀 있으면 그대로 적어요.")
    cup_notes = st.text_input("커피 노트 (선택)", placeholder="예: 견과류, 초콜릿", key="nb-notes",
                              help="로스터리가 봉투에 적어 둔 맛과 향 표현이에요.")
    amount = st.number_input("양 (g, 선택)", min_value=0.0, step=50.0, format="%.0f", value=None, key="nb-amount")
    status = cal.rest_status(roast_level, roast_date.isoformat(), date.today())
    html(f'<div class="dc-caption"><b>{escape(status.label)}</b><br>{escape(status.message)}</div>')
    if st.button("추가하기", type="primary", width="stretch", key="nb-save"):
        try:
            bean_id = cal.append_bean({"name": name, "roaster": roaster, "roast_date": roast_date.isoformat(),
                                       "roast_level": roast_level, "origin": origin, "process": process,
                                       "cup_notes": cup_notes, "amount_g": amount}, BEANS_PATH)
        except cal.BeanError as error:
            notice(str(error), error=True)
        else:
            st.session_state["new_bean"] = bean_id  # 다음 실행에서 1단계 원두 칸에 넣는다 (칸을 그리기 전에만 바꿀 수 있어서)
            st.rerun()


# 1단계에서 장비 종류마다 고르는 칸의 key
EQUIPMENT_KEYS = {"드리퍼": "flow-dripper", "필터": "flow-filter", "그라인더": "flow-grinder"}
EQUIPMENT_HELP = {
    "드리퍼": "물을 붓는 깔때기 모양 도구예요. 목록에 없으면 아래 '＋ 장비 추가'로 더해요.",
    "필터": "드리퍼에 끼우는 종이예요. 종이마다 물 빠지는 속도가 달라질 수 있어 기록에 함께 남겨요.",
    "그라인더": "원두를 가는 기계예요. 클릭 수를 알려 줄 때 써요.",
}


def equipment_select(where, equipment, kind, key, chosen=None):
    """장비 목록에서만 고르는 칸. 고른 것이 없고 그 종류가 하나뿐이면 그것을 채운다.
    이름을 직접 치는 칸은 두지 않는다(일부만 치고 Enter를 누르면 '홀츠' 같은 새 이름이 생겨서). 새 장비는 '장비 추가'로."""
    names = [e["name"] for e in equipment[kind]]
    if chosen not in names:
        chosen = names[0] if len(names) == 1 else None
    return where.selectbox(kind, names, index=names.index(chosen) if chosen else None,
                           placeholder=f"{kind}를 골라 주세요", key=key, help=EQUIPMENT_HELP[kind])


def dripper_shape(equipment, dripper):
    return next((e["detail"] for e in equipment["드리퍼"] if e["name"] == dripper), "")


NOT_CONE = "이 앱의 물 빠짐 계산은 V60 같은 원뿔형 드리퍼를 기준으로 해요. 다른 모양은 시간 예측이 많이 다를 수 있어요."


@st.dialog("장비 추가")
def add_equipment_dialog():
    """장비 목록에 없는 드리퍼·필터·그라인더를 더한다. 저장하면 1단계 칸에서 바로 고른 상태가 된다."""
    html('<div class="dc-caption">오늘 쓰는 도구를 더해 두면 다음부터 목록에서 고를 수 있어요.</div>')
    kind = st.segmented_control("종류", cal.EQUIPMENT_KINDS, default=cal.EQUIPMENT_KINDS[0], key="ne-kind")
    kind = kind or cal.EQUIPMENT_KINDS[0]
    examples = {"드리퍼": "예: 칼리타 웨이브 185", "필터": "예: 칼리타 웨이브 필터 185", "그라인더": "예: 코만단테 C40"}
    name = st.text_input("이름", key=f"ne-name-{kind}", placeholder=examples[kind])
    detail = ""
    if kind == "드리퍼":
        detail = st.radio("모양", cal.DRIPPER_SHAPES, key="ne-shape", horizontal=True,
                          help="옆에서 봤을 때 아래로 갈수록 좁아지면 원뿔형, 바닥이 넓고 평평하면 평평한 형이에요.")
        if detail != "원뿔형":
            notice(NOT_CONE)
    elif kind == "그라인더":
        if name.strip() and cal.table_clicks(1000, name) is not None:
            html('<div class="dc-caption">클릭표가 있는 그라인더예요. 처음에는 클릭표로 시작 클릭을 알려 드려요.</div>')
        else:
            html('<div class="dc-caption">처음 쓰는 그라인더는 클릭 수를 알려 줄 근거가 없어요. 가루 사진으로 잰 기록이 '
                 "쌓이면 몇 클릭에서 시작하면 되는지 알려 드려요.</div>")
    else:
        html('<div class="dc-caption">상자에 적힌 이름을 그대로 적어요. 드리퍼 크기에 맞는 필터인지 확인해 주세요.</div>')
    if st.button("추가하기", type="primary", width="stretch", key="ne-save"):
        try:
            cal.append_equipment(kind, name, detail, EQUIPMENT_PATH)
        except cal.EquipmentError as error:
            notice(str(error), error=True)
        else:
            st.session_state["new_equipment"] = (kind, name.strip())  # 다음 실행에서 1단계 칸에 넣는다
            st.rerun()


def rest_notice(bean):
    """로스팅 뒤 디게싱이 얼마나 됐는지. 좋을 때는 작은 글씨로, 이르거나 상미 기간이 지났으면 상자로 알린다."""
    try:
        status = cal.rest_status(bean["roast_level"], bean["roast_date"], date.today())
    except ValueError:  # 로스팅 날짜가 없거나 형식이 다름
        return
    notes = f"<br>커피 노트: {escape(bean['cup_notes'])}" if bean.get("cup_notes") else ""
    if status.state == "좋음":
        html(f'<div class="dc-caption"><b>{escape(status.label)}</b> · {escape(status.message)}{notes}</div>')
    else:
        html(f'<div class="dc-notice"><b>{escape(status.label)}</b><br>{escape(status.message)}{notes}</div>')


def flow_equipment(flow):
    """1단계: 원두와 장비."""
    html('<div class="dc-lead">원두와 장비를 알려 주면 오늘 한 잔을 처음부터 끝까지 안내해요.<br>'
         "레시피를 추천하고, 간 가루를 사진으로 확인해 맞추고, 내린 결과와 맛을 기록해 다음 추천에 반영해요.</div>")
    brews = cal.load_brews(BREWS_PATH)
    beans = cal.load_beans(BEANS_PATH)
    bean_ids = list(beans)
    last = brews[-1] if brews else None
    added = st.session_state.pop("new_bean", None)
    if added in beans:
        st.session_state["flow-bean"] = added
        notice(f"{added} {cal.with_object(beans[added]['name'])} 원두 목록에 더했어요.")
    if not beans:  # 처음 쓰는 사람: 원두부터 더한다
        notice("아직 원두가 없어요. 먼저 오늘 내릴 원두를 추가해 주세요.")
        if st.button("＋ 새 원두 추가", type="primary", width="stretch", key="flow-add-first-bean"):
            add_bean_dialog()
        return
    # 원두는 늘 직접 고른다. 지난 원두가 저절로 넘어가 잘못 기록되는 일을 막으려고 비워 둔 채 시작한다
    # (시작 화면에서 '같은 원두로 내리기'를 누른 때만 그 원두를 채운다)
    chosen_bean = flow.get("bean_id")
    c1, c2 = st.columns([2, 1])
    bean_id = c1.selectbox("원두", bean_ids, index=bean_ids.index(chosen_bean) if chosen_bean in beans else None,
                           placeholder="원두를 골라 주세요",
                           format_func=lambda b: f"{b} {beans[b]['name']} ({beans[b]['roast_level']})", key="flow-bean")
    dose = c2.number_input("원두 무게 (g)", min_value=5.0, max_value=40.0, step=0.1, format="%.1f",
                           value=float(flow.get("dose_g", 20.0)), key="flow-dose",
                           help="저울로 잰 원두 무게예요. 보통 한 잔에 15–20 g을 써요.")
    if st.button("＋ 새 원두 추가", type="tertiary", key="flow-add-bean"):
        add_bean_dialog()
    if bean_id:
        if beans[bean_id]["role"] == "시험":
            notice("시험용 원두예요. 모델을 동결한 뒤 최종 시험에서만 내려요 (MODEL.md 6절).")
        rest_notice(beans[bean_id])
    # 장비는 목록에서만 고른다. 드리퍼·그라인더는 지난 잔 것을 채우고, 필터는 장소마다 달라서 매번 고른다
    # (목록에 하나뿐이면 그것을 채운다)
    equipment = cal.load_equipment(EQUIPMENT_PATH)
    new = st.session_state.pop("new_equipment", None)
    if new:
        kind, name = new
        st.session_state[EQUIPMENT_KEYS[kind]] = name
        notice(f"{cal.with_object(name)} 장비 목록에 더했어요.")
    c1, c2, c3 = st.columns(3)
    dripper = equipment_select(c1, equipment, "드리퍼", "flow-dripper",
                               flow.get("dripper") or (last.row.get("dripper") if last else None))
    filter_paper = equipment_select(c2, equipment, "필터", "flow-filter", flow.get("filter"))
    grinder = equipment_select(c3, equipment, "그라인더", "flow-grinder",
                               flow.get("grinder") or (last.grinder if last else None))
    if st.button("＋ 장비 추가", type="tertiary", key="flow-add-equipment"):
        add_equipment_dialog()
    if dripper and dripper_shape(equipment, dripper) != "원뿔형":
        notice(NOT_CONE)
    ready = bool(bean_id and dripper and filter_paper and grinder)
    if bottom_cta("다음: 기초 추천 받기", "flow-next-1", ready, "원두·드리퍼·필터·그라인더를 고르면 다음으로 갈 수 있어요"):
        flow.update(bean_id=bean_id, dose_g=float(dose), dripper=dripper, filter=filter_paper, grinder=grinder)
        go_to(2)


LITERATURE_PASSED = (13, 14)  # 문헌 비교 판정 통과 수 / 전체 (MODEL.md 6절)
INTENT_ICONS = {"평소대로": "coffee", "더 진하게": "dark_mode", "더 연하게": "water_drop",
                "더 우러나게": "local_fire_department", "더 산뜻하게": "eco"}
EFFECTS = {  # 레시피 값이 맛에 주는 영향 (모델의 방향 테스트와 같은 방향, MODEL.md 6절)
    "분쇄": "곱게 갈수록 더 우러나고 물이 늦게 빠져요",
    "물 온도": "높일수록 더 우러나요 · 쓴맛과 바디가 늘어요",
    "물": "많이 부을수록 더 우러나지만 묽어져요",
    "가수": "많이 탈수록 연해져요 · 우러난 정도는 그대로",
}


@st.cache_data(show_spinner=False, max_entries=4)
def drawdown_error_s(log_version):
    """표시 없는 잔의 끊김 예측 오차 크기 평균(초). 기록이 없으면 None."""
    comparisons = [c for c in cal.compare_with_log(cal.load_brews(BREWS_PATH)) if not c.brew.flag]
    errors = [abs(c.end_error_s) for c in comparisons if c.end_error_s is not None]
    return sum(errors) / len(errors) if errors else None


def evidence_chips(brews):
    """추천이 무엇에 기대는지: 내 기록, 예측 오차, 문헌 비교."""
    error = drawdown_error_s(BREWS_PATH.stat().st_mtime)
    chips = [f"내 기록 {len(brews)}잔 반영" if brews else "기록이 없어 골든컵에서 시작",
             *([f"예측 오차 평균 {error:.0f}초"] if error is not None else []),
             f"문헌 비교 {LITERATURE_PASSED[0]}/{LITERATURE_PASSED[1]} 통과"]
    html('<div class="dc-evidence">' + "".join(f"<span>✓ {escape(c)}</span>" for c in chips) + "</div>")


def base_changes(r, hint, target, brews, intent):
    """베이스(평소 레시피와 그 잔)에서 무엇을 왜 바꿨는지 줄마다."""
    cup = next((b for b in brews if b.brew_id == r.usual.brew_id), None)
    recipe, usual = r.recipe, r.usual
    low, high = r.temp_range
    rows = [
        change_row("device_thermostat", "물 온도", usual.temp_c, recipe.water_temp_c, "°C", f"배전도 범위 {low}–{high} °C 안"),
        change_row("water_drop", "물", usual.water_ratio * recipe.dose_g, recipe.water_g, "g",
                   f"원두의 {r.water_ratio}배 · {rec.pattern_text(r.pattern)}배 · {r.interval_s:g}초 간격"),
    ]
    if cup and cup.grinder_clicks is not None and hint and hint.clicks is not None:
        rows.append(change_row("grain", "분쇄", cup.grinder_clicks, hint.clicks, "클릭",
                               f"목표 추출수율 {target.ey_pct:.1f}%에 맞춰 · {grind_feel(r.d50_um)}"))
    if cup:
        rows.append(change_row("water_full", "가수", cup.bypass_g, recipe.bypass_g, "g",
                               f"목표 농도 {target.tds_pct:.2f}%에 맞춰"
                               + (f" · 오늘은 '{intent}'" if intent != "평소대로" else "")))
    html('<div class="dc-change">' + "".join(rows) + "</div>")
    where = (f"{short_date(cup.date)} {cup.bean_id} {cup.row.get('bean_name', '')} 잔".strip() if cup else "기본 레시피")
    same = cup is not None and r.usual.source.startswith("같은 원두")
    html(f'<div class="dc-caption" style="margin-top:8px">베이스: {escape(base_name(usual))} · {escape(where)}'
         + ("" if same or not cup else " (기록이 없는 원두라 같은 배전도 묶음의 마지막 잔)") + "</div>")


def reset_what_if(keys):
    for key in keys:
        st.session_state.pop(key, None)


def what_if(r, target, intent):
    """분쇄와 온도만 바꿔 모델로 다시 계산해 본다 (붓기와 가수는 그대로). 레시피는 바꾸지 않는다."""
    section("이렇게 바꾸면?", "모델로 바로 계산해요. 붓기와 가수는 그대로 두고 분쇄와 온도만 바꿔 봐요.")
    center = int(round(r.d50_um / 50) * 50)
    sizes = [d for d in range(center - 300, center + 301, 50) if 700 <= d <= 1800]
    low, high = r.temp_range
    temps = list(range(min(low, int(r.recipe.water_temp_c)), max(high, int(r.recipe.water_temp_c)) + 1))
    keys = (f"whatif-grind-{intent}", f"whatif-temp-{intent}")
    grind = st.select_slider("분쇄", sizes, value=center, key=keys[0], format_func=lambda d: f"{d} µm · {grind_feel(d)}")
    temp = st.select_slider("물 온도", temps, value=int(r.recipe.water_temp_c), key=keys[1], format_func=lambda t: f"{t} °C")
    recipe = em.Recipe(r.recipe.dose_g, float(temp), r.recipe.pours, r.recipe.bypass_g)
    try:
        p = em.predict(recipe, em.grind_from_d50(float(grind)))
    except em.RecipeError as error:
        notice(str(error), error=True)
        return
    ref = em.predict(em.Recipe(r.recipe.dose_g, r.recipe.water_temp_c, r.recipe.pours, r.recipe.bypass_g),
                     em.grind_from_d50(float(center)))

    def versus(value, base, unit, digits):
        diff = round(value - base, digits)
        return "추천과 같아요" if diff == 0 else f"추천보다 {diff:+g}{unit}"

    html('<div class="dc-summary three">'
         + tile("timer", "끊김", clock(p.drawdown_end_s), "", versus(p.drawdown_end_s, ref.drawdown_end_s, "초", 0))
         + tile("science", "추출수율", f"{p.ey_pct:.1f}", "%", versus(p.ey_pct, ref.ey_pct, "%p", 1))
         + tile("opacity", "농도", f"{p.tds_cup_pct:.2f}", "%", versus(p.tds_cup_pct, ref.tds_cup_pct, "%p", 2))
         + "</div>")
    gap = p.ey_pct - target.ey_pct
    verdict = ("더 우러나요. 단맛과 바디가 늘지만 쓴맛도 같이 늘 수 있어요." if gap > 0.7
               else "덜 우러나요. 산뜻하고 가볍지만 신맛이 튀거나 밍밍할 수 있어요." if gap < -0.7
               else "내 입맛 목표 근처예요.")
    if p.drawdown_end_s > rec.GOOD_DRAWDOWN_S[1]:
        verdict += " 물 빠짐이 3:30을 넘어서 추천에서는 고르지 않는 조합이에요."
    html(f'<div class="dc-verdict">{escape(verdict)}</div>')
    st.button("↺ 추천값으로 되돌리기", type="tertiary", key="whatif-reset", on_click=reset_what_if, args=(keys,))


def flow_base_recipe(flow):
    """2단계: 기초 추천. 근거, 오늘의 요청, 베이스에서 바뀐 것, 레시피와 각 값의 영향, 붓기 일정, '이렇게 바꾸면?'."""
    brews = cal.load_brews(BREWS_PATH)
    beans = cal.load_beans(BEANS_PATH)
    bean = beans[flow["bean_id"]]
    html(f'<div class="dc-lead">{escape(bean["bean_id"])} {escape(bean["name"])} · {escape(bean["roast_level"])} · '
         f'원두 {flow["dose_g"]:g} g</div>')
    evidence_chips(brews)
    section("오늘은 어떻게?", "고르면 이번 잔의 목표만 옮겨요. 내 입맛 학습은 그대로예요.")
    intent = st.segmented_control("오늘은 어떻게?", list(rec.INTENTS), default=flow.get("intent", "평소대로"),
                                  key="flow-intent", label_visibility="collapsed",
                                  format_func=lambda name: f":material/{INTENT_ICONS[name]}: {name}") or "평소대로"
    flow["intent"] = intent
    try:
        with st.spinner("레시피 후보를 계산하는 중..."):
            target, r = cached_recommendation(flow["bean_id"], flow["dose_g"], BREWS_PATH.stat().st_mtime, intent)
    except rec.NoRecipeError as error:
        notice(str(error), error=True)
        if st.button("이전", key="flow-back-2"):
            go_to(1)
        return
    hint = cal.click_hint(r.d50_um, flow["grinder"], bean["bean_id"], bean["roast_date"], brews)
    flow.update(target_d50_um=r.d50_um, hint_clicks=hint.clicks if hint else None)
    recipe = r.recipe
    low, high = r.temp_range

    section("베이스에서 이렇게 바꿨어요")
    base_changes(r, hint, target, brews, intent)
    section("이번 레시피", f"목표: 추출수율 {target.ey_pct:.1f}%, 농도 {target.tds_pct:.2f}%")
    summary(
        tile("grain", "목표 분쇄", f"{r.d50_um:.0f}", "µm", f"{hint_short(hint, flow['grinder'])} · {grind_feel(r.d50_um)}",
             EFFECTS["분쇄"]),
        tile("device_thermostat", "물 온도", f"{recipe.water_temp_c:g}", "°C", f"{bean['roast_level']} {low}–{high} °C",
             EFFECTS["물 온도"]),
        tile("water_drop", "물", f"{recipe.water_g:g}", "g", f"원두의 {r.water_ratio}배", EFFECTS["물"]),
        tile("water_full", "가수", f"{recipe.bypass_g:g}", "g", "드리퍼를 뺀 뒤 더해요", EFFECTS["가수"]),
    )
    html(f'<div class="dc-caption">{escape(hint_text(hint, flow["grinder"]))}</div>')
    time_limit_notice(r, target)
    section("붓기 일정", f"{r.interval_s:g}초마다 부어요 · 원두의 {rec.pattern_text(r.pattern)}배 · "
                     "붓는 방식(가운데로, 원을 그리며)은 평소대로 해요.")
    pour_timeline(r.recipe, r.prediction.drawdown_end_s)
    html(f'<div class="dc-caption">예상 끊김 {clock(r.prediction.drawdown_end_s)} '
         "(모델 예측이라 원두에 따라 20초쯤 다를 수 있어요).</div>")
    what_if(r, target, intent)
    notice("이제 위 클릭으로 원두를 갈고, 사진용으로 조금 덜어 축척 용지에 얇게 펴 주세요. "
           "다음 단계에서 사진으로 실제 크기를 재고, 목표와 다르면 그 가루에 맞게 레시피를 고쳐요.")
    if bottom_cta("다음: 갈고 사진 찍기", "flow-next-2", back_to=1, back_key="flow-back-2"):
        go_to(3)


def time_limit_notice(r, target):
    """끊김 3:30 안에서는 목표 추출수율에 닿지 못해 가장 가까운 레시피를 골랐을 때만 그 까닭을 알린다 (MODEL.md 9.2)."""
    if not r.ey_gap_pct:
        return
    why = "막혔던 기록 때문에 더 곱게 갈 수 없어서, " if r.clog_limit_um else ""
    notice(f"오래 우리면 쓴맛이 나기 쉬워서 물 빠짐은 3:30을 넘지 않게 해요. {why}그 안에서 가장 많이 우러나는 "
           f"레시피예요. 목표보다 조금 덜 우러나요 (예상 추출수율 {r.prediction.ey_pct:.1f}% · 목표 {target.ey_pct:.1f}%).")


def size_verdict(measured_um, target_um):
    """잰 크기가 목표와 얼마나 다른지 한 문장으로. 3% 안이면 같다고 본다(사진 측정의 흔들림보다 작음)."""
    diff = measured_um - target_um
    if abs(diff) <= 0.03 * target_um:
        return "목표와 거의 같아요"
    return f"목표보다 {abs(diff):.0f} µm {'굵어요' if diff > 0 else '고와요'}"


def interval_text(change_s):
    if change_s == 0:
        return "추천 그대로예요."
    if change_s > 0:
        return f"추천보다 {change_s:g}초 늘렸어요. 굵은 가루가 덜 우러나지 않게 물에 더 오래 닿게 해요."
    return f"추천보다 {-change_s:g}초 줄였어요. 고운 가루가 너무 우러나지 않게 물에 덜 닿게 해요."


def flow_grind(flow):
    """3단계: 분쇄 확인. 간 가루를 사진으로 재고, 목표와 다르면 붓기 간격과 가수량을 그 가루에 맞춘다(MODEL.md 9.3)."""
    target, r = cached_recommendation(flow["bean_id"], flow["dose_g"], BREWS_PATH.stat().st_mtime, flow.get("intent", "평소대로"))
    html('<div class="dc-lead">간 가루를 조금 덜어 축척 용지의 <b>A칸(큰 칸)</b>에 <b>서로 닿지 않게</b> 펴고, '
         "네 모서리의 검은 네모가 모두 보이게 위에서 찍어 주세요.<br>"
         f"목표 대표 크기는 <b>{r.d50_um:.0f} µm</b>예요. 대표 크기는 가루 무게의 절반이 이보다 작은 크기예요.</div>")
    clicks = st.number_input("실제로 간 클릭", min_value=0, max_value=180, step=1,
                             value=flow.get("clicks", flow.get("hint_clicks")), key="flow-clicks",
                             help="그라인더에서 맞춘 숫자예요. 기록에 남겨 다음에 같은 원두를 갈 때 클릭을 알려 드려요.")
    uploads = st.file_uploader("가루 사진 (여러 장 가능)", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                               key="flow-photos",
                               help="폰 카메라로 찍은 원본을 올려 주세요. 여러 장을 올리면 합쳐서 더 정확해요.") or []
    measurements, names, shown, errors = [], [], [], []
    for upload in uploads:
        with st.spinner(f"{upload.name} 재는 중..."):
            rectified, measurement, error = analyze_upload(upload.getvalue())
        if error:
            errors.append((upload.name, error))
            continue
        shown.append((upload.name, rectified, measurement))
        if len(measurement.diameters_um):
            measurements.append(measurement)
            names.append(upload.name)

    if errors:
        toast(photo_errors_text(errors))
    if not measurements:
        if uploads and shown:
            toast("입자를 잰 사진이 없어요. 가루를 칸 안쪽에 듬성듬성 펴고 다시 찍어 주세요.")
        elif not uploads:
            notice("사진을 올리면 가루의 실제 크기를 재고, 목표와 다르면 그 가루에 맞게 레시피를 고쳐 드려요. "
                   "사진 없이 내리면 크기를 몰라서, 이 잔으로는 입맛을 배우기 어려워요.")
        if st.button("사진 없이 추천대로 내리기", type="tertiary", key="flow-skip-3"):
            flow.update(clicks=clicks, measured_d50_um=None, photo_names=[], profile=None, recipe=r.recipe,
                        pattern=r.pattern, interval_s=r.interval_s, prediction=r.prediction)
            go_to(4)
        bottom_cta("다음: 이 레시피로 내리기", "flow-next-3", False, "가루 사진을 올리면 다음으로 갈 수 있어요",
                   back_to=2, back_key="flow-back-3")
        return

    profile = ga.combine_measurements(measurements)
    fit = rec.refit_to_grind(r, profile.d50_um, target)
    low, high = profile.d50_range_um
    change = fit.interval_change_s
    summary(
        tile("grain", "잰 대표 크기", f"{profile.d50_um:.0f}", "µm", size_verdict(profile.d50_um, r.d50_um)),
        tile("schedule", "붓기 간격", f"{fit.interval_s:g}", "초",
             "추천 그대로" if change == 0 else f"추천보다 {abs(change):g}초 {'늘림' if change > 0 else '줄임'}"),
        tile("water_full", "가수", f"{fit.recipe.bypass_g:g}", "g", f"추천 {r.recipe.bypass_g:g} g"),
        tile("timer", "예상 끊김", clock(fit.prediction.drawdown_end_s), "", "물줄기가 끊기는 때"),
    )
    html(f'<div class="dc-caption">목표 {r.d50_um:.0f} µm. {escape(interval_text(change))} 같은 가루를 다시 찍으면 '
         f"{low:.0f}~{high:.0f} µm로 나올 수 있어요.</div>")
    if not fit.reached:
        need = "더 곱게" if profile.d50_um > r.d50_um else "더 굵게"
        notice(f"간격을 15초까지 바꿔도 목표만큼 {'우러나지 않아요' if fit.ey_gap_pct < 0 else '덜 우러나게 하지 못해요'}"
               f"(추출수율 {abs(fit.ey_gap_pct):.1f}%p 차이). 이대로 내려도 되고, 다음에는 {need} 갈아 주세요.")
    for warning in profile.warnings:
        notice(warning)
    section("이 가루에 맞춘 붓기 일정", f"원두 {fit.recipe.dose_g:g} g · {fit.recipe.water_temp_c:g} °C · "
                                     f"물 {fit.recipe.water_g:g} g · 그다음 가수 {fit.recipe.bypass_g:g} g")
    pour_timeline(fit.recipe, fit.prediction.drawdown_end_s)
    with st.expander(f"사진에서 잰 입자 보기 ({len(shown)}장 · 입자 {profile.n_particles}개)"):
        html('<div class="dc-caption">파란 선은 잰 입자, 회색 선은 뺀 것(여러 알이 엉킨 덩어리, 칸 테두리에 걸린 입자, '
             "은피)이에요.</div>")
        for name, rectified, measurement in shown:
            html(f'<div class="dc-caption"><b>{escape(name)}</b></div>')
            show_photo_result(rectified, measurement)

    if bottom_cta("다음: 이 레시피로 내리기", "flow-next-3", back_to=2, back_key="flow-back-3"):
        flow.update(clicks=clicks, measured_d50_um=profile.d50_um, photo_names=names, profile=profile,
                    recipe=fit.recipe, pattern=r.pattern, interval_s=fit.interval_s, prediction=fit.prediction)
        go_to(4)


# 4단계 '문제가 있었나요?'에서 고르는 표시. 처음 쓰는 사람도 알 수 있게 풀어 쓴다.
FLAG_CHOICES = {"": "없었어요", "막힘": "물이 고여 잘 안 빠졌어요", "흘림": "물이나 커피를 흘렸어요",
                "측정실수": "시각이나 무게를 잘못 쟀어요", "연습": "연습으로 내렸어요", "기타": "그 밖의 문제가 있었어요"}


def versus_time(predicted_s, actual_s):
    """끊긴 시각을 예측과 비교한 한 문장."""
    diff = actual_s - predicted_s
    if abs(diff) < 3:
        return f"예측({clock(predicted_s)})과 거의 같아요."
    return f"예측({clock(predicted_s)})보다 {abs(diff):.0f}초 {'늦게' if diff > 0 else '빨리'} 끊겼어요."


def versus_grams(predicted_g, actual_g):
    """음료 무게를 예측과 비교한 한 문장."""
    diff = actual_g - predicted_g
    if abs(diff) < 1:
        return f"예측({predicted_g:.0f} g)과 거의 같아요."
    return f"예측({predicted_g:.0f} g)보다 {abs(diff):.0f} g {'많아요' if diff > 0 else '적어요'}."


def flow_brew(flow):
    """4단계: 내리기와 기록. 맞춘 붓기 일정대로 내리고, 끊긴 시각과 음료 무게를 적으면 예측과 비교한다."""
    recipe, prediction = flow["recipe"], flow["prediction"]
    html('<div class="dc-lead">타이머를 켜고 첫 물(뜸)을 붓는 순간을 <b>0:00</b>으로 해요. 아래 시각에 맞춰 부어 주세요.<br>'
         "물줄기가 끊기고 방울로 바뀌면 드리퍼를 바로 빼고, 그 시각을 적어요. "
         "뜸은 처음에 물을 조금 부어 가루를 고루 적시는 단계예요.</div>")
    measured = flow.get("measured_d50_um")
    clicks = f"{flow['clicks']}클릭" if flow.get("clicks") is not None else "클릭 기록 없음"
    # 사진이 없으면 예측에 쓴 목표 크기를 보여 준다('–'로 두면 예상 시각이 무엇을 기준으로 했는지 안 보여서).
    # 기록에는 잰 값이 아니므로 남기지 않는다
    size = measured or flow["target_d50_um"]
    summary(
        tile("coffee", "원두", f"{recipe.dose_g:g}", "g", flow.get("bean_id", "")),
        tile("device_thermostat", "물 온도", f"{recipe.water_temp_c:g}", "°C", "주전자 물"),
        tile("grain", "분쇄" if measured else "목표 분쇄", f"{size:.0f}", "µm",
             f"{clicks} · " + ("사진으로 잼" if measured else f"사진 없음 · {grind_feel(size)}")),
        tile("water_drop", "물", f"{recipe.water_g:g}", "g", f"그다음 가수 {recipe.bypass_g:g} g"),
    )
    section("붓기 일정", "타이머 0:00부터 이 시각에 부어요. 저울이 아래 숫자가 될 때까지예요.")
    pour_timeline(recipe, prediction.drawdown_end_s)
    html(f'<div class="dc-caption">예상: 물줄기는 {clock(prediction.drawdown_end_s)}쯤 끊기고, 가수 전 음료는 '
         f"약 {prediction.beverage_g:.0f} g이에요. 모델 예측이라 원두에 따라 20초쯤 다를 수 있어요.</div>")

    section("결과 적기", "드리퍼를 뺀 시각과, 가수하기 전에 잰 음료 무게예요.")
    c1, c2 = st.columns(2)
    drawdown = c1.text_input("물줄기가 끊긴 시각", value=flow.get("drawdown_text", ""), placeholder="예: 2:25",
                             key="flow-drawdown", help="분:초(2:25)나 초(145)로 적어요.")
    beverage = c2.number_input("음료 무게 (g, 가수 전)", min_value=0.0, step=0.1, format="%.1f",
                               value=flow.get("beverage_g"), key="flow-beverage",
                               help="서버째 잰 무게에서 빈 서버 무게를 빼요. 가수하기 전에 재요.")
    with st.expander("문제가 있었나요? (선택)"):
        flags = list(FLAG_CHOICES)
        flag = st.radio("문제", flags, index=flags.index(flow.get("flag", "")), format_func=FLAG_CHOICES.get,
                        key="flow-flag", label_visibility="collapsed")
        html('<div class="dc-caption">문제가 있었던 잔도 기록은 남기지만, 다음 추천을 배울 때는 빼요.</div>')

    seconds = None
    try:
        seconds = cal.parse_clock(drawdown)
    except cal.BrewLogError as error:
        toast(str(error))
    if seconds and beverage:
        html(
            '<div class="dc-cards">'
            + card("물줄기 끊김", clock(seconds), "", versus_time(prediction.drawdown_end_s, seconds))
            + card("음료 (가수 전)", f"{beverage:g}", "g", versus_grams(prediction.beverage_g, beverage))
            + "</div>"
        )
        st.altair_chart(timeline_chart(prediction, recipe, actual_end=seconds), width="stretch")
        html('<div class="dc-caption">파란 선은 모델이 예측한 음료가 나오는 모습이에요. 진한 회색 점선이 실제로 끊긴 시각이에요.</div>')

    if bottom_cta("다음: 맛 평가", "flow-next-4", bool(seconds and beverage),
                  "끊긴 시각과 음료 무게를 적으면 다음으로 갈 수 있어요", back_to=3, back_key="flow-back-4"):
        flow.update(drawdown_text=drawdown, drawdown_end_s=seconds, beverage_g=float(beverage), flag=flag)
        go_to(5)


# 5단계 진하기 질문: 기록에 남기는 값 → 화면에 보이는 말
STRENGTH_CHOICES = {"연함": "연했어요", "좋음": "딱 좋았어요", "진함": "진했어요"}


def flow_record(flow, taste, strength, memo):
    """흐름에서 모은 값으로 brews.csv 한 줄을 만든다."""
    brews = cal.load_brews(BREWS_PATH)
    bean = cal.load_beans(BEANS_PATH)[flow["bean_id"]]
    intent = flow.get("intent", "평소대로")
    target, r = cached_recommendation(flow["bean_id"], flow["dose_g"], BREWS_PATH.stat().st_mtime, intent)
    recipe, prediction = flow["recipe"], flow["prediction"]
    measured = flow.get("measured_d50_um")
    today = date.today().isoformat()
    # 내리기 전에 낸 예측을 함께 남긴다. 검증(MODEL.md 6절)은 이 예측과 실제를 비교한다
    note = f"앱 예측(내리기 전): 끊김 {clock(prediction.drawdown_end_s)}, 음료 {prediction.beverage_g:.0f} g."
    return {
        "brew_id": cal.next_brew_id(today, brews), "date": today, "bean_id": flow["bean_id"],
        **{brew_col: bean[bean_col] for bean_col, brew_col in BEAN_TO_BREW.items()},
        "roast_date": bean["roast_date"], "dripper": flow.get("dripper", ""), "filter": flow["filter"],
        "grinder": flow["grinder"], "grinder_clicks": flow.get("clicks"),
        "grind_d50_um": round(measured) if measured else None, "recipe": "추천 레시피",
        "dose_g": recipe.dose_g, "water_temp_c": recipe.water_temp_c, "pours": rec.pours_text(recipe.pours),
        "drawdown_end_s": flow["drawdown_end_s"], "beverage_g": flow["beverage_g"], "bypass_g": recipe.bypass_g,
        **taste, "strength": strength, "memo": " ".join(part for part in (memo.strip(), note) if part),
        "flag": flow.get("flag", ""),
        # 사진에 맞춰 붓기 간격을 바꿨으면 바꾸기 전 간격을 남긴다. 다음 원두의 '평소 레시피'는 바꾸기 전 간격을 쓴다
        "followed_recommendation": f"예 (한 잔 내리기, 추천 {r.d50_um:.0f} µm"
                                   + (f" → 잰 {measured:.0f} µm에 맞춤" if measured else "")
                                   + (f" · 간격 {r.interval_s:g} → {flow['interval_s']:g}초"
                                      if flow.get("interval_s", r.interval_s) != r.interval_s else "")
                                   + (f" · 오늘은 '{intent}'" if intent != "평소대로" else "")
                                   + f", 목표 {target.ey_pct:.1f}% · {target.tds_pct:.2f}%)",
    }


STRENGTH_ICONS = {"연했어요": "water_drop", "딱 좋았어요": "check_circle", "진했어요": "coffee"}


def flow_taste(flow):
    """5단계: 맛 평가. 가수하고 바로 맛본 점수와 진하기를 받아 이 잔을 기록에 저장한다.
    가수 안내는 들어올 때 한 번 위에서 내려왔다 사라지는 알림으로 띄우고(누를 필요 없게), 화면에는 작은 표시만 둔다.
    만족도 육각형을 위에 두고
    슬라이더는 '좋았던 정도'와 '세기' 두 묶음으로 나눈다. 진하기는 큰 선택 카드, 메모는 접어 둔다."""
    recipe = flow["recipe"]
    if not flow.get("taste_intro_shown"):
        flow["taste_intro_shown"] = True
        toast(f"물 {recipe.bypass_g:g} g 넣고 바로 맛봐요. 식기 전에 매겨야 잔끼리 비교돼요.", icon="water_full")
    html(f'<div class="dc-chip"><span class="dc-icon" aria-hidden="true">water_full</span>'
         f"가수 {recipe.bypass_g:g} g · 바로 맛보기</div>")

    def key(name):
        return f"flow-taste-{name}"

    guide = "\n".join(f"- **{label}** ({TASTE_GUIDE[name][0]}): {TASTE_GUIDE[name][1]}"
                       for name, label, _ in cal.TASTE_ITEMS)
    st.container(key="taste-title").subheader("맛", help=guide, anchor=False)
    html('<div class="dc-caption">0.5점 단위, 3점이 보통이에요. 움직이기 전에는 \'–\'(안 매김)로 보여요.</div>')
    hexagon = st.empty()  # 슬라이더를 다 그린 뒤 점수로 채운다
    with st.container(key="taste-good"):
        html('<div class="dc-group">좋았던 정도<span>1~5점</span></div>')
        taste = taste_inputs(key, cal.QUALITY_ITEMS, short=True)
    with st.container(key="taste-strength"):
        html('<div class="dc-group">세기<span>0~5점 · 클수록 좋은 게 아니에요</span></div>')
        taste.update(taste_inputs(key, ("acidity", "body"), short=True))
    taste = {name: taste[name] for name, _, _ in cal.TASTE_ITEMS}  # 기록지 순서로
    hexagon.markdown(taste_hexagon(taste, compact=True), unsafe_allow_html=True)

    section("진하기는 어땠나요?", "다음 잔의 가수 물을 맞추는 데 써요.")
    labels = list(STRENGTH_CHOICES.values())
    picked = st.segmented_control("진하기는 어땠나요?", labels, key="flow-strength", label_visibility="collapsed",
                                  format_func=lambda label: f":material/{STRENGTH_ICONS[label]}: {label}")
    with st.expander("메모 남기기 (선택)", icon=":material/edit:"):
        memo = st.text_area("메모", key="flow-memo", label_visibility="collapsed",
                            placeholder="나만 보는 기록이에요. 예: 식으면서 단맛이 올라옴")

    rated = all(v is not None for v in taste.values())
    hint = "맛 6항목을 모두 매기면 기록할 수 있어요" if not rated else "진하기를 고르면 기록할 수 있어요"
    if bottom_cta("이 잔 기록하기", "flow-save", rated and bool(picked), hint, back_to=4, back_key="flow-back-5"):
        strength = next(code for code, label in STRENGTH_CHOICES.items() if label == picked)
        before, r = cached_recommendation(flow["bean_id"], flow["dose_g"], BREWS_PATH.stat().st_mtime)
        record = flow_record(flow, taste, strength, memo)
        try:
            cal.append_brew(record, BREWS_PATH)
        except cal.BrewLogError as error:
            toast(str(error), error=True)
            return
        if flow.get("profile") is not None:  # 사진 측정도 이 추출번호로 저장해 둔다 ('사진만 재기'의 저장한 측정)
            ga.save_profile(ga.profile_record(flow["profile"], flow["bean_id"], flow["grinder"], flow.get("clicks"),
                                              record["brew_id"], flow.get("photo_names", ())), PROFILE_DIR)
        flow.update(saved_brew_id=record["brew_id"], target_before=(before.ey_pct, before.tds_pct),
                    bypass_before=r.recipe.bypass_g, d50_before=r.d50_um)
        go_to(6)


def vs_short(diff, unit, more, less):
    """예측과 실제의 차이를 짧게. diff는 실제 − 예측."""
    if abs(diff) < (3 if unit == "초" else 1):
        return "예측과 거의 같아요"
    return f"예측보다 {abs(diff):.0f}{unit} {more if diff > 0 else less}"


def change_row(icon, name, before, after, unit, why):
    """다음 잔에서 바뀌는 값 한 줄: 아이콘, 이름과 바뀐 이유, 전 → 후, 변화량."""
    delta = after - before
    pill = (f'<span class="dc-pill">{delta:+.0f} {escape(unit)}</span>' if abs(delta) >= 1
            else '<span class="dc-pill same">그대로</span>')
    return (f'<div class="dc-change-row"><span class="dc-icon" aria-hidden="true">{icon}</span>'
            f'<span class="dc-change-name">{escape(name)}<small>{escape(why)}</small></span>'
            f'<span class="dc-change-from">{before:.0f} →</span>'
            f'<span class="dc-change-to">{after:.0f} {escape(unit)}</span>{pill}</div>')


def bypass_reason(strength, delta):
    """가수가 바뀐 이유를 사용자가 고른 진하기 말로. 진하기 답과 바뀐 방향이 맞을 때만 그 답을 이유로 든다
    (목표 농도는 진하기 답으로 배운다, MODEL.md 9.1)."""
    answer = STRENGTH_CHOICES.get(strength)
    consistent = {"진함": delta >= 1, "연함": delta <= -1, "좋음": abs(delta) < 1}.get(strength, False)
    return f"'{answer}'라고 답해서" if consistent else "맛 점수와 진하기를 함께 반영해서"


def flow_finish(flow):
    """6단계: 마무리. 이번 잔이 내 선호 영역의 어디에 있고, 다음 추천이 어떻게 바뀌었는지 보여 준다."""
    brews = cal.load_brews(BREWS_PATH)
    saved = next((b for b in brews if b.brew_id == flow.get("saved_brew_id")), None)
    if saved is None:  # 저장하지 않고 이 단계에 온 경우
        notice("아직 기록한 잔이 없어요. 5단계에서 맛을 매기고 기록해 주세요.", error=True)
        if st.button("이전", key="flow-back-6"):
            go_to(5)
        return
    points = rec.brew_points(brews)
    after = rec.preference_target(points)
    prediction = flow["prediction"]
    html(f'<div class="dc-done"><div class="dc-icon" aria-hidden="true">check</div>'
         f'<div class="dc-done-title">기록했어요</div><div class="dc-done-sub">{escape(saved.brew_id)} · '
         f"{escape(saved.bean_id)} {escape(saved.row.get('bean_name', ''))}</div></div>")
    summary(
        tile("star", "만족도", f"{saved.quality_score:.2f}", "점", "맛 4항목 평균 (산미·Body 빼고)"),
        tile("coffee", "진하기", STRENGTH_CHOICES.get(saved.strength, "–"), "", "다음 잔 가수에 반영"),
        tile("timer", "물줄기 끊김", clock(saved.drawdown_end_s), "", vs_short(saved.drawdown_end_s - prediction.drawdown_end_s,
                                                                          "초", "늦어요", "빨라요")),
        tile("local_cafe", "음료", f"{saved.beverage_g:g}", "g", vs_short(saved.beverage_g - prediction.beverage_g,
                                                                       " g", "많아요", "적어요")),
    )
    if saved.flag:
        html('<div class="dc-caption">문제가 있었던 잔이라 다음 추천을 배울 때는 뺐어요.</div>')
    if not saved.flag and not any(p.brew_id == saved.brew_id for p in points):
        html('<div class="dc-caption">분쇄 크기를 몰라서 이번 잔은 다음 추천을 배울 때 쓰지 못했어요. '
             "다음에는 3단계에서 가루 사진을 찍어 주세요.</div>")

    try:
        _, nxt = cached_recommendation(flow["bean_id"], flow["dose_g"], BREWS_PATH.stat().st_mtime)
    except rec.NoRecipeError:
        nxt = None
    if nxt:
        section("다음 잔은 이렇게 바뀌어요", f"{saved.bean_id}를 같은 양으로 다시 내릴 때예요.")
        bypass_delta = nxt.recipe.bypass_g - flow["bypass_before"]
        html('<div class="dc-change">'
             + change_row("water_full", "가수", flow["bypass_before"], nxt.recipe.bypass_g, "g",
                          bypass_reason(saved.strength, bypass_delta))
             + change_row("grain", "분쇄", flow["d50_before"], nxt.d50_um, "µm",
                          "맛 점수를 반영해 다시 맞춰서" if abs(nxt.d50_um - flow["d50_before"]) >= 1 else "바뀌지 않았어요")
             + "</div>")
    with st.expander("자세히 보기"):
        ey0, tds0 = flow["target_before"]
        html(rows([("목표 추출수율", f"{ey0:.1f}% → {after.ey_pct:.1f}%"),
                   ("목표 농도 (가수 후)", f"{tds0:.2f}% → {after.tds_pct:.2f}%")]))
        html('<div class="dc-caption">얼마나 우러나게(추출수율)는 맛 점수로, 얼마나 진하게(농도)는 진하기로 배워요. '
             "기록한 잔 전체의 위치는 '내 기록 보기'의 내 선호 영역에서 볼 수 있어요.</div>")

    with FLOW_CTA:
        left, right = st.columns([1, 2])
        left.button("내 기록 보기", width="stretch", key="flow-finish-history", on_click=open_page, args=("history",))
        right.button("완료", type="primary", width="stretch", key="flow-finish-home", on_click=open_page,
                     args=("home",))


def flow_later(flow):
    """정해진 단계 밖이면 처음부터 다시 시작하게 한다."""
    notice("이 단계는 없어요. 처음부터 다시 시작해 주세요.", error=True)
    c1, c2 = st.columns([1, 2])
    if c1.button("이전", width="stretch", key="flow-back-later"):
        go_to(flow["step"] - 1)
    c2.button("처음으로", width="stretch", key="flow-restart", on_click=open_page, args=("home",))


def flow_page():
    back_link("flow")
    flow = st.session_state.setdefault("flow", {"step": 1})
    step = flow["step"]
    global FLOW_CTA
    screen = st.container(key=f"flow-step-{step}")  # 단계가 바뀌면 새 화면이 올라오며 나타난다 (STYLE)
    FLOW_CTA = st.container(key="flow-cta")  # 화면 아래에 붙는 버튼 자리 (bottom_cta)
    with screen:
        flow_header(min(step, len(FLOW_STEPS)))
        {1: flow_equipment, 2: flow_base_recipe, 3: flow_grind, 4: flow_brew, 5: flow_taste,
         6: flow_finish}.get(step, flow_later)(flow)


def open_page(page):
    """화면을 바꾼다. 한 잔 내리기는 늘 1단계부터, 사용법은 첫 장부터 시작한다."""
    st.session_state["page"] = page
    if page == "flow":
        st.session_state["flow"] = {"step": 1}
    elif page == "tutorial":
        st.session_state["tutorial"] = 0


def back_link(name):
    st.button("← 처음으로", type="tertiary", key=f"{name}-home", on_click=open_page, args=("home",))


def page_header(name, title, lead, eyebrow="Drip Coffee"):
    back_link(name)
    html(f'<div class="dc-eyebrow">{escape(eyebrow)}</div><div class="dc-title">{escape(title)}</div>'
         f'<div class="dc-lead">{lead}</div>')


def list_row(key, markup, label, on_click, args=()):
    """누를 수 있는 목록 한 줄 (토스 ListRow). 보이는 내용은 HTML로 그리고, 같은 크기의 투명한 버튼을 그 위에 덮어
    줄 어디를 눌러도 눌리게 한다. 화면 읽기 프로그램은 버튼 이름(label)을 읽는다."""
    with st.container(key=f"row-{key}"):
        html(markup)
        st.button(label, key=key, on_click=on_click, args=args, width="stretch")


def row_markup(lead, title, sub="", action="", muted=False, wide_sub=""):
    """목록 한 줄의 모양: 왼쪽 동그라미(lead), 제목과 작은 설명, 오른쪽 글(할 일은 파랗게, 개수는 회색)과 '>'.
    wide_sub는 넓은 화면에서만 설명 앞에 붙인다 (폰에서는 설명이 두 줄로 꺾이지 않게 뺀다)."""
    wide = f'<span class="dc-wide">{escape(wide_sub)}</span>' if wide_sub else ""
    sub = f"<small>{wide}{escape(sub)}</small>" if sub else ""
    return (f'<div class="dc-lrow">{lead}<span class="dc-lrow-text"><b>{escape(title)}</b>{sub}</span>'
            f'<span class="dc-lrow-go{" muted" if muted else ""}">{escape(action)}</span><span class="dc-lrow-chev" aria-hidden="true">chevron_right</span></div>')


def short_date(iso):
    """'2026-10-19' → '10월 19일'."""
    try:
        day = date.fromisoformat(iso)
    except ValueError:
        return iso
    return f"{day.month}월 {day.day}일"


def brew_same_bean(last):
    """지난 잔과 같은 원두로 한 잔 내리기를 시작한다. 원두와 양만 채우고(드리퍼·그라인더는 원래 지난 잔 것을 채운다)
    필터는 장소마다 달라서 비워 둔다. 레시피는 2단계에서 지금까지의 기록으로 새로 계산한다."""
    open_page("flow")
    st.session_state["flow"].update(bean_id=last.bean_id, dose_g=last.dose_g)


def home_page():
    """첫 화면: 아래에 붙은 '시작하기' 하나를 크게 두고, 지난 잔과 내 기록은 누를 수 있는 목록 한 줄씩.
    모델 점검은 사용자 화면이 아니라 모델이 실제 잔을 얼마나 맞히는지 보는 검증 화면이라 맨 아래 작게 둔다."""
    html('<div class="dc-home"><div class="dc-icon dc-home-icon" aria-hidden="true">coffee</div>'
         '<div class="dc-eyebrow">Drip Coffee</div><div class="dc-hero">오늘의 한 잔</div>'
         '<div class="dc-lead">원두와 장비만 알려 주면 레시피부터 기록까지 안내해요.<br>'
         "맛을 매길수록 다음 잔이 내 입맛에 가까워져요.</div></div>")
    brews = cal.load_brews(BREWS_PATH)
    # 사용법 링크: 처음 쓰는 사람에게는 파랗게, 기록이 있는 사람에게는 '다시 보기'로 회색으로 작게
    with st.container(key="home-guide-again" if brews else "home-guide-new"):
        st.button("사용법 다시 보기 ›" if brews else "처음이세요? 사용법 보기 ›", type="tertiary", key="home-tutorial",
                  on_click=open_page, args=("tutorial",))
    with st.container(key="home-list"):
        if brews:
            last = brews[-1]
            score = f" · {last.quality_score:.2f}점" if last.quality_score is not None else ""
            name = last.row.get("bean_name", "") or last.bean_id
            list_row("home-same-bean",
                     row_markup(f'<span class="dc-avatar">{escape(last.bean_id)}</span>', name,
                                f"{short_date(last.date)}{score}", "같은 원두로 내리기", wide_sub="지난 잔 "),
                     f"같은 원두로 내리기: {last.bean_id} {name}", brew_same_bean, (last,))
        list_row("home-history",
                 row_markup('<span class="dc-icon" aria-hidden="true">history</span>', "내 기록",
                            action=f"{len(brews)}잔" if brews else "아직 없어요", muted=True),
                 "내 기록 보기", open_page, ("history",))
    st.button("모델 점검 (검증용)", type="tertiary", width="stretch", key="home-check", on_click=open_page, args=("check",))
    with st.container(key="home-cta"):  # 화면 아래에 붙는 버튼 (한 잔 내리기의 아래 버튼과 같은 모양)
        # 기록이 하나도 없으면 사용법부터 보여 준다 (이번 접속에서 이미 봤으면 바로 1단계로)
        first = not brews and not st.session_state.get("tutorial_seen")
        st.button("시작하기", type="primary", width="stretch", key="home-start", on_click=open_page,
                  args=("tutorial" if first else "flow",))


# 사용법 1장의 도구 그림 (선으로 직접 그림, 88×88). 이름 → SVG 안쪽
TOOL_ICONS = (
    ("원두", '<g transform="rotate(-30 44 44)"><ellipse cx="44" cy="44" rx="17" ry="25"/>'
             '<path d="M44 19 C 34 33, 54 55, 44 69"/></g>'),
    ("드리퍼", '<path d="M16 23 H70 L54 55 H32 Z"/><path d="M25 65 H61"/><path d="M36 55 V65 M50 55 V65"/>'
              '<path d="M70 25 C 83 28, 83 43, 63 41"/>'),
    ("필터", '<path d="M14 23 Q44 12 74 23 L49 69 H39 Z"/><path d="M22 23.5 L40 68" stroke-width="2.4" opacity=".55"/>'),
    ("그라인더", '<path d="M35 32 H53 V71 Q53 75 49 75 H39 Q35 75 35 71 Z"/><path d="M35 46 H53 M35 52 H53"/>'
                f'<path d="M32 32 H56"/><path d="M44 32 V20 H69 V16"/><circle cx="69" cy="12" r="5" fill="{BRAND}"/>'),
)
TOOL_DONE = (f'<circle cx="44" cy="44" r="28" fill="{BRAND}" stroke="none"/>'
             '<path d="M31 44 L40 53 L58 35" stroke="#FFFFFF" stroke-width="5"/>')
TOOL_START_S, TOOL_STEP_S = 0.3, 0.75  # 첫 도구가 나오는 때, 다음 도구까지 간격(초)


def tutorial_tools():
    """1장: 원두 → 드리퍼 → 필터 → 그라인더가 차례로 나오고 완료 표시로 끝난다."""
    delays = [TOOL_START_S + i * TOOL_STEP_S for i in range(len(TOOL_ICONS) + 1)]
    icons = "".join(f'<svg class="dc-tool" style="--d:{d:.2f}s" viewBox="0 0 88 88">{svg}</svg>'
                    for d, (_, svg) in zip(delays, TOOL_ICONS))
    done = f'<svg class="dc-tool last" style="--d:{delays[-1]:.2f}s" viewBox="0 0 88 88">{TOOL_DONE}</svg>'
    names = "".join(f'<span class="dc-tool-slot" style="--d:{d:.2f}s"><span class="dc-tool-name">'
                    f'<span class="dc-tool-check">✓</span>{name}</span></span>' for d, (name, _) in zip(delays, TOOL_ICONS))
    return icons + done + f'<div class="dc-tool-names">{names}</div>'


# 2장의 가루 알갱이: (가로 %, 세로 %, 지름 px)
TUTORIAL_GRAINS = [(8, 20, 9), (18, 62, 7), (27, 35, 11), (33, 75, 8), (41, 18, 7), (47, 52, 10), (55, 30, 8), (60, 70, 9),
                   (68, 45, 7), (74, 15, 10), (80, 62, 8), (87, 35, 9), (13, 45, 6), (38, 48, 5), (64, 22, 6), (91, 72, 6)]


def tutorial_photo():
    """2장: 선이 왼쪽에서 오른쪽으로 훑으면 지나간 알갱이에 테두리가 생긴다(선이 지나가는 때에 맞춰 늦게 시작).
    네 모서리의 검은 네모는 실제 축척 용지의 표시처럼."""
    corners = "".join(f'<b style="{v}:8px;{h}:8px"></b>' for v in ("top", "bottom") for h in ("left", "right"))
    grains = "".join(f'<span class="dc-grain" style="left:{x}%;top:{y}%;width:{d}px;height:{d * 0.85:.1f}px;'
                     f'animation-delay:{0.3 + x / 100 * 1.45:.2f}s"></span>' for x, y, d in TUTORIAL_GRAINS)
    return (f'<div class="dc-tut-paper">{corners}{grains}<span class="dc-scan"></span>'
            '<span class="dc-tut-size">대표 크기 1090 µm</span></div>')


# 3장의 붓는 칸: (시작 %, 길이 %, 시각). STYLE의 dc-kettle-tilt 값과 맞춘다
TUTORIAL_POURS = [(4, 14, "0:00 뜸"), (34, 16, "0:45"), (64, 16, "1:30")]
TUTORIAL_POUR_S, TUTORIAL_POUR_START_S = 4.5, 0.3  # 포트가 시간표를 지나는 시간, 시작하는 때(초)
# 커피포트 (72×72): 아래로 넓어지는 몸통, 왼쪽 손잡이, 오른쪽 위로 곧게 뻗은 주둥이(끝 67, 31)
KETTLE = ('<path d="M21 30 H45 L50 60 Q50 63 47 63 H19 Q16 63 16 60 Z"/><path d="M18 36 C 8 38, 8 52, 17 54"/>'
          '<path d="M49 54 L67 31"/>')


def tutorial_pours():
    """3장: 커피포트가 시간표를 지나가며 파란 칸에서 기울어 붓고, 칸은 포트가 지나간 만큼 채워진다."""
    def at(percent):
        return TUTORIAL_POUR_START_S + TUTORIAL_POUR_S * percent / 100

    pours = "".join(f'<span class="dc-tut-pour" style="left:{x}%;width:{w}%;--d:{at(x):.2f}s;'
                    f'--t:{TUTORIAL_POUR_S * w / 100:.2f}s"></span><span class="dc-tut-time" style="left:{x}%">{t}</span>'
                    for x, w, t in TUTORIAL_POURS)
    return (f'<div class="dc-tut-track">{pours}<span class="dc-tut-pourer"><svg class="dc-kettle" viewBox="0 0 72 72">'
            f'{KETTLE}</svg></span></div><div class="dc-tut-note">물 빠짐까지 예측 2:32</div>')


def tutorial_taste():
    """4장: 맛 육각형이 커지고 '다음 잔 가수 125 → 139 g'이 올라온다."""
    def at(i, r):
        angle = math.radians(-90 + 60 * i)
        return f"{75 + r * math.cos(angle):.1f},{70 + r * math.sin(angle):.1f}"

    grid = "".join(f'<polygon points="{" ".join(at(i, 58 * k) for i in range(6))}" fill="none" stroke="#E5E8EB"/>'
                   for k in (1, 0.66, 0.33))
    shape = " ".join(at(i, 58 * v) for i, v in enumerate([0.8, 0.7, 0.35, 0.55, 0.85, 0.75]))
    return (f'<svg class="dc-tut-hex" viewBox="0 0 150 140">{grid}<polygon class="dc-tut-shape" points="{shape}" '
            f'fill="{BRAND}" fill-opacity="0.2" stroke="{BRAND}" stroke-width="2" stroke-linejoin="round"/></svg>'
            '<div class="dc-tut-change"><span class="dc-icon">water_full</span><span style="flex:1">다음 잔 가수</span>'
            '125 → <b>139 g</b><span class="dc-pill">+14 g</span></div>')


# 사용법 네 장: (제목, 설명, 그림)
TUTORIAL = (
    ("원두와 장비를 골라요", "봉투에 적힌 원두 이름과 로스팅 날짜면 충분해요. 드리퍼·필터·그라인더는 목록에서 골라요.",
     tutorial_tools),
    ("간 가루를 사진 한 장으로 재요", "가루를 축척 용지에 펴서 찍으면 알갱이 크기를 재서, 레시피를 내 가루에 맞춰요.",
     tutorial_photo),
    ("추천 레시피에 맞춰 추출해요", "언제, 얼마나 부을지 시간표로 보여 줘요. 저울과 타이머만 보면서 따라 하면 돼요.",
     tutorial_pours),
    ("맛을 매기면 다음 잔이 맞춰져요", "Aroma·Flavor 같은 맛 6가지를 매기고 진하기를 고르면, 다음 잔의 물과 분쇄가 "
     "내 입맛 쪽으로 바뀌어요.", tutorial_taste),
)


def tutorial_prev():
    st.session_state["tutorial"] = max(st.session_state.get("tutorial", 0) - 1, 0)


def tutorial_next():
    page = st.session_state.get("tutorial", 0)
    if page >= len(TUTORIAL) - 1:
        open_page("flow")
    else:
        st.session_state["tutorial"] = page + 1


def tutorial_page():
    """사용법: 한 잔 내리기의 흐름을 네 장의 움직이는 그림으로 보여 준다(토스 온보딩처럼). 기록이 하나도 없을 때
    '시작하기'를 누르면 먼저 나오고, 시작 화면의 '사용법 보기'로 언제든 다시 볼 수 있다."""
    page = min(st.session_state.setdefault("tutorial", 0), len(TUTORIAL) - 1)
    last = page == len(TUTORIAL) - 1
    st.session_state["tutorial_seen"] = True
    with st.container(key="tutorial-top", horizontal=True, horizontal_alignment="distribute"):
        back_link("tutorial")
        if not last:  # 마지막 장은 '시작하기'가 같은 일을 한다
            st.button("건너뛰기", type="tertiary", key="tutorial-skip", on_click=open_page, args=("flow",))
    title, lead, art = TUTORIAL[page]
    dots = "".join(f'<i class="{"on" if i == page else ""}"></i>' for i in range(len(TUTORIAL)))
    with st.container(key=f"tutorial-{page}"):  # 장이 바뀌면 새로 그려져 그림이 처음부터 움직인다
        # 그림에 SVG가 있어서 st.html이 아니라 markdown으로 넣는다
        st.markdown(f'<div class="dc-tut-art" aria-hidden="true">{art()}</div>', unsafe_allow_html=True)
        html(f'<div class="dc-eyebrow">{page + 1} / {len(TUTORIAL)}</div><div class="dc-tut-title">{escape(title)}</div>'
             f'<div class="dc-tut-lead">{escape(lead)}</div><div class="dc-dots">{dots}</div>')
    with st.container(key="flow-cta"):  # 한 잔 내리기와 같은 아래 버튼 자리. 첫 장 말고는 '이전 | 다음'
        if page == 0:
            right = st.container()
        else:
            left, right = st.columns([1, 2])
            left.button("이전", width="stretch", key="tutorial-prev", on_click=tutorial_prev)
        right.button("시작하기" if last else "다음", type="primary", width="stretch", key="tutorial-next",
                     on_click=tutorial_next)


def history_page():
    """내 기록: 요약 카드, 원두 칩, '기록한 잔 | 맛 변화 | 내 선호 영역' 세 보기, 직접 기록하기. 잔이나 점을 누르면 자세히."""
    brews = cal.load_brews(BREWS_PATH)
    page_header("history", "내 기록", f"{len(brews)}잔을 기록했어요." if brews else "아직 기록한 잔이 없어요.")
    points = rec.brew_points(brews)
    target = rec.preference_target(points)
    history_summary(brews, target)
    views = ["기록한 잔", "맛 변화", "내 선호 영역"]
    view = st.segmented_control("보기", views, default=views[0], key="history-view", label_visibility="collapsed") or views[0]
    beans = list(dict.fromkeys(b.bean_id for b in brews))
    names = {b.bean_id: b.row.get("bean_name", "") for b in brews}
    bean = "전체"
    if len(beans) > 1:
        bean = st.pills("원두", ["전체", *beans], default="전체", required=True, key="history-bean",
                        label_visibility="collapsed",
                        format_func=lambda x: x if x == "전체" else f"{x} {names[x].split(' ')[0]}") or "전체"
    shown = brews if bean == "전체" else [b for b in brews if b.bean_id == bean]
    if view == views[0]:
        cup_list(shown)
    elif view == views[1]:
        taste_trend(shown)
    else:
        preference_section(shown, [p for p in points if bean == "전체" or p.bean_id == bean], target)
    opened = st.session_state.pop("history-open", None)
    if opened:
        cup_detail(opened)
    writing = "log_prefill" in st.session_state or "log_saved" in st.session_state
    with st.expander("직접 기록하기", expanded=writing, icon=":material/edit_note:"):
        log_form()


def check_page():
    """모델 점검: 예측과 실제 비교, 레시피 직접 계산, 추천 계산 자세히, 커피 없이 분쇄 사진만 재기."""
    page_header("check", "모델 점검", "모델이 실제 잔을 얼마나 맞히는지 봐요. 사용자 화면이 아니라 모델을 확인하는 "
                "화면이에요. 식과 근거는 docs/MODEL.md에 있어요.", eyebrow="검증용")
    views = ["예측과 실제", "레시피 계산", "추천 자세히", "사진만 재기"]
    view = st.segmented_control("보기", views, default=views[0], key="check-view", label_visibility="collapsed") or views[0]
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
TOASTS = st.container(key="dc-toasts")  # 위에서 내려오는 알림 자리 (toast)
PAGES = {"home": home_page, "flow": flow_page, "tutorial": tutorial_page, "history": history_page,
         "check": check_page}
PAGES[st.session_state.setdefault("page", "home")]()
