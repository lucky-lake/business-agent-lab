"""핸드드립 추출 분석·추천 도구의 계산 패키지.

경로는 모두 이 파일 위치를 기준으로 만든다. 그래야 drip-coffee/ 안에서 실행하든
저장소 루트에서 실행하든 같은 폴더를 가리킨다.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LOCAL_DATA_DIR = DATA_DIR / "local"  # 앱이 읽고 쓰는 작업 데이터 (커밋하지 않음)
OUT_DIR = PROJECT_ROOT / "out"  # 도구가 만드는 출력물 (커밋하지 않음)
