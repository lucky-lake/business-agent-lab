# 참고 데이터

- `maverick_click_table.csv`: 홀츠클로츠 메버릭1 클릭 수별 분쇄 크기(µm). 인터넷 기준 값이며, 사진으로 잰 분쇄도와 비교하는 용도로만 쓴다.
- `literature_brews.csv`: 추출 모델의 농도·추출수율 예측을 검증하려고 문헌에서 옮긴 실험값 (조사 노트 4, `docs/research/04_검증데이터.md`).
  - `caa_v60`: Coffee ad Astra V60 반복 추출 3회 (블로그)
  - `batali_drip`: Batali 2020 보충 표 S1 중 7회 (배치 브루어)
  - `varady_v60`: Váradý 2022 표 1의 V60 분쇄 4단계
  - 행마다 출처와 표·그림 위치를 적었다. 사용자가 원문과 대조하기 전까지 `status`는 '미확인'이다. 원문에 없는 값을 재구성했거나 해석한 부분은 `note`에 적었다.
