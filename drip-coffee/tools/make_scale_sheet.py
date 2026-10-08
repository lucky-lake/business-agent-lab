"""분쇄도 측정용 축척 용지(A4)를 PDF와 PNG로 만든다.

사용법 (drip-coffee 폴더에서):
    python tools/make_scale_sheet.py
결과: out/scale_sheet_v1.pdf (인쇄용), out/scale_sheet_v1.png (미리보기)

인쇄할 때는 반드시 '실제 크기(100%)'로 출력하고, 기준 막대가 100 mm인지 자로 확인한다.
"""
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dripcoffee import OUT_DIR, sheet  # noqa: E402

MM_PER_INCH = 25.4
KOREAN_FONTS = ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]

INSTRUCTIONS = [
    "1. '실제 크기(100%)'로 인쇄하세요. '페이지에 맞춤'은 끄세요.",
    "2. 인쇄 후 자로 위 기준 막대가 100 mm인지 확인하세요.",
    "3. 원두를 갈아 한 꼬집(0.05~0.1 g)을 가루 칸 안에 놓고, 카드나 붓으로 입자끼리 닿지 않게 얇게 펴세요.",
    "4. 네 모서리 마커가 모두 보이게, 폰을 종이와 평행하게 들고 찍으세요.",
    "5. 플래시는 끄고, 그림자가 지지 않는 밝은 곳에서 찍으세요. 마커 위에는 가루를 올리지 마세요.",
    "6. 해상도를 높이려면 B 칸을 2배 줌이나 매크로 모드로 가까이 찍으세요.",
]
MEMO_FIELDS = ["원두", "그라인더 / 클릭", "측정 분쇄도 (µm)\n분석 후 기입", "날짜", "시료 번호"]


def pick_korean_font():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in KOREAN_FONTS:
        if name in available:
            return name
    return "DejaVu Sans"  # 한글이 깨질 수 있지만 마커와 축척에는 영향 없음


def marker_bits(marker_id):
    """ArUco 마커를 칸 단위 행렬로 만든다. True가 검은 칸."""
    img = cv2.aruco.generateImageMarker(
        sheet.aruco_dictionary(), marker_id, sheet.ARUCO_GRID_CELLS, borderBits=1
    )
    return img < 128


def draw_matrix(ax, dark, x, y, size):
    """검은 칸 행렬(마커)을 (x, y)부터 한 변 size mm 크기로 그린다.
    화면에서 칸 사이에 가는 틈이 보이지 않도록 가로로 이어진 칸은 한 사각형으로 합친다."""
    n_rows, n_cols = dark.shape
    cell = size / n_cols
    for r in range(n_rows):
        c = 0
        while c < n_cols:
            if not dark[r, c]:
                c += 1
                continue
            start = c
            while c < n_cols and dark[r, c]:
                c += 1
            ax.add_patch(
                Rectangle(
                    (x + start * cell, y + r * cell),
                    (c - start) * cell,
                    cell,
                    facecolor="black",
                    edgecolor="none",
                    antialiased=False,
                )
            )


def draw_zone(ax, zone):
    ax.add_patch(
        Rectangle(
            (zone.square_x, zone.square_y),
            zone.square_size,
            zone.square_size,
            facecolor="white",
            edgecolor="#b0b0b0",
            linewidth=0.5,
        )
    )
    for marker_id, (mx, my) in zip(zone.marker_ids, zone.marker_origins()):
        draw_matrix(ax, marker_bits(marker_id), mx, my, zone.marker_size)
    top = zone.square_y - zone.gap - zone.marker_size
    ax.text(zone.square_x - zone.gap - zone.marker_size, top - 2.0, zone.label, fontsize=9, va="bottom")


def draw_reference_bar(ax):
    x0, y0, length = sheet.REF_BAR_X_MM, sheet.REF_BAR_Y_MM, sheet.REF_BAR_LEN_MM
    ax.add_patch(Rectangle((x0, y0), length, 1.0, facecolor="black", edgecolor="none"))
    for mm in range(int(length) + 1):
        tick = 4.0 if mm % 10 == 0 else (2.5 if mm % 5 == 0 else 1.5)
        ax.plot([x0 + mm, x0 + mm], [y0 + 1.0, y0 + 1.0 + tick], color="black", linewidth=0.3)
        if mm % 10 == 0:
            ax.text(x0 + mm, y0 + 6.0, str(mm), fontsize=6, ha="center", va="top")
    ax.text(x0, y0 - 1.5, "기준 막대 100 mm: 인쇄 후 자로 확인", fontsize=8, va="bottom")


def build_figure():
    plt.rcParams["font.family"] = pick_korean_font()
    plt.rcParams["pdf.fonttype"] = 42  # 글꼴을 PDF에 그대로 넣어 다른 컴퓨터에서도 같게 보이게

    fig = plt.figure(figsize=(sheet.PAGE_W_MM / MM_PER_INCH, sheet.PAGE_H_MM / MM_PER_INCH))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, sheet.PAGE_W_MM)
    ax.set_ylim(sheet.PAGE_H_MM, 0)  # y는 아래로 커지게 (용지 좌표와 같게)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(20, 14, f"분쇄도 측정 용지 v{sheet.SHEET_VERSION}", fontsize=14, fontweight="bold", va="bottom")
    ax.text(20, 19, "Drip Coffee: 핸드드립 추출 분석·추천 도구", fontsize=8, va="bottom", color="#555555")

    for zone in sheet.ZONES:
        draw_zone(ax, zone)

    draw_reference_bar(ax)

    y = 150.0
    ax.text(20, y, "사용법", fontsize=10, fontweight="bold", va="bottom")
    for line in INSTRUCTIONS:
        y += 7.0
        ax.text(20, y, line, fontsize=8, va="bottom")

    box_gap, box_h, box_y = 2.5, 16.0, 215.0
    box_w = (170.0 - box_gap * (len(MEMO_FIELDS) - 1)) / len(MEMO_FIELDS)
    for i, field in enumerate(MEMO_FIELDS):
        bx = 20 + i * (box_w + box_gap)
        ax.add_patch(Rectangle((bx, box_y), box_w, box_h, facecolor="none", edgecolor="black", linewidth=0.5))
        ax.text(bx + 1.5, box_y + 1.5, field, fontsize=7, va="top")

    ax.text(
        20,
        285,
        "마커의 위치·크기는 dripcoffee/sheet.py에 정의되어 있으며, 사진 분석도 같은 값을 씁니다.",
        fontsize=6,
        color="#777777",
        va="bottom",
    )
    return fig


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    pdf_path = OUT_DIR / f"scale_sheet_v{sheet.SHEET_VERSION}.pdf"
    png_path = OUT_DIR / f"scale_sheet_v{sheet.SHEET_VERSION}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"저장: {pdf_path}")
    print(f"저장: {png_path}")


if __name__ == "__main__":
    main()
