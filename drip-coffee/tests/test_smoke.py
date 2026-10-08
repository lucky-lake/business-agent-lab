"""설치와 경로 설정이 맞는지 확인하는 기본 테스트."""
import cv2

from dripcoffee import PROJECT_ROOT


def test_project_root_is_drip_coffee_folder():
    assert (PROJECT_ROOT / "app.py").exists()
    assert (PROJECT_ROOT / "requirements.txt").exists()


def test_opencv_has_aruco():
    # 축척 용지의 모서리 마커를 찾는 데 필요하다.
    assert hasattr(cv2, "aruco")
