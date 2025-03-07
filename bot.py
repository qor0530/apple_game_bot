import mss
import mss.tools
from PIL import Image
import numpy as np
import cv2
import pyautogui
import time
import keyboard  # keyboard 라이브러리 추가
import math

def merge_nearby_locations(locations, threshold=3):
    """
    주변 위치들을 병합하는 함수

    Args:
        locations (list): (x, y) 튜플의 리스트
        threshold (int): 병합을 위한 최대 거리 (픽셀)

    Returns:
        list: 병합된 위치들의 리스트
    """
    merged_locations = []
    used = [False] * len(locations)

    for i in range(len(locations)):
        if used[i]:
            continue

        x_sum = locations[i][0]
        y_sum = locations[i][1]
        count = 1
        used[i] = True

        for j in range(i + 1, len(locations)):
            if not used[j]:
                x_diff = abs(locations[i][0] - locations[j][0])
                y_diff = abs(locations[i][1] - locations[j][1])

                if x_diff <= threshold and y_diff <= threshold:
                    x_sum += locations[j][0]
                    y_sum += locations[j][1]
                    count += 1
                    used[j] = True

        merged_x = x_sum // count
        merged_y = y_sum // count
        merged_locations.append((merged_x, merged_y))

    return merged_locations

# 1. 화면 캡처
with mss.mss() as sct:
    monitor_number = 0
    mon = sct.monitors[monitor_number]
    capture_area = {
        'left': mon['left'],
        'top': mon['top'],
        'width': mon['width'],
        'height': mon['height'],
        'mon': monitor_number
    }
    sct_img = sct.grab(capture_area)
    img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
    img_np = np.array(img)
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)  # 화면 이미지를 grayscale로 변환

# 숫자 템플릿 준비
number_locations = {}
total_count = 0
for number in range(1, 10):
    # 2. 템플릿 준비 (파일에서 읽기)
    template_filename = f'{number}.png'
    template = cv2.imread(template_filename, cv2.IMREAD_COLOR)
    if template is None:
        print(f"Error: Could not read template image {template_filename}")
        continue

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template.shape[:2]

    # 3. 이미지 매칭
    res = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)

    # 4. 결과 추출 및 필터링
    threshold = 0.965  # 유사도 임계값 0.97로 변경
    loc = np.where(res >= threshold)

    locations = []
    for pt in zip(*loc[::-1]):
        # 중심 좌표 계산
        center_x = pt[0] + w // 2
        center_y = pt[1] + h // 2
        locations.append((center_x, center_y))

    # 새로운 위치 병합 함수 적용
    merged_locations = merge_nearby_locations(locations, threshold=3)

    number_locations[number] = merged_locations
    count = len(merged_locations)
    total_count += count

    # 5. 숫자별 결과 출력
    print(f"Number {number} locations: {merged_locations}")
    print(f"Number {number} count: {count}")

    # 사각형 그리기 (선택 사항) - 병합된 위치에 그리기
    for (center_x, center_y) in merged_locations:
        cv2.rectangle(img_np, (center_x - w//2, center_y - h//2), (center_x + w//2, center_y + h//2), (0, 255, 0), 2)
        cv2.circle(img_np, (center_x, center_y), 5, (0, 0, 255), -1) # 좌표에 점 찍기


# 6. 전체 결과 출력
print(f"Total count: {total_count}")

# (선택 사항) 이미지 저장
cv2.imwrite('numbers_detected.png', img_np)

# 7. 숫자 위치에 클릭
for number, locations in number_locations.items():
    for (center_x, center_y) in locations:
        if keyboard.is_pressed('q'): # 반복문 시작 전에 확인
            print("Stopped by user.")
            cv2.destroyAllWindows()
            exit() # 또는 break 후 clear_numbers() 호출을 중단

        pyautogui.click(center_x, center_y)
        time.sleep(0.1)
        # 드래그 앤 드롭
        pyautogui.moveTo(center_x, center_y)
        pyautogui.mouseDown()
        pyautogui.moveTo(center_x -10, center_y -10, duration=0.1) # duration 추가
        pyautogui.mouseUp()
        time.sleep(0.1)

        if keyboard.is_pressed('q'): # 드래그 앤 드롭 후에도 확인
            print("Stopped by user.")
            cv2.destroyAllWindows()
            exit() # 또는 break 후 clear_numbers() 호출을 중단
