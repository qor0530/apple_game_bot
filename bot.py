import mss
import mss.tools
from PIL import Image
import numpy as np
import cv2
import pyautogui
import time

def merge_nearby_locations(locations, threshold=3):
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

def calculate_sum_in_range(appletables, start, end):
    total_sum = 0
    for i in range(start[0], end[0] + 1):
        for j in range(start[1], end[1] + 1):
            total_sum += appletables[i][j].num
    return total_sum

def find_best_combination(appletables):
    rows = len(appletables)
    cols = len(appletables[0]) if rows > 0 else 0
    min_non_zero_count = float('inf')
    best_combination = None
    
    for i in range(rows):
        for j in range(cols):
            for ni in range(i, rows):
                for nj in range(j, cols):
                    total_sum = 0
                    non_zero_count = 0
                    for x in range(i, ni + 1):
                        for y in range(j, nj + 1):
                            num = appletables[x][y].num
                            total_sum += num
                            if num != 0:
                                non_zero_count += 1
                    if total_sum == 10 and non_zero_count < min_non_zero_count:
                        min_non_zero_count = non_zero_count
                        best_combination = ((i, j), (ni, nj))
    return best_combination

def drag_and_remove_combinations(appletables):
    box_padding = 30 # 드래그 범위 패딩

    while True:
        best_combination = find_best_combination(appletables)
        if best_combination:
            remove_combination(appletables, best_combination[0], best_combination[1], box_padding)
        else:
            break

def remove_combination(appletables, start, end, box_padding):
    if calculate_sum_in_range(appletables, start, end) != 10:
        return

    start_x, start_y = appletables[start[0]][start[1]].x, appletables[start[0]][start[1]].y
    end_x, end_y = appletables[end[0]][end[1]].x, appletables[end[0]][end[1]].y

    start_drag_x = start_x - box_padding
    start_drag_y = start_y - box_padding
    end_drag_x = end_x + box_padding
    end_drag_y = end_y + box_padding 

    # 드래그
    pyautogui.moveTo(start_drag_x, start_drag_y, duration=0.1)
    time.sleep(0.1)
    pyautogui.mouseDown()
    pyautogui.moveTo(end_drag_x, end_drag_y, duration=0.6)
    time.sleep(0.2)
    pyautogui.mouseUp()

    # 조합 제거
    for i in range(start[0], end[0] + 1):
        for j in range(start[1], end[1] + 1):
            appletables[i][j].num = 0

    print_table(appletables)

def print_table(appletable):
    print()
    for row in appletable:
        for apple in row:
            if apple.num == 0:
                print('0', end=' ')
            else:
                print(f'{apple.num}', end=' ')
        print()

class Apple:
    def __init__(self, x, y, num):
        self.x = x
        self.y = y
        self.num = num

# 화면 캡처 및 숫자 인식
with mss.mss() as sct:
    monitor_number = 0
    mon = sct.monitors[monitor_number]
    capture_area = {'left': mon['left'], 'top': mon['top'], 'width': mon['width'], 'height': mon['height'], 'mon': monitor_number}
    sct_img = sct.grab(capture_area)
    img = Image.frombytes('RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX')
    img_np = np.array(img)
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

number_locations = {}
total_count = 0
for number in range(1, 10):
    template_filename = f'{number}.png'
    template = cv2.imread(template_filename, cv2.IMREAD_COLOR)
    if template is None:
        print(f"Error: Could not read template image {template_filename}")
        continue

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template.shape[:2]

    res = cv2.matchTemplate(gray_img, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.965
    loc = np.where(res >= threshold)

    locations = []
    for pt in zip(*loc[::-1]):
        center_x = pt[0] + w // 2
        center_y = pt[1] + h // 2
        locations.append((center_x, center_y))

    merged_locations = merge_nearby_locations(locations, threshold=3)

    number_locations[number] = merged_locations
    count = len(merged_locations)
    total_count += count

    print(f"Number {number} locations: {merged_locations}")
    print(f"Number {number} count: {count}")

print(f"Total count: {total_count}")

applelist = []
for number, locations in number_locations.items():
    applelist.extend(locations)

applelist2 = sorted(applelist, key=lambda coord: coord[1])

applelist3 = []
for i in range(0, len(applelist2), 17):
    chunk = applelist2[i:i+17]
    chunk_sorted = sorted(chunk, key=lambda coord: coord[0])
    applelist3.extend(chunk_sorted)

applelist3_objects = []
for coord in applelist3:
    found_number = 0
    for number, locations in number_locations.items():
        if coord in locations:
            found_number = number
            break
    apple = Apple(coord[0], coord[1], found_number)
    applelist3_objects.append(apple)

appletable = [[Apple(0, 0, 0) for col in range(17)] for row in range(10)]

index = 0
for c in range(10):
    for b in range(17):
        if index < len(applelist3_objects):
            appletable[c][b] = applelist3_objects[index]
            index += 1

print_table(appletable)

drag_and_remove_combinations(appletable)
