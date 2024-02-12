import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# 첫 번째 이미지를 불러옵니다
target = load_img('/hs/HeartSignal/data/v1/train/85345_AV_2_label.png', color_mode='grayscale')
target = img_to_array(target)
target = target.astype("uint8")

# 고유한 색상 값 추출
unique_colors = np.unique(target)
# 각 색상 값의 빈도수 계산
color_counts = np.bincount(target.flatten())
# 빈도수가 0이 아닌 색상 값과 해당 빈도수 추출
colors_with_counts = [(color, count) for color, count in enumerate(color_counts) if count > 0]

# 빈도수에 따라 정렬 후 상위 5개 선택
top_colors_with_counts = sorted(colors_with_counts, key=lambda x: x[1], reverse=True)[:5]

# 상위 5개 색상 값과 빈도수 분리
top_colors, top_counts = zip(*top_colors_with_counts)

print("Top 5 Colors and Their Frequencies:")
for color, count in zip(top_colors, top_counts):
    print(f"Color: {color}, Frequency: {count}")

# 빈도수 시각화
plt.figure(figsize=(12, 6))
plt.bar(top_colors, top_counts, color='gray')
plt.title("Frequency of Top 5 Color Values")
plt.xlabel("Color Value")
plt.ylabel("Frequency")
plt.xticks(top_colors, top_colors)

# plt.show() 대신에 파일로 저장
plt.savefig('/hs/HeartSignal/eda/top_color_frequency.png')