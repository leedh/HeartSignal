import cv2
import numpy as np

# Blank Region Clipping (BRC) from RespireNet
def blank_region_clipping(mel_list, n_mels):
    mel_list_prep = []
    for i in range(len(mel_list)):
        mel_color = cv2.cvtColor(mel_list[i], cv2.COLOR_BGR2RGB) # BRG to RGB
        mel_gray = cv2.cvtColor(mel_list[i], cv2.COLOR_BGR2GRAY) # BRG to Gray

        mel_gray[mel_gray < 10] = 0
        for row in range(mel_gray.shape[0]):
            # 위쪽 row부터 블랙 픽셀(0) 비율 계산. 80% 이상이면 break
            black_percent = len(np.where(mel_gray[row,:]==0)[0])/len(mel_gray[row,:])
            if black_percent < 0.80:
                break

        if (row+1) < mel_color.shape[0]:
            mel_color = mel_color[(row+1):, :, :]
        mel_color = cv2.resize(mel_color, (mel_color.shape[1], n_mels), interpolation=cv2.INTER_LINEAR)
        
        mel_list_prep.append(mel_color)
    return mel_list_prep




