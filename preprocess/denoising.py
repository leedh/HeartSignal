import pywt
import numpy as np

# Denoise audio using wavelet transform
def denoise_audio(audio, wavelet='db1', level=1):
    # 다중 레벨 웨이블릿 변환
    coeffs = pywt.wavedec(audio, wavelet, mode='per', level=level)
    # 임계값을 사용한 노이즈 제거
    threshold = (np.median(np.abs(coeffs[-level])) / 0.6745) * np.sqrt(2 * np.log(len(audio)))
    new_coeffs = list(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
    # 웨이블릿 변환 역변환
    denoised_audio = pywt.waverec(new_coeffs, wavelet, mode='per')
    return denoised_audio

# 결과 플롯
# plt.figure(figsize=(12, 8))
# plt.subplot(2, 1, 1)
# plt.title("Original Audio")
# plt.plot(audio)
# plt.subplot(2, 1, 2)
# plt.title("Denoised Audio")
# plt.plot(denoised_audio[:len(audio)])  # 재구성된 신호의 길이 조정
# plt.tight_layout()
# plt.show()
