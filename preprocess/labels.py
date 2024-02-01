import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as patches
from shutil import copy2
from PIL import Image
from collections import Counter

# Mel 범위로 변환
def get_mel_spectrogram(wave_form, sample_rate, n_mels, frame_length, frame_stride):
    """
    Convert an audio waveform to a Mel spectrogram.

    :param wave_form: NumPy array of audio waveform.
    :param sample_rate: Sampling rate of the audio.
    :param n_mels: Number of Mel bands to generate.
    :param frame_length: Length of each frame in seconds.
    :param frame_stride: Stride between successive frames in seconds.
    :return: Mel spectrogram (2D NumPy array).
    """
    # Calculate the number of samples per frame and stride
    win_length = int(round(sample_rate * frame_length))
    hop_length = int(round(sample_rate * frame_stride))

    # Calculate the number of FFT components (n_fft) as the next power of two from win_length
    n_fft = 2 ** int(np.ceil(np.log2(win_length)))

    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=wave_form, sr=sample_rate, n_fft=n_fft,
                                                     hop_length=hop_length, win_length=win_length,
                                                     n_mels=n_mels)
    return mel_spectrogram

# 경계 시간 계산
def calculate_boundaries_time(file_path, sample_rate, frame_stride):
    """
    Calculate boundary times for audio events from a TSV file and convert them to sample indices.

    :param file_path: Path to the TSV file containing start and end times with labels.
    :param sample_rate: Sampling rate of the audio.
    :param frame_stride: Stride between successive frames in seconds.
    :return: A list of tuples with start and end indices and labels.
    """

    boundaries = []
    with open(file_path, 'r') as file:
        tsv_reader = pd.read_csv(file, sep='\t', header=None)

        # Read each line from the TSV file
        for n_row in range(len(tsv_reader)):
            row = tsv_reader.loc[n_row]

            # Select rows where label is '1'() or '3'()
            if row[2] == 1 or row[2] == 3:
                start_time, end_time, label = float(row[0]), float(row[1]), int(row[2])

                # Append the start and end times with labels to the list
                boundaries.append((start_time, end_time, label))

    # Convert boundary times to sample indices
    hop_length = int(round(sample_rate * frame_stride))
    boundaries_in_samples = [
        (int(round(start * sample_rate / hop_length)),
         int(round(end * sample_rate / hop_length)), label)
        for start, end, label in boundaries
    ]

    return boundaries_in_samples

# 심음 경계 정보의 시작 및 종료 시점 추출
def calculate_cropped_indices(boundaries_time):
    """
    Calculate the average length of the heart sound events and use it to determine
    the start and end indices for cropping the Mel spectrogram.

    :param boundaries_time: List of tuples containing start and end indices with labels.
    :return: Tuple of integers representing the start and end indices for cropping.
    """

    # Calculate the average length of the heart sound events
    avg_length = np.mean([end - start for start, end, _ in boundaries_time]) / 2

    # Determine the start and end indices for cropping
    cropped_start = int(boundaries_time[0][0] - avg_length)
    cropped_end = int(boundaries_time[-1][1] + avg_length)

    # Ensure that the start and end indices are not negative
    cropped_start = max(0, cropped_start)
    cropped_end = max(0, cropped_end)

    return cropped_start, cropped_end

# 멜 스펙트로그램의 시작 및 종료 시점 기준 크롭
def get_cropped_mel(mel, cropped_start, cropped_end):
    """
    Crop the Mel spectrogram using the specified start and end indices.

    :param mel: 2D NumPy array of Mel spectrogram.
    :param cropped_start: Start index for cropping.
    :param cropped_end: End index for cropping.
    :return: Cropped Mel spectrogram.
    """

    # Ensure the cropping indices are within the bounds of the Mel spectrogram
    cropped_start = max(0, cropped_start)
    cropped_end = min(mel.shape[1], cropped_end)

    # Crop and return the Mel spectrogram
    return mel[:, cropped_start:cropped_end]

# Adjust the boundaries according to the cropping
def adjust_boundaries(boundaries_in_samples, cropped_start):
    """
    Adjust the start and end indices of each heart sound event based on the cropping start index.

    :param boundaries_in_samples: List of tuples with original start and end indices and labels.
    :param cropped_start: The start index used for cropping the Mel spectrogram.
    :return: List of tuples with adjusted start and end indices and labels.
    """

    adjusted_boundaries = []
    for start, end, label in boundaries_in_samples:
        # Adjust the start and end indices based on the cropped start index
        new_start = max(0, start - cropped_start)
        new_end = max(0, end - cropped_start)

        # Add the boundary to the list if it's valid (start index is before the end index)
        if new_start < new_end:
            adjusted_boundaries.append((new_start, new_end, label))

    return adjusted_boundaries

# 지정된 초(n_seconds) 이내로 범위 조정
def process_mel_and_boundaries(mel, boundaries, sample_rate, frame_stride, n_seconds):
    """
    Process Mel spectrogram and boundaries to fit a specified duration (n_seconds).

    This function adjusts the length of the Mel spectrogram and its associated boundaries.
    If the Mel spectrogram is shorter than the specified duration, it will be repeated
    until it fits. The boundaries will be recalculated to align with the modified Mel spectrogram.

    :param mel: 2D NumPy array representing the Mel spectrogram.
    :param boundaries: List of tuples representing start, end, and label of each boundary.
    :param sample_rate: Sampling rate of the audio.
    :param frame_stride: Stride between successive frames in seconds.
    :param n_seconds: Desired duration of the Mel spectrogram in seconds.
    :return: Tuple of adjusted Mel spectrogram and its boundaries.
    """

    # Calculate hop_length and target_length in samples
    hop_length = sample_rate * frame_stride
    target_length = int(n_seconds * sample_rate / hop_length)

    # Check if the original Mel spectrogram needs to be repeated
    original_length = mel.shape[1]
    if original_length >= target_length:
        # If the spectrogram is already long enough, crop it to the target length
        mel = mel[:, :target_length]
        repeat_times = 1
    else:
        # If the spectrogram is too short, repeat it until it reaches the target length
        repeat_times = int(np.ceil(target_length / original_length))
        mel = np.tile(mel, (1, repeat_times))[:, :target_length]

    # Process boundaries for the adjusted Mel spectrogram
    processed_boundaries = []
    for start, end, label in boundaries:
        # Iterate over the repeated segments
        for i in range(repeat_times):
            # Calculate new start and end points for the current segment
            new_start = start + i * original_length
            new_end = end + i * original_length

            # Check if the boundary is within the current segment
            segment_start = i * original_length
            segment_end = min((i + 1) * original_length, target_length)
            if new_start >= segment_start and new_end <= segment_end:
                # Add the adjusted boundary to the list
                processed_boundaries.append((new_start, new_end, label))

    # Return the adjusted Mel spectrogram and its boundaries
    return mel, processed_boundaries

# 특정 초 이내로 멜 크롭 (시작 시간(0)으로부터 seconds 까지)
def crop_mel_by_sec(mel, sample_rate, frame_stride, seconds):
    """
    Crop the Mel spectrogram to a specified length in seconds.

    :param mel: Mel spectrogram (2D NumPy array).
    :param sample_rate: Sampling rate of the original audio.
    :param frame_stride: Stride between successive frames in seconds.
    :param seconds: Desired length of the Mel spectrogram in seconds.
    :return: Cropped Mel spectrogram.
    """
    # Calculate hop length to align time axis
    hop_length = int(round(sample_rate * frame_stride))
    mel_duration = mel.shape[1] * hop_length / sample_rate

    # Return original Mel if specified seconds exceed its duration
    if seconds >= mel_duration:
        return mel

    # Calculate target length in samples and crop
    target_length = int(seconds * sample_rate / hop_length)
    return mel[:, :target_length]

# 정사각형 스펙트로그램 그리고 저장하기
def draw_mel_square_spec(mel, sample_rate, frame_stride):
    """
    Draw a square Mel spectrogram and save it to a file.

    :param mel: Mel spectrogram (2D NumPy array).
    :param sample_rate: Sampling rate of the original audio.
    :param frame_stride: Stride between successive frames in seconds.
    :param save_path: Path to save the spectrogram image.
    :param filename: Filename for saving the image.
    """
    # Convert Mel spectrogram to Decibel scale
    S_dB = librosa.power_to_db(mel, ref=np.max)

    # Calculate hop length for time alignment
    hop_length = int(round(sample_rate * frame_stride))
    
    return S_dB, hop_length


# 바운딩박스 레이블 그리고 저장하기
# def draw_bbox_label_and_save(mel, sample_rate, frame_stride, boundaries_time, save_path, filename, label_1_height=1.0, label_3_height=1.0):
#     """
#     Draw bounding boxes on Mel spectrogram based on given boundaries and save it.

#     :param mel: Mel spectrogram (2D NumPy array).
#     :param sample_rate: Sampling rate of the original audio.
#     :param frame_stride: Stride between successive frames in seconds.
#     :param boundaries_time: List of tuples with start and end indices and label.
#     :param save_path: Path to save the image.
#     :param filename: Filename for saving the image.
#     :param label_1_height: Height ratio of the bounding box for label 1.
#     :param label_3_height: Height ratio of the bounding box for label 3.
#     """
#     # Set S_dB to a constant minimum value for a black background
#     S_dB = np.full(mel.shape, -80)  # For example, -80 dB

#     # Calculate hop length for time alignment
#     hop_length = int(round(sample_rate * frame_stride))

#     # Create a square figure
#     fig, ax = plt.subplots(figsize=(3, 3), dpi=95)  # 256x256 pixels at 95 DPI

#     # Display the Mel spectrogram with a gray color map
#     librosa.display.specshow(S_dB, sr=sample_rate, hop_length=hop_length, cmap='gray')
#     ax.set_aspect('auto')

#     # Draw bounding boxes for each boundary
#     for start_idx, end_idx, label in boundaries_time:
#         color = 'blue' if label == 1 else 'red'
#         height_ratio = label_1_height if label == 1 else label_3_height
#         height = int(mel.shape[0] * height_ratio)
#         rect = patches.Rectangle((start_idx, 0), end_idx - start_idx, height, linewidth=1, edgecolor=color, facecolor=color)
#         ax.add_patch(rect)

#     # Save and display the bounding box image
#     plt.tight_layout()
#     full_save_path = os.path.join(save_path, filename + '_label')
#     plt.savefig(full_save_path, bbox_inches='tight', pad_inches=0)
#     plt.show()
#     print(f"{filename} bounding box segments label image saved\n")

# 강도를 지정하여 이미지 그리기
def draw_filtered_mel_square_spec_with_boundaries(mel, boundaries_in_samples, filter_limit, sample_rate, frame_stride, save_path, filename):
    """
    Draw a square Mel spectrogram within specific boundaries of a label.

    :param mel: Mel spectrogram (2D NumPy array).
    :param boundaries_in_samples: List of tuples with start and end indices and labels.
    :param sample_rate: Sampling rate of the original audio.
    :param frame_stride: Stride between successive frames in seconds.
    :param save_path: Directory path where the image will be saved.
    :param filename: Name of the file to save the image.
    """
    # Convert Mel spectrogram to Decibel scale
    S_dB = librosa.power_to_db(mel, ref=np.max)

    min_val = np.min(S_dB)
    S_dB = np.where(S_dB > filter_limit, S_dB, min_val)

    # Calculate hop length for time alignment
    hop_length = int(round(sample_rate * frame_stride))

    # Mask regions outside the label boundaries
    for i in range(S_dB.shape[1]):
        if not any(start <= i < end for start, end, boundary_label in boundaries_in_samples if boundary_label):
            S_dB[:, i] = min_val

    # Create a square figure
    fig, ax = plt.subplots(figsize=(3, 3), dpi=95)

    # Display the Mel spectrogram
    librosa.display.specshow(S_dB, sr=sample_rate, hop_length=hop_length)
    ax.set_aspect('auto')
    plt.tight_layout()
    plt.show()

# 심음별 시작/종료 레이블 정보를 바탕으로 강도를 지정하여 세그먼트 및 이미지 저장
def draw_mel_square_spec_with_boundaries_to_label_and_filter(mel, boundaries_in_samples, filter_limit, sample_rate, frame_stride):
    """
    Draw a square Mel spectrogram within specific boundaries of a label.

    :param mel: Mel spectrogram (2D NumPy array).
    :param boundaries_in_samples: List of tuples with start and end indices and labels.
    :param sample_rate: Sampling rate of the original audio.
    :param frame_stride: Stride between successive frames in seconds.
    :param save_path: Directory path where the image will be saved.
    :param filename: Name of the file to save the image.
    """
    # Convert Mel spectrogram to Decibel scale
    S_dB = librosa.power_to_db(mel, ref=np.max)

    min_val = -80
    max_val = np.max(S_dB)
    mid_val = (min_val + max_val) / 2
    S_dB = np.where(S_dB > filter_limit, S_dB, min_val)

    # Calculate hop length for time alignment
    hop_length = int(round(sample_rate * frame_stride))

    # Process each column based on label boundaries
    for i in range(S_dB.shape[1]):
        label_found = False
        for start, end, boundary_label in boundaries_in_samples:
            if start <= i < end:
                label_found = True
                if boundary_label == 1:
                    S_dB[:, i] = np.where(S_dB[:, i] > min_val, max_val, S_dB[:, i])
                elif boundary_label == 3:
                    S_dB[:, i] = np.where(S_dB[:, i] > min_val, mid_val, S_dB[:, i])
                break  # Break the loop once the matching label is found and applied

        if not label_found:
            S_dB[:, i] = min_val

    return S_dB