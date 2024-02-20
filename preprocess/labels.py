import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 경계 시간 계산
def calculate_boundaries_time(file_path, sample_rate, hop_length):
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

def process_and_chunk_mel_and_boundaries(mel, boundaries, sample_rate, frame_stride, n_seconds):
    """
    Process and chunk Mel spectrogram and boundaries to fit multiple segments of specified duration (n_seconds).

    If the Mel spectrogram is shorter than the specified duration, it will be repeated
    until it fits the duration. If it's longer, the Mel spectrogram and its associated boundaries
    are divided into chunks, each fitting the specified duration.

    :param mel: 2D NumPy array representing the Mel spectrogram.
    :param boundaries: List of tuples representing start, end, and label of each boundary.
    :param sample_rate: Sampling rate of the audio.
    :param frame_stride: Stride between successive frames in seconds.
    :param n_seconds: Desired duration of each Mel spectrogram segment in seconds.
    :return: List of tuples, each containing a chunk of the Mel spectrogram and its boundaries.
    """

    # Calculate hop_length and target_length in frames
    hop_length = sample_rate * frame_stride
    target_length = int(n_seconds * sample_rate / hop_length)
    original_length = mel.shape[1]
    
    mel_list = []
    boundaries_list = []

    if original_length < target_length:
        # If the spectrogram is too short, repeat it until it reaches the target length
        repeat_times = int(np.ceil(target_length / original_length))
        mel = np.tile(mel, (1, repeat_times,1))[:, :target_length]
        original_length = mel.shape[1]  # Update original_length after repeating
        mel_list.append(mel)

        # Repeat boundaries accordingly
        repeated_boundaries = []
        for i in range(repeat_times):
            for start, end, label in boundaries:
                new_start = start + i * original_length
                new_end = end + i * original_length
                
                # Check if the boundary is within the current segment
                segment_start = i * original_length
                segment_end = min((i + 1) * original_length, target_length)
                if new_start >= segment_start and new_end <= segment_end:
                    # Add the adjusted boundary to the list
                    repeated_boundaries.append((new_start, new_end, label))
        boundaries_list.append(repeated_boundaries)
    
    # Calculate the number of chunks needed
    num_chunks = int(np.floor(original_length / target_length))

    for i in range(num_chunks):
        # Calculate start and end indices for the current chunk
        start_idx = i * target_length
        end_idx = min((i + 1) * target_length, original_length)
        
        # Slice the mel spectrogram for the current chunk
        mel_chunk = mel[:, start_idx:end_idx]
        mel_list.append(mel_chunk)

        # Process boundaries for the current chunk
        chunk_boundaries = []
        for start, end, label in boundaries:
            # Adjust boundaries to the current chunk's timeframe
            chunk_start = max(start - start_idx, 0)
            chunk_end = min(end - start_idx, end_idx - start_idx)

            # Add the boundary to the chunk's list if it's within the current chunk
            if chunk_start < chunk_end:
                chunk_boundaries.append((chunk_start, chunk_end, label))
        boundaries_list.append(chunk_boundaries)

    return mel_list, boundaries_list

# 심음별 시작/종료 레이블 정보를 바탕으로 강도를 지정하여 세그먼트 및 이미지 저장
def get_segmentation_labels(mel_list, boundary_list, filter_value, hop_length):
    """
    Draw a square Mel spectrogram within specific boundaries of a label.
    """

    label_list = []
    
    for mel, boundaries in zip(mel_list, boundary_list):
        background = 0
        S1 = 127
        S2 = 255
        mel = np.where(mel > filter_value, mel, background) # filtering
        mel[:, :, 1] = 0
        mel = np.squeeze(mel[:, :, 0:1])
        #mel = np.dot(mel[..., :3], [0.299, 0.587, 0.114]) # RGB to grayscale

        # Process each column based on label boundaries
        for timepoint in range(mel.shape[1]):
            label_found = False
            for start, end, boundary_label in boundaries:
                if start <= timepoint < end:
                    label_found = True
                    if boundary_label == 1: # S1
                        mel[:, timepoint] = np.where(mel[:, timepoint] > background, S1, background).astype(int)
                    elif boundary_label == 3: # S2
                        mel[:, timepoint] = np.where(mel[:, timepoint] > background, S2, background).astype(int)
                    break  # Break the loop once the matching label is found and applied

            if not label_found:
                mel[:, timepoint] = background
        
        label_list.append(mel)

    return label_list

# segmentation mask에서 빈 영역을 채우기
def fill_blank_regions(mel_list):
    filled_mel_list = []

    for mel in mel_list:
        height, width = mel.shape
        # 각 열을 순회하며 특정 색상의 픽셀 아래 부분을 같은 색으로 채우기
        for x in range(width):
            for y in range(height):
                pixel_color = mel[y, x]
                if pixel_color == 255 or pixel_color == 127:  # 흰색(255) 또는 회색(127) 픽셀 발견
                    mel[y:, x] = pixel_color  # 해당 위치부터 아래까지 같은 색으로 채움
                    break  # 해당 열에서 색상을 찾으면 나머지 행은 무시
        filled_mel_list.append(mel)

    return filled_mel_list