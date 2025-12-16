import numpy as np

def add_noise(x, snr_db_range=(10, 30)):
    """
    신호에 가우시안 노이즈를 추가.
    snr_db_range: (min_dB, max_dB) - SNR 범위 (클수록 노이즈 작음)
    """
    x = np.asarray(x)
    power = np.mean(x ** 2) + 1e-8  # 0 division 방지
    snr_db = np.random.uniform(*snr_db_range)
    snr = 10 ** (snr_db / 10.0)
    noise_power = power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), size=x.shape)
    return x + noise


def scale_amplitude(x, scale_range=(0.8, 1.2)):
    """
    전체 진폭을 랜덤 배율로 스케일링.
    """
    x = np.asarray(x)
    scale = np.random.uniform(*scale_range)
    return x * scale


def time_shift(x, max_shift_ratio=0.1):
    """
    신호를 좌우로 회전(roll) 시켜 시간 축 이동.
    max_shift_ratio: 신호 길이에 대한 최대 이동 비율 (예: 0.1 -> 최대 10% 길이 이동)
    """
    x = np.asarray(x)
    n = len(x)
    max_shift = int(n * max_shift_ratio)
    if max_shift < 1:
        return x.copy()
    shift = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, shift)
