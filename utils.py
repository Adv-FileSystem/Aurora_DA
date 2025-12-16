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
