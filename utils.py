import numpy as np

ArrayLike = Union[np.ndarray, Iterable[float]]


def _to_1d_array(x: ArrayLike) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x
    

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


def time_stretch(x, stretch_range=(0.8, 1.2)):
    """
    신호를 시간축에서 늘이거나 줄이기.
    출력 길이는 원래 길이와 동일하게 리샘플링.
    stretch < 1: 압축 (빨라짐)
    stretch > 1: 스트레치 (느려짐)
    """
    x = np.asarray(x)
    n = len(x)
    factor = np.random.uniform(*stretch_range)

    # 새로운 길이 결정
    new_len = max(2, int(n * factor))  # 최소 2 포인트
    t_orig = np.arange(n)
    t_new = np.linspace(0, n - 1, new_len)

    # 1D 선형 보간
    x_stretched = np.interp(t_new, t_orig, x)

    # 다시 원래 길이로 맞춰주기 (crop or pad)
    if new_len > n:
        return x_stretched[:n]
    elif new_len < n:
        pad_len = n - new_len
        # 끝값을 반복해서 패딩
        return np.pad(x_stretched, (0, pad_len), mode="edge")
    else:
        return x_stretched


def permute_segments(x, n_segments=4):
    """
    신호를 n_segments 개의 구간으로 나누고 순서를 랜덤하게 섞음.
    길이가 딱 나누어떨어지지 않아도 대략적으로 분할.
    """
    x = np.asarray(x)
    n = len(x)

    if n_segments <= 1 or n < n_segments:
        return x.copy()

    # 균등하게 segment 인덱스 구분
    indices = np.linspace(0, n, n_segments + 1, dtype=int)
    segments = [x[indices[i]:indices[i + 1]] for i in range(n_segments)]

    perm = np.random.permutation(n_segments)
    permuted = [segments[i] for i in perm]
    return np.concatenate(permuted)


class SignalAugmentor:
    """
    여러 증강 기법을 확률적으로 적용해서 새로운 신호를 만들어주는 클래스.
    """

    def __init__(
        self,
        p_noise=0.7,
        p_scale=0.5,
        p_shift=0.5,
        p_stretch=0.5,
        p_permute=0.3,
        noise_snr_db_range=(10, 30),
        scale_range=(0.8, 1.2),
        max_shift_ratio=0.1,
        stretch_range=(0.9, 1.1),
        n_segments=4,
    ):
        self.p_noise = p_noise
        self.p_scale = p_scale
        self.p_shift = p_shift
        self.p_stretch = p_stretch
        self.p_permute = p_permute

        self.noise_snr_db_range = noise_snr_db_range
        self.scale_range = scale_range
        self.max_shift_ratio = max_shift_ratio
        self.stretch_range = stretch_range
        self.n_segments = n_segments

    def augment_once(self, x):
        """
        신호 x (1D array)를 한 번 증강해서 반환.
        """
        x_aug = np.asarray(x).copy()

        if np.random.rand() < self.p_noise:
            x_aug = add_noise(x_aug, self.noise_snr_db_range)

        if np.random.rand() < self.p_scale:
            x_aug = scale_amplitude(x_aug, self.scale_range)

        if np.random.rand() < self.p_shift:
            x_aug = time_shift(x_aug, self.max_shift_ratio)

        if np.random.rand() < self.p_stretch:
            x_aug = time_stretch(x_aug, self.stretch_range)

        if np.random.rand() < self.p_permute:
            x_aug = permute_segments(x_aug, self.n_segments)

        return x_aug

    def augment_batch(self, x, n_aug=5):
        """
        한 개의 원본 신호 x에 대해 n_aug개의 증강 샘플 리스트 반환.
        """
        return [self.augment_once(x) for _ in range(n_aug)]
