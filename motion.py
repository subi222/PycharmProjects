import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.signal import spectrogram
import os
import glob
import re
import pandas as pd



ROOT = "/home/subi/PycharmProjects/Cohface"

def get_session0_avis(root=ROOT):
    paths = glob.glob(os.path.join(root, "cohface*", "*", "0", "data.avi"))
    # 숫자 client-id 정렬 (1,2,3,...,40)
    def client_id_key(p):
        # .../cohfaceX/<client-id>/0/data.avi
        m = re.search(r"/cohface\d+/(\d+)/0/data\.avi$", p)
        return int(m.group(1)) if m else 10**9
    return sorted(paths, key=client_id_key)


def moving_average_1s(signal: np.ndarray, fps: float) -> np.ndarray:
    """1초 창 크기의 이동평균을 적용."""
    if len(signal) == 0:
        return signal
    win = max(1, int(round(fps)))   #이동평균 창의 크기
    # 간단한 누적합 기반 이동평균
    cumsum = np.cumsum(np.insert(signal, 0, 0.0))
    ma = (cumsum[win:] - cumsum[:-win]) / float(win)  #슬라이딩 윈도우 평균 계산식
    # 길이 맞추기 위해 앞쪽은 가장 첫 평균값으로 패딩
    pad = np.full(win - 1, ma[0] if len(ma) > 0 else signal[0])
    return np.concatenate([pad, ma])

def diff_consecutive(signal: np.ndarray) -> np.ndarray:
    """연속된 값의 차 (속도 신호). 길이 맞추기 위해 앞쪽에 0 패딩."""
    if len(signal) == 0:
        return signal
    d = np.diff(signal, prepend=signal[0])
    return d

def landmarks_to_bbox(landmarks, w, h):
    """랜드마크 전체의 바운딩 박스 (픽셀 좌표) 반환."""
    xs = [int(l.x * w) for l in landmarks]  #랜드마크들의 x좌표 리스트
    ys = [int(l.y * h) for l in landmarks]
    x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
    y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
    return x_min, y_min, x_max, y_max   #바운딩 박스 왼쪽 x좌표,바운딩 박스 위쪽 y좌표,바운딩 박스 오른쪽 x좌표,바운딩 박스 아래쪽 y좌표

def define_rois_from_bbox(bbox, w, h):
    """
    FaceMesh 전체 박스 기준으로 두 개의 직사각형 ROI를 정의. 비율:너비 0.30/0.22, 높이 0.15/0.20
    """
    x_min, y_min, x_max, y_max = bbox
    bw, bh = (x_max - x_min), (y_max - y_min)

    # Forehead ROI (top-center)
    fw = int(bw * 0.30)
    fh = int(bh * 0.15)
    fx = int(x_min + (bw - fw) / 2)
    fy = int(y_min + bh * 0.10)  # 상단에서 조금 아래
    forehead = (fx, fy, fx + fw, fy + fh)

    # Nose ROI (center)
    nw = int(bw * 0.22)
    nh = int(bh * 0.20)
    nx = int(x_min + (bw - nw) / 2)
    ny = int(y_min + bh * 0.40)  # 중앙 조금 위쪽
    nose = (nx, ny, nx + nw, ny + nh)

    # 화면 경계 클램프
    def clamp_roi(roi):
        x1, y1, x2, y2 = roi
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        return (x1, y1, max(x1 + 1, x2), max(y1 + 1, y2))

    return clamp_roi(forehead), clamp_roi(nose)

def mean_landmark_y_in_roi(landmarks, w, h, roi):
    """ROI 내부에 포함되는 랜드마크들의 Y 픽셀좌표 평균을 움직임 신호로 사용."""
    x1, y1, x2, y2 = roi
    ys = []
    for l in landmarks:
        px = int(l.x * w)
        py = int(l.y * h)
        if x1 <= px <= x2 and y1 <= py <= y2:
            ys.append(py)
    if len(ys) == 0:
        return None
    return float(np.mean(ys))

def interpolate_nans(arr: np.ndarray) -> np.ndarray:
    """결측(None/NaN) 프레임을 선형 보간."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    x = np.arange(n)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return np.zeros_like(arr)
    if mask.sum() == n:
        return arr
    arr_interp = np.copy(arr)
    arr_interp[~mask] = np.interp(x[~mask], x[mask], arr[mask])
    return arr_interp

def percentile_threshold(abs_signal: np.ndarray, p: float = 90.0) -> float:
    """
    절대값 신호의 상위 p%에 해당하는 동적 임계값 계산.
    예) p=90 -> 절대값 분포의 90퍼센타일 값.
    """
    if len(abs_signal) == 0:
        return 0.0
    return float(np.percentile(abs_signal, p))

def kurtosis_noise_suppress(vel: np.ndarray, fps: float, p: float = 90.0,
                            pad_sec: float = 0.10) -> tuple[np.ndarray, dict]:
    if len(vel) == 0:
        return vel, {'threshold': 0.0, 'noise_ratio': 0.0, 'mask': np.zeros(0, dtype=bool)}

    abs_vel = np.abs(vel)
    T = percentile_threshold(abs_vel, p=p)
    mask = abs_vel >= T  # 임계값 이상 = 노이즈로 간주할 구간

    # 주변 몇 프레임까지 같이 눌러주기(옵션): 너무 날카로운 스파이크만 남는 것 방지
    pad = max(0, int(round(pad_sec * fps)))
    if pad > 0 and mask.any():
        # 1D dilation 유사: 이동창 합으로 확장
        kernel = np.ones(2 * pad + 1, dtype=int)
        # valid 영역 유지 위해 패딩
        padded = np.pad(mask.astype(int), (pad, pad), mode='edge')
        expanded = np.convolve(padded, kernel, mode='valid') > 0
        mask = expanded

    clean = vel.copy()
    clean[mask] = 0.0  # 요구사항: 잡음 구간 0으로 처리

    info = {
        'threshold': float(T),
        'noise_ratio': float(mask.mean()),  # 전체 프레임 중 눌린 비율
        'mask': mask
    }
    return clean, info

# --- 3단계: Savitzky-Golay 필터 ---
def savgol_2s(signal: np.ndarray, fps: float, polyorder: int = 2) -> np.ndarray:
    from scipy.signal import savgol_filter

    if len(signal) == 0:
        return signal

    win = max(polyorder + 3, int(round(2.0 * fps)))  # 최소한 polyorder+2보다 크게
    if win % 2 == 0:
        win += 1
    if win > len(signal):
        # 신호가 짧으면 가능한 홀수 창으로 줄임
        win = len(signal) if len(signal) % 2 == 1 else len(signal) - 1
        win = max(polyorder + 3, win)

    return savgol_filter(signal, window_length=win, polyorder=polyorder, mode='interp')

# --- 4단계: Welch PSD 기반 SNR 계산 ---
def psd_snr(signal: np.ndarray,
            fs: float,
            band: tuple = (0.1, 0.5),     # 호흡대역(Hz) ≈ 6~42 BPM
            nperseg_sec: float = 8.0,     # Welch 창 길이(초)
            peak_bw: float = 0.05         # 피크 주변 대역폭(±Hz) → 신호파워 집계
            ) -> dict:
    """
    Welch PSD로 호흡대역 내 최대 피크를 찾고,
    - Psig: 피크 주변(±peak_bw) 평균 전력
    - Pnoise: band 내(피크 주변 제외) 전력의 중앙값
    - SNR = Psig / Pnoise  (선형 스케일)
    반환: {'snr': float, 'peak_freq': float, 'f': f, 'psd': Pxx, 'band': band}
    """
    from scipy.signal import welch

    if len(signal) == 0:
        return {'snr': 0.0, 'peak_freq': np.nan, 'f': np.array([]), 'psd': np.array([]), 'band': band}

    # Welch 파라미터
    nperseg = max(16, int(round(nperseg_sec * fs)))
    nperseg = min(nperseg, len(signal))
    if nperseg < 8:
        nperseg = len(signal)

    f, Pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=nperseg // 2, detrend='constant')

    # 호흡 대역 선택
    bmask = (f >= band[0]) & (f <= band[1])
    if bmask.sum() < 3:
        return {'snr': 0.0, 'peak_freq': np.nan, 'f': f, 'psd': Pxx, 'band': band}

    fb, Pb = f[bmask], Pxx[bmask]

    # 호흡대역 피크 주파수
    idx = int(np.argmax(Pb))
    f0 = float(fb[idx])

    # 신호 파워: 피크 주변 ±peak_bw
    sigmask = (f >= max(band[0], f0 - peak_bw)) & (f <= min(band[1], f0 + peak_bw))
    Psig = float(Pxx[sigmask].mean()) if sigmask.any() else float(Pb.max())

    # 노이즈 플로어: band 내(피크 주변 제외) 중앙값
    noise_band_mask = bmask & (~sigmask)
    if noise_band_mask.sum() >= 3:
        Pnoise = float(np.median(Pxx[noise_band_mask]))
    else:
        Pnoise = float(np.median(Pb))

    snr = float(Psig / (Pnoise + 1e-12))
    return {'snr': snr, 'peak_freq': f0, 'f': f, 'psd': Pxx, 'band': band}


# --- 4단계: SNR 가중 통합 ---
def combine_by_snr(sig_a: np.ndarray, sig_b: np.ndarray, fs: float,
                   band: tuple = (0.1, 0.7)) -> tuple[np.ndarray, dict]:
    """
    두 신호를 그대로 사용하고, Welch-PSD 기반 SNR을 가중치로 사용해 통합.
    - 가중치 = SNR / (SNR_a + SNR_b). (합이 0이면 0.5/0.5)
    반환: (combined, info)
      info = {'w_a':..., 'w_b':..., 'snr_a':..., 'snr_b':..., 'peak_a':..., 'peak_b':...}
    """
    # 신호 그대로 사용
    a = np.asarray(sig_a, dtype=float)
    b = np.asarray(sig_b, dtype=float)

    # 각 신호의 SNR 계산
    info_a = psd_snr(a, fs=fs, band=band)
    info_b = psd_snr(b, fs=fs, band=band)

    snr_a, snr_b = info_a['snr'], info_b['snr']
    s = snr_a + snr_b
    if s <= 0:
        w_a = w_b = 0.5
    else:
        w_a = float(snr_a / s)
        w_b = float(snr_b / s)

    # 가중합
    combined = w_a * a + w_b * b
    info = {
        'w_a': w_a, 'w_b': w_b,
        'snr_a': snr_a, 'snr_b': snr_b,
        'peak_a': info_a['peak_freq'], 'peak_b': info_b['peak_freq'],
        'band': band
    }
    return combined, info

def process_video(video_path: str, draw_preview: bool = False, save_csv: bool = False, out_dir: str = "./results/motion_signals"):
    mp_face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0  # 기본값
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    forehead_Y = []
    nose_Y = []

    # Face Mesh 설정 (정적 이미지가 아니므로 refine_landmarks=False로 충분)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                bbox = landmarks_to_bbox(lm, width, height)
                roi_forehead, roi_nose = define_rois_from_bbox(bbox, width, height)

                fy = mean_landmark_y_in_roi(lm, width, height, roi_forehead)
                ny = mean_landmark_y_in_roi(lm, width, height, roi_nose)
            else:
                fy, ny = None, None

            forehead_Y.append(np.nan if fy is None else fy)
            nose_Y.append(np.nan if ny is None else ny)

            if draw_preview:
                # 디버그 미리보기(원하면 True로)
                vis = frame.copy()
                if res.multi_face_landmarks:
                    x1, y1, x2, y2 = roi_forehead
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    x1, y1, x2, y2 = roi_nose
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.imshow("preview (q to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    cap.release()
    if draw_preview:
        cv2.destroyAllWindows()

    # 결측 프레임 보간
    forehead_Y = interpolate_nans(np.array(forehead_Y, dtype=float))
    nose_Y = interpolate_nans(np.array(nose_Y, dtype=float))

    # 1초 이동평균
    forehead_Y_smooth = moving_average_1s(forehead_Y, fps)
    nose_Y_smooth = moving_average_1s(nose_Y, fps)

    # 속도 신호(차분)
    forehead_vel = diff_consecutive(forehead_Y_smooth)
    nose_vel = diff_consecutive(nose_Y_smooth)

     # [2단계] 첨도-아이디어 기반(상위 10% 퍼센타일) 잡음 제거
    forehead_vel_clean, forehead_denoise = kurtosis_noise_suppress(
        forehead_vel, fps=fps, p=90.0, pad_sec=0.10
    )
    nose_vel_clean, nose_denoise = kurtosis_noise_suppress(
        nose_vel, fps=fps, p=90.0, pad_sec=0.10
    )

    # === [3단계] Savitzky–Golay(차수2, 2초 창)로 한 번 더 스무딩 ===
    forehead_vel_filt = savgol_2s(forehead_vel_clean, fps=fps, polyorder=2)
    nose_vel_filt     = savgol_2s(nose_vel_clean,     fps=fps, polyorder=2)

    # === [4단계] Welch PSD SNR 기반 가중 평균으로 통합 ===
    # 호흡 대역(Hz): 실험에 맞게 필요시 (0.08~0.8) 등으로 조정 가능
    resp_band = (0.1, 0.7)

    motion_resp_signal, snr_info = combine_by_snr(
        forehead_vel_filt, nose_vel_filt, fs=fps, band=resp_band
    )

    out = {
        "fps": fps,
        "forehead_Y": forehead_Y,
        "nose_Y": nose_Y,
        "forehead_Y_smooth": forehead_Y_smooth,
        "nose_Y_smooth": nose_Y_smooth,
        "forehead_vel": forehead_vel,
        "nose_vel": nose_vel,
        # 2단계 결과
        "forehead_vel_clean": forehead_vel_clean,
        "nose_vel_clean": nose_vel_clean,
        "forehead_denoise_info": forehead_denoise,  # threshold, ratio, mask 포함
        "nose_denoise_info": nose_denoise,
        # 3단계
        "forehead_vel_filt": forehead_vel_filt,
        "nose_vel_filt": nose_vel_filt,
        # 4단계: 최종 움직임 기반 호흡 신호 + SNR 가중 정보
        "motion_resp_signal": motion_resp_signal,
        "snr_info": snr_info,
    }

    if save_csv:
        os.makedirs(out_dir, exist_ok=True)
        sid = _extract_subject_id(video_path)
        t = np.arange(len(motion_resp_signal)) / fps
        pd.DataFrame({"t_sec": t, "motion_resp_signal": motion_resp_signal}) \
            .to_csv(os.path.join(out_dir, f"{sid}.csv"), index=False)
        print(f"✅ Saved motion signal: {os.path.join(out_dir, f'{sid}.csv')}")

    return out

def _extract_subject_id(video_path: str) -> str:
    m = re.search(r"/cohface\d+/(\d+)/0/data\.avi$", video_path)
    return m.group(1) if m else os.path.splitext(os.path.basename(video_path))[0]

def process_multiple_videos(video_paths, draw_preview=False,
                            save_csv=False, out_dir="./results/motion_signals"):
    results = {}
    for idx, path in enumerate(video_paths, 1):
        print(f"[{idx}/{len(video_paths)}] Processing: {path}")
        try:
            results[path] = process_video(path,
                                          draw_preview=draw_preview,
                                          save_csv=save_csv,
                                          out_dir=out_dir)
        except Exception as e:
            print(f" {path} 처리 실패: {e}")
    return results


if __name__ == "__main__":
    video_paths = get_session0_avis(ROOT)
    all_results = process_multiple_videos(video_paths, draw_preview=False,save_csv=True, out_dir="./results/motion_signals")
    print("총 처리 완료:", len(all_results), "개")