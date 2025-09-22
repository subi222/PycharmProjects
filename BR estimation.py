import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.signal import get_window, detrend
import os
from pathlib import Path

def read_time_value_csv(path: str) -> Tuple[np.ndarray, float]:
    """2열 [time(sec), value] CSV → (value 신호, fs) 반환"""
    df = pd.read_csv(path)
    t = df.iloc[:, 0].to_numpy(dtype=float)
    x = df.iloc[:, 1].to_numpy(dtype=float)
    if len(t) < 2:
        raise ValueError(f"Not enough samples in {path}")
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    return x, fs

def run_for_csv(input_csv: str, output_csv: str,
                win_sec: float = 30.0, hop_sec: float = 5.0,
                br_min_bpm: float = 8.0, br_max_bpm: float = 30.0) -> None:
    sig, fs = read_time_value_csv(input_csv)
    df_br = estimate_br_series(
        signal=sig, fs=fs,
        win_sec=win_sec, hop_sec=hop_sec,
        br_min_bpm=br_min_bpm, br_max_bpm=br_max_bpm
    )
    df_br.to_csv(output_csv, index=False)

def run_batch(input_dir: str, output_dir: str,
                  win_sec: float = 30.0, hop_sec: float = 5.0,
                  br_min_bpm: float = 8.0, br_max_bpm: float = 30.0,
                  file_range: range = range(1, 41)) -> None:
        """
        input_dir에 1.csv~40.csv가 있다고 가정하고 일괄 처리.
        결과는 output_dir/1_br.csv ... 로 저장.
        """
        in_dir = Path(input_dir)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in file_range:
            in_path = in_dir / f"{i}.csv"
            out_path = out_dir / f"{i}_br.csv"
            if not in_path.exists():
                print(f"[skip] {in_path} 없음")
                continue
            try:
                run_for_csv(
                    input_csv=str(in_path),
                    output_csv=str(out_path),
                    win_sec=win_sec, hop_sec=hop_sec,
                    br_min_bpm=br_min_bpm, br_max_bpm=br_max_bpm
                )
                print(f"[ok] {in_path.name} → {out_path.name}")
            except Exception as e:
                print(f"[fail] {in_path.name}: {e}")


def frobenius_normalize(x: np.ndarray) -> np.ndarray:
    """프로베니우스(L2) 노름으로 1D 신호 정규화."""
    x = np.asarray(x, dtype=float)
    norm = np.sqrt(np.sum(x ** 2))
    if norm == 0 or not np.isfinite(norm):
        return x.copy()
    return x / norm


def sliding_indices(n: int, win: int, hop: int) -> List[Tuple[int, int]]:
    """길이 n 신호를 창(win)·홉(hop)으로 자를 (s,e) 인덱스 리스트."""
    out: List[Tuple[int, int]] = []
    s = 0
    while s + win <= n:
        out.append((s, s + win))
        s += hop
    return out


def _next_pow2(n: int) -> int:
    """n 이상인 가장 가까운 2의 거듭제곱."""
    return 1 if n <= 1 else 1 << (int(np.ceil(np.log2(n))))


# ========= 윈도우 단위 BR 추정 =========
def psd_peak_bpm(
    window_sig: np.ndarray,
    fs: float,
    *,
    zero_pad_factor: int = 8,
    br_min_bpm: float = 8.0,
    br_max_bpm: float = 30.0,
    band_guard_hz: float = 0.03,   # 대역 경계 가드(≈1.8 BPM)
    min_snr_ratio: float = 3.0     # (피크/대역 중앙값) 최소 SNR
) -> Optional[float]:
    """
    한 윈도우에서 PSD 피크를 찾아 BPM으로 반환. 신뢰 낮으면 None.
    """
    x = np.asarray(window_sig, dtype=float)
    if (not np.isfinite(x).any()) or (np.nanstd(x) < 1e-9):
        return None

    # (1) DC/추세 제거
    x = detrend(x, type="linear")
    x = x - np.mean(x)

    # (2) 창 곱 & FFT
    w = get_window("hann", x.size, fftbins=True)
    xw = x * w
    n_fft = _next_pow2(len(xw)) * max(1, int(zero_pad_factor))
    X = np.fft.rfft(xw, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    psd = (np.abs(X) ** 2)
    df = freqs[1] - freqs[0]

    # (3) 호흡 대역 마스크
    lo_hz, hi_hz = br_min_bpm / 60.0, br_max_bpm / 60.0
    band_mask = (freqs >= lo_hz) & (freqs <= hi_hz)
    if not np.any(band_mask):
        return None
    band_idx = np.where(band_mask)[0]
    if band_idx.size < 7:  # 너무 빈약하면 스킵
        return None

    # (4) 대역 경계 가드(한두 칸 잘라 경계 오인 방지)
    guard = max(1, int(round(band_guard_hz / df)))
    band_idx = band_idx[guard:-guard] if band_idx.size > 2*guard else band_idx
    if band_idx.size == 0:
        return None

    # (5) SNR 검사(피크/대역 중앙값)
    P_band = psd[band_idx]
    Pmed = float(np.median(P_band) + 1e-12)
    if (P_band.max() / Pmed) < min_snr_ratio:
        return None

    # (6) 최강 피크 위치
    i_local = int(np.argmax(P_band))
    i_peak = band_idx[i_local]

    # (7) 로그-포물선 보간으로 서브빈 정밀화
    def _interp_log_parabola(P: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= P.size - 1:
            return 0.0
        y0, y1, y2 = np.log(P[idx - 1] + 1e-20), np.log(P[idx] + 1e-20), np.log(P[idx + 1] + 1e-20)
        denom = (y0 - 2*y1 + y2)
        return 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0

    delta_bins = _interp_log_parabola(psd, i_peak)
    f_peak = freqs[i_peak] + delta_bins * df

    # (8) 안정성 체크 & BPM 변환
    if not (lo_hz < f_peak < hi_hz):
        return None
    return float(f_peak * 60.0)

import numpy as np
import pandas as pd

def br_series_from_raw_csv(input_csv: str,
                           win_sec: float = 30.0, hop_sec: float = 5.0,
                           br_min_bpm: float = 8.0, br_max_bpm: float = 30.0,
                           time_mode: str = "center") -> pd.DataFrame:
    """
    원본 2열 CSV(시간,값) → BR 시계열 DataFrame 반환.
    반환 컬럼: t_sec, br_bpm
    time_mode: 'center'면 윈도우 중심시간, 'hop'이면 5,10,15,...로 강제.
    """
    sig, fs = read_time_value_csv(input_csv)
    df = estimate_br_series(signal=sig, fs=fs,
                            win_sec=win_sec, hop_sec=hop_sec,
                            br_min_bpm=br_min_bpm, br_max_bpm=br_max_bpm)
    t = df["t_center_sec"].to_numpy()

    if time_mode == "hop":
        # 윈도우 k(0-base)에 대해 시간 = (k+1)*hop_sec  → 5,10,15,…
        t = (np.arange(len(t)) + 1) * hop_sec

    out = pd.DataFrame({"t_sec": t, "br_bpm": df["br_bpm"].to_numpy()})
    return out

# (기존 함수 수정) GT를 읽어와 병합/플롯/CSV에 br_gt 추가
def plot_motion_vs_rppg_pair(index_i: int,
                             motion_dir: str, rppg_dir: str,
                             win_sec: float = 30.0, hop_sec: float = 5.0,
                             br_min_bpm: float = 8.0, br_max_bpm: float = 30.0,
                             use_hop_time: bool = True,
                             interpolate_to_motion_axis: bool = True,
                             save_png_dir: str | None = None,
                             save_merged_csv: str | None = None,
                             cohface_root: str | None = None) -> None:   # ← 인자 추가
    motion_csv = str(Path(motion_dir) / f"{index_i}.csv")
    rppg_csv   = str(Path(rppg_dir)   / f"{index_i}.csv")
    time_mode = "hop" if use_hop_time else "center"

    dm = br_series_from_raw_csv(motion_csv, win_sec, hop_sec, br_min_bpm, br_max_bpm, time_mode=time_mode).rename(columns={"br_bpm":"br_motion"})
    dr = br_series_from_raw_csv(rppg_csv,   win_sec, hop_sec, br_min_bpm, br_max_bpm, time_mode=time_mode).rename(columns={"br_bpm":"br_rppg"})

    dg = None
    if cohface_root is not None:
        try:
            dg = br_series_from_gt(index_i, cohface_root, win_sec, hop_sec, br_min_bpm, br_max_bpm, time_mode=time_mode)
        except Exception as e:
            print(f"[warn] GT load failed for {index_i}: {e}")

    if interpolate_to_motion_axis:
        base_t = dm["t_sec"]
        dr_i = dr.set_index("t_sec").reindex(base_t).interpolate(method="index")
        parts = [dm.set_index("t_sec"), dr_i]
        if dg is not None:
            dg_i = dg.set_index("t_sec").reindex(base_t).interpolate(method="index")
            parts.append(dg_i)
        df = pd.concat(parts, axis=1).reset_index()
    else:
        df = pd.merge(dm, dr, on="t_sec", how="outer")
        if dg is not None:
            df = pd.merge(df, dg, on="t_sec", how="outer")
        df = df.sort_values("t_sec", ignore_index=True)

    # ==== Plot ====
    plt.figure(figsize=(10,4))
    plt.plot(df["t_sec"], df["br_motion"], marker="o", linestyle="-", label="Motion BR (bpm)")
    plt.plot(df["t_sec"], df["br_rppg"],   marker="s", linestyle="-", label="rPPG BR (bpm)")
    if "br_gt" in df.columns:
        plt.plot(df["t_sec"], df["br_gt"], marker="^", linestyle="-", label="GT BR (bpm)")
    plt.xlabel("Time (sec)"); plt.ylabel("Breathing Rate (bpm)")
    plt.title(f"Motion vs rPPG vs GT — {index_i}")
    plt.grid(True, alpha=0.3); plt.legend()

    if save_png_dir:
        Path(save_png_dir).mkdir(parents=True, exist_ok=True)
        out_png = Path(save_png_dir) / f"{index_i}_compare.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[ok] saved plot: {out_png}")
    else:
        plt.show()

    if save_merged_csv:
        Path(save_merged_csv).mkdir(parents=True, exist_ok=True)
        out_csv = Path(save_merged_csv) / f"{index_i}_merged.csv"
        df.to_csv(out_csv, index=False)
        print(f"[ok] saved merged csv: {out_csv}")

# ========= 전체 시계열 BR =========
def estimate_br_series(
    signal: np.ndarray,
    fs: float,
    *,
    win_sec: float = 30.0,
    hop_sec: float = 5.0,
    zero_pad_factor: int = 8,
    br_min_bpm: float = 8.0,
    br_max_bpm: float = 30.0,
    band_guard_hz: float = 0.03,
    min_snr_ratio: float = 3.0,
) -> pd.DataFrame:
    """
    입력 신호에 대해 슬라이딩 윈도우 BR 시계열을 추정.
    반환: DataFrame(columns=["t_center_sec","br_bpm"])
    """
    x = frobenius_normalize(np.asarray(signal, dtype=float))

    win = int(round(win_sec * fs))
    hop = int(round(hop_sec * fs))
    idx_pairs = sliding_indices(len(x), win, hop)

    t_centers: List[float] = []
    brs: List[float] = []

    for s, e in idx_pairs:
        seg = x[s:e]  # 전체 한 번만 정규화(이중 정규화 제거)
        br = psd_peak_bpm(
            seg, fs,
            zero_pad_factor=zero_pad_factor,
            br_min_bpm=br_min_bpm,
            br_max_bpm=br_max_bpm,
            band_guard_hz=band_guard_hz,
            min_snr_ratio=min_snr_ratio,
        )
        t_center = (s + e) / 2.0 / fs
        t_centers.append(t_center)
        brs.append(np.nan if br is None else br)

    return pd.DataFrame({"t_center_sec": t_centers, "br_bpm": brs})

def run_compare_batch(motion_dir: str,
                      rppg_dir: str,
                      out_plot_dir: str,
                      out_merged_dir: str,
                      start_idx: int = 1,
                      end_idx: int = 40,
                      win_sec: float = 30.0,
                      hop_sec: float = 5.0,
                      br_min_bpm: float = 8.0,
                      br_max_bpm: float = 30.0,
                      use_hop_time: bool = True,
                      interpolate_to_motion_axis: bool = True,
                      cohface_root: str | None = None) -> None:         # ← 인자 추가
    Path(out_plot_dir).mkdir(parents=True, exist_ok=True)
    Path(out_merged_dir).mkdir(parents=True, exist_ok=True)

    for i in range(start_idx, end_idx + 1):
        try:
            plot_motion_vs_rppg_pair(
                index_i=i,
                motion_dir=motion_dir,
                rppg_dir=rppg_dir,
                win_sec=win_sec, hop_sec=hop_sec,
                br_min_bpm=br_min_bpm, br_max_bpm=br_max_bpm,
                use_hop_time=use_hop_time,
                interpolate_to_motion_axis=interpolate_to_motion_axis,
                save_png_dir=out_plot_dir,
                save_merged_csv=out_merged_dir,
                cohface_root=cohface_root
            )
        except Exception as e:
            print(f"[fail] {i}.csv → {e}")
        else:
            print(f"[ok] {i}.csv")

if __name__ == "__main__":
    motion_dir = "/home/subi/PycharmProjects/results/motion_signals"
    rppg_dir   = "/home/subi/PycharmProjects/results/rPPG_signals"
    out_plot_dir   = "/home/subi/PycharmProjects/BR_series/plots"
    out_merged_dir = "/home/subi/PycharmProjects/BR_series/merged"
    cohface_root   = "/home/subi/PycharmProjects/Cohface"

    run_compare_batch(
        motion_dir, rppg_dir,
        out_plot_dir, out_merged_dir,
        start_idx=1, end_idx=40,
        win_sec=30.0, hop_sec=5.0,
        br_min_bpm=8.0, br_max_bpm=30.0,
        use_hop_time=True,
        interpolate_to_motion_axis=True,
        cohface_root=cohface_root                              #  GT 활성화
    )
