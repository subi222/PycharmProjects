import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from scipy.signal import get_window, detrend
import os, h5py
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List
import glob



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

def _find_h5_for_index(cohface_root: str, idx: int, session: str = "0") -> Optional[str]:
    """
    /<root>/cohface*/<idx>/<session>/data.hdf5 를 검색해 첫 매치를 반환.
    예: /home/subi/PycharmProjects/Cohface/cohface1/1/0/data.hdf5
    """
    pats = [
        os.path.join(cohface_root, "cohface*", str(idx), session, "data.hdf5"),
        os.path.join(cohface_root, "cohface*", f"{idx}", session, "data.hdf5"),
    ]
    for p in pats:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None

def _infer_fs_from_time(t: np.ndarray) -> float:
    """time(초) 배열에서 fs 추정 (median Δt)."""
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.nan
    return float(1.0 / np.median(dt))

# ---------- 핵심: 한 파일 → br_gt 시계열 ----------
def br_series_from_gt(idx: int,
                      cohface_root: str,
                      win_sec: float = 30.0,
                      hop_sec: float = 5.0,
                      br_min_bpm: float = 8.0, # ← 안 쓰이지만 호출 호환성용
                      br_max_bpm: float = 30.0, # ← 안 쓰이지만 호출 호환성용
                      time_mode: str = "hop",
                      min_coverage_ratio: float = 0.5
                      ) -> pd.DataFrame:

    h5_path = _find_h5_for_index(cohface_root, idx, session="0")
    if not h5_path or not os.path.exists(h5_path):
        raise FileNotFoundError(f"GT HDF5 not found for idx={idx}: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        # 데이터셋 이름은 COHFACE 표준: 'time', 'respiration' (둘 다 float)
        t  = np.asarray(f["time"][:], dtype=float)
        gt = np.asarray(f["respiration"][:], dtype=float)

    # 안전 처리: 시간/신호 길이 맞추기 + 유효구간만 사용
    n = min(len(t), len(gt))
    t, gt = t[:n], gt[:n]
    mask = np.isfinite(t) & np.isfinite(gt)
    t, gt = t[mask], gt[mask]

    if t.size < 2:
        raise ValueError(f"Not enough GT samples in {h5_path}")

    fs = _infer_fs_from_time(t)
    if not np.isfinite(fs):
        raise ValueError(f"Cannot infer fs from time in {h5_path}")

    # --- 윈도우/홉 기반 시간축 구성 ---
    t_start = float(t[0])
    t_end   = float(t[-1])
    win = float(win_sec)
    hop = float(hop_sec)

    # hop 모드: 5,10,15,... 형태의 고정 시간축(파일 기준으로는 t_start 이후로 맞춤)
    # 관용적으로 5,10,…을 원하면 기준을 0으로 두고 생성 후 파일 범위 내만 사용
    # 여기서는 window가 완전히 데이터 범위에 들어오는 지점까지만 생성
    centers: List[float] = []
    k = 1
    while True:
        tc = k * hop if time_mode == "hop" else (t_start + win/2 + (k-1)*hop)
        w_start, w_end = tc - win/2, tc + win/2
        # 윈도우가 데이터 범위에 완전히 들어오지 않으면 종료
        if w_end > t_end or w_start < t_start:
            if time_mode == "center":
                break  # center 모드는 시작부터 범위 밖이면 바로 종료
            # hop 모드에서는 w_end가 범위를 넘기면 종료
            if w_end > t_end:
                break
        # 범위 내면 추가
        centers.append(tc)
        k += 1

    # --- 각 창에서 평균(BPM) 산출 ---
    br_vals: List[float] = []
    for tc in centers:
        w_start, w_end = tc - win/2, tc + win/2
        sel = (t >= w_start) & (t < w_end)
        if not np.any(sel):
            br_vals.append(np.nan)
            continue
        # 커버리지 체크 (너무 빈약하면 NaN)
        cov = sel.sum() / max(1, int(round(win * fs)))
        if cov < min_coverage_ratio:
            br_vals.append(np.nan)
            continue
        br_vals.append(float(np.nanmean(gt[sel])))

    return pd.DataFrame({"t_sec": np.asarray(centers, dtype=float),
                         "br_gt": np.asarray(br_vals, dtype=float)})

# ---------- 배치: 1~40 저장 ----------
def run_gt_batch(cohface_root: str,
                 out_dir: str,
                 start_idx: int = 1,
                 end_idx: int = 40,
                 win_sec: float = 30.0,
                 hop_sec: float = 5.0,
                 time_mode: str = "hop"):
    """
    각 참가자 idx(1..40)의 session0 GT → {out_dir}/{idx}_gt.csv 로 저장.
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    for i in range(start_idx, end_idx + 1):
        try:
            df = br_series_from_gt(i, cohface_root, win_sec, hop_sec, time_mode=time_mode)
            csv_path = outp / f"{i}_gt.csv"
            df.to_csv(csv_path, index=False)
            print(f"[ok] {i}: {csv_path.name} (rows={len(df)})")
        except Exception as e:
            print(f"[fail] {i}: {e}")

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
    out_dir = "/home/subi/PycharmProjects/BR_series/gt"
    run_gt_batch(cohface_root, out_dir,
                 start_idx=1, end_idx=40,
                 win_sec=30.0, hop_sec=5.0, time_mode="hop")

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
