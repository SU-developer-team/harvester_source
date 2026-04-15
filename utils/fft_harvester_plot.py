# fft_harvester_plot.py
# 1) Рисует сигнал катушки (последний столбец) во времени
# 2) Делает FFT, находит значимые частоты
# 3) Снизу рисует сумму синусоид + (по желанию) отдельные синусоиды
# 4) Печатает: сколько синусоид (частотных компонент) прошло порог

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_coil_signal(csv_path: str) -> np.ndarray:
    """
    CSV: разделитель ';', десятичная запятая ',', в строках часто есть финальная ';'
    Берём последний НЕпустой столбец.
    """
    df = pd.read_csv(
        csv_path,
        sep=";",
        header=None,
        decimal=",",
        engine="python",
    )
    # убрать полностью пустые столбцы (из-за финальной ';')
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 1:
        raise ValueError("Не нашёл ни одного столбца с данными в CSV.")
    y = df.iloc[:, -1].astype(float).to_numpy()
    return y


def rfft_components(y: np.ndarray, fs: float, window: str):
    """
    Возвращает частоты, комплексный спектр, амплитуды (в тех же единицах, что y),
    и фазы для косинусов: A*cos(2πft + phi)
    """
    y = np.asarray(y, dtype=float)

    # убрать DC (смещение)
    y0 = y - np.mean(y)

    # окно
    if window.lower() == "hann":
        w = np.hanning(len(y0))
    elif window.lower() == "hamming":
        w = np.hamming(len(y0))
    elif window.lower() == "none":
        w = np.ones(len(y0))
    else:
        raise ValueError("window должен быть: hann | hamming | none")

    yw = y0 * w
    N = len(yw)

    X = np.fft.rfft(yw)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Амплитуды для реконструкции косинусами.
    # Из-за окна амплитуды чуть «проседают», поэтому нормируем на среднее окна.
    win_correction = np.mean(w)
    amps = np.abs(X) / (N * win_correction)
    amps = amps * 2.0  # односторонний спектр -> двусторонний эквивалент

    # DC (0 Гц) не удваиваем
    amps[0] = np.abs(X[0]) / (N * win_correction)

    # если N чётное, Nyquist тоже не удваиваем (последний бин)
    if (N % 2 == 0) and (len(amps) > 1):
        amps[-1] = np.abs(X[-1]) / (N * win_correction)

    phases = np.angle(X)  # фаза для cos
    return freqs, X, amps, phases, y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default=r"C:\Users\wwwai\projects\harvester\harvester\experiments\harvester_50mm\exp_1\data\2.csv",
        help="Путь к CSV",
    )
    ap.add_argument("--fs", type=float, default=1000.0, help="Частота дискретизации, Гц (1000 = 1000 точек/сек)")
    ap.add_argument("--duration", type=float, default=20.0, help="Ожидаемая длительность, сек (для проверки)")
    ap.add_argument("--window", default="hann", help="hann | hamming | none")
    ap.add_argument("--min_hz", type=float, default=0.5, help="Игнорировать частоты ниже этого порога (убрать дрейф)")
    ap.add_argument("--max_hz", type=float, default=None, help="Ограничить максимум частоты (по умолчанию до Nyquist)")
    ap.add_argument("--min_rel", type=float, default=0.03, help="Порог значимости относительно max амплитуды (например 0.03 = 3%)")
    ap.add_argument("--min_abs", type=float, default=None, help="Абсолютный порог амплитуды (если хочешь вместо min_rel)")
    ap.add_argument("--plot_components", type=int, default=30, help="Сколько отдельных синусоид рисовать (0 = не рисовать)")
    ap.add_argument("--out_png", default="fft_harvester.png", help="Куда сохранить картинку")
    ap.add_argument("--out_components_csv", default="fft_components.csv", help="Куда сохранить список компонент (freq, amp, phase)")
    args = ap.parse_args()

    y = read_coil_signal(args.csv)
    N = len(y)
    t = np.arange(N) / args.fs
    real_duration = N / args.fs

    # быстрый sanity-check
    if abs(real_duration - args.duration) > 0.5:
        print(f"[!] Предупреждение: длительность по данным = {real_duration:.3f} сек, а ожидалось ~ {args.duration} сек.")

    freqs, X, amps, phases, y0 = rfft_components(y, fs=args.fs, window=args.window)

    # ограничение по частотам
    nyq = args.fs / 2.0
    fmax = args.max_hz if args.max_hz is not None else nyq

    valid = (freqs >= args.min_hz) & (freqs <= fmax)
    freqs_v = freqs[valid]
    amps_v = amps[valid]
    phases_v = phases[valid]

    # порог: либо абсолютный, либо относительный
    max_amp = float(np.max(amps_v)) if len(amps_v) else 0.0
    if args.min_abs is not None:
        thr = float(args.min_abs)
    else:
        thr = max_amp * float(args.min_rel)

    sel = amps_v >= thr
    sel_freqs = freqs_v[sel]
    sel_amps = amps_v[sel]
    sel_phases = phases_v[sel]

    # отсортируем по амплитуде (сильные сверху)
    order = np.argsort(sel_amps)[::-1]
    sel_freqs = sel_freqs[order]
    sel_amps = sel_amps[order]
    sel_phases = sel_phases[order]

    print(f"Всего строк: {N}")
    print(f"Fs: {args.fs} Гц, длительность: {real_duration:.3f} сек, Nyquist: {nyq:.1f} Гц")
    print(f"Окно: {args.window}")
    print(f"Порог амплитуды: {thr:.6g} (max={max_amp:.6g})")
    print(f"Найдено значимых синусоид (частотных компонент): {len(sel_freqs)}")

    # сохраним список компонент
    comp_df = pd.DataFrame({
        "freq_hz": sel_freqs,
        "amp": sel_amps,
        "phase_rad": sel_phases,
    })
    comp_df.to_csv(args.out_components_csv, index=False)
    print(f"Список компонент сохранён: {Path(args.out_components_csv).resolve()}")

    # реконструкция суммы синусоид
    # x_rec(t) = Σ A_k cos(2π f_k t + phi_k)
    y_rec = np.zeros_like(t, dtype=float)
    for f, a, ph in zip(sel_freqs, sel_amps, sel_phases):
        y_rec += a * np.cos(2.0 * np.pi * f * t + ph)

    # графики
    fig = plt.figure(figsize=(12, 7))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(t, y, linewidth=0.9)
    ax1.set_title("Сигнал катушки (последний столбец CSV)")
    ax1.set_xlabel("Время, сек")
    ax1.set_ylabel("Амплитуда")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(2, 1, 2)
    # сумма выбранных синусоид
    ax2.plot(t, y_rec, linewidth=1.2, label=f"Сумма {len(sel_freqs)} синусоид")

    # отдельные синусоиды (чтобы не превратить график в лапшу)
    k = int(args.plot_components)
    if k > 0 and len(sel_freqs) > 0:
        k = min(k, len(sel_freqs))
        for i in range(k):
            f = sel_freqs[i]
            a = sel_amps[i]
            ph = sel_phases[i]
            comp = a * np.cos(2.0 * np.pi * f * t + ph)
            ax2.plot(t, comp, linewidth=0.8, alpha=0.7, label=f"{f:.2f} Hz")

    ax2.set_title("Реконструкция из FFT: синусоиды и их частоты (сильнейшие сверху)")
    ax2.set_xlabel("Время, сек")
    ax2.set_ylabel("Амплитуда")
    ax2.grid(True, alpha=0.3)

    # легенду ограничим, иначе будет простыня
    if (k > 0) and (k <= 12):
        ax2.legend(loc="upper right", fontsize=9)
    else:
        # если синусоид много — легенду не рисуем, частоты смотри в csv
        ax2.text(
            0.01, 0.98,
            f"Частоты/амплитуды см. {args.out_components_csv}\nПоказано отдельных синусоид: {min(k, len(sel_freqs))}",
            transform=ax2.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", alpha=0.15)
        )

    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Картинка сохранена: {Path(args.out_png).resolve()}")
    plt.show()


if __name__ == "__main__":
    main()

# python fft_harvester_plot.py --csv "experiments\harvester_80mm\exp_1\data\20.csv" --fs 1000 --duration 20 --min_rel 0.03 --plot_components 30 --window none
# python fft_harvester_plot.py --csv "D:\PROJECTs\magnet\harvester\graphs\20hz_v0_0\timeseries_emf_position_velocity.csv" --fs 1000 --duration 20 --min_rel 0.03 --plot_components 30 --window none
