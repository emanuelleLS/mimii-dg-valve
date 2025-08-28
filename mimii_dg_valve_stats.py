#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Título: Estatística Descritiva de Áudio para o Subconjunto 'valve' do MIMII-DG
Autora: Emanuelle Lino Scheifer
Versão: 1.2 (2025-08-28)  |  Licença: MIT

Descrição geral
---------------
Analisa arquivos .wav do subconjunto 'valve' (MIMII-DG, 2022), extrai variáveis no
tempo e na frequência, gera CSVs e gráficos (histogramas e boxplots) por classe.

Uso (exemplo)
-------------
python mimii_dg_valve_stats.py \
  --base_dir "C:/.../valve/test" \
  --normal_dir normal \
  --anom_dir anomaly \
  --n_per_class 200 \
  --seed 42 \
  --box_whis 1.5 \
  --show_fliers 1 \
  --out_dir "saida_mimii_valve"
"""

from __future__ import annotations
import argparse, glob, os, random, sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Leitura de WAV (robusta)
# -----------------------------
def read_wav(path: str | os.PathLike) -> Tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore
        data, sr = sf.read(path, always_2d=False)
        data = data.astype(np.float32, copy=False) if np.issubdtype(data.dtype, np.floating) else data.astype(np.float32)
        return data, int(sr)
    except Exception:
        from scipy.io import wavfile  # type: ignore
        sr, data = wavfile.read(path)
        if not np.issubdtype(data.dtype, np.floating):
            if np.issubdtype(data.dtype, np.integer):
                max_val = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max) or 1
                data = data.astype(np.float32) / float(max_val)
            else:
                data = data.astype(np.float32)
        else:
            data = data.astype(np.float32, copy=False)
        return data, int(sr)

# -----------------------------
# Métricas no tempo/frequência
# -----------------------------
def _dominant_freq(x: np.ndarray, sr: int) -> float:
    n = len(x)
    if n == 0 or sr <= 0:
        return float("nan")
    nfft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(x, n=nfft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    if mag.size < 2:
        return float("nan")
    k = int(np.argmax(mag[1:]) + 1)  # ignora DC
    return float(freqs[k])

def _spectral_centroid_bandwidth(x: np.ndarray, sr: int) -> Tuple[float, float]:
    n = len(x)
    if n == 0 or sr <= 0:
        return float("nan"), float("nan")
    nfft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(x, n=nfft)
    mag = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    return centroid, bw

def extract_features(x: np.ndarray, sr: int) -> Dict[str, float]:
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x - np.mean(x)  # remove DC
    mean_abs = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    cv = float(std / (mean_abs + 1e-12))
    zcr = float(((x[:-1] * x[1:]) < 0).mean()) if len(x) > 1 else float("nan")
    p2p = float(np.max(x) - np.min(x)) if len(x) else float("nan")
    dom = _dominant_freq(x, sr)
    sc, bw = _spectral_centroid_bandwidth(x, sr)
    return {
        "mean_abs": mean_abs,              # média do valor absoluto (intensidade)
        "rms": rms,                        # root mean square (energia)
        "std": std,                        # desvio-padrão
        "cv": cv,                          # coeficiente de variação
        "zcr": zcr,                        # zero-crossing rate
        "p2p": p2p,                        # pico-a-pico
        "dom_freq_hz": dom,                # frequência dominante (Hz)
        "spec_centroid_hz": sc,            # centróide espectral (Hz)
        "spec_bandwidth_hz": bw,           # largura de banda espectral (Hz)
        "sr": float(sr),
        "n_samples": int(len(x)),
    }

# -----------------------------
# Coleta e amostragem de arquivos
# -----------------------------
def collect_files(base_dir: str | os.PathLike,
                  normal_dir: str = "normal",
                  anom_dir: str = "anomaly",
                  n_per_class: int | None = 200,
                  seed: int | None = None) -> Tuple[List[str], List[str]]:
    base = Path(base_dir)
    n_path = base / normal_dir
    a_path = base / anom_dir
    if not n_path.exists():
        raise FileNotFoundError(f"Pasta de normais não encontrada: {n_path}")
    if not a_path.exists():
        for cand in ("anomalous", "anomaly", "abnormal"):
            if (base / cand).exists():
                a_path = base / cand
                break
        else:
            raise FileNotFoundError(f"Pasta de anomalias não encontrada: {a_path}. Tente --anom_dir anomalous")
    n_files = sorted(glob.glob(str(n_path / "**/*.wav"), recursive=True))
    a_files = sorted(glob.glob(str(a_path / "**/*.wav"), recursive=True))
    if seed is not None:
        random.seed(seed)
    if n_per_class is not None:
        if len(n_files) > n_per_class:
            n_files = random.sample(n_files, k=n_per_class)
        if len(a_files) > n_per_class:
            a_files = random.sample(a_files, k=n_per_class)
    return n_files, a_files

# -----------------------------
# Resumo por classe
# -----------------------------
def summarize_by_class(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    g = df.groupby("classe")[list(value_cols)]
    return pd.concat(
        [
            g.count().rename(columns=lambda c: f"n_{c}"),
            g.mean().rename(columns=lambda c: f"media_{c}"),
            g.median().rename(columns=lambda c: f"mediana_{c}"),
            g.std().rename(columns=lambda c: f"dp_{c}"),
            (g.std() / g.mean()).rename(columns=lambda c: f"cv_{c}"),
            g.quantile(0.25).rename(columns=lambda c: f"q1_{c}"),
            g.quantile(0.75).rename(columns=lambda c: f"q3_{c}"),
            g.min().rename(columns=lambda c: f"min_{c}"),
            g.max().rename(columns=lambda c: f"max_{c}"),
        ],
        axis=1,
    )

# -----------------------------
# Visualização
# -----------------------------
# Mapeia nomes curtos -> rótulos completos (com significado)
DISPLAY = {
    "rms": "rms (energia média; root mean square)",
    "std": "std (desvio-padrão)",
    "cv": "cv (coeficiente de variação)",
    "dom_freq_hz": "dom_freq_hz (frequência dominante, Hz)",
    "spec_centroid_hz": "spec_centroid_hz (centróide espectral, Hz)",
    "zcr": "zcr (taxa de cruzamento por zero)",
    "p2p": "p2p (amplitude pico-a-pico)",
    "mean_abs": "mean_abs (intensidade média |x|)",
    "spec_bandwidth_hz": "spec_bandwidth_hz (largura de banda espectral, Hz)",
}

def plot_hist(df: pd.DataFrame, col: str, out_png: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    for cls in ("normal", "anomalous"):
        subset = df[df["classe"] == cls][col].dropna().values
        if subset.size == 0:
            continue
        plt.hist(subset, bins=40, alpha=0.6, label=cls)
    plt.title(f"Histograma de {DISPLAY.get(col, col)}")
    plt.xlabel(DISPLAY.get(col, col))
    plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def plot_box(df: pd.DataFrame, col: str, out_png: Path, whis: float = 1.5, show_fliers: bool = True) -> None:
    plt.figure(figsize=(6.4, 4.8))
    data = [
        df[df["classe"] == "normal"][col].dropna().values,
        df[df["classe"] == "anomalous"][col].dropna().values,
    ]
    # Matplotlib 3.9+: use tick_labels (evita deprecation warning)
    plt.boxplot(
        data,
        tick_labels=["normal", "anomalous"],
        showmeans=True,
        whis=whis,
        showfliers=show_fliers,
    )
    full = DISPLAY.get(col, col)
    plt.title(f"Boxplot de {full} por classe")
    plt.ylabel(full)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mimii_dg_valve_stats.py",
        description=("Analisa MIMII-DG/valve, extrai variáveis e gera CSVs/figuras por classe."),
    )
    ap.add_argument("--base_dir", required=True, help="Raiz do subconjunto 'valve' (contendo normal/ e anomaly/|anomalous/).")
    ap.add_argument("--normal_dir", default="normal", help="Pasta de sinais normais (padrão: normal).")
    ap.add_argument("--anom_dir", default="anomaly", help="Pasta de anomalias (ex.: anomalous, anomaly, abnormal).")
    ap.add_argument("--n_per_class", type=int, default=200, help="Tamanho máximo da amostra por classe.")
    ap.add_argument("--seed", type=int, default=None, help="Semente para reprodutibilidade da amostragem.")
    ap.add_argument("--out_dir", default="saida_mimii_valve", help="Pasta de saída para CSVs e PNGs.")
    ap.add_argument("--box_whis", type=float, default=1.5, help="Comprimento do bigode no boxplot (em IQR).")
    ap.add_argument("--show_fliers", type=int, default=1, help="1=mostrar outliers; 0=ocultar.")
    return ap

def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("Coletando arquivos...")
    n_files, a_files = collect_files(args.base_dir, args.normal_dir, args.anom_dir, args.n_per_class, args.seed)
    print(f"Arquivos normal: {len(n_files)} | anomalous: {len(a_files)}")

    rows: List[Dict[str, float]] = []
    for cls, files in (("normal", n_files), ("anomalous", a_files)):
        for fp in files:
            try:
                x, sr = read_wav(fp)
                feats = extract_features(x, sr)
                feats.update({"classe": cls, "arquivo": str(fp)})
                rows.append(feats)
            except Exception as e:
                print(f"[AVISO] Falha ao processar {fp}: {e}", file=sys.stderr)

    if not rows:
        print("Nenhum arquivo processado. Verifique caminhos e extensões .wav.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # CSV 1: características por arquivo
    out_csv = out_dir / "amostra_features.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"CSV salvo em {out_csv}")

    # CSV 2: resumo por classe
    value_cols = ["mean_abs","rms","std","cv","zcr","p2p","dom_freq_hz","spec_centroid_hz","spec_bandwidth_hz"]
    resumo = summarize_by_class(df, value_cols)
    out_sum = out_dir / "resumo_por_classe.csv"
    resumo.to_csv(out_sum, encoding="utf-8")
    print(f"Resumo salvo em {out_sum}")

    # Gráficos
    for col in ("rms","std","cv","dom_freq_hz","spec_centroid_hz"):
        plot_hist(df, col, out_dir / f"hist_{col}.png")
        plot_box(df, col, out_dir / f"box_{col}.png", whis=args.box_whis, show_fliers=bool(args.show_fliers))

    print("Gráficos gerados (PNG) e prontos para o relatório.")
    print("Concluído.")

if __name__ == "__main__":
    main()
