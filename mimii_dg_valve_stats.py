#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIMII-DG (2022) - Estatística descritiva para 'valve'
Autora: Emanuelle Lino Scheifer
Descrição:
    - Lê arquivos WAV de duas pastas (normal e anômalo) do conjunto MIMII-DG (valve)
    - Extrai variáveis numéricas por arquivo
    - Gera tabelas (CSV) e gráficos (PNG) para o relatório de Probabilidade e Estatística
Requisitos:
    Python 3.9+
    pip install numpy scipy soundfile librosa pandas matplotlib
Uso (exemplo):
    python mimii_dg_valve_stats.py --base_dir "/caminho/para/valve" --normal_dir "normal" --anom_dir "anomaly" --n_per_class 100
Observação:
    Alguns zips do MIMII-DG usam "anomaly" e outros "anomalous". Ajuste a flag --anom_dir se necessário.
"""

import argparse
import os
import sys
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read_wav(path):
    """Lê WAV retornando float32 np.array e sr."""

    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=False)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32, copy=False)
        else:
            data = data.astype(np.float32, copy=False)
        return data, sr
    except Exception:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if not np.issubdtype(data.dtype, np.floating):
            # normalizar se inteiro
            if np.issubdtype(data.dtype, np.integer):
                max_val = np.iinfo(data.dtype).max
                data = data.astype(np.float32) / max_val
            else:
                data = data.astype(np.float32)
        else:
            data = data.astype(np.float32, copy=False)
        return data, sr

def dominant_freq(x, sr):
    # Frequência dominante (Hz) via pico no módulo do FFT.
    n = len(x)
    if n == 0 or sr <= 0:
        return np.nan
    nfft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(x, n=nfft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    if mag.size < 2:
        return np.nan
    k = np.argmax(mag[1:]) + 1  # ignora DC
    return float(freqs[k])

def spectral_centroid_bandwidth(x, sr):
    # Centroid e bandwidth espectrais usando magnitude do FFT
    n = len(x)
    if n == 0 or sr <= 0:
        return np.nan, np.nan
    nfft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(x, n=nfft)
    mag = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(nfft, 1.0/sr)
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    bw = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    return centroid, bw

def extract_features(x, sr):
    # Extrai variáveis estatísticas de um sinal mono.
    if x.ndim > 1:
        x = np.mean(x, axis=1)
    x = x - np.mean(x)  # remove DC
    # Medidas no tempo
    mean_abs = float(np.mean(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2)))
    std = float(np.std(x, ddof=1)) if len(x) > 1 else np.nan
    cv = float(std / (np.mean(np.abs(x)) + 1e-12))
    # Zero-crossing rate (simples)
    zcr = float(((x[:-1] * x[1:]) < 0).mean()) if len(x) > 1 else np.nan
    p2p = float(np.max(x) - np.min(x)) if len(x) else np.nan
    # Medidas no espectro
    dom = dominant_freq(x, sr)
    sc, bw = spectral_centroid_bandwidth(x, sr)
    return {
        "mean_abs": mean_abs,
        "rms": rms,
        "std": std,
        "cv": cv,
        "zcr": zcr,
        "p2p": p2p,
        "dom_freq_hz": dom,
        "spec_centroid_hz": sc,
        "spec_bandwidth_hz": bw,
        "sr": float(sr),
        "n_samples": int(len(x)),
    }

def collect_files(base_dir, normal_dir, anom_dir, n_per_class=None):
    base = Path(base_dir)
    n_path = base / normal_dir
    a_path = base / anom_dir
    if not n_path.exists():
        raise FileNotFoundError(f"Pasta de normais não encontrada: {n_path}")
    if not a_path.exists():
        # tenta nomes alternativos comuns
        for cand in ["anomalous", "anomaly", "abnormal"]:
            if (base / cand).exists():
                a_path = base / cand
                break
        else:
            raise FileNotFoundError(f"Pasta de anomalias não encontrada: {a_path}. Tente --anom_dir anomalous")
    n_files = sorted(glob.glob(str(n_path / "**/*.wav"), recursive=True))
    a_files = sorted(glob.glob(str(a_path / "**/*.wav"), recursive=True))
    if n_per_class is not None:
        n_files = n_files[:n_per_class]
        a_files = a_files[:n_per_class]
    return n_files, a_files

def summarize_by_class(df, value_cols):
    g = df.groupby("classe")[value_cols]
    summary = pd.concat([
        g.count().rename(columns=lambda c: f"n_{c}"),
        g.mean().rename(columns=lambda c: f"media_{c}"),
        g.median().rename(columns=lambda c: f"mediana_{c}"),
        g.std().rename(columns=lambda c: f"dp_{c}"),
        (g.std()/g.mean()).rename(columns=lambda c: f"cv_{c}"),
        g.quantile(0.25).rename(columns=lambda c: f"q1_{c}"),
        g.quantile(0.75).rename(columns=lambda c: f"q3_{c}"),
        g.min().rename(columns=lambda c: f"min_{c}"),
        g.max().rename(columns=lambda c: f"max_{c}"),
    ], axis=1)
    return summary

def plot_hist(df, col, out_png):
    plt.figure(figsize=(7,4.5))
    for cls in ["normal", "anomalous"]:
        subset = df[df["classe"]==cls][col].dropna().values
        if subset.size == 0:
            continue
        plt.hist(subset, bins=40, alpha=0.6, label=cls)
    plt.title(f"Histograma de {col}")
    plt.xlabel(col); plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def plot_box(df, col, out_png):
    plt.figure(figsize=(6,4.5))
    data = [
        df[df["classe"]=="normal"][col].dropna().values,
        df[df["classe"]=="anomalous"][col].dropna().values
    ]
    plt.boxplot(data, labels=["normal","anomalous"], showmeans=True)
    plt.title(f"Boxplot de {col} por classe")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="pasta 'valve' extraída do Zenodo (contendo subpastas normal/ e anomaly/ ou anomalous/)")
    ap.add_argument("--normal_dir", default="normal", help="nome da pasta de normais (padrão: normal)")
    ap.add_argument("--anom_dir", default="anomaly", help="nome da pasta de anomalias (alternativas: anomalous, abnormal)")
    ap.add_argument("--n_per_class", type=int, default=100, help="número máximo de arquivos por classe para a amostra")
    ap.add_argument("--out_dir", default="saida_mimii_valve", help="pasta de saída para CSV/PNGs")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Coletando arquivos...")
    n_files, a_files = collect_files(args.base_dir, args.normal_dir, args.anom_dir, args.n_per_class)
    print(f"Arquivos normal: {len(n_files)} | anomalous: {len(a_files)}")

    rows = []
    for cls, files in [("normal", n_files), ("anomalous", a_files)]:
        for fp in files:
            try:
                x, sr = read_wav(fp)
                feats = extract_features(x, sr)
                feats.update({"classe": cls, "arquivo": str(fp)})
                rows.append(feats)
            except Exception as e:
                print(f"[AVISO] Falha ao processar {fp}: {e}", file=sys.stderr)

    if len(rows) == 0:
        print("Nenhum arquivo processado. Verifique caminhos e extensões .wav.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    out_csv = Path(args.out_dir)/"amostra_features.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"CSV salvo em {out_csv}")

    value_cols = ["mean_abs","rms","std","cv","zcr","p2p","dom_freq_hz","spec_centroid_hz","spec_bandwidth_hz"]
    resumo = summarize_by_class(df, value_cols)
    out_sum = Path(args.out_dir)/"resumo_por_classe.csv"
    resumo.to_csv(out_sum, encoding="utf-8")
    print(f"Resumo salvo em {out_sum}")

    for col in ["rms","std","cv","dom_freq_hz","spec_centroid_hz"]:
        plot_hist(df, col, Path(args.out_dir)/f"hist_{col}.png")
        plot_box(df, col, Path(args.out_dir)/f"box_{col}.png")

    print("Gráficos gerados (PNG) e prontos para inserir no relatório.")
    print("Concluído.")

if __name__ == "__main__":
    main()
