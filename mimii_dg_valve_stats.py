#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Título: Estatística Descritiva de Áudio para o Subconjunto 'valve' do MIMII-DG
Autora: Emanuelle Lino Scheifer
Versão: 1.1 (2025-08-27)
Licença: MIT

Descrição geral
---------------
Este script realiza uma análise descritiva de arquivos de áudio (.wav) do
subconjunto 'valve' do MIMII-DG (2022). Ele:

1) Lê amostras de duas pastas: sinais normais e anômalos.
2) Extrai um conjunto de variáveis numéricas *no domínio do tempo e da frequência*.
3) Gera:
   - Uma amostra de características por arquivo (CSV).
   - Um resumo estatístico por classe (CSV).
   - Gráficos (histogramas e boxplots) comparando as classes.

Público-alvo
------------
Relatórios acadêmicos de disciplinas de Probabilidade e Estatística / Sinais
e Sistemas / Processamento de Áudio, mantendo linguagem objetiva e reprodutível.

Requisitos
----------
- Python 3.9+
- pip install numpy scipy soundfile librosa pandas matplotlib

Uso (exemplo)
-------------
python mimii_dg_valve_stats.py \
    --base_dir "/caminho/para/valve" \
    --normal_dir "normal" \
    --anom_dir "anomaly" \
    --n_per_class 100 \
    --seed 42 \
    --out_dir "saida_mimii_valve"

Observações
-----------
- Alguns zips do MIMII-DG usam "anomaly" e outros "anomalous". Ajuste --anom_dir se necessário.
- Para garantir reprodutibilidade da amostra quando há muitos arquivos, use --seed.

Definições das variáveis (por linha/arquivo)
--------------------------------------------
As variáveis extraídas são salvas no CSV `amostra_features.csv`:

- mean_abs               : média do valor absoluto do sinal (intensidade média).
- rms                    : *root mean square* (energia média do sinal no tempo).
- std                    : desvio-padrão não-viesado (ddof=1) da amplitude.
- cv                     : coeficiente de variação = std / (mean_abs + 1e-12).
- zcr                    : *zero-crossing rate* (proporção de trocas de sinal).
- p2p                    : pico-a-pico = max(x) - min(x).
- dom_freq_hz            : frequência dominante (Hz), pico de |FFT| (ignora DC).
- spec_centroid_hz       : centróide espectral (Hz): média ponderada das frequências por |FFT|.
- spec_bandwidth_hz      : largura de banda espectral (Hz): desvio-padrão em torno do centróide.
- sr                     : taxa de amostragem (Hz) reportada pelo arquivo.
- n_samples              : número de amostras do sinal após conversão para mono.

Definições das colunas do resumo por classe
-------------------------------------------
O CSV `resumo_por_classe.csv` agrega por classe (normal/anomalous) e cria,
para cada variável v em {mean_abs, rms, std, cv, zcr, p2p, dom_freq_hz,
spec_centroid_hz, spec_bandwidth_hz}, os seguintes indicadores:

- n_v       : contagem de observações válidas para v.
- media_v   : média aritmética de v.
- mediana_v : mediana de v.
- dp_v      : desvio-padrão de v.
- cv_v      : coeficiente de variação = dp_v / media_v.
- q1_v      : primeiro quartil (25%) de v.
- q3_v      : terceiro quartil (75%) de v.
- min_v     : valor mínimo observado de v.
- max_v     : valor máximo observado de v.

Saídas gráficas
---------------
Para um subconjunto de variáveis (rms, std, cv, dom_freq_hz, spec_centroid_hz),
o script gera, em `out_dir/`, dois tipos de visualizações por variável:
- hist_<var>.png : histogramas sobrepostos por classe.
- box_<var>.png  : boxplots comparando a distribuição por classe.

Contato
-------
Emanuelle Lino Scheifer — Projeto acadêmico UEPG (2025).
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==============================
# Utilidades de leitura de WAVs
# ==============================
def read_wav(path: str | os.PathLike) -> Tuple[np.ndarray, int]:
    """
    Lê um arquivo WAV e retorna (x, sr).

    - x é um np.ndarray float32, mono ou estéreo; se estéreo, o chamador decide
      como combinar (neste script, fazemos média dos canais).
    - sr é a taxa de amostragem (Hz).

    Estratégia:
      1) Tenta `soundfile` (robusto para diferentes bit-depths).
      2) Faz *fallback* para `scipy.io.wavfile` se necessário.
    """
    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(path, always_2d=False)
        # Converte para float32 preservando escala; se inteiro, apenas *cast*.
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32, copy=False)
        else:
            data = data.astype(np.float32, copy=False)
        return data, int(sr)
    except Exception:
        from scipy.io import wavfile  # type: ignore

        sr, data = wavfile.read(path)
        if not np.issubdtype(data.dtype, np.floating):
            # Normaliza inteiros para [-1, 1] aproximadamente.
            if np.issubdtype(data.dtype, np.integer):
                max_val = max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max)
                data = data.astype(np.float32) / float(max_val if max_val != 0 else 1.0)
            else:
                data = data.astype(np.float32)
        else:
            data = data.astype(np.float32, copy=False)
        return data, int(sr)


# =========================================
# Métricas no domínio do tempo e frequência
# =========================================
def _dominant_freq(x: np.ndarray, sr: int) -> float:
    """Frequência dominante (Hz) via pico no módulo da FFT (ignora a componente DC)."""
    n = len(x)
    if n == 0 or sr <= 0:
        return float("nan")
    # Próxima potência de 2 para FFT eficiente
    nfft = 1 << (n - 1).bit_length()
    spec = np.fft.rfft(x, n=nfft)
    mag = np.abs(spec)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    if mag.size < 2:
        return float("nan")
    k = int(np.argmax(mag[1:]) + 1)  # ignora DC
    return float(freqs[k])


def _spectral_centroid_bandwidth(x: np.ndarray, sr: int) -> Tuple[float, float]:
    """Retorna (centróide espectral, largura de banda), ambos em Hz, usando |FFT| como pesos."""
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
    """
    Extrai variáveis estatísticas de um sinal (converte para mono por média dos canais).

    Retorna um dicionário com as chaves documentadas no cabeçalho do arquivo.
    """
    # Converte para mono, se necessário
    if x.ndim > 1:
        x = np.mean(x, axis=1)

    # Remove componente DC para focar em oscilações
    x = x - np.mean(x)

    # Medidas no tempo
    mean_abs = float(np.mean(np.abs(x)))  # intensidade média
    rms = float(np.sqrt(np.mean(x**2)))  # energia média
    std = float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
    cv = float(std / (np.mean(np.abs(x)) + 1e-12))
    # Zero-crossing rate (proporção de transições de sinal)
    zcr = float(((x[:-1] * x[1:]) < 0).mean()) if len(x) > 1 else float("nan")
    # Amplitude pico-a-pico
    p2p = float(np.max(x) - np.min(x)) if len(x) else float("nan")

    # Medidas no espectro
    dom = _dominant_freq(x, sr)
    sc, bw = _spectral_centroid_bandwidth(x, sr)

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


# =========================
# Coleta e amostragem de IO
# =========================
def collect_files(
    base_dir: str | os.PathLike,
    normal_dir: str = "normal",
    anom_dir: str = "anomaly",
    n_per_class: int | None = None,
    seed: int | None = None,
) -> Tuple[List[str], List[str]]:
    """
    Retorna listas de caminhos .wav para cada classe.

    - Se n_per_class for fornecido, realiza amostragem aleatória (reprodutível com --seed).
    - Aceita nomes alternativos para pasta de anomalias: {"anomalous", "anomaly", "abnormal"}.
    """
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
            raise FileNotFoundError(
                f"Pasta de anomalias não encontrada: {a_path}. Tente --anom_dir anomalous"
            )

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


# ========================
# Estatística por agregado
# ========================
def summarize_by_class(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
    """
    Agrega por 'classe' e computa métricas resumo padronizadas para cada variável.
    """
    g = df.groupby("classe")[list(value_cols)]
    summary = pd.concat(
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
    return summary


# ============
# Visualização
# ============
def plot_hist(df: pd.DataFrame, col: str, out_png: Path) -> None:
    """Gera histograma sobreposto por classe para a coluna indicada."""
    plt.figure(figsize=(7, 4.5))
    for cls in ("normal", "anomalous"):
        subset = df[df["classe"] == cls][col].dropna().values
        if subset.size == 0:
            continue
        plt.hist(subset, bins=40, alpha=0.6, label=cls)
    plt.title(f"Histograma de {col}")
    plt.xlabel(col)
    plt.ylabel("Frequência")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_box(df: pd.DataFrame, col: str, out_png: Path) -> None:
    """Gera boxplot comparando a distribuição por classe para a coluna indicada."""
    plt.figure(figsize=(6, 4.5))
    data = [
        df[df["classe"] == "normal"][col].dropna().values,
        df[df["classe"] == "anomalous"][col].dropna().values,
    ]
    plt.boxplot(data, labels=["normal", "anomalous"], showmeans=True)
    plt.title(f"Boxplot de {col} por classe")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =====
# Main
# =====
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mimii_dg_valve_stats.py",
        description=(
            "Analisa sinais de áudio (MIMII-DG/valve), extrai variáveis no tempo e frequência, "
            "gera CSVs e gráficos comparando classes normal vs anomalous."
        ),
        epilog=(
            "Exemplo: python mimii_dg_valve_stats.py --base_dir ./valve --normal_dir normal "
            "--anom_dir anomalous --n_per_class 100 --seed 42 --out_dir saida_mimii_valve"
        ),
    )
    ap.add_argument(
        "--base_dir",
        required=True,
        help="Diretório raiz do subconjunto 'valve' (contendo subpastas normal/ e anomaly/ ou anomalous/).",
    )
    ap.add_argument(
        "--normal_dir",
        default="normal",
        help="Nome da pasta de sinais normais (padrão: normal).",
    )
    ap.add_argument(
        "--anom_dir",
        default="anomaly",
        help="Nome da pasta de anomalias (ex.: anomalous, anomaly, abnormal).",
    )
    ap.add_argument(
        "--n_per_class",
        type=int,
        default=100,
        help="Tamanho máximo da amostra por classe (amostragem aleatória se houver excesso).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semente para reprodutibilidade da amostragem (opcional).",
    )
    ap.add_argument(
        "--out_dir",
        default="saida_mimii_valve",
        help="Pasta de saída para CSVs e PNGs (será criada se não existir).",
    )
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Coletando arquivos...")
    n_files, a_files = collect_files(
        args.base_dir, args.normal_dir, args.anom_dir, args.n_per_class, args.seed
    )
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

    if len(rows) == 0:
        print("Nenhum arquivo processado. Verifique caminhos e extensões .wav.")
        sys.exit(1)

    df = pd.DataFrame(rows)

    # CSV 1: amostra de características por arquivo
    out_csv = out_dir / "amostra_features.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"CSV salvo em {out_csv}")

    # CSV 2: resumo por classe
    value_cols = [
        "mean_abs",
        "rms",
        "std",
        "cv",
        "zcr",
        "p2p",
        "dom_freq_hz",
        "spec_centroid_hz",
        "spec_bandwidth_hz",
    ]
    resumo = summarize_by_class(df, value_cols)
    out_sum = out_dir / "resumo_por_classe.csv"
    resumo.to_csv(out_sum, encoding="utf-8")
    print(f"Resumo salvo em {out_sum}")

    # Gráficos
    for col in ("rms", "std", "cv", "dom_freq_hz", "spec_centroid_hz"):
        plot_hist(df, col, out_dir / f"hist_{col}.png")
        plot_box(df, col, out_dir / f"box_{col}.png")

    print("Gráficos gerados (PNG) e prontos para inserir no relatório.")
    print("Concluído.")


if __name__ == "__main__":
    main()
