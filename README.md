# Estatística Descritiva — MIMII‑DG (valve)

**Autora:** Emanuelle Lino Scheifer  
**Versão:** 1.1 — 27 ago 2025  
**Objetivo:** descrever, de forma acadêmica e reprodutível, as variáveis extraídas do áudio e os produtos gerados pelo script `mimii_dg_valve_stats.py`.

---

## 1. Metodologia (resumo)

- **Amostras:** arquivos `.wav` de duas classes (normal e anômalo) do subconjunto *valve* do MIMII‑DG.  
- **Conversão para mono:** sinais estéreo são convertidos por **média dos canais**.  
- **Remoção de DC:** subtração da média do sinal para evitar viés de nível.  
- **Domínios analisados:** tempo e frequência (usando FFT).  
- **Amostragem por classe:** se `--n_per_class` for definido, é feita **amostragem aleatória** (use `--seed` para reprodutibilidade).

---

## 2. Variáveis extraídas por arquivo

As seguintes variáveis são computadas e salvas no CSV `amostra_features.csv`:

1. **`mean_abs`** *(média do valor absoluto)* — Mede a **intensidade média** do sinal no tempo, útil para comparar níveis gerais de amplitude entre amostras.

2. **`rms`** *(root mean square)* — Estima a **energia média** do sinal no tempo. Sinais mais energéticos produzem valores de RMS maiores.

3. **`std`** *(desvio‑padrão não‑viesado; ddof=1)* — Quantifica a **dispersão** da amplitude ao redor da média (após remoção de DC).

4. **`cv`** *(coeficiente de variação)* — Razão entre `std` e `mean_abs` (com estabilização numérica `+ 1e-12`). É uma medida **adimensional** de variabilidade relativa.

5. **`zcr`** *(zero‑crossing rate)* — Proporção de **trocas de sinal** entre amostras consecutivas. Em geral, sinais com mais conteúdo de alta frequência têm ZCR maior.

6. **`p2p`** *(pico‑a‑pico)* — Diferença **máximo − mínimo** da amplitude. Indica a **faixa dinâmica instantânea** do trecho.

7. **`dom_freq_hz`** *(frequência dominante)* — Frequência (Hz) do maior pico da magnitude da FFT, ignorando a componente DC. Aproxima a **frequência mais proeminente** do espectro.

8. **`spec_centroid_hz`** *(centróide espectral)* — Média ponderada das frequências em Hz, usando |FFT| como pesos. Interpreta‑se como o “**centro de massa**” do espectro (brilho espectral).

9. **`spec_bandwidth_hz`** *(largura de banda espectral)* — **Desvio‑padrão** das frequências em torno do centróide, ponderado por |FFT|. Indica o **espalhamento** do espectro.

10. **`sr`** *(sample rate)* — **Taxa de amostragem** (Hz) do arquivo de áudio.

11. **`n_samples`** — **Número de amostras** do sinal após conversão para mono.

> **Nota:** quando a leitura usa `scipy.io.wavfile` para formatos inteiros, o script normaliza para aproximadamente `[-1, 1]` antes de calcular as métricas, garantindo comparabilidade.

---

## 3. Resumo estatístico por classe

O CSV `resumo_por_classe.csv` agrega as amostras por **classe** (`normal` e `anomalous`) e computa, para cada variável *v*, os indicadores:

- **`n_v`** — tamanho da amostra válida;
- **`media_v`** — média aritmética;
- **`mediana_v`** — mediana;
- **`dp_v`** — desvio‑padrão;
- **`cv_v`** — coeficiente de variação = `dp_v / media_v`;
- **`q1_v`** — primeiro quartil (25%);
- **`q3_v`** — terceiro quartil (75%);
- **`min_v`** — mínimo;
- **`max_v`** — máximo.

Esse resumo permite comparar **tendência central**, **dispersão** e **assimetria** (por meio de `q1`/`q3`) entre as classes.

---

## 4. Saídas gráficas

Para as variáveis `rms`, `std`, `cv`, `dom_freq_hz` e `spec_centroid_hz`, são gerados:
- **Histogramas** (`hist_<var>.png`): mostram a **distribuição de frequências** para cada classe.
- **Boxplots** (`box_<var>.png`): sintetizam **mediana, quartis e possíveis outliers**, facilitando a comparação direta entre classes.

---

## 5. Boas práticas e reprodutibilidade

- **Controle de amostra:** utilize `--n_per_class` para limitar o tamanho por classe de forma consistente em diferentes execuções.  
- **Semente fixa:** forneça `--seed` (e.g., `--seed 42`) sempre que for importante reproduzir exatamente a mesma seleção de arquivos.  
- **Organização das pastas:** confirme os nomes das pastas de anomalia (`anomaly`/`anomalous`/`abnormal`).

---

## 6. Execução (exemplo)

```bash
python mimii_dg_valve_stats.py \
  --base_dir "/caminho/para/valve" \
  --normal_dir "normal" \
  --anom_dir "anomalous" \
  --n_per_class 100 \
  --seed 42 \
  --out_dir "saida_mimii_valve"
```

Os resultados (CSVs e PNGs) ficarão disponíveis em `saida_mimii_valve/`.

---

## 7. Citação sugerida (trabalho acadêmico)

> Scheifer, E. L. (2025). **Estatística descritiva de áudio para o subconjunto ‘valve’ do MIMII‑DG** (Versão 1.1) [Software]. UEPG.  
> Disponível mediante execução local: `mimii_dg_valve_stats.py`.

---

**Contato:** Emanuelle Lino Scheifer — UEPG (2025).
