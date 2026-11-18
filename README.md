# Concentrated shocks in panel regressions — United States

Replication code for the US results of the paper *Concentrated shocks in panel regressions*: SEC EDGAR (XBRL) quarterly firm data, the Bauer–Swanson monetary-policy shock (`MPS_ORTH`, summed by quarter, 1988Q1–2023Q4), and local projections of the shock × cash interaction.

The Brazilian results are in the companion repository: <https://github.com/LivMondini/Concentrated-shocks-in-panel-regressions---EUA>.


## 1. Repository layout

```
extrair_sec.py, tratar_choque.py, tratar_epu.py   raw data -> tidy series
painel_eua.py, filtrar_amostra.py                 builds the regression panels
projecoes_locais.py, bloco_checklist.py           local projections, episodes, block bootstrap
monte_carlo_checklist.py                          Monte Carlo of the episode-equality test
revisao_eua.py                                    audit tables and Monte Carlo (Tables A26, A27)
saidas_revisao/                                   small output tables (CSV) used in the paper
```

All paths are relative to the repository root; run every command from that folder.

## 2. Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Main packages: numpy 2.5.4, pandas, statsmodels 0.15.0, linearmodels, matplotlib 3.11.2, scipy, openpyxl. Random seeds are fixed (section 5).

<!-- TODO (author): pin the exact Python / pandas / linearmodels versions used for the paper (local .venv had Python 3.15.0, pandas 3.1.0, linearmodels 7.1) -->

## 3. Data (not included)

Raw data are **not redistributed**. Download them and respect their terms of use.

| Input | Source | Location |
|---|---|---|
| Financial Statement Data Sets (quarterly ZIPs containing `sub.txt` and `num.txt`) | U.S. SEC, <https://www.sec.gov/data-research/sec-markets-data/financial-statement-data-sets> | `dados_brutos/sec/` (e.g. `2023q4.zip`) |
| Monetary policy surprises, file `monetary-policy-surprises-data.xlsx` (columns `Date`, `MPS`, `MPS_ORTH`) | Bauer and Swanson (2023), authors' data page | `dados_brutos/fomc/` |
| Economic Policy Uncertainty, file `US_Policy_Uncertainty_Data.csv` | <https://www.policyuncertainty.com/us_monthly.html> | `dados_brutos/epu/` |

<!-- TODO (author): range of SEC quarters needed; exact URL of the Bauer–Swanson file -->

## 4. How to run

```bash
python extrair_sec.py        # SEC ZIPs        -> dados_tratados/sec_longo.csv
python tratar_choque.py      # FOMC spreadsheet -> dados_tratados/choque_fomc_trimestral.csv
python tratar_epu.py         # EPU             -> dados_tratados/epu_eua_trimestral.csv
python painel_eua.py         # -> dados_tratados/painel_regressao_eua*.csv (main, no median, robust)
python filtrar_amostra.py    # -> dados_tratados/painel_regressao_eua_amostra*.csv
python projecoes_locais.py   # local projections; also runs bloco_checklist.py
python revisao_eua.py        # audit tables -> saidas_revisao/
```

`dados_tratados/` is created by these scripts and is not part of the repository (the SEC extract alone is about 460 MB).

### Treatment of missing debt in the panels

| Version | Definition |
|---|---|
| main (median) | observed debt → firm's own fill → zero if the firm never reports short- or long-term debt → median by quarter, SIC2 and quartile |
| `sem_med` (no median) | removes only the median-filled rows |
| `robust` | observed debt only |
| `_adj` | winsorization / z-score adjustment conditional on a valid lagged leverage (`projecoes_locais.py`) |

## 5. Outputs used in the paper

| Paper item | Script | Output |
|---|---|---|
| Table A26 (United States): shock concentration, 45 quarters, effective number 17.8, episodes A and B | `revisao_eua.py` | `saidas_revisao/u3_concentracao_trimestre.csv` |
| Table A27: Monte Carlo of the episode-equality test | `revisao_eua.py` (loads `monte_carlo_checklist.py`) | `saidas_revisao/u9_monte_carlo_t54.csv` |

<!-- TODO (author): confirm u3 has 45 rows (window of Table A26); list any other US results shown in the manuscript (checklist_*.csv) -->

Seeds: `20261024` in `bloco_checklist.py` (bootstrap), `monte_carlo_checklist.py` and `revisao_eua.py` (Monte Carlo with R = 2,000 replications and B = 300 permutations).

## 6. Check your run

| Quantity | Expected | File |
|---|---|---|
| Quarters in the concentration table | 45 | `u3_concentracao_trimestre.csv` |
| Effective number of quarters | 17.8 | `u3_concentracao_trimestre.csv` |
| Share of each episode (A: 2015Q4–2018Q4, B: 2022Q1–2023Q3) | 33.7% | `u3_concentracao_trimestre.csv` |

## 7. License

Code: MIT (see `LICENSE`). The data belong to their original providers; see their terms of use.