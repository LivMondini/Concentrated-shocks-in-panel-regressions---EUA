import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# py filtrar_amostra.py principal      -> painel_regressao_eua.csv
# py filtrar_amostra.py sem_mediana    -> painel_regressao_eua_sem_mediana.csv
# py filtrar_amostra.py robustez       -> painel_regressao_eua_robustez.csv
_arg = sys.argv[1] if len(sys.argv) > 1 else ""
DIV_VERSAO = "" if _arg == "principal" else _arg
if DIV_VERSAO not in ("", "sem_mediana", "robustez"):
    raise ValueError("Versão inválida. Use: principal | sem_mediana | robustez")

_rb = f"_{DIV_VERSAO}" if DIV_VERSAO else ""
ENTRADA = ROOT / "dados_tratados" / f"painel_regressao_eua{_rb}.csv"
SAIDA = ROOT / "dados_tratados" / f"painel_regressao_eua_amostra{_rb}.csv"
T_INI = pd.Period("2011Q4", "Q").ordinal
MIN_ATIVO = 1e7        # mediana do ativo total da firma, em US$
MIN_TRIM = 12          # trimestres com ativo fixo válido

print(f">>> Versão: '{DIV_VERSAO or 'principal'}' | {ENTRADA} -> {SAIDA}")

p = pd.read_csv(ENTRADA, dtype={"CNPJ_CIA": str})
p = p[p["t"] >= T_INI]
n0 = p["CNPJ_CIA"].nunique()

# 2. tamanho mínimo
med = p.groupby("CNPJ_CIA")["Ativo_Total"].median()
p = p[p["CNPJ_CIA"].isin(med[med >= MIN_ATIVO].index)]
n1 = p["CNPJ_CIA"].nunique()

# 3. histórico mínimo
n_af = p.groupby("CNPJ_CIA")["Ativo_Fixo"].count()
p = p[p["CNPJ_CIA"].isin(n_af[n_af >= MIN_TRIM].index)]
n2 = p["CNPJ_CIA"].nunique()

# 4. grade completa por firma
lim = p.groupby("CNPJ_CIA")["t"].agg(["min", "max"])
grade = pd.concat(
    [pd.DataFrame({"CNPJ_CIA": c, "t": np.arange(r["min"], r["max"] + 1)})
     for c, r in lim.iterrows()],
    ignore_index=True,
)
p = grade.merge(p, on=["CNPJ_CIA", "t"], how="left")
p["obs_real"] = p["obs_real"].eq(True)
p["SETOR_ATIV"] = p.groupby("CNPJ_CIA")["SETOR_ATIV"].transform("first")
for c in ["Shock_t", "Choque_HFI", "EPU"]:
    p[c] = p.groupby("t")[c].transform("first")
p["Trimestre"] = (1970 + p["t"] // 4).astype(str) + "Q" + (p["t"] % 4 + 1).astype(str)
p = p.sort_values(["CNPJ_CIA", "t"])

SAIDA.parent.mkdir(parents=True, exist_ok=True)
p.to_csv(SAIDA, index=False)

print(f"Firmas: {n0} -> {n1} (tamanho) -> {n2} (histórico)")
print(f"Gravado {SAIDA}: {len(p)} linhas, {int(p['obs_real'].sum())} observações reais")

p["ano"] = p["Trimestre"].str[:4]
print("\nFirmas com ativo fixo, por ano (média dos trimestres):")
print(p[p["Ativo_Fixo"].notna()].groupby(["ano", "Trimestre"])["CNPJ_CIA"]
      .nunique().groupby("ano").mean().round(0).to_string())

d = p["Delta_log_Ativo_Fixo"].dropna()
print("\nDelta_log_Ativo_Fixo, percentis:")
print(d.quantile([.01, .05, .25, .5, .75, .95, .99]).round(3).to_string())
print(f"Participação com |variação| > 0,5: {(d.abs() > 0.5).mean():.1%}")

cols = ["log_Ativo_Fixo", "Cash_Ratio", "Alavancagem", "Tamanho",
        "Fluxo_Caixa_Op_Ativo", "Shock_t"]
print("\nObservações com todas as variáveis da especificação base:")
print(p.dropna(subset=cols).groupby("ano").size().to_string())