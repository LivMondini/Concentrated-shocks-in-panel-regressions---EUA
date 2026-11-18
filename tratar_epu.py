import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARQUIVO = ROOT / "dados_brutos" / "epu" / "US_Policy_Uncertainty_Data.csv"  # ajuste o nome
SAIDA = ROOT / "dados_tratados" / "epu_eua_trimestral.csv"

df = pd.read_csv(ARQUIVO, sep=";")
df = df.dropna(subset=["Year", "Month"])
df["Year"] = df["Year"].astype(int)
df["Month"] = df["Month"].astype(int)
df["EPU"] = pd.to_numeric(df["News_Based_Policy_Uncert_Index"], errors="coerce")
df = df.dropna(subset=["EPU"])

df["Trimestre"] = df["Year"].astype(str) + "Q" + ((df["Month"] - 1) // 3 + 1).astype(str)

epu = df.groupby("Trimestre")["EPU"].agg(EPU="mean", n_meses="count").reset_index()
epu = epu[epu["n_meses"] == 3].drop(columns="n_meses").sort_values("Trimestre")

SAIDA.parent.mkdir(parents=True, exist_ok=True)
epu.to_csv(SAIDA, index=False)
print(epu.head(), "\n...\n", epu.tail(), f"\n{len(epu)} trimestres")