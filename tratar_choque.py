import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARQUIVO = ROOT / "dados_brutos" / "fomc" / "monetary-policy-surprises-data.xlsx"  # confira o nome
ABA = "FOMC (update 2023)"
SAIDA = ROOT / "dados_tratados" / "choque_fomc_trimestral.csv"
INCLUIR_NAO_AGENDADAS = True  # reuniões extraordinárias (por exemplo, março de 2020)

df = pd.read_excel(ARQUIVO, sheet_name=ABA, na_values=["NA"])
df["Date"] = pd.to_datetime(df["Date"])
if not INCLUIR_NAO_AGENDADAS:
    df = df[df["Unscheduled"] == 0]
xls = pd.ExcelFile(ARQUIVO)
print("Abas:", xls.sheet_names)
print("Última data da aba:", df["Date"].max())

ausentes = df[(df["Date"] >= "2011-01-01") & df["MPS_ORTH"].isna()]
print("\nAnúncios sem MPS_ORTH desde 2011:")
print(ausentes[["Date", "MPS", "Unscheduled"]].to_string(index=False))
df = df[["Date", "MPS", "MPS_ORTH"]].copy()
df["Trimestre"] = df["Date"].dt.to_period("Q").astype(str)


def soma_completa(s):
    # se algum anúncio do trimestre não tem choque ortogonalizado, o trimestre fica ausente
    return s.sum() if s.notna().all() else float("nan")


choque = (
    df.groupby("Trimestre")
    .agg(
        Shock_t=("MPS_ORTH", soma_completa),
        Shock_bruto=("MPS", "sum"),
        n_anuncios=("MPS", "size"),
    )
    .reset_index()
    .sort_values("Trimestre")
)

SAIDA.parent.mkdir(parents=True, exist_ok=True)
choque.to_csv(SAIDA, index=False)

print(choque[choque["Trimestre"] >= "2011Q1"].to_string(index=False))
print("\nMaior |Shock_t| desde 2011:",
      choque.loc[choque["Trimestre"] >= "2011Q1", "Shock_t"].abs().max())
choque = (
    df.groupby("Trimestre")
    .agg(
        Shock_t=("MPS_ORTH", soma_completa),
        Shock_bruto=("MPS", "sum"),
        n_anuncios=("MPS", "size"),
        n_com_orto=("MPS_ORTH", "count"),     # quantos anúncios têm MPS_ORTH
    )
    .reset_index()
    .sort_values("Trimestre")
)
choque["orto_completo"] = choque["n_com_orto"] == choque["n_anuncios"]