import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRADA = ROOT / "dados_tratados" / "sec_longo.csv"
CHOQUE = ROOT / "dados_tratados" / "choque_fomc_trimestral.csv"
EPU = ROOT / "dados_tratados" / "epu_eua_trimestral.csv"
SAIDA = ROOT / "dados_tratados" / "painel_regressao_eua.csv"                  # principal (todas as etapas)
SAIDA_SEM_MED = ROOT / "dados_tratados" / "painel_regressao_eua_sem_mediana.csv"  # sem imputação por mediana
SAIDA_ROBUSTEZ = ROOT / "dados_tratados" / "painel_regressao_eua_robustez.csv"    # só dívida observada

INCLUIR_INTANGIVEL = True  # se False, Ativo_Fixo = PropertyPlantAndEquipmentNet
T_INI = pd.Period("2010Q4", "Q").ordinal
T_FIM = pd.Period("2026Q2", "Q").ordinal

# ---- Regras de tratamento da dívida ausente (hierarquia) ----
K_FILL = 4          # etapa 1: preenche NA com valor observado da própria firma até K trimestres
                    #          de distância (anterior; se não houver, posterior)
ZERO_REGRA = "ambos"  # etapa 2: "ambos"      -> zero só se a firma NUNCA reporta CP nem LP
                      #          "componente" -> zero se nunca reporta aquele componente
MIN_OBS = 5         # etapa 3: mediana Trim x SIC2 x Quartil só com >= MIN_OBS firmas observadas

CFO_TAGS = [
    "NetCashProvidedByUsedInOperatingActivities",
    "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
]
CX = ["CashAndCashEquivalentsAtCarryingValue",
      "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents", "Cash"]
INV = ["ShortTermInvestments", "MarketableSecuritiesCurrent",
       "AvailableForSaleSecuritiesDebtSecuritiesCurrent"]
PPE = ["PropertyPlantAndEquipmentNet",
       "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization"]
INT = ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"]
CP_ALT = ["LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
          "ShortTermBorrowings", "NotesPayableCurrent", "LoansPayableCurrent",
          "ConvertibleNotesPayableCurrent", "ConvertibleDebtCurrent"]
LTD = ["LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
       "LongTermNotesPayable", "LongTermLineOfCredit",
       "ConvertibleLongTermNotesPayable", "SecuredDebt", "SubordinatedDebt",
       "NotesPayable", "LineOfCredit", "OtherBorrowings"]
# "Liabilities" só entra num diagnóstico (se a tag não estiver no sec_longo, o diagnóstico é omitido)
ESTOQUE = list(dict.fromkeys(
    ["Assets", "DebtCurrent", "LongTermDebt", "Liabilities"] + CX + INV + PPE + INT + CP_ALT + LTD
))

df = pd.read_csv(
    ENTRADA, dtype=str,
    usecols=["cik", "sic", "period", "qtrs", "tag", "value", "filed"],
)
df["qtrs"] = pd.to_numeric(df["qtrs"], errors="coerce")
df["value"] = pd.to_numeric(df["value"], errors="coerce")
df["data"] = pd.to_datetime(df["period"], format="%Y%m%d", errors="coerce")
df = df.dropna(subset=["data", "qtrs", "value"])
df["tq"] = (df["data"] - pd.Timedelta(days=45)).dt.to_period("Q")


def coalesce(e, cols):
    return e[cols].bfill(axis=1).iloc[:, 0]


# ---------- SIC por firma ----------
sic = (
    df.dropna(subset=["sic"]).sort_values("filed").groupby("cik")["sic"].last()
    .str.strip().str.zfill(4)
)
sic_num = pd.to_numeric(sic, errors="coerce")

# ---------- estoques (balanço) ----------
s = df[(df["qtrs"] == 0) & df["tag"].isin(ESTOQUE)]
s = s.sort_values(["cik", "tq", "tag", "data", "filed"])
s = s.drop_duplicates(["cik", "tq", "tag"], keep="last")
e = s.set_index(["cik", "tq", "tag"])["value"].unstack("tag").reindex(columns=ESTOQUE)
e = e.sort_index()

p = pd.DataFrame(index=e.index)
p["Ativo_Total"] = e["Assets"].where(e["Assets"] > 0)
p["Caixa_Equiv"] = coalesce(e, CX)
p["Aplic_Financeiras"] = coalesce(e, INV)
p["Caixa"] = p["Caixa_Equiv"] + p["Aplic_Financeiras"].fillna(0)

ppe = coalesce(e, PPE)
if INCLUIR_INTANGIVEL:
    af = ppe + coalesce(e, INT).fillna(0)
else:
    af = ppe
p["Ativo_Fixo"] = af.where(af > 0)

# ---------- dívida observada ----------
cp_alt = e[CP_ALT].sum(axis=1, min_count=1)
cp = e["DebtCurrent"].combine_first(cp_alt)
lp_base = coalesce(e, LTD)
# LongTermDebt costuma incluir a parcela circulante: tira-se a parte de curto prazo
ld_liq = e["LongTermDebt"] - e["LongTermDebtCurrent"].fillna(0)
lp = lp_base.combine_first(ld_liq.where(ld_liq > 0))

# ---------- hierarquia de tratamento do NA de dívida ----------
#  etapa 0: observada
#  etapa 1: preenchida com valor observado da própria firma (até K_FILL trimestres)
#  etapa 2: zero, porque a firma nunca reporta dívida (NA = "não tem dívida")
#  etapa 3: mediana de Trimestre x SIC2 x Quartil de Ativo_Total (só doadoras observadas)
imp = pd.DataFrame({"cp": cp, "lp": lp, "Ativo_Total": p["Ativo_Total"]}, index=p.index)
imp["tq"] = imp.index.get_level_values("tq")
imp["t"] = imp.index.get_level_values("tq").asi8
sic_firma = imp.index.get_level_values("cik").map(sic_num)
imp["sic2"] = sic_firma // 100
fora = pd.Series(sic_firma, index=imp.index)
fora = fora.isna() | fora.between(6000, 6999) | fora.between(4900, 4999)
at_escopo = imp["Ativo_Total"].where(~fora)
imp["q_tam"] = np.ceil(at_escopo.groupby(imp["tq"]).rank(pct=True) * 4)  # 1..4

# firma "nunca reporta"
n_cp = imp.groupby(level="cik")["cp"].transform("count")
n_lp = imp.groupby(level="cik")["lp"].transform("count")
if ZERO_REGRA == "ambos":
    nunca_cp = nunca_lp = (n_cp == 0) & (n_lp == 0)
elif ZERO_REGRA == "componente":
    nunca_cp, nunca_lp = (n_cp == 0), (n_lp == 0)
else:
    raise ValueError("ZERO_REGRA deve ser 'ambos' ou 'componente'")


def preenche_firma(col):
    """Etapa 1: valor observado mais próximo da própria firma, até K_FILL trimestres
    (prefere o anterior; usa o posterior se não houver anterior dentro da janela)."""
    x, t = imp[col], imp["t"]
    t_obs = t.where(x.notna())
    last_t = t_obs.groupby(level="cik").ffill()
    last_v = x.groupby(level="cik").ffill()
    next_t = t_obs.groupby(level="cik").bfill()
    next_v = x.groupby(level="cik").bfill()
    ok_f = x.isna() & last_v.notna() & ((t - last_t) <= K_FILL)
    ok_b = x.isna() & ~ok_f & next_v.notna() & ((next_t - t) <= K_FILL)
    out = x.copy()
    out[ok_f] = last_v[ok_f]
    out[ok_b] = next_v[ok_b]
    return out, (ok_f | ok_b)


def mediana_celula(col):
    """Mediana Trim x SIC2 x Quartil (>= MIN_OBS); fallback Trim x Quartil; senão NA."""
    x = imp[col]
    g1 = x.groupby([imp["tq"], imp["sic2"], imp["q_tam"]])
    g2 = x.groupby([imp["tq"], imp["q_tam"]])
    med = g1.transform("median").where(g1.transform("count") >= MIN_OBS)
    return med.fillna(g2.transform("median").where(g2.transform("count") >= MIN_OBS))


def hierarquia(col, nunca):
    x = imp[col]
    a, f_fill = preenche_firma(col)
    z = a.isna() & nunca
    b = a.mask(z, 0.0)
    m = mediana_celula(col)
    f_med = b.isna() & m.notna()
    final = b.fillna(m)
    etapa = pd.Series(np.nan, index=imp.index)
    etapa[x.notna()] = 0
    etapa[f_fill] = 1
    etapa[z] = 2
    etapa[f_med] = 3
    return final, f_fill, z, f_med, etapa


(p["Emprestimos_CP"], p["cp_fill_firma"], p["cp_zero_nunca"],
 p["cp_mediana"], et_cp) = hierarquia("cp", nunca_cp)
(p["Emprestimos_LP"], p["lp_fill_firma"], p["lp_zero_nunca"],
 p["lp_mediana"], et_lp) = hierarquia("lp", nunca_lp)
p["div_etapa"] = np.maximum(et_cp, et_lp)          # pior etapa entre CP e LP (0..3; NA = sem valor)
p["div_imputada"] = p["div_etapa"] >= 1            # qualquer tratamento além do dado observado

# diagnóstico: as firmas tratadas como "sem dívida" têm passivo/ativo compatível?
pa = (e["Liabilities"] / e["Assets"]).where(e["Assets"] > 0)
imp_diag = pd.DataFrame({"pa": pa, "nunca": nunca_cp & nunca_lp})
n_firmas = imp.index.get_level_values("cik").nunique()
n_nunca = imp_diag.groupby(level="cik")["nunca"].first().sum()
print(f"Firmas que nunca reportam dívida (tratadas como zero): {int(n_nunca)} de {n_firmas} "
      f"({n_nunca / n_firmas:.1%}) | regra: {ZERO_REGRA}")
if imp_diag["pa"].notna().any():
    print("Passivo/Ativo mediano (Liabilities/Assets):")
    print(imp_diag.groupby("nunca")["pa"].median().rename({False: "reporta dívida",
                                                          True: "nunca reporta"}).round(3).to_string())
else:
    print("(tag Liabilities ausente no sec_longo: diagnóstico de passivo omitido)")
p = p.reset_index()

# ---------- fluxo de caixa operacional (desacumulado) ----------
f = df[df["tag"].isin(CFO_TAGS) & df["qtrs"].between(1, 4)].copy()
f["prio"] = (f["tag"] != CFO_TAGS[0]).astype(int)
f = f.sort_values(
    ["cik", "tq", "qtrs", "prio", "data", "filed"],
    ascending=[True, True, True, True, False, False],
).drop_duplicates(["cik", "tq", "qtrs"], keep="first")
f = f[["cik", "tq", "qtrs", "value"]]
ant = f.copy()
ant["tq"] = ant["tq"] + 1
ant["qtrs"] = ant["qtrs"] + 1
ant = ant.rename(columns={"value": "ytd_ant"})
g = f.merge(ant, on=["cik", "tq", "qtrs"], how="left")
g["CFO"] = np.where(g["qtrs"] == 1, g["value"], g["value"] - g["ytd_ant"])
g = g.dropna(subset=["CFO"]).sort_values(["cik", "tq", "qtrs"])
g = g.drop_duplicates(["cik", "tq"], keep="first")[["cik", "tq", "CFO"]]
p = p.merge(g, on=["cik", "tq"], how="left")

# ---------- tempo e defasagens ----------
p["t"] = (p["tq"].dt.year - 1970) * 4 + p["tq"].dt.quarter - 1
p["Trimestre"] = p["tq"].astype(str)
p = p.drop_duplicates(["cik", "t"], keep="last")

lagvars = ["Ativo_Total", "Caixa", "Ativo_Fixo"]
ant = p[["cik", "t"] + lagvars].copy()
ant["t"] += 1
ant = ant.rename(columns={v: v + "_L1" for v in lagvars})
p = p.merge(ant, on=["cik", "t"], how="left")


def com_alavancagem_l1(q):
    """(Re)calcula Alavancagem_L1 a partir de Alavancagem (merge em t+1: não pula trimestre)."""
    ant = q[["cik", "t", "Alavancagem"]].copy()
    ant["t"] += 1
    ant = ant.rename(columns={"Alavancagem": "Alavancagem_L1"})
    return q.drop(columns="Alavancagem_L1", errors="ignore").merge(ant, on=["cik", "t"], how="left")


p["Alavancagem"] = (p["Emprestimos_CP"] + p["Emprestimos_LP"]) / p["Ativo_Total"]
p = com_alavancagem_l1(p)

p["Cash_Ratio"] = p["Caixa_L1"] / p["Ativo_Total_L1"]
p["Tamanho"] = np.log(p["Ativo_Total_L1"])
p["log_Ativo_Fixo"] = np.log(p["Ativo_Fixo"])
p["log_Ativo_Fixo_L1"] = np.log(p["Ativo_Fixo_L1"])
p["Delta_log_Ativo_Fixo"] = p["log_Ativo_Fixo"] - p["log_Ativo_Fixo_L1"]
p["Delta_Capex"] = (p["Ativo_Fixo"] - p["Ativo_Fixo_L1"]) / p["Ativo_Total_L1"]
p["Fluxo_Caixa_Op_Ativo"] = p["CFO"] / p["Ativo_Total_L1"]
p["Tobin_Q"] = np.nan
p["Tobin_Q_L1"] = np.nan
p["obs_real"] = True

# ---------- setor e filtros ----------
p["sic"] = p["cik"].map(sic_num)
p = p.dropna(subset=["sic"])
p = p[~p["sic"].between(6000, 6999) & ~p["sic"].between(4900, 4999)]
p["SETOR_ATIV"] = "SIC" + (p["sic"] // 100).astype(int).astype(str).str.zfill(2)

# ---------- choque e EPU ----------
ch = pd.read_csv(CHOQUE).rename(columns={"Shock_bruto": "Choque_HFI"})
ch = ch[["Trimestre", "Shock_t", "Choque_HFI"]]
ep = pd.read_csv(EPU)
p = p.merge(ch, on="Trimestre", how="left").merge(ep, on="Trimestre", how="left")


# ---------- versões de robustez ----------
def versao(q, mascara):
    """Cópia em que a dívida (e a alavancagem) das linhas em 'mascara' vira NA.
    Alavancagem_L1 é recalculada, então a defasagem também respeita a máscara."""
    r = q.copy()
    r.loc[mascara, ["Emprestimos_CP", "Emprestimos_LP", "Alavancagem"]] = np.nan
    return com_alavancagem_l1(r)


def finaliza(q):
    q = q[(q["t"] >= T_INI) & (q["t"] <= T_FIM) & q["Ativo_Total"].notna()]
    q = q.rename(columns={"cik": "CNPJ_CIA"}).drop(columns=["tq", "sic"])
    return q.sort_values(["CNPJ_CIA", "t"])


p_sem_med = versao(p, p["div_etapa"] == 3)    # tira só a mediana; mantém firma-fill e zeros
p_obs = versao(p, p["div_etapa"] >= 1)        # só dívida efetivamente observada

SAIDA.parent.mkdir(parents=True, exist_ok=True)
p = finaliza(p)
p.to_csv(SAIDA, index=False)
finaliza(p_sem_med).to_csv(SAIDA_SEM_MED, index=False)
finaliza(p_obs).to_csv(SAIDA_ROBUSTEZ, index=False)

# ---------- diagnósticos ----------
print(f"\nGravado {SAIDA}: {len(p)} linhas, {p['CNPJ_CIA'].nunique()} firmas, "
      f"{p['SETOR_ATIV'].nunique()} setores")
print(f"Alavancagem válida: principal {p['Alavancagem'].notna().mean():.1%} | "
      f"sem_mediana {finaliza(p_sem_med)['Alavancagem'].notna().mean():.1%} | "
      f"robustez (só observada) {finaliza(p_obs)['Alavancagem'].notna().mean():.1%}")
print("\nFirmas por trimestre (a cada 4º):")
print(p.groupby("Trimestre")["CNPJ_CIA"].nunique().iloc[::4].to_string())
cols = ["Cash_Ratio", "Alavancagem", "Ativo_Fixo", "Delta_log_Ativo_Fixo",
        "Fluxo_Caixa_Op_Ativo", "Shock_t", "EPU"]
print("\nFração preenchida, por ano:")
print(p.assign(ano=p["Trimestre"].str[:4]).groupby("ano")[cols]
      .agg(lambda x: x.notna().mean()).round(2).to_string())
nomes = {0: "0 observada", 1: "1 preenchida (própria firma)", 2: "2 zero (nunca reporta)",
         3: "3 mediana de célula"}
dist = p["div_etapa"].value_counts(normalize=True, dropna=False).sort_index()
print("\nOrigem da dívida (pior etapa entre CP e LP), % das linhas:")
for k, v in dist.items():
    print(f"  {nomes.get(k, 'sem valor (célula pequena)'):<32s} {v:6.1%}")