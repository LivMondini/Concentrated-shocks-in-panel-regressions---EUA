import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parent
_T0 = time.time()


def _tic(msg):
    print(f"[{time.time() - _T0:6.0f}s] {msg}", flush=True)


# ======================= COMO RODAR =======================
#   py projecoes_locais.py principal          -> DIV_VERSAO = ""  (sem ajuste, não se aplica)
#   py projecoes_locais.py sem_mediana        -> com ajuste (_adj)
#   py projecoes_locais.py sem_mediana noadj  -> sem ajuste
#   py projecoes_locais.py robustez           -> com ajuste (_adj)
#   py projecoes_locais.py robustez noadj     -> sem ajuste
# ==========================================================
_args = [a for a in sys.argv[1:] if a != "noadj"]
_arg = _args[0] if _args else ""
DIV_VERSAO = "" if _arg == "principal" else _arg
if DIV_VERSAO not in ("", "sem_mediana", "robustez"):
    raise ValueError("Versão inválida. Use: principal | sem_mediana | robustez")

# ======================= CONFIGURAÇÃO =======================
# Desfecho: y_{i,t+h} = log_Ativo_Fixo_{i,t+h} - log_Ativo_Fixo_{i,t-1}  (Delta_log_Ativo_Fixo)
H_MIN, H_MAX = -4, 8
_rb = f"_{DIV_VERSAO}" if DIV_VERSAO else ""
ARQ = ROOT / "dados_tratados" / f"painel_regressao_eua_amostra{_rb}.csv"
ARQ_CHOQUE = ROOT / "dados_tratados" / "choque_fomc_trimestral.csv"
MIN_OBS = 200          # mínimo de observações para estimar um horizonte
H_PRE = (-4, -3, -2)   # horizontes do teste de pré-tendências

# Inferência: "dk" (Driscoll-Kraay, lags = |h|+1) | "firma" | "dupla"
INFERENCIA = "dk"

WINSOR_P = (0.01, 0.99)

UNIDADE_SHOCK_ORIGEM = "pp"      # "bp" ou "pp"
ESCALA_SHOCK = "zscore"          # "100bp" ou "zscore"
inverter_sinal_choque = False

RODAR_DIAGNOSTICO_SELECAO = False
CONTROLE_SHOCK_L1 = True
EXCLUIR_ANOS: tuple = ()
INTERACOES_L2 = False
AMOSTRA_COMUM = True

# ---- Chaves de velocidade ----
RODAR_SPECS = ["base", "ctrl_int", "setor_t"]
RODAR_SPLIT_ALAV = False             # base por grupo de alavancagem (Baixa/Alta)
RODAR_GRAFICOS = False
RODAR_EXTRAS = True                  # VIF, influência por ano, escudo v2/v3, entrantes (diagnóstico)
# ============================================================

DESFECHO = "Delta_log_Ativo_Fixo"
SUFIXO = "_Delta_log_Ativo_Fixo_FIXO_ctrlL2"
if DIV_VERSAO:
    SUFIXO += f"_div_{DIV_VERSAO}"

# Winsor/z-score só nas linhas com Alavancagem_L1 válida (só em sem_mediana e robustez)
AJUSTE_DIV = bool(DIV_VERSAO) and ("noadj" not in sys.argv)
if AJUSTE_DIV:
    SUFIXO += "_adj"

if EXCLUIR_ANOS:
    SUFIXO += "_sem" + "_".join(str(a) for a in EXCLUIR_ANOS)

if INFERENCIA not in ("dupla", "firma", "dk"):
    raise ValueError("INFERENCIA deve ser 'dupla', 'firma' ou 'dk'")
if UNIDADE_SHOCK_ORIGEM not in ("bp", "pp"):
    raise ValueError("UNIDADE_SHOCK_ORIGEM deve ser 'bp' ou 'pp'")
if ESCALA_SHOCK not in ("100bp", "zscore"):
    raise ValueError("ESCALA_SHOCK deve ser '100bp' ou 'zscore'")

print(f">>> Versão: '{DIV_VERSAO or 'principal'}' | AJUSTE_DIV = {AJUSTE_DIV} | SUFIXO = {SUFIXO}")
print(f">>> Specs: {RODAR_SPECS} | split alav: {RODAR_SPLIT_ALAV} | gráficos: {RODAR_GRAFICOS} "
      f"| extras: {RODAR_EXTRAS}")


# ---------------- FUNÇÕES AUXILIARES ----------------
def zscore(s):
    """Z-score; devolve NaN se não houver variação."""
    sd = s.std()
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def winsor(s, lo=WINSOR_P[0], hi=WINSOR_P[1]):
    """Winsorização por clip nos quantis lo/hi (ignora NaN)."""
    s = pd.Series(s)
    if s.notna().sum() == 0:
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def winsor_m(s, m, lo=WINSOR_P[0], hi=WINSOR_P[1]):
    """Winsoriza todas as linhas, mas calcula os quantis só onde m é True."""
    q = s[m]
    return s.clip(q.quantile(lo), q.quantile(hi)) if q.notna().any() else s


def zscore_m(df, v, m):
    """Z-score por trimestre, com média e desvio calculados só nas linhas m."""
    g = df[v].where(m).groupby(df["t"])
    sd = g.transform("std").replace(0, np.nan)
    return (df[v] - g.transform("mean")) / sd


def n_lags(h, n_t):
    """Lags de Newey-West/Bartlett: lags = |h| + 1 (máx. n_t - 1)."""
    return int(min(abs(h) + 1, n_t - 1))


def cov_kwargs(h, n_t):
    if INFERENCIA == "dupla":
        return dict(cov_type="clustered", cluster_entity=True, cluster_time=True)
    if INFERENCIA == "firma":
        return dict(cov_type="clustered", cluster_entity=True)
    return dict(cov_type="kernel", kernel="bartlett", bandwidth=n_lags(h, n_t))


def ano_de_t(t):
    """Ano a partir do ordinal trimestral (ordinal 0 = 1970Q1)."""
    return 1970 + np.asarray(t) // 4


def prepara_choque(p):
    f = pd.read_csv(ARQ_CHOQUE)
    f["t"] = pd.PeriodIndex(f["Trimestre"], freq="Q").asi8
    tri = f.set_index("t")["Shock_t"].astype(float).sort_index()
    tri = tri.reindex(np.arange(tri.index.min(), tri.index.max() + 1))   # grade contínua
    if UNIDADE_SHOCK_ORIGEM == "bp":
        tri = tri / 100.0
    if inverter_sinal_choque:
        tri = -tri
        print(">>> Sinal do choque INVERTIDO (inverter_sinal_choque=True).")

    amostra = tri.reindex(np.unique(p["t"])).dropna()      # trimestres usados na estimação
    sd = amostra.std()
    faltam = [str(pd.Period(ordinal=int(o), freq="Q")) for o in np.unique(p["t"])
              if o not in amostra.index]
    print(f"Choque (pp, antes da escala): média = {amostra.mean():.4f} | desvio = {sd:.4f} | "
          f"min = {amostra.min():.4f} | max = {amostra.max():.4f} | T = {len(amostra)}")
    print(f"Trimestres do painel SEM choque (ficam ausentes): {faltam}")
    anos = [pd.Period(ordinal=int(o), freq="Q").year for o in amostra.index]
    print("Choque médio por ano (pp):")
    print(amostra.groupby(anos).mean().round(4).to_string(), "\n")

    ac1, ac2 = tri.autocorr(1), tri.autocorr(2)
    print(f"Autocorrelação do choque: lag1 = {ac1:.3f} | lag2 = {ac2:.3f} "
          f"(|valor| > ~{2 / np.sqrt(tri.notna().sum()):.2f} sugere persistência relevante)\n")

    if ESCALA_SHOCK == "zscore":
        tri = (tri - amostra.mean()) / sd                  # escala pela amostra de estimação
    return tri.rename("Shock")


# ---------------- DADOS ----------------
p = pd.read_csv(ARQ, dtype={"CNPJ_CIA": str})

obrig = ["CNPJ_CIA", "log_Ativo_Fixo", "Shock_t", "SETOR_ATIV",
         "Cash_Ratio", "Alavancagem", "Tobin_Q", "Tamanho", "Fluxo_Caixa_Op_Ativo"]
faltam = [c for c in obrig if c not in p.columns]
if faltam:
    raise KeyError(f"Colunas ausentes em {ARQ}: {faltam}")

if "Trimestre" in p.columns:
    p["t"] = pd.PeriodIndex(p["Trimestre"], freq="Q").asi8
elif "t" not in p.columns:
    raise KeyError(f"{ARQ} precisa ter 'Trimestre' ou 't'.")
p["t"] = p["t"].astype("int64")
p = p.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)

if p.duplicated(["CNPJ_CIA", "t"]).any():
    raise AssertionError("Há pares (CNPJ_CIA, t) duplicados na base.")

p["log_Ativo_Fixo"] = p["log_Ativo_Fixo"].where(np.isfinite(p["log_Ativo_Fixo"]))
print(f"Desfecho: {DESFECHO} | obs. com log_Ativo_Fixo válido: "
      f"{int(p['log_Ativo_Fixo'].notna().sum())} de {len(p)} "
      f"({p.loc[p['log_Ativo_Fixo'].notna(), 'CNPJ_CIA'].nunique()} empresas)")
if EXCLUIR_ANOS:
    print(f">>> Robustez: observações dos anos {EXCLUIR_ANOS} excluídas das regressões.")

# Winsorização 1%/99% das variáveis de balanço em nível, ANTES dos lags e do z-score
NIVEL = ["Cash_Ratio", "Alavancagem", "Tobin_Q", "Tamanho", "Fluxo_Caixa_Op_Ativo"]
for v in NIVEL:
    if AJUSTE_DIV and v in ("Cash_Ratio", "Tamanho", "Fluxo_Caixa_Op_Ativo"):
        continue          # winsorizados depois, só nas linhas ok_div
    p[v] = winsor(p[v])

p = p.drop(columns=[c for c in ("Alavancagem_L1", "Tobin_Q_L1", "Fluxo_Caixa_Op_Ativo_L1")
                    if c in p.columns])

# Controles pré-determinados: tudo em t-1 (lag por MERGE em t+1: nunca pula trimestre)
for variable in ["Alavancagem", "Tobin_Q", "Fluxo_Caixa_Op_Ativo"]:
    lag = p[["CNPJ_CIA", "t", variable]].copy()
    lag["t"] += 1
    lag = lag.rename(columns={variable: variable + "_L1"})
    p = p.merge(lag, on=["CNPJ_CIA", "t"], how="left")

# Choque na escala/sinal escolhidos (ANTES de construir as interações)
tri = prepara_choque(p)
p = p.merge(tri, left_on="t", right_index=True, how="left")

# Choque defasado: série trimestral contínua -> shift(1) == trimestre t-1
tri_cont = tri.reindex(np.arange(tri.index.min(), tri.index.max() + 1))
p = p.merge(tri_cont.shift(1).rename("Shock_L1"), left_on="t", right_index=True, how="left")
p = p.merge(tri_cont.shift(2).rename("Shock_L2"), left_on="t", right_index=True, how="left")

# Padronização cross-sectional dentro de cada trimestre
VZ = ["Cash_Ratio", "Alavancagem_L1", "Tobin_Q_L1", "Tamanho", "Fluxo_Caixa_Op_Ativo_L1"]
if AJUSTE_DIV:
    ok_div = p["Alavancagem_L1"].notna()
    for v in ["Cash_Ratio", "Tamanho", "Fluxo_Caixa_Op_Ativo_L1"]:
        p[v] = winsor_m(p[v], ok_div)
    for v in VZ:
        p[v + "_z"] = zscore_m(p, v, ok_div)
    zz = p.loc[ok_div, ["t", "Cash_Ratio_z"]].groupby("t")["Cash_Ratio_z"].agg(["mean", "std"])
    print(f">>> AJUSTE_DIV: {int(ok_div.sum())} de {len(p)} linhas na base de winsor/z-score. "
          f"Cash_Ratio_z: média {zz['mean'].abs().max():.1e} (máx. |média| por trimestre), "
          f"dp médio {zz['std'].mean():.3f}")
else:
    for v in VZ:
        p[v + "_z"] = p.groupby("t")[v].transform(zscore)

# Interações construídas APÓS o z-score trimestral
p["Ch_Caixa"] = p["Shock"] * p["Cash_Ratio_z"]
p["Ch_Alav"] = p["Shock"] * p["Alavancagem_L1_z"]
p["Ch_Q"] = p["Shock"] * p["Tobin_Q_L1_z"]
p["Ch_Tam"] = p["Shock"] * p["Tamanho_z"]
p["Ch_FCO"] = p["Shock"] * p["Fluxo_Caixa_Op_Ativo_L1_z"]

# Interações defasadas: Shock_{t-k} x z_{t-k}, por firma
for ch in ["Ch_Caixa", "Ch_Alav", "Ch_Q", "Ch_Tam", "Ch_FCO"]:
    for k in (1, 2):
        lag = p[["CNPJ_CIA", "t", ch]].copy()
        lag["t"] += k
        lag = lag.rename(columns={ch: f"{ch}_L{k}"})
        p = p.merge(lag, on=["CNPJ_CIA", "t"], how="left")
_tic("dados preparados")

# ---------------- DEFINIÇÃO DAS AMOSTRAS ----------------
p["in_restrita"] = p["Tobin_Q_L1_z"].notna()
print(f"\nAmostra Total: {len(p)} obs. | Restrita (Tobin_Q_L1 válido): "
      f"{int(p['in_restrita'].sum())} obs. "
      f"({p.loc[p['in_restrita'], 'CNPJ_CIA'].nunique()} de {p['CNPJ_CIA'].nunique()} empresas)")

AMOSTRAS = {"Total": dict(com_q=False)}
if RODAR_DIAGNOSTICO_SELECAO:
    AMOSTRAS["Restrita_semQ"] = dict(com_q=False)


def dados_da_amostra(nome):
    return p if nome == "Total" else p[p["in_restrita"]]


def monta_specs(com_q):
    """Especificações de uma amostra. 'req' força amostra idêntica entre specs (AMOSTRA_COMUM)."""
    ctrl = ["Cash_Ratio_z", "Alavancagem_L1_z"] + (["Tobin_Q_L1_z"] if com_q else []) + \
           ["Tamanho_z", "Fluxo_Caixa_Op_Ativo_L1_z"]
    ctrl_t = (["Shock_L1"] if CONTROLE_SHOCK_L1 else []) + ctrl
    ints = ["Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO"] + (["Ch_Q"] if com_q else [])
    ints_L1 = [v + "_L1" for v in ints]
    ints_L2 = [v + "_L2" for v in ints] if INTERACOES_L2 else []
    ints_din = ints_L1 + ints_L2
    req = (["Shock_L2"] + ints_din) if AMOSTRA_COMUM else []
    return {
        "base":       dict(x_int=["Shock"], x=["Shock"] + ctrl_t, tempo_fe=False, req=req),
        "ctrl_int":   dict(x_int=["Shock"] + ints, x=["Shock"] + ints + ctrl_t,
                           tempo_fe=False, req=req),
        "ctrl_int_L": dict(x_int=["Shock", "Shock_L1", "Shock_L2"] + ints + ints_din,
                           x=["Shock", "Shock_L1", "Shock_L2"] + ints + ints_din + ctrl,
                           tempo_fe=False, req=req),
        "setor_t":    dict(x_int=ints, x=ints + ctrl, tempo_fe=True, req=req),
        "setor_t_L":  dict(x_int=ints + ints_din, x=ints + ints_din + ctrl,
                           tempo_fe=True, req=req),
    }


# ---------------- DIAGNÓSTICO DE SELEÇÃO ----------------
def diagnostico_selecao(df):
    """Diferença padronizada de médias: firmas na amostra Restrita vs. fora dela."""
    vars_ = ["log_Ativo_Fixo", "Cash_Ratio", "Alavancagem_L1", "Tamanho", "Fluxo_Caixa_Op_Ativo_L1"]
    a, b = df[df["in_restrita"]], df[~df["in_restrita"]]
    linhas = []
    for v in vars_:
        sd = np.sqrt((a[v].var() + b[v].var()) / 2)
        linhas.append({"var": v, "media_restrita": a[v].mean(), "media_fora": b[v].mean(),
                       "dif_padronizada": (a[v].mean() - b[v].mean()) / sd if sd > 0 else np.nan,
                       "n_restrita": int(a[v].notna().sum()), "n_fora": int(b[v].notna().sum())})
    return pd.DataFrame(linhas)


sel = diagnostico_selecao(p)
print("\n===== Seleção: Restrita vs. fora da Restrita (|dif. padronizada| > 0,2 = relevante) =====")
print(sel.round(4).to_string(index=False))
sel.to_csv(ROOT / f"diagnostico_selecao{SUFIXO}.csv", index=False)

# ---------------- split por mediana da alavancagem pré-choque ----------------
mediana_alav = p["Alavancagem_L1"].median()
p["grupo_alav"] = (
    pd.Series(np.where(p["Alavancagem_L1"] <= mediana_alav, "Baixa", "Alta"), index=p.index)
    .where(p["Alavancagem_L1"].notna())
)
print(f"\nInferência: {INFERENCIA}" + (" (lags = |h| + 1)" if INFERENCIA == "dk" else ""))
print(f"Controle Shock_L1: {CONTROLE_SHOCK_L1}")
print(f"Interações L2: {INTERACOES_L2} | Amostra comum: {AMOSTRA_COMUM}")
print(f"Mediana Alavancagem(t-1) = {mediana_alav:.4f}")
print(p["grupo_alav"].value_counts(dropna=False).to_string(), "\n")

# ---------------- SÉRIE DEPENDENTE (com cache) ----------------
T_ALL = np.arange(p["t"].min(), p["t"].max() + 1)
piv_log_af = (p.pivot(index="CNPJ_CIA", columns="t", values="log_Ativo_Fixo")
              .reindex(columns=T_ALL))
piv_log_af.columns.name = "t"

_Y = {}


def serie_y(h):
    """y = log AtivoFixo(t+h) - log AtivoFixo(t-1); h pode ser negativo. Cacheado por horizonte."""
    if h not in _Y:
        lead = piv_log_af.shift(-h, axis=1)
        baseline = piv_log_af.shift(1, axis=1)
        _Y[h] = (lead - baseline).stack().rename("y").reset_index()
    return _Y[h]


def base_h(dados, spec, h):
    """Mesma amostra que a função lp() usa naquele horizonte."""
    y = serie_y(h)
    d = dados.merge(y, on=["CNPJ_CIA", "t"], how="inner")
    d = d.dropna(subset=["y"] + spec["x"] + ["SETOR_ATIV"] + spec.get("req", [])).copy()
    if EXCLUIR_ANOS:
        d = d[~np.isin(ano_de_t(d["t"]), EXCLUIR_ANOS)]
    d["y"] = winsor(d["y"])
    d["ano"] = ano_de_t(d["t"])
    return d.reset_index(drop=True)


# ---------------- PROJEÇÕES LOCAIS ----------------
RES_COLS = ["amostra", "subgrupo", "spec", "h", "var", "coef", "se", "p", "n", "n_firmas",
            "inferencia", "lags_dk"]


def lp(spec_nome, dados, amostra, subgrupo, specs):
    spec = specs[spec_nome]
    x, x_int = spec["x"], spec["x_int"]
    tag = f"[{amostra}|{subgrupo}|{spec_nome}]"
    res = []
    for h in range(H_MIN, H_MAX + 1):
        if h == -1:   # zero por construção (baseline = t-1)
            continue
        y = serie_y(h)
        d = dados.merge(y, on=["CNPJ_CIA", "t"], how="inner")
        d = d.dropna(subset=["y"] + x + ["SETOR_ATIV"] + spec.get("req", [])).copy()
        if EXCLUIR_ANOS:
            d = d[~np.isin(ano_de_t(d["t"]), EXCLUIR_ANOS)]
        if len(d) < MIN_OBS:
            print(f"{tag} h={h}: poucas obs ({len(d)}), pulado")
            continue

        d["y"] = winsor(d["y"])

        n_t = d["t"].nunique()
        if spec["tempo_fe"]:
            d["setor_t"] = d["SETOR_ATIV"].astype(str) + "|" + d["t"].astype(str)
        d = d.set_index(["CNPJ_CIA", "t"])

        if spec["tempo_fe"]:
            model = PanelOLS(d["y"], d[x], entity_effects=True, other_effects=d[["setor_t"]])
        else:
            model = PanelOLS(d["y"], d[x], entity_effects=True)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = model.fit(**cov_kwargs(h, n_t))
        if any("sqrt" in str(wi.message) for wi in w) or r.std_errors.isna().any():
            print(f"{tag} h={h}: AVISO - variância inválida (SE NaN) com inferencia='{INFERENCIA}'")

        nf = d.index.get_level_values(0).nunique()
        bw = n_lags(h, n_t) if INFERENCIA == "dk" else np.nan

        for v in x_int:
            res.append({"amostra": amostra, "subgrupo": subgrupo, "spec": spec_nome, "h": h,
                        "var": v, "coef": r.params[v], "se": r.std_errors[v], "p": r.pvalues[v],
                        "n": int(r.nobs), "n_firmas": nf, "inferencia": INFERENCIA, "lags_dk": bw})
        print(f"{tag} h={h}: n={int(r.nobs)}, empresas={nf}, T={n_t}"
              + (f", lags DK={bw}" if INFERENCIA == "dk" else ""))
    out = pd.DataFrame(res, columns=RES_COLS)
    out["lo95"] = out["coef"] - 1.96 * out["se"]
    out["hi95"] = out["coef"] + 1.96 * out["se"]
    out["lo90"] = out["coef"] - 1.645 * out["se"]
    out["hi90"] = out["coef"] + 1.645 * out["se"]
    return out


todos = []
for nome, cfg in AMOSTRAS.items():
    dados = dados_da_amostra(nome)
    specs = monta_specs(cfg["com_q"])
    print(f"\n########## AMOSTRA {nome}: {len(dados)} obs. | controle de Q: {cfg['com_q']} ##########")
    todos.append(lp("base", dados, nome, "Cheia", specs))
    if RODAR_SPLIT_ALAV:
        for g in ["Baixa", "Alta"]:
            todos.append(lp("base", dados[dados["grupo_alav"] == g], nome, g, specs))
    for s_ in RODAR_SPECS:
        if s_ != "base":
            todos.append(lp(s_, dados, nome, "Cheia", specs))
            _tic(f"spec {s_} concluída")

res = pd.concat(todos, ignore_index=True)
_tic("projeções locais concluídas")

# ---------------- EXPORTAÇÕES ----------------
cols_irf = ["amostra", "subgrupo", "spec", "var", "h", "coef", "se", "p",
            "lo95", "hi95", "lo90", "hi90", "n", "n_firmas", "inferencia", "lags_dk"]
res[cols_irf].to_csv(ROOT / f"irf_consolidado{SUFIXO}.csv", index=False)


def tabela_comparativa(df, var, a1="Total", a2="Restrita"):
    d = df[(df["var"] == var) & (df["subgrupo"] == "Cheia") & (df["amostra"].isin([a1, a2]))]
    if d.empty:
        return pd.DataFrame()
    w = d.pivot_table(index=["spec", "h"], columns="amostra", values=["coef", "p", "se", "n"])
    w.columns = [f"{m}_{a}" for m, a in w.columns]
    w = w.reset_index()
    ordem = ([f"{m}_{a}" for a in (a1, a2) for m in ("coef", "p")] +
             [f"{m}_{a}" for m in ("se", "n") for a in (a1, a2)])
    ordem = [c for c in ordem if c in w.columns]
    if f"coef_{a1}" in w.columns and f"coef_{a2}" in w.columns:
        w["dif_coef"] = w[f"coef_{a2}"] - w[f"coef_{a1}"]
        ordem.append("dif_coef")
    w.insert(0, "var", var)
    return w[["var", "spec", "h"] + ordem]


comp = pd.concat([tabela_comparativa(res, v)
                  for v in ("Ch_Caixa", "Ch_Tam", "Ch_FCO", "Shock")], ignore_index=True)
comp.to_csv(ROOT / f"comparacao_total_vs_restrita{SUFIXO}.csv", index=False)

print("\n===== Ch_Caixa: Total (sem Q) vs. Restrita (com Q) =====")
cc = comp[comp["var"] == "Ch_Caixa"] if not comp.empty else comp
print(cc.drop(columns="var").round(5).to_string(index=False) if len(cc)
      else "(Ch_Caixa não estimado)")

for nome, df in [("BASE (β1)", res[(res["spec"] == "base") & (res["subgrupo"] == "Cheia")]),
                 ("CTRL_INT", res[res["spec"] == "ctrl_int"]),
                 ("CTRL_INT_L", res[res["spec"] == "ctrl_int_L"]),
                 ("SETOR_T", res[res["spec"] == "setor_t"]),
                 ("SETOR_T_L", res[res["spec"] == "setor_t_L"])]:
    print(f"\n===== {nome} - {DESFECHO}: coeficientes =====")
    if df.empty:
        print("(spec não rodada ou sem horizontes estimados)")
        continue
    print(df.pivot_table(index="h", columns=["amostra", "var"], values="coef").round(5).to_string())
    print(f"===== {nome}: p-valores =====")
    print(df.pivot_table(index="h", columns=["amostra", "var"], values="p").round(4).to_string())


# ---------------- TESTE AUTOMÁTICO DE PRÉ-TENDÊNCIAS (h = -4, -3, -2) ----------------
def teste_pre_tendencia(df, rotulo, vars_teste):
    """β deve ser estatisticamente nulo antes do choque."""
    pre = df[df["h"].isin(H_PRE) & df["var"].isin(vars_teste)].copy()
    print(f"\n=== Pré-tendências [{rotulo}] - {DESFECHO} (h = {list(H_PRE)}) ===")
    if pre.empty:
        print("Nenhum horizonte pré-choque estimado (spec não rodada, ou verifique H_MIN e MIN_OBS).")
        pre["rejeita_5pct"] = pd.Series(dtype=bool)
        return pre
    pre["rejeita_5pct"] = pre["p"] < 0.05
    print(pre[["var", "h", "coef", "se", "p", "n", "rejeita_5pct"]].round(5).to_string(index=False))
    n_rej = int(pre["rejeita_5pct"].sum())
    print(f"Rejeições a 5%: {n_rej} de {len(pre)} testes "
          f"(esperado ao acaso sob H0: ~{0.05 * len(pre):.1f}).")
    return pre


VARS_CTRL_INT = ["Shock", "Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO", "Ch_Q"]
VARS_L = VARS_CTRL_INT + ["Shock_L1", "Shock_L2"] + \
         [f"{v}_{l}" for v in ("Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO") for l in ("L1", "L2")]
pre_all = []
for nome in AMOSTRAS:
    cheia = res[(res["amostra"] == nome) & (res["subgrupo"] == "Cheia")]
    pre_all.append(teste_pre_tendencia(cheia[cheia["spec"] == "base"],
                                       f"{nome} | base, Shock", ["Shock"])
                   .assign(amostra_teste=nome, spec_teste="base"))
    pre_all.append(teste_pre_tendencia(cheia[cheia["spec"] == "ctrl_int"],
                                       f"{nome} | ctrl_int", VARS_CTRL_INT)
                   .assign(amostra_teste=nome, spec_teste="ctrl_int"))
    if "ctrl_int_L" in RODAR_SPECS:
        pre_all.append(teste_pre_tendencia(cheia[cheia["spec"] == "ctrl_int_L"],
                                           f"{nome} | ctrl_int_L", VARS_L)
                       .assign(amostra_teste=nome, spec_teste="ctrl_int_L"))
pd.concat(pre_all, ignore_index=True).to_csv(ROOT / f"pre_tendencias{SUFIXO}.csv", index=False)

# ---------------- GRÁFICOS (opcional) ----------------
if RODAR_GRAFICOS:
    CORES = {"Total": "C0", "Restrita": "C3", "Restrita_semQ": "C7"}
    NOMES = {"Shock": "β1 (Shock)", "Ch_Caixa": "Shock × Caixa",
             "Ch_Alav": "Shock × Alav(t-1)", "Ch_Q": "Shock × Q(t-1)",
             "Ch_Tam": "Shock × Tamanho", "Ch_FCO": "Shock × FCO(t-1)"}

    def desenha(ax, s, label, cor, ls="-"):
        s = s.sort_values("h")
        if s.empty:
            return
        ax.plot(s["h"], s["coef"], marker="o", color=cor, ls=ls, label=label)
        ax.fill_between(s["h"], s["lo95"], s["hi95"], color=cor, alpha=0.12)

    def formata(ax, titulo):
        ax.axhline(0, color="gray", lw=0.8)
        ax.axvline(-0.5, color="gray", lw=0.8, ls=":")
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel("Horizonte (trimestres)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)

    def plot_comparativo(spec, vars_, arquivo):
        fig, axs = plt.subplots(1, len(vars_), figsize=(5.2 * len(vars_), 4), squeeze=False)
        for ax, v in zip(axs[0], vars_):
            for a, cor in CORES.items():
                s = res[(res["spec"] == spec) & (res["subgrupo"] == "Cheia")
                        & (res["amostra"] == a) & (res["var"] == v)]
                desenha(ax, s, a, cor, "--" if a == "Restrita_semQ" else "-")
            formata(ax, f"{NOMES[v]} - {DESFECHO}")
        plt.tight_layout(); plt.savefig(ROOT / arquivo, dpi=150); plt.close(fig)

    plot_comparativo("base", ["Shock"], f"lp_irf_base{SUFIXO}.png")
    plot_comparativo("ctrl_int", ["Shock", "Ch_Caixa", "Ch_Alav", "Ch_Q"],
                     f"lp_irf_interacoes{SUFIXO}.png")
    plot_comparativo("ctrl_int", ["Ch_Caixa", "Ch_Tam", "Ch_FCO"],
                     f"lp_irf_interacoes_extra{SUFIXO}.png")
    if "ctrl_int_L" in RODAR_SPECS:
        plot_comparativo("ctrl_int_L", ["Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO"],
                         f"lp_irf_interacoes_L{SUFIXO}.png")
    if "setor_t" in RODAR_SPECS:
        plot_comparativo("setor_t", ["Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO", "Ch_Q"],
                         f"lp_irf_setor_t{SUFIXO}.png")
    if RODAR_SPLIT_ALAV:
        fig, axs = plt.subplots(1, len(AMOSTRAS), figsize=(5.2 * len(AMOSTRAS), 4), squeeze=False)
        for ax, a in zip(axs[0], AMOSTRAS):
            for g, cor in [("Baixa", "C2"), ("Alta", "C3")]:
                s = res[(res["spec"] == "base") & (res["amostra"] == a)
                        & (res["subgrupo"] == g) & (res["var"] == "Shock")]
                desenha(ax, s, f"{g} alavancagem", cor)
            formata(ax, f"β1 por alavancagem - amostra {a}")
        plt.tight_layout(); plt.savefig(ROOT / f"lp_irf_split_alav{SUFIXO}.png", dpi=150); plt.close(fig)
    _tic("gráficos concluídos")

print(f"\nSalvos: irf_consolidado{SUFIXO}.csv, comparacao_total_vs_restrita{SUFIXO}.csv, "
      f"diagnostico_selecao{SUFIXO}.csv, pre_tendencias{SUFIXO}.csv")

# ---------------- ENTRANTES (coluna usada pelo checklist; sempre calculada) ----------------
ENTR_JAN = ("2019Q1", "2021Q4")
_e0 = pd.Period(ENTR_JAN[0], freq="Q").ordinal
_e1 = pd.Period(ENTR_JAN[1], freq="Q").ordinal
prim = p.loc[p["log_Ativo_Fixo"].notna()].groupby("CNPJ_CIA")["t"].min().rename("t_prim")
p = (p.drop(columns=["t_prim", "entrante"], errors="ignore")
       .merge(prim, left_on="CNPJ_CIA", right_index=True, how="left"))
p["entrante"] = p["t_prim"].between(_e0, _e1)

# =====================================================================================
# BLOCOS EXTRAS (diagnóstico; desligados por padrão). Ligue com RODAR_EXTRAS = True
# =====================================================================================
if RODAR_EXTRAS:
    # ---------- VIF (spec ctrl_int) ----------
    def calcula_vif(X):
        X = X.assign(const=1.0)
        cols = list(X.columns)
        return pd.Series([variance_inflation_factor(X.values, i) for i in range(len(cols))],
                         index=cols).drop("const")

    vifs = []
    for nome in ["Total"]:
        x = monta_specs(AMOSTRAS[nome]["com_q"])["ctrl_int"]["x"]
        Xv = dados_da_amostra(nome)[x + ["CNPJ_CIA"]].dropna()
        Xw = Xv[x] - Xv.groupby("CNPJ_CIA")[x].transform("mean")
        vifs.append(pd.DataFrame({"amostra": nome, "VIF_nivel": calcula_vif(Xv[x]),
                                  "VIF_within_firma": calcula_vif(Xw)}))
        print(f"===== VIF (ctrl_int, amostra {nome}, n = {len(Xv)}) =====")
        print(vifs[-1][["VIF_nivel", "VIF_within_firma"]].round(3).to_string())
    pd.concat(vifs).to_csv(ROOT / f"vif_ctrl_int{SUFIXO}.csv")

    # ---------- INFLUÊNCIA POR ANO / TRIMESTRE (FWL) ----------
    INFL_SPECS = ["base", "ctrl_int"]
    INFL_VARS = {"base": ["Shock"],
                 "ctrl_int": ["Shock", "Ch_Alav", "Ch_FCO", "Ch_Tam", "Ch_Caixa"]}
    INFL_H = [2, 4, 6, 8]
    specs_infl = monta_specs(False)

    def fwl(d, x, k):
        """Contribuição de cada observação ao coeficiente de k (a soma é o coeficiente)."""
        g = d.groupby("CNPJ_CIA")
        W = d[x] - g[x].transform("mean")
        yw = d["y"] - g["y"].transform("mean")
        outros = [c for c in x if c != k]
        if outros:
            gam = np.linalg.lstsq(W[outros].values, W[k].values, rcond=None)[0]
            xt = W[k].values - W[outros].values @ gam
        else:
            xt = W[k].values
        return xt * yw.values / (xt ** 2).sum()

    linhas_ano, linhas_tri = [], []
    for spec_nome in INFL_SPECS:
        spec = specs_infl[spec_nome]
        x = spec["x"]
        for h in INFL_H:
            d = base_h(p, spec, h)
            shock_ano = d.groupby("ano")["Shock"].mean()
            n_ano = d.groupby("ano").size()
            for k in INFL_VARS[spec_nome]:
                c = fwl(d, x, k)
                beta = c.sum()
                ref = res[(res["spec"] == spec_nome) & (res["h"] == h) & (res["var"] == k)
                          & (res["subgrupo"] == "Cheia")]["coef"]
                dif = abs(beta - ref.iloc[0]) if len(ref) else np.nan
                cs = pd.Series(c, index=d.index).groupby(d["ano"]).sum()
                mesmo = cs[np.sign(cs) == np.sign(beta)].sum()
                for g_ in cs.index:
                    beta_sem = fwl(d[d["ano"] != g_], x, k).sum()
                    linhas_ano.append({
                        "spec": spec_nome, "h": h, "var": k, "ano": int(g_),
                        "coef_total": beta, "contrib": cs[g_],
                        "parcela_mesmo_sinal": (cs[g_] / mesmo) if np.sign(cs[g_]) == np.sign(beta) else 0.0,
                        "coef_sem_ano": beta_sem, "dif_vs_total": beta_sem - beta,
                        "n_obs_ano": int(n_ano[g_]), "choque_medio_ano": shock_ano[g_],
                        "checa_vs_PanelOLS": dif})
                if spec_nome == "base" and k == "Shock":
                    ct = pd.Series(c, index=d.index).groupby(d["t"]).sum()
                    for t_, v in ct.items():
                        linhas_tri.append({"h": h, "trimestre": str(pd.Period(ordinal=int(t_), freq="Q")),
                                           "contrib": v, "coef_total": beta})

    infl = pd.DataFrame(linhas_ano)
    infl_tri = pd.DataFrame(linhas_tri)
    infl.to_csv(ROOT / f"influencia_ano{SUFIXO}.csv", index=False)
    infl_tri.to_csv(ROOT / f"influencia_trimestre{SUFIXO}.csv", index=False)

    print("\n\n################ INFLUÊNCIA POR ANO ################")
    print(f"Checagem (decomposição vs PanelOLS): dif. máxima = {infl['checa_vs_PanelOLS'].max():.2e} "
          "(deve ser ~0)")
    n_anos = infl.groupby(["spec", "h", "var"])["ano"].nunique().iloc[0]
    print(f"Referência: com {n_anos} anos, uma contribuição uniforme seria ~{100 / n_anos:.1f}% por ano.")

    sb = infl[(infl["spec"] == "base") & (infl["var"] == "Shock")]
    print("\n--- Shock (base): contribuição de cada ano ao coeficiente ---")
    tab = sb.pivot_table(index="ano", columns="h", values="contrib")
    tab.loc["TOTAL"] = tab.sum()
    print(tab.round(4).to_string())
    print("\n--- Shock (base): parcela do ano nas contribuições de MESMO SINAL (%) ---")
    print((100 * sb.pivot_table(index="ano", columns="h", values="parcela_mesmo_sinal")).round(1).to_string())
    print("\n--- Shock (base): coeficiente re-estimado SEM cada ano ---")
    tab2 = sb.pivot_table(index="ano", columns="h", values="coef_sem_ano")
    tab2.loc["TOTAL (todos os anos)"] = sb.groupby("h")["coef_total"].first()
    print(tab2.round(4).to_string())

    print("\n--- ctrl_int: ano de maior influência em cada coeficiente ---")
    linhas = []
    for (spec_, h, var), g in infl[infl["spec"] == "ctrl_int"].groupby(["spec", "h", "var"]):
        top = g.loc[g["parcela_mesmo_sinal"].idxmax()]
        linhas.append({"var": var, "h": h, "coef_total": top["coef_total"], "ano_top": int(top["ano"]),
                       "parcela_%": 100 * top["parcela_mesmo_sinal"], "coef_sem_ano_top": top["coef_sem_ano"]})
    print(pd.DataFrame(linhas).sort_values(["var", "h"]).round(3).to_string(index=False))

    print("\n--- Shock (base): 3 trimestres com maior |contribuição| por horizonte ---")
    for h, g in infl_tri.groupby("h"):
        t3 = g.reindex(g["contrib"].abs().sort_values(ascending=False).index).head(3)
        print(f"h={h} (coef total {g['coef_total'].iloc[0]:.4f}): "
              + " | ".join(f"{r.trimestre}: {r.contrib:+.4f}" for r in t3.itertuples()))
    _tic("influência concluída")

    # ---------- ESCUDO v2: Ch_Caixa por ano (anos bem identificados) ----------
    HET_H = [-4, -3, -2, 0, 1, 2, 4, 6, 8]
    LIM_ID = 2.6
    spec_h = monta_specs(False)["ctrl_int"]

    tri_ch = p.groupby("t")["Shock"].first().dropna()
    ss_ano = (tri_ch ** 2).groupby(ano_de_t(tri_ch.index.values)).sum()

    linhas_het = []
    for h in HET_H:
        d = base_h(p, spec_h, h)
        anos = sorted(d["ano"].unique())
        cols = []
        for a in anos:
            c_ = f"Ch_Caixa_{a}"
            d[c_] = d["Ch_Caixa"] * (d["ano"] == a)
            cols.append(c_)
        x = [c for c in spec_h["x"] if c != "Ch_Caixa"] + cols
        n_t = d["t"].nunique()
        sd_y = d["y"].std()
        d = d.set_index(["CNPJ_CIA", "t"])
        r = PanelOLS(d["y"], d[x], entity_effects=True).fit(**cov_kwargs(h, n_t))
        for a, c_ in zip(anos, cols):
            linhas_het.append({"h": h, "ano": int(a), "coef": r.params[c_], "se": r.std_errors[c_],
                               "p": r.pvalues[c_], "sd_y": sd_y})

    het = pd.DataFrame(linhas_het)
    het["ss_ano"] = het["ano"].map(ss_ano)
    het["bem_id"] = het["ss_ano"] >= LIM_ID
    het["coef_pad"] = het["coef"] / het["sd_y"]
    het.to_csv(ROOT / f"escudo_por_ano_v2{SUFIXO}.csv", index=False)

    print("\n\n################ ESCUDO DE LIQUIDEZ POR ANO (v2) ################")
    print("\nIdentificação por ano (soma dos quadrados do choque):")
    print(ss_ano.round(4).to_string())
    print(f"Anos bem identificados (>= {LIM_ID}): {sorted(het.loc[het['bem_id'], 'ano'].unique())}")

    bi = het[het["bem_id"]]
    print("\n--- Coeficiente de Ch_Caixa por ano (só bem identificados; h<0 = placebo) ---")
    print(bi.pivot_table(index="ano", columns="h", values="coef").round(3).to_string())

    gb = bi.groupby("h")
    resumo = pd.DataFrame({
        "n_anos": gb.size(),
        "dp_coef": gb["coef"].std(),
        "dp_coef_pad": gb["coef_pad"].std(),
        "dp_pad_sem2021": bi[bi["ano"] != 2021].groupby("h")["coef_pad"].std(),
        "pct_p_menor_5": gb["p"].apply(lambda s: 100 * (s < 0.05).mean()),
    })
    ref = resumo.loc[resumo.index < 0, "dp_coef_pad"].mean()
    resumo["razao_vs_placebo"] = resumo["dp_coef_pad"] / ref
    print("\n--- Dispersão entre anos ---")
    print(resumo.round(3).to_string())
    print(f"Placebo (média de dp_coef_pad em h<0) = {ref:.4f}")
    _tic("escudo v2 concluído")

    # ---------- ESCUDO v3: com vs sem controle de crescimento pré-choque ----------
    HET_H3 = [-3, -2, 0, 1, 2, 4, 6, 8]

    cr = (piv_log_af.shift(1, axis=1) - piv_log_af.shift(5, axis=1)).stack().rename("Cresc_pre").reset_index()
    p = p.drop(columns=[c for c in ("Cresc_pre", "Cresc_pre_z", "Ch_Cresc") if c in p.columns]) \
         .merge(cr, on=["CNPJ_CIA", "t"], how="left")
    if AJUSTE_DIV:
        ok_div = p["Alavancagem_L1"].notna()      # recalcula: o merge recriou o índice
        p["Cresc_pre"] = winsor_m(p["Cresc_pre"], ok_div)
        p["Cresc_pre_z"] = zscore_m(p, "Cresc_pre", ok_div)
    else:
        p["Cresc_pre"] = winsor(p["Cresc_pre"])
        p["Cresc_pre_z"] = p.groupby("t")["Cresc_pre"].transform(zscore)
    p["Ch_Cresc"] = p["Shock"] * p["Cresc_pre_z"]

    spec3 = monta_specs(False)["ctrl_int"]
    x_sem = spec3["x"]
    x_com = x_sem + ["Cresc_pre_z", "Ch_Cresc"]

    def ajusta(di, x, h, n_t):
        return PanelOLS(di["y"], di[x], entity_effects=True).fit(**cov_kwargs(h, n_t))

    lin_pool, lin_ano = [], []
    for h in HET_H3:
        d = base_h(p, {"x": x_com, "req": spec3.get("req", [])}, h)
        n_t, sd_y = d["t"].nunique(), d["y"].std()
        anos = sorted(d["ano"].unique())
        for a in anos:
            d[f"Ch_Caixa_{a}"] = d["Ch_Caixa"] * (d["ano"] == a)
        cols = [f"Ch_Caixa_{a}" for a in anos]
        di = d.set_index(["CNPJ_CIA", "t"])
        for rot, xb in (("sem_ctrl", x_sem), ("com_ctrl", x_com)):
            r0 = ajusta(di, xb, h, n_t)
            lin_pool.append({"h": h, "controle": rot, "coef": r0.params["Ch_Caixa"],
                             "p": r0.pvalues["Ch_Caixa"], "n": int(r0.nobs)})
            r1 = ajusta(di, [c for c in xb if c != "Ch_Caixa"] + cols, h, n_t)
            for a, c_ in zip(anos, cols):
                lin_ano.append({"h": h, "controle": rot, "ano": int(a), "coef": r1.params[c_],
                                "p": r1.pvalues[c_], "sd_y": sd_y})

    pool3 = pd.DataFrame(lin_pool)
    ya = pd.DataFrame(lin_ano)
    ya = ya[ya["ano"].map(lambda a: ss_ano.get(a, 0) >= LIM_ID)].copy()
    ya["coef_pad"] = ya["coef"] / ya["sd_y"]
    pool3.to_csv(ROOT / f"escudo_v3_agrupado{SUFIXO}.csv", index=False)
    ya.to_csv(ROOT / f"escudo_v3_por_ano{SUFIXO}.csv", index=False)

    print("\n\n################ ESCUDO v3: COM vs SEM controle de crescimento pré-choque ################")
    print(pool3.pivot_table(index="h", columns="controle", values=["coef", "p"]).round(4).to_string())
    print(f"n da amostra comum por horizonte: {pool3.groupby('h')['n'].first().to_dict()}")
    print("\n--- Coeficiente por ano COM controle ---")
    print(ya[ya["controle"] == "com_ctrl"].pivot_table(index="ano", columns="h", values="coef").round(3).to_string())

    res3 = []
    for rot, g in ya.groupby("controle"):
        s = g.groupby("h")["coef_pad"].std()
        s21 = g[g["ano"] != 2021].groupby("h")["coef_pad"].std()
        res3.append(pd.DataFrame({"controle": rot, "dp_pad": s, "razao_vs_placebo": s / s[s.index < 0].mean(),
                                  "dp_pad_sem2021": s21,
                                  "razao_sem2021": s21 / s21[s21.index < 0].mean()}))
    print("\n--- Dispersão padronizada entre anos ---")
    print(pd.concat(res3).round(3).to_string())

    esp = ya.pivot_table(index=["controle", "ano"], columns="h", values="coef")
    pre, post = esp[[-3, -2]].mean(axis=1), esp[[1, 2, 4]].mean(axis=1)
    espelho = pd.DataFrame({"pre(h=-3,-2)": pre, "post(h=1,2,4)": post})
    espelho["sinais_opostos"] = (pre * post) < 0
    print("\n--- Espelho pré/pós por ano ---")
    print(espelho.round(3).to_string())
    print(espelho.groupby("controle")["sinais_opostos"].agg(["sum", "count"]).to_string())
    _tic("escudo v3 concluído")

    # ---------- ENTRANTES: diagnóstico e escudo todas vs sem entrantes ----------
    ini = p[(p["t"] - p["t_prim"]).between(0, 3)]
    por_firma = ini.groupby("CNPJ_CIA").agg(cash=("Cash_Ratio", "mean"), alav=("Alavancagem", "mean"))
    por_firma["ano_prim"] = ano_de_t(prim.reindex(por_firma.index).values)
    diag = por_firma.groupby("ano_prim").agg(n_firmas=("cash", "size"), cash_mediana=("cash", "median"),
                                             alav_mediana=("alav", "median"))
    print("\n\n################ ENTRANTES (IPOs/SPACs 2020-21) ################")
    print(f"\nCash_Ratio mediano de todas as observações: {p['Cash_Ratio'].median():.4f}")
    print(diag.round(4).to_string())
    o21 = p[ano_de_t(p["t"].values) == 2021]
    print(f"\nEntrantes ({ENTR_JAN[0]} a {ENTR_JAN[1]}): {p.loc[p['entrante'], 'CNPJ_CIA'].nunique()} firmas; "
          f"{100 * o21['entrante'].mean():.1f}% das observações de 2021.")
    n67 = p.loc[p["SETOR_ATIV"].astype(str).str.upper() == "SIC67", "CNPJ_CIA"].nunique()
    print(f"Firmas com SETOR_ATIV = SIC67: {n67}")

    def escudo_ano(dados, rot):
        lin_p, lin_a = [], []
        for h in HET_H:
            d = base_h(dados, spec_h, h)
            anos = sorted(d["ano"].unique())
            for a in anos:
                d[f"Ch_Caixa_{a}"] = d["Ch_Caixa"] * (d["ano"] == a)
            cols = [f"Ch_Caixa_{a}" for a in anos]
            n_t, sd_y = d["t"].nunique(), d["y"].std()
            di = d.set_index(["CNPJ_CIA", "t"])
            r0 = PanelOLS(di["y"], di[spec_h["x"]], entity_effects=True).fit(**cov_kwargs(h, n_t))
            x1 = [c for c in spec_h["x"] if c != "Ch_Caixa"] + cols
            r1 = PanelOLS(di["y"], di[x1], entity_effects=True).fit(**cov_kwargs(h, n_t))
            lin_p.append({"amostra": rot, "h": h, "coef": r0.params["Ch_Caixa"],
                          "p": r0.pvalues["Ch_Caixa"], "n": int(r0.nobs)})
            for a, c_ in zip(anos, cols):
                lin_a.append({"amostra": rot, "h": h, "ano": int(a), "coef": r1.params[c_],
                              "p": r1.pvalues[c_], "sd_y": sd_y})
        return pd.DataFrame(lin_p), pd.DataFrame(lin_a)

    p_t, a_t = escudo_ano(p, "todas")
    p_s, a_s = escudo_ano(p[~p["entrante"]], "sem_entrantes")
    pool_e = pd.concat([p_t, p_s], ignore_index=True)
    ano_e = pd.concat([a_t, a_s], ignore_index=True)
    ano_e = ano_e[ano_e["ano"].map(lambda a: ss_ano.get(a, 0) >= LIM_ID)].copy()
    ano_e["coef_pad"] = ano_e["coef"] / ano_e["sd_y"]
    pool_e.to_csv(ROOT / f"escudo_entrantes_agrupado{SUFIXO}.csv", index=False)
    ano_e.to_csv(ROOT / f"escudo_entrantes_por_ano{SUFIXO}.csv", index=False)

    print("\n--- Ch_Caixa agrupado: todas vs sem entrantes ---")
    print(pool_e.pivot_table(index="h", columns="amostra", values=["coef", "p", "n"]).round(4).to_string())
    for rot, g in ano_e.groupby("amostra"):
        print(f"\n--- Ch_Caixa por ano: {rot} (h<0 = placebo) ---")
        print(g.pivot_table(index="ano", columns="h", values="coef").round(3).to_string())
    disp = []
    for rot, g in ano_e.groupby("amostra"):
        s = g.groupby("h")["coef_pad"].std()
        disp.append(pd.DataFrame({"amostra": rot, "dp_pad": s, "razao_vs_placebo": s / s[s.index < 0].mean()}))
    print("\n--- Dispersão padronizada entre anos ---")
    print(pd.concat(disp).round(3).to_string())
    _tic("entrantes concluído")

# ---------------- CHECKLIST (concentração, episódios, bootstrap) ----------------
exec(open(ROOT / "bloco_checklist.py", encoding="utf-8").read())
_tic("checklist concluído")