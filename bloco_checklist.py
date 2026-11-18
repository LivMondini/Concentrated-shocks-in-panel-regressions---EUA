# pyright: reportUndefinedVariable=false
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if "base_h" not in globals():
    raise SystemExit("Este bloco não roda sozinho. Rode projecoes_locais.py "
                     "(com o exec no final dele).")

# =====================================================================================
# BLOCO CHECKLIST DO ARTIGO (versão EUA)
# Como usar: salve este arquivo na mesma pasta do projecoes_locais.py e, no FINAL dele, acrescente
#     exec(open(ROOT / "bloco_checklist.py", encoding="utf-8").read())
# Ele reaproveita p, base_h, monta_specs, cov_kwargs, PanelOLS, SUFIXO etc. já definidos no script.
#
# Passo 1  Concentração do choque: participação de cada trimestre/ano/episódio e nº efetivo de trimestres.
# Passo 2  Gradiente de caixa (Ch_Caixa) por episódio, e excluindo cada episódio (inclusive a
#          versão "estrita", que também remove observações cuja janela [t-1, t+h] toca o episódio).
# Passo 3  Bootstrap em blocos móveis sobre trimestres (blocos de comprimento h+1), com teste de
#          igualdade entre os dois episódios.
#
# Especificação: a do artigo (ctrl_int, com Shock_L1, sem Shock_L2 nem interações defasadas),
# independentemente de AMOSTRA_COMUM. Horizontes: 3, 5 e 7.
#
# ATENÇÃO sobre os episódios: as janelas abaixo seguem a cronologia dos ciclos de alta do Fed
# (dez/2015-dez/2018 e mar/2022-jul/2023), não os coeficientes. Mas você já viu resultados por
# ano, então declare essas análises como exploratórias, como o artigo faz com 2013-15.
# =====================================================================================
import math

CK_N_BOOT = 1000                 # sorteios do bootstrap (o artigo usa 1.000)
CK_SEED = 20261024
CK_H = (3, 5, 7)
CK_MIN_Q = 3    # trimestres distintos mínimos de cada episódio por sorteio
CK_EPIS = {"A": ("2015Q4", "2018Q4"), "B": ("2022Q1", "2023Q3")}
CK_SEM_CHOQUE = ("2020Q1", "2020Q4")   # trimestres sem MPS_ORTH na fonte


def _q(s):
    return pd.Period(s, freq="Q").ordinal


CK_EP = {k: (_q(a), _q(b)) for k, (a, b) in CK_EPIS.items()}
CK_2020 = (_q(CK_SEM_CHOQUE[0]), _q(CK_SEM_CHOQUE[1]))

spec_art = dict(monta_specs(False)["ctrl_int"])
spec_art["req"] = []                       # não exige Shock_L2 nem interações defasadas
CK_X = spec_art["x"]
CK_J = CK_X.index("Ch_Caixa")


def _lbl(t):
    return str(pd.Period(ordinal=int(t), freq="Q"))


def _dentro(t, ep):
    return (t >= ep[0]) & (t <= ep[1])


def _toca(t, h, ep):
    """Janela do desfecho [t-1, t+h] intersecta o episódio."""
    return ((t - 1) <= ep[1]) & ((t + h) >= ep[0])


def wls_fe(X, y, firm, rw, n_firm):
    """OLS com efeito fixo de firma e pesos por linha rw (equivale a duplicar linhas)."""
    sw = np.bincount(firm, weights=rw, minlength=n_firm)
    sw[sw == 0] = 1.0
    yd = y - (np.bincount(firm, weights=rw * y, minlength=n_firm) / sw)[firm]
    Xd = np.empty_like(X)
    for j in range(X.shape[1]):
        Xd[:, j] = X[:, j] - (np.bincount(firm, weights=rw * X[:, j], minlength=n_firm) / sw)[firm]
    XtW = Xd.T * rw
    try:
        return np.linalg.solve(XtW @ Xd, XtW @ yd)
    except np.linalg.LinAlgError:
        return np.full(X.shape[1], np.nan)


def sorteio_blocos(T, L, rng):
    """Blocos móveis de comprimento L sobre T trimestres ordenados; devolve T índices."""
    L = max(1, min(L, T))
    nb = int(math.ceil(T / L))
    ini = rng.integers(0, T - L + 1, size=nb)
    return (ini[:, None] + np.arange(L)[None, :]).ravel()[:T]


def p_normal(z):
    return math.erfc(abs(z) / math.sqrt(2))


# ---- FIM DAS FUNÇÕES AUXILIARES ----

# ------------------------------------------------------------------ PASSO 1
print("\n\n################ CHECKLIST 1: CONCENTRAÇÃO DO CHOQUE ################")
tri_ck = p.groupby("t")["Shock"].first().dropna()
dev2 = (tri_ck - tri_ck.mean()) ** 2
share = dev2 / dev2.sum()
n_eff = 1.0 / (share ** 2).sum()
print(f"Trimestres com choque: {len(tri_ck)} | desvio-padrão (100 pb): {tri_ck.std():.4f}")
print(f"Número efetivo de trimestres: {n_eff:.1f} de {len(tri_ck)} "
      f"(artigo, Brasil: 9,8 de 54)")

por_ano = 100 * share.groupby(ano_de_t(share.index.values)).sum()
print("\nParticipação de cada ano na variação quadrática do choque (%):")
print(por_ano.round(1).to_string())
print("\nCinco trimestres de maior participação (%):")
top = share.nlargest(5)
print(pd.DataFrame({"trimestre": [_lbl(t) for t in top.index],
                    "choque": tri_ck.loc[top.index].round(4).values,
                    "participacao_%": (100 * top.values).round(1)}).to_string(index=False))
linhas_conc = []
for k, ep in CK_EP.items():
    m = _dentro(share.index.values, ep)
    print(f"Episódio {k} ({CK_EPIS[k][0]}-{CK_EPIS[k][1]}): {m.sum()} trimestres, "
          f"{100 * share[m].sum():.1f}% da variação do choque")
    linhas_conc.append({"item": f"episodio_{k}", "n_trim": int(m.sum()),
                        "participacao_pct": 100 * share[m].sum()})
linhas_conc.append({"item": "n_efetivo", "n_trim": len(tri_ck), "participacao_pct": n_eff})
pd.DataFrame(linhas_conc).to_csv(ROOT / f"checklist_concentracao{SUFIXO}.csv", index=False)

# ------------------------------------------------------------------ PASSO 2
print("\n\n################ CHECKLIST 2: EPISÓDIOS E EXCLUSÕES (spec do artigo) ################")
print("Erros-padrão: Driscoll-Kraay. Em subamostras curtas (episódio B tem ~7 trimestres) "
      "o DK é pouco confiável; use o bootstrap (passo 3) como referência.")
EA, EB = CK_EP["A"], CK_EP["B"]
SUBAMOSTRAS = {
    "Pooled":          lambda t, h: np.ones(len(t), bool),
    "Episodio_A":      lambda t, h: _dentro(t, EA),
    "Episodio_B":      lambda t, h: _dentro(t, EB),
    "Calmos":          lambda t, h: ~_dentro(t, EA) & ~_dentro(t, EB),
    "Sem_A":           lambda t, h: ~_dentro(t, EA),
    "Sem_B":           lambda t, h: ~_dentro(t, EB),
    "Sem_A_e_B":       lambda t, h: ~_dentro(t, EA) & ~_dentro(t, EB),
    "Sem_A_estrito":   lambda t, h: ~_toca(t, h, EA),
    "Sem_B_estrito":   lambda t, h: ~_toca(t, h, EB),
    "Sem_janela_2020": lambda t, h: ~_toca(t, h, CK_2020),
}


def reg_sub(h, nome):
    d = base_h(p, spec_art, h)
    t = d["t"].to_numpy()
    if nome == "Sem_entrantes":
        keep = ~d["entrante"].to_numpy(bool) if "entrante" in d.columns else np.ones(len(d), bool)
    else:
        keep = SUBAMOSTRAS[nome](t, h)
    d = d[keep]
    n_t = d["t"].nunique()
    if len(d) < MIN_OBS or n_t < CK_MIN_Q:
        return dict(coef=np.nan, se=np.nan, p=np.nan, n=len(d), n_trim=n_t)
    di = d.set_index(["CNPJ_CIA", "t"])
    try:
        r = PanelOLS(di["y"], di[CK_X], entity_effects=True).fit(**cov_kwargs(h, n_t))
        return dict(coef=r.params["Ch_Caixa"], se=r.std_errors["Ch_Caixa"],
                    p=r.pvalues["Ch_Caixa"], n=int(r.nobs), n_trim=n_t)
    except Exception as e:
        print(f"  [{nome}|h={h}] falhou: {e}")
        return dict(coef=np.nan, se=np.nan, p=np.nan, n=len(d), n_trim=n_t)


nomes_sub = list(SUBAMOSTRAS) + (["Sem_entrantes"] if "entrante" in p.columns else [])
lin_ep = []
for h in CK_H:
    for nome in nomes_sub:
        r = reg_sub(h, nome)
        lin_ep.append({"h": h, "subamostra": nome, **r})
ep_df = pd.DataFrame(lin_ep)

print("\n--- Ch_Caixa por subamostra (coef | se | p | N | trimestres) ---")
for h in CK_H:
    print(f"\nh = {h}")
    print(ep_df[ep_df["h"] == h].drop(columns="h").round(4).to_string(index=False))

dif_rows = []
for h in CK_H:
    a = ep_df[(ep_df.h == h) & (ep_df.subamostra == "Episodio_A")].iloc[0]
    b = ep_df[(ep_df.h == h) & (ep_df.subamostra == "Episodio_B")].iloc[0]
    se = math.sqrt(a.se ** 2 + b.se ** 2)
    z = (a.coef - b.coef) / se if se > 0 else np.nan
    dif_rows.append({"h": h, "coef_A": a.coef, "coef_B": b.coef, "dif_A_menos_B": a.coef - b.coef,
                     "se_dif": se, "p_normal": p_normal(z) if np.isfinite(z) else np.nan})
dif_df = pd.DataFrame(dif_rows)
print("\n--- Igualdade dos coeficientes dos dois episódios (aprox. normal; independência assumida) ---")
print(dif_df.round(4).to_string(index=False))
ep_df.to_csv(ROOT / f"checklist_episodios{SUFIXO}.csv", index=False)
dif_df.to_csv(ROOT / f"checklist_diferenca_episodios{SUFIXO}.csv", index=False)

# ------------------------------------------------------------------ PASSO 3
print("\n\n################ CHECKLIST 3: BOOTSTRAP EM BLOCOS SOBRE TRIMESTRES ################")
print(f"{CK_N_BOOT} sorteios; blocos de comprimento h+1; sorteios sem trimestre de algum episódio "
      "são descartados. Pode levar vários minutos.")
rng = np.random.default_rng(CK_SEED)
lin_boot = []
for h in CK_H:
    d = base_h(p, spec_art, h)
    X = d[CK_X].to_numpy(float)
    y = d["y"].to_numpy(float)
    firm = pd.factorize(d["CNPJ_CIA"])[0]
    n_firm = int(firm.max()) + 1
    tq = d["t"].to_numpy()
    qs = np.sort(np.unique(tq))
    T = len(qs)
    qidx = np.searchsorted(qs, tq)
    inA, inB = _dentro(tq, EA), _dentro(tq, EB)
    grupos = {"Pooled": np.ones(len(d), bool), "A": inA, "B": inB, "Calmos": ~inA & ~inB}
    G = {g: dict(X=X[m], y=y[m], f=firm[m], q=qidx[m]) for g, m in grupos.items()}
    qA = np.where(_dentro(qs, EA))[0]
    qB = np.where(_dentro(qs, EB))[0]
    if len(qA) < CK_MIN_Q or len(qB) < CK_MIN_Q:
        print(f"h={h}: algum episódio com menos de {CK_MIN_Q} trimestres na amostra; pulado.")
        continue

    est = {g: wls_fe(G[g]["X"], G[g]["y"], G[g]["f"], np.ones(len(G[g]["y"])), n_firm)[CK_J]
           for g in G}
    chk = ep_df[(ep_df.h == h) & (ep_df.subamostra == "Pooled")]["coef"].iloc[0]
    print(f"\nh={h}: T={T} trimestres, {n_firm} firmas. Estimativa pontual pooled = {est['Pooled']:.5f} "
          f"(PanelOLS: {chk:.5f}; devem coincidir)")

    L = h + 1
    draws = {g: [] for g in G}
    aceitos, tentativas = 0, 0
    while aceitos < CK_N_BOOT and tentativas < 20 * CK_N_BOOT:
        tentativas += 1
        cont = np.bincount(sorteio_blocos(T, L, rng), minlength=T).astype(float)
        if cont[qA].sum() < CK_MIN_Q or cont[qB].sum() < CK_MIN_Q:
            continue
        for g in G:
            rw = cont[G[g]["q"]]
            draws[g].append(wls_fe(G[g]["X"], G[g]["y"], G[g]["f"], rw, n_firm)[CK_J])
        aceitos += 1
        if aceitos % 100 == 0:
            print(f"   h={h}: {aceitos}/{CK_N_BOOT} sorteios aceitos ({tentativas} tentativas)")
    draws = {g: np.array(v) for g, v in draws.items()}
    print(f"   h={h}: {aceitos} sorteios aceitos de {tentativas} tentativas")

    def resume(nome, pt, v):
        v = v[np.isfinite(v)]
        lo, hi = np.percentile(v, [2.5, 97.5])
        pb = min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))
        return {"h": h, "alvo": nome, "estimativa": pt, "se_boot": v.std(ddof=1),
                "ic95_lo": lo, "ic95_hi": hi, "p_boot": pb, "n_sorteios": len(v)}

    for g, nome in [("Pooled", "Pooled"), ("A", "Episodio_A"), ("B", "Episodio_B"), ("Calmos", "Calmos")]:
        lin_boot.append(resume(nome, est[g], draws[g]))
    lin_boot.append(resume("Dif_A_menos_B", est["A"] - est["B"], draws["A"] - draws["B"]))

boot_df = pd.DataFrame(lin_boot)
print("\n--- Resultados do bootstrap em blocos (Ch_Caixa) ---")
print("p_boot: p-valor bilateral pelo percentil do bootstrap (aproximado).")
print(boot_df.round(4).to_string(index=False))
boot_df.to_csv(ROOT / f"checklist_bootstrap{SUFIXO}.csv", index=False)
print(f"\nSalvos: checklist_concentracao{SUFIXO}.csv, checklist_episodios{SUFIXO}.csv, "
      f"checklist_diferenca_episodios{SUFIXO}.csv e checklist_bootstrap{SUFIXO}.csv")