"""
monte_carlo_checklist.py
Simulação para o artigo: quando um coeficiente agregado de interação (choque comum x característica
da firma) é significativo, o que o checklist consegue detectar?

Desenho (simples de propósito):
  y_it = alpha_i + beta_t * S_t * c_it + e_it
  - S_t   : choque comum, com variância maior em episódios (concentração)
  - c_it  : característica da firma, padronizada dentro de cada trimestre (como Cash_Ratio_z)
  - beta_t: coeficiente do episódio + ruído trimestral (heterogeneidade não modelada)
  - alpha_i: efeito fixo de firma (removido por demeaning)
Cenários: nulo, canal estável, saldo de episódios com sinais opostos, um único episódio.

Estimação: aproxima o painel por somas trimestrais (após remover efeito fixo de firma):
  Sxy_t = sum_i x~ y~,  Sxx_t = sum_i x~^2.  O coeficiente agregado é sum Sxy / sum Sxx, e tudo
  (Driscoll-Kraay, bootstrap em blocos, episódios) é feito sobre essas somas. Isso é rápido.
Simplificações: o efeito fixo não é recalculado dentro de cada sorteio do bootstrap; não há horizonte h
nem sobreposição de janelas; c_it é independente entre firmas e trimestres.

Uso:  py monte_carlo_checklist.py            (R=500, ~1 minuto)
      py monte_carlo_checklist.py --R 2000 --B 500
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--R", type=int, default=500, help="replicações por cenário")
ap.add_argument("--B", type=int, default=300, help="sorteios do bootstrap por replicação")
ap.add_argument("--N", type=int, default=300, help="firmas")
ap.add_argument("--seed", type=int, default=20261024)
args = ap.parse_args()

T, L, NEED = 54, 4, 3                 # trimestres, banda DK = comprimento do bloco, mín. de trimestres/episódio
SIGMA_E, ETA = 2.0, 0.03              # ruído idiossincrático; ruído trimestral do coeficiente
EP = np.zeros(T, int)                 # 0 = calmo, 1 = episódio dominante, 2 = episódio oposto
EP[10:18] = 2                         # E2: 8 trimestres
EP[38:44] = 1                         # E1: 6 trimestres
SD_EP = np.array([1.0, 2.5, 1.5])     # desvio do choque em (calmo, E1, E2)
M1, M2 = EP == 1, EP == 2

CENARIOS = {                          # beta em (calmo, E1, E2)
    "Nulo (beta = 0)":             (0.00, 0.00, 0.00),
    "Estável (beta = 0,05)":       (0.05, 0.05, 0.05),
    "Saldo de episódios":          (0.03, 0.10, -0.08),
    "Um único episódio":           (0.00, 0.12, 0.00),
}


def p_norm(z):
    return math.erfc(abs(z) / math.sqrt(2))


def dk_se(g, S, band):
    n = len(g)
    band = min(band, n - 1)
    v = float(g @ g)
    for l in range(1, band + 1):
        v += 2 * (1 - l / (band + 1)) * float(g[l:] @ g[:-l])
    return math.sqrt(max(v, 0.0)) / S


def estima(Sxy, Sxx, mask):
    sxy, sxx = Sxy[mask], Sxx[mask]
    S = sxx.sum()
    b = sxy.sum() / S
    se = dk_se(sxy - b * sxx, S, L)
    return b, se


def p_boot_bilateral(v):
    return min(1.0, 2 * min((v <= 0).mean(), (v >= 0).mean()))


def uma_rep(rng, beta_ep):
    S = rng.standard_normal(T) * SD_EP[EP]
    beta_t = np.array(beta_ep)[EP] + ETA * rng.standard_normal(T)
    c = rng.standard_normal((args.N, T))
    c = (c - c.mean(0)) / c.std(0)                       # padroniza dentro do trimestre
    x = S * c
    y = rng.standard_normal((args.N, 1)) + beta_t * x + SIGMA_E * rng.standard_normal((args.N, T))
    xd = x - x.mean(1, keepdims=True)
    yd = y - y.mean(1, keepdims=True)
    Sxy, Sxx = (xd * yd).sum(0), (xd * xd).sum(0)

    # --- Passo 1: concentração
    dev2 = (S - S.mean()) ** 2
    sh = dev2 / dev2.sum()
    n_eff = 1.0 / float((sh ** 2).sum())

    # --- Passo 2: agregado, episódios, exclusão do episódio dominante
    b, se = estima(Sxy, Sxx, np.ones(T, bool))
    b1, se1 = estima(Sxy, Sxx, M1)
    b2, se2 = estima(Sxy, Sxx, M2)
    bx, sex = estima(Sxy, Sxx, ~M1)
    p_dk = p_norm(b / se)
    p_dif_n = p_norm((b1 - b2) / math.sqrt(se1 ** 2 + se2 ** 2))
    p_sem1 = p_norm(bx / sex)

    # --- Passo 3: bootstrap em blocos móveis sobre trimestres (descarta sorteios sem E1 ou E2)
    nb = math.ceil(T / L)
    starts = rng.integers(0, T - L + 1, size=(args.B, nb))
    idx = (starts[:, :, None] + np.arange(L)).reshape(args.B, -1)[:, :T]
    cnt = (idx[:, :, None] == np.arange(T)).sum(1).astype(float)
    ok = (cnt[:, M1].sum(1) >= NEED) & (cnt[:, M2].sum(1) >= NEED)
    cnt = cnt[ok]
    bp = (cnt @ Sxy) / (cnt @ Sxx)
    b1s = (cnt[:, M1] @ Sxy[M1]) / (cnt[:, M1] @ Sxx[M1])
    b2s = (cnt[:, M2] @ Sxy[M2]) / (cnt[:, M2] @ Sxx[M2])
    p_bt = p_boot_bilateral(bp)
    p_dif_b = p_boot_bilateral(b1s - b2s)

    return dict(n_eff=n_eff, sh_E1=sh[M1].sum(), b=b, b1=b1, b2=b2,
                sig_dk=p_dk < 0.05, sig_boot=p_bt < 0.05,
                dif_normal=p_dif_n < 0.05, dif_boot=p_dif_b < 0.05,
                sinais_opostos=(b1 * b2) < 0,
                perde_sem_E1=(p_dk < 0.05) and (p_sem1 >= 0.05))


def main():
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    linhas = []
    for nome, beta_ep in CENARIOS.items():
        reps = pd.DataFrame([uma_rep(rng, beta_ep) for _ in range(args.R)])
        sig = reps["sig_dk"]
        linhas.append({
            "cenario": nome,
            "beta_agregado_verdadeiro": float(np.sum(np.array(beta_ep)[EP] * SD_EP[EP] ** 2) / np.sum(SD_EP[EP] ** 2)),
            "n_efetivo_trimestres": reps["n_eff"].mean(),
            "part_E1_%": 100 * reps["sh_E1"].mean(),
            "agregado_signif_DK_%": 100 * sig.mean(),
            "agregado_signif_boot_%": 100 * reps["sig_boot"].mean(),
            "sinais_opostos_%": 100 * reps["sinais_opostos"].mean(),
            "igualdade_rejeitada_normal_%": 100 * reps["dif_normal"].mean(),
            "igualdade_rejeitada_boot_%": 100 * reps["dif_boot"].mean(),
            "perde_signif_sem_E1_|signif_%": 100 * reps["perde_sem_E1"].sum() / max(sig.sum(), 1),
            # Condicionais a o agregado ser significativo (a situação que o artigo discute):
            "sinais_opostos_|signif_%": 100 * reps.loc[sig, "sinais_opostos"].mean() if sig.any() else np.nan,
            "igualdade_rejeitada_boot_|signif_%": 100 * reps.loc[sig, "dif_boot"].mean() if sig.any() else np.nan,
            "n_replicacoes_signif": int(sig.sum()),
        })
        print(f"  {nome}: ok ({time.time() - t0:.0f}s)")
    res = pd.DataFrame(linhas)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(f"\nN={args.N} firmas, T={T} trimestres, R={args.R} replicações, B={args.B} sorteios\n")
    print(res.round(3).T.to_string(header=False))
    resultados = ROOT / "saidas_revisao" / "monte_carlo_resultados.csv"
    resultados.parent.mkdir(parents=True, exist_ok=True)
    res.round(4).to_csv(resultados, index=False)
    print("\nSalvo: monte_carlo_resultados.csv")


if __name__ == "__main__":
    main()