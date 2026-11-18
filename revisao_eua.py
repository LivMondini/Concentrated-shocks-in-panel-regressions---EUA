"""Reproduce labelled US panel evidence without changing the estimation design."""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dados_tratados"
OUT = ROOT / "saidas_revisao"
SHOCK_FILE = DATA / "choque_fomc_trimestral.csv"
ROUNDS = {
    "obs_adj": (DATA / "painel_regressao_eua_amostra_robustez.csv", True),
    "obs_unadj": (DATA / "painel_regressao_eua_amostra_robustez.csv", False),
    "sem_med_adj": (DATA / "painel_regressao_eua_amostra_sem_mediana.csv", True),
    "mediana": (DATA / "painel_regressao_eua_amostra.csv", False),
}
ZERO_NOTE = "não reproduzível: não há painel/rotina identificável da rodada zero"
CHECKLIST_H = (3, 5, 7)
COMMON_H = (2, 4, 6, 8)
CONTROL = ["Cash_Ratio_z", "Alavancagem_L1_z", "Tamanho_z",
           "Fluxo_Caixa_Op_Ativo_L1_z"]
INTERACTIONS = ["Ch_Caixa", "Ch_Alav", "Ch_Tam", "Ch_FCO"]
REQUIRED_COMMON = ["Shock_L2", "Ch_Caixa_L1", "Ch_Alav_L1", "Ch_Tam_L1", "Ch_FCO_L1"]
TAG_PPE = [
    "PropertyPlantAndEquipmentNet",
    "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization",
]
TAG_INTANGIBLES = ["IntangibleAssetsNetExcludingGoodwill", "FiniteLivedIntangibleAssetsNet"]
TAG_CP = [
    "DebtCurrent", "LongTermDebtCurrent", "LongTermDebtAndCapitalLeaseObligationsCurrent",
    "ShortTermBorrowings", "NotesPayableCurrent", "LoansPayableCurrent",
    "ConvertibleNotesPayableCurrent", "ConvertibleDebtCurrent",
]
TAG_LP = [
    "LongTermDebt", "LongTermDebtNoncurrent", "LongTermDebtAndCapitalLeaseObligations",
    "LongTermNotesPayable", "LongTermLineOfCredit", "ConvertibleLongTermNotesPayable",
    "SecuredDebt", "SubordinatedDebt", "NotesPayable", "LineOfCredit", "OtherBorrowings",
]


def winsor(s: pd.Series, lo: float = 0.01, hi: float = 0.99) -> pd.Series:
    if not s.notna().any():
        return s
    return s.clip(s.quantile(lo), s.quantile(hi))


def winsor_masked(s: pd.Series, mask: pd.Series) -> pd.Series:
    q = s[mask]
    return s.clip(q.quantile(0.01), q.quantile(0.99)) if q.notna().any() else s


def n_lags(h: int, n_t: int) -> int:
    return min(abs(h) + 1, n_t - 1)


def read_shock() -> pd.Series:
    frame = pd.read_csv(SHOCK_FILE)
    frame["t"] = pd.PeriodIndex(frame["Trimestre"], freq="Q").asi8
    s = frame.set_index("t")["Shock_t"].astype(float).sort_index()
    return s.reindex(np.arange(s.index.min(), s.index.max() + 1))


def prepare_panel(path: Path, adjusted: bool, shock_raw: pd.Series) -> pd.DataFrame:
    p = pd.read_csv(path, dtype={"CNPJ_CIA": str})
    if "t" not in p:
        p["t"] = pd.PeriodIndex(p["Trimestre"], freq="Q").asi8
    p["t"] = p["t"].astype("int64")
    p = p.sort_values(["CNPJ_CIA", "t"]).reset_index(drop=True)
    if p.duplicated(["CNPJ_CIA", "t"]).any():
        raise AssertionError(f"Duplicated firm-quarter keys in {path.name}")
    p["log_Ativo_Fixo"] = p["log_Ativo_Fixo"].where(np.isfinite(p["log_Ativo_Fixo"]))

    level = ["Cash_Ratio", "Alavancagem", "Tobin_Q", "Tamanho", "Fluxo_Caixa_Op_Ativo"]
    for col in level:
        if not (adjusted and col in ("Cash_Ratio", "Tamanho", "Fluxo_Caixa_Op_Ativo")):
            p[col] = winsor(p[col])

    lag_vars = ["Alavancagem", "Tobin_Q", "Fluxo_Caixa_Op_Ativo"]
    p = p.drop(columns=[v + "_L1" for v in lag_vars if v + "_L1" in p], errors="ignore")
    for v in lag_vars:
        lag = p[["CNPJ_CIA", "t", v]].copy()
        lag["t"] += 1
        p = p.merge(lag.rename(columns={v: v + "_L1"}), on=["CNPJ_CIA", "t"], how="left")

    used_shock = shock_raw.reindex(np.unique(p["t"])).dropna()
    sd = used_shock.std()
    if not np.isfinite(sd) or sd == 0:
        raise ValueError(f"Shock has no usable variation for {path.name}")
    shock = (shock_raw - used_shock.mean()) / sd
    p = p.merge(shock.rename("Shock"), left_on="t", right_index=True, how="left")
    continuous = shock.reindex(np.arange(shock.index.min(), shock.index.max() + 1))
    for lag in (1, 2):
        p = p.merge(continuous.shift(lag).rename(f"Shock_L{lag}"),
                    left_on="t", right_index=True, how="left")

    if adjusted:
        ok_div = p["Alavancagem_L1"].notna()
        for v in ("Cash_Ratio", "Tamanho", "Fluxo_Caixa_Op_Ativo_L1"):
            p[v] = winsor_masked(p[v], ok_div)
        vz = ["Cash_Ratio", "Alavancagem_L1", "Tobin_Q_L1", "Tamanho",
              "Fluxo_Caixa_Op_Ativo_L1"]
        for v in vz:
            grouped = p[v].where(ok_div).groupby(p["t"])
            sd_t = grouped.transform("std").replace(0, np.nan)
            p[v + "_z"] = (p[v] - grouped.transform("mean")) / sd_t
    else:
        vz = ["Cash_Ratio", "Alavancagem_L1", "Tobin_Q_L1", "Tamanho",
              "Fluxo_Caixa_Op_Ativo_L1"]
        for v in vz:
            grouped = p[v].groupby(p["t"])
            sd_t = grouped.transform("std").replace(0, np.nan)
            p[v + "_z"] = (p[v] - grouped.transform("mean")) / sd_t

    p["Ch_Caixa"] = p["Shock"] * p["Cash_Ratio_z"]
    p["Ch_Alav"] = p["Shock"] * p["Alavancagem_L1_z"]
    p["Ch_Tam"] = p["Shock"] * p["Tamanho_z"]
    p["Ch_FCO"] = p["Shock"] * p["Fluxo_Caixa_Op_Ativo_L1_z"]
    for ch in INTERACTIONS:
        for lag in (1, 2):
            x = p[["CNPJ_CIA", "t", ch]].copy()
            x["t"] += lag
            p = p.merge(x.rename(columns={ch: f"{ch}_L{lag}"}),
                        on=["CNPJ_CIA", "t"], how="left")

    # First valid outcome observation defines entrant status, as in the original LP script.
    first = p.loc[p["log_Ativo_Fixo"].notna()].groupby("CNPJ_CIA")["t"].min()
    p["t_prim"] = p["CNPJ_CIA"].map(first)
    lo, hi = pd.Period("2019Q1", freq="Q").ordinal, pd.Period("2021Q4", freq="Q").ordinal
    p["entrante"] = p["t_prim"].between(lo, hi)
    return p


def spec_x(which: str) -> list[str]:
    ctrl_t = ["Shock_L1"] + CONTROL
    if which == "base":
        return ["Shock"] + ctrl_t
    if which == "ctrl_int":
        return ["Shock"] + INTERACTIONS + ctrl_t
    if which == "ctrl_int_L":
        return (["Shock", "Shock_L1", "Shock_L2"] + INTERACTIONS
                + [v + "_L1" for v in INTERACTIONS] + CONTROL)
    if which == "u6_binary":
        return ["Shock", "Ch_Caixa_obs", "Ch_Caixa_imp"] + INTERACTIONS[1:] + ctrl_t
    if which == "u6_stage":
        return (["Shock"] + [f"Ch_Caixa_s{i}" for i in range(4)]
                + INTERACTIONS[1:] + ctrl_t)
    raise ValueError(which)


def common_required(which: str) -> list[str]:
    if which == "ctrl_int_L":
        return []
    return REQUIRED_COMMON if which in ("ctrl_int", "base") else []


def build_y(p: pd.DataFrame, h: int, pivot: pd.DataFrame | None = None) -> pd.DataFrame:
    if pivot is None:
        all_t = np.arange(p["t"].min(), p["t"].max() + 1)
        pivot = p.pivot(index="CNPJ_CIA", columns="t", values="log_Ativo_Fixo").reindex(columns=all_t)
    return (pivot.shift(-h, axis=1) - pivot.shift(1, axis=1)).stack(
        future_stack=True
    ).rename("y").reset_index()


def base_h(p: pd.DataFrame, h: int, x: list[str], req: list[str] | None = None,
           pivot: pd.DataFrame | None = None) -> pd.DataFrame:
    d = p.merge(build_y(p, h, pivot), on=["CNPJ_CIA", "t"], how="inner")
    d = d.dropna(subset=["y"] + x + ["SETOR_ATIV"] + (req or [])).copy()
    if d.empty:
        return d
    d["y"] = winsor(d["y"])
    d["ano"] = 1970 + d["t"].to_numpy() // 4
    return d.reset_index(drop=True)


def estimate(d: pd.DataFrame, x: list[str], h: int, var: str = "Ch_Caixa") -> dict:
    n_t = int(d["t"].nunique())
    if len(d) < 200 or n_t < 2:
        return {"coef": np.nan, "se": np.nan, "p": np.nan, "n": len(d), "n_firmas": d["CNPJ_CIA"].nunique(), "T": n_t}
    panel = d.set_index(["CNPJ_CIA", "t"])
    model = PanelOLS(panel["y"], panel[x], entity_effects=True)
    result = model.fit(cov_type="kernel", kernel="bartlett", bandwidth=n_lags(h, n_t))
    return {"coef": result.params.get(var, np.nan), "se": result.std_errors.get(var, np.nan),
            "p": result.pvalues.get(var, np.nan), "n": int(result.nobs),
            "n_firmas": d["CNPJ_CIA"].nunique(), "T": n_t,
            "df_resid": int(result.df_resid), "df_model": int(result.df_model)}


def append_lp(rows: list[dict], round_name: str, sample: str, h: int,
              result: dict, spec: str) -> None:
    rows.append({"rodada": round_name, "amostra": sample, "h": h,
                 "coef": result["coef"], "se": result["se"], "p": result["p"],
                 "n": result["n"], "n_firmas": result["n_firmas"], "T": result["T"],
                 "spec": spec})


def fwl_contributions(d: pd.DataFrame, x: list[str], target: str) -> np.ndarray:
    groups = d.groupby("CNPJ_CIA", sort=False)
    w = d[x] - groups[x].transform("mean")
    yw = d["y"] - groups["y"].transform("mean")
    others = [c for c in x if c != target]
    if others:
        beta = np.linalg.lstsq(w[others].to_numpy(), w[target].to_numpy(), rcond=None)[0]
        residual = w[target].to_numpy() - w[others].to_numpy() @ beta
    else:
        residual = w[target].to_numpy()
    return residual * yw.to_numpy() / np.square(residual).sum()


def write_table(name: str, rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    frame.to_csv(OUT / name, index=False)
    print(f"\n--- {name} ---")
    print(frame.round(6).to_string(index=False) if not frame.empty else "(sem resultados)")
    return frame


def lp_tasks() -> tuple[list[dict], dict[str, pd.DataFrame]]:
    shock_raw = read_shock()
    estimates: list[dict] = []
    fwl_rows, entrant_rows, spec_rows, n_rows, u6_rows = [], [], [], [], []
    prepared: dict[str, pd.DataFrame] = {}
    entrant_firms = None

    for round_name, (path, adjusted) in ROUNDS.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing panel for {round_name}: {path}")
        print(f"\n[rodada {round_name}] preparando {path.name}", flush=True)
        p = prepare_panel(path, adjusted, shock_raw)
        if round_name in ("obs_adj", "mediana"):
            prepared[round_name] = p
        if entrant_firms is None:
            entrant_firms = p.loc[p["entrante"], "CNPJ_CIA"].nunique()
            print(f"Firmas entrantes (primeira observação válida em 2019Q1-2021Q4): {entrant_firms}")
        pivot = p.pivot(index="CNPJ_CIA", columns="t", values="log_Ativo_Fixo")
        xbase, xint, xlag = spec_x("base"), spec_x("ctrl_int"), spec_x("ctrl_int_L")

        # Observation counts over the full LP window in both sample definitions.
        for sample, req in (("checklist", []), ("comum", REQUIRED_COMMON)):
            for h in range(-4, 9):
                if h == -1:
                    continue
                d = base_h(p, h, xint, req, pivot)
                n_rows.append({"rodada": round_name, "amostra": sample, "h": h,
                               "n": len(d), "n_firmas": d["CNPJ_CIA"].nunique(),
                               "T": d["t"].nunique()})

        # All round-specific LPs requested in the checklist and common samples.
        for sample, horizons, req in (
            ("checklist", CHECKLIST_H, []),
            ("comum", COMMON_H, REQUIRED_COMMON),
        ):
            for h in horizons:
                for spec_name, x in (("base", xbase), ("ctrl_int", xint)):
                    d = base_h(p, h, x, req, pivot)
                    result = estimate(d, x, h, "Ch_Caixa" if spec_name == "ctrl_int" else "Shock")
                    append_lp(estimates, round_name, sample, h, result, spec_name)
                spec_rows.append({"rodada": round_name, "amostra": sample, "horizontes": ",".join(map(str, horizons)),
                                  "requisitos_amostra": ";".join(req),
                                  "regressores_baseline": ";".join(xbase),
                                  "regressores_ctrl_int": ";".join(xint),
                                  "regressores_ctrl_int_L": ";".join(xlag)})
                if sample == "comum":
                    d_lag = base_h(p, h, xlag, [], pivot)
                    append_lp(estimates, round_name, sample, h, estimate(d_lag, xlag, h), "ctrl_int_L")

        # Directly compare h=2 and h=5 under both sample definitions for the text claims.
        for sample, h, req in (
            ("checklist", 2, []), ("checklist", 5, []),
            ("comum", 2, REQUIRED_COMMON), ("comum", 5, REQUIRED_COMMON),
        ):
            if not any(r["rodada"] == round_name and r["amostra"] == sample and r["h"] == h
                       and r["spec"] == "ctrl_int" for r in estimates):
                d = base_h(p, h, xint, req, pivot)
                append_lp(estimates, round_name, sample, h, estimate(d, xint, h), "ctrl_int")
            if not any(r["rodada"] == round_name and r["amostra"] == sample and r["h"] == h
                       and r["spec"] == "ctrl_int_L" for r in estimates):
                d = base_h(p, h, xlag, req, pivot)
                append_lp(estimates, round_name, sample, h, estimate(d, xlag, h), "ctrl_int_L")

        # U1: year-level FWL for the common ctrl_int sample, h=0..8.
        if round_name in ("obs_adj", "mediana"):
            for h in range(0, 9):
                d = base_h(p, h, xint, REQUIRED_COMMON, pivot)
                if d.empty:
                    continue
                contrib = fwl_contributions(d, xint, "Ch_Caixa")
                d = d.copy()
                d["contrib"] = contrib
                beta = float(np.sum(contrib))
                by_year = d.groupby("ano", sort=True)["contrib"].sum()
                same_sign_total = float(by_year[np.sign(by_year) == np.sign(beta)].sum())
                model = estimate(d, xint, h)
                check_error = abs(beta - model["coef"])
                for year, value in by_year.items():
                    fwl_rows.append({
                        "rodada": round_name, "amostra": "comum", "h": h, "ano": int(year),
                        "coef_2022_total": beta, "contrib_ano": value,
                        "participacao_mesmo_sinal_pct": (100 * value / same_sign_total
                                                         if np.sign(value) == np.sign(beta) and same_sign_total else 0.0),
                        "coef_sem_ano": float(fwl_contributions(d[d["ano"] != year], xint, "Ch_Caixa").sum()),
                        "erro_decomp_panelols": check_error,
                    })

        # U2: pooled and only-2022 with/without entrant firms.
        for sample, horizons, req in (
            ("checklist", CHECKLIST_H, []),
            ("comum", COMMON_H, REQUIRED_COMMON),
        ):
            for h in horizons:
                d = base_h(p, h, xint, req, pivot)
                for subset, mask in (
                    ("pooled_com_entrantes", np.ones(len(d), dtype=bool)),
                    ("pooled_sem_entrantes", ~d["entrante"].to_numpy(bool)),
                    ("somente_2022_com_entrantes", (1970 + d["t"].to_numpy() // 4) == 2022),
                    ("somente_2022_sem_entrantes", ((1970 + d["t"].to_numpy() // 4) == 2022)
                     & ~d["entrante"].to_numpy(bool)),
                    ("pooled_sem_2022", (1970 + d["t"].to_numpy() // 4) != 2022),
                ):
                    sub = d.loc[mask]
                    result = estimate(sub, xint, h)
                    entrant_rows.append({"rodada": round_name, "amostra": sample, "h": h,
                                         "subamostra": subset, **result})

        # U6: split the cash-shock interaction by debt-imputation status/stage.
        if round_name == "mediana":
            stage = pd.to_numeric(p.get("div_etapa", pd.Series(np.nan, index=p.index)), errors="coerce")
            p["Ch_Caixa_obs"] = p["Ch_Caixa"] * (stage == 0)
            p["Ch_Caixa_imp"] = p["Ch_Caixa"] * (stage >= 1)
            for s in range(4):
                p[f"Ch_Caixa_s{s}"] = p["Ch_Caixa"] * (stage == s)
            for h in (-4, -3, -2, 2, 3, 4, 5, 6, 7, 8):
                for spec_name in ("u6_binary", "u6_stage"):
                    x = spec_x(spec_name)
                    d = base_h(p, h, x, REQUIRED_COMMON, pivot)
                    if spec_name == "u6_binary":
                        names = ("Ch_Caixa_obs", "Ch_Caixa_imp")
                    else:
                        names = tuple(f"Ch_Caixa_s{s}" for s in range(4))
                    for var in names:
                        r = estimate(d, x, h, var)
                        u6_rows.append({"rodada": "mediana", "amostra": "comum", "h": h,
                                        "modelo": spec_name, "termo": var, **r})

        print(f"  {round_name}: painel preparado ({len(p):,} linhas; {p['CNPJ_CIA'].nunique():,} firmas)", flush=True)

    for h in range(0, 9):
        for round_name in ("obs_adj", "mediana"):
            candidates = [r for r in fwl_rows if r["rodada"] == round_name and r["h"] == h]
            err = max((r["erro_decomp_panelols"] for r in candidates), default=np.nan)
            if np.isfinite(err):
                print(f"U1 erro FWL vs PanelOLS {round_name}, h={h}: {err:.3e}")

    for h in range(0, 9):
        fwl_rows.append({"rodada": "zero", "amostra": "comum", "h": h,
                         "ano": np.nan, "status": ZERO_NOTE})
    entrant_rows.append({"rodada": "zero", "amostra": ZERO_NOTE, "h": np.nan,
                         "subamostra": ZERO_NOTE, "coef": np.nan, "se": np.nan,
                         "p": np.nan, "n": np.nan})
    spec_rows.append({"rodada": "zero", "amostra": ZERO_NOTE,
                      "regressores_baseline": ZERO_NOTE,
                      "regressores_ctrl_int": ZERO_NOTE,
                      "regressores_ctrl_int_L": ZERO_NOTE})

    files = {
        "fwl": write_table("u1_fwl_ano.csv", fwl_rows),
        "entrants": write_table("u2_entrantes.csv", entrant_rows),
        "specs": write_table("u5_regressores.csv", spec_rows),
        "n": write_table("u8_n_por_h.csv", n_rows),
        "u6": write_table("u6_interacao_imputacao.csv", u6_rows),
    }
    estimates.append({"rodada": "zero", "amostra": ZERO_NOTE, "h": np.nan,
                      "coef": np.nan, "se": np.nan, "p": np.nan, "n": np.nan,
                      "n_firmas": np.nan, "T": np.nan, "spec": "não reproduzível"})
    return estimates, {**files, "prepared": prepared}


def concentration_task(shock_raw: pd.Series) -> pd.DataFrame:
    periods = pd.PeriodIndex([pd.Period(ordinal=int(t), freq="Q") for t in shock_raw.index])
    q = pd.Series(shock_raw.to_numpy(), index=periods)
    q = q.loc[(q.index >= pd.Period("2011Q4", freq="Q")) &
              (q.index <= pd.Period("2023Q4", freq="Q"))].dropna()
    dev2 = (q - q.mean()) ** 2
    share = dev2 / dev2.sum()
    ep_a = (q.index >= pd.Period("2015Q4", freq="Q")) & (q.index <= pd.Period("2018Q4", freq="Q"))
    ep_b = (q.index >= pd.Period("2022Q1", freq="Q")) & (q.index <= pd.Period("2023Q3", freq="Q"))
    drop = {pd.Period("2011Q4", freq="Q"), pd.Period("2021Q1", freq="Q")}
    checklist = q.loc[~q.index.isin(drop)]
    labels = []
    for period, shock_value in q.items():
        i = q.index.get_loc(period)
        episode = "A" if ep_a[i] else "B" if ep_b[i] else "calm"
        labels.append({"trimestre": str(period), "choque": shock_value,
                       "share_desvios2_pct": 100 * share.loc[period],
                       "episodio": episode, "removido_checklist": period in drop})
    detail = pd.DataFrame(labels)
    detail.insert(0, "rodada", "choque_comum")
    detail.insert(1, "amostra", "checklist")
    print("\nU3 trimestres com choque:", len(q), "| checklist:", len(checklist),
          "| contagem A/B/calm:",
          sum(ep_a & ~q.index.isin(drop)), sum(ep_b & ~q.index.isin(drop)),
          len(checklist) - sum(ep_a & ~q.index.isin(drop)) - sum(ep_b & ~q.index.isin(drop)))
    print("Removidos checklist:", ", ".join(sorted(map(str, drop))))
    print(f"Participação episódio A={100 * share.iloc[np.flatnonzero(ep_a)].sum():.3f}% | "
          f"B={100 * share.iloc[np.flatnonzero(ep_b)].sum():.3f}% | "
          f"N efetivo={1 / np.square(share).sum():.3f} ({100 * (1 / np.square(share).sum()) / len(q):.1f}%)")
    print(f"Maior trimestre: {str(share.idxmax())} ({100 * share.max():.2f}%)")
    by_year = detail.assign(ano=detail["trimestre"].str[:4]).groupby("ano")["share_desvios2_pct"].sum()
    print("\nShare desvios quadrados por ano (%):\n", by_year.round(3).to_string())
    print("\nLista trimestral U3:\n", detail.to_string(index=False))
    write_table("u3_concentracao_trimestre.csv", detail)
    return detail


def imputation_task(prepared: dict[str, pd.DataFrame]) -> pd.DataFrame:
    src = DATA / "sec_longo.csv"
    sec = pd.read_csv(src, usecols=["cik", "tag"], dtype={"cik": str}, low_memory=False)
    cp_tags, lp_tags = set(TAG_CP), set(TAG_LP)
    sec["_cp"] = sec["tag"].isin(cp_tags)
    sec["_lp"] = sec["tag"].isin(lp_tags)
    grouped = sec.groupby("cik")[["_cp", "_lp"]].any()
    sec_n = len(grouped)
    sec_never = int((~grouped["_cp"] & ~grouped["_lp"]).sum())
    del sec, grouped

    stage_panels = {
        "SEC_extração": (sec_n, sec_never),
    }
    for label, path in (
        ("painel_pré_filtro", DATA / "painel_regressao_eua.csv"),
        ("final_4787", DATA / "painel_regressao_eua_amostra.csv"),
    ):
        d = pd.read_csv(path, usecols=lambda c: c in ("CNPJ_CIA", "cp_zero_nunca", "lp_zero_nunca",
                                                       "div_etapa", "div_imputada"),
                        dtype={"CNPJ_CIA": str})
        if "cp_zero_nunca" in d and "lp_zero_nunca" in d:
            by_firm = d.groupby("CNPJ_CIA")[["cp_zero_nunca", "lp_zero_nunca"]].max()
            never = int((by_firm["cp_zero_nunca"].astype(bool) & by_firm["lp_zero_nunca"].astype(bool)).sum())
            nfirm = len(by_firm)
        else:
            raise KeyError(f"No never-report flags in {path.name}")
        stage_panels[label] = (nfirm, never)
        print(f"\n{label}: {never}/{nfirm} firm as nunca reportam CP nem LP ({never / nfirm:.2%})")
    print(f"\nSEC_extração: {sec_never}/{sec_n} firmas nunca reportam CP nem LP ({sec_never / sec_n:.2%})")
    write_table("u4_denominadores.csv", [
        {"rodada": "mediana", "amostra": label, "firmas": n, "nunca_reportam_cp_lp": never,
         "pct_nunca_reportam": 100 * never / n}
        for label, (n, never) in stage_panels.items()
    ])

    rows = []
    prefilter_path = DATA / "painel_regressao_eua.csv"
    prefilter_stage = pd.read_csv(prefilter_path, dtype={"CNPJ_CIA": str},
                                  usecols=["CNPJ_CIA", "div_etapa", "div_imputada"])
    for label, d in (
        ("painel_pre_filtro_10178", prefilter_stage),
        ("amostra_final_4787_painel", prepared["mediana"]),
    ):
        levels = d["div_etapa"].fillna(4).value_counts()
        for s in range(5):
            rows.append({"rodada": "mediana", "amostra": label, "estagio": s,
                         "rotulo_estagio": "sem classificação" if s == 4 else f"estágio_{s}",
                         "n_linhas": int(levels.get(s, 0)),
                         "share_pct": 100 * levels.get(s, 0) / len(d), "n_total": len(d)})

    final = prepared["mediana"]
    for sample, d in (
        ("checklist_h3", base_h(final, 3, spec_x("ctrl_int"), [], None)),
        ("checklist_h5", base_h(final, 5, spec_x("ctrl_int"), [], None)),
        ("checklist_h7", base_h(final, 7, spec_x("ctrl_int"), [], None)),
        ("comum_h2", base_h(final, 2, spec_x("ctrl_int"), REQUIRED_COMMON, None)),
        ("comum_h4", base_h(final, 4, spec_x("ctrl_int"), REQUIRED_COMMON, None)),
        ("comum_h6", base_h(final, 6, spec_x("ctrl_int"), REQUIRED_COMMON, None)),
        ("comum_h8", base_h(final, 8, spec_x("ctrl_int"), REQUIRED_COMMON, None)),
    ):
        if "div_etapa" not in d:
            continue
        counts = d["div_etapa"].fillna(-1).value_counts(normalize=True)
        n = len(d)
        for s in range(4):
            rows.append({"rodada": "mediana", "amostra": sample, "estagio": s,
                         "n_linhas": int((d["div_etapa"] == s).sum()),
                         "share_pct": 100 * counts.get(s, 0.0), "n_total": n})
    stage_df = write_table("u4_estagios_amostrais.csv", rows)
    print("\nParticipação por estágio na extração, pré-filtro e final:")
    print(pd.DataFrame([{"estagio": label, "firmas": n, "nunca_reportam": never,
                         "pct_nunca": 100 * never / n} for label, (n, never) in stage_panels.items()])
          .round(2).to_string(index=False))

    prefilter_firm_imp = prefilter_stage.groupby("CNPJ_CIA")["div_imputada"].max()
    final_firm_imp = final.groupby("CNPJ_CIA")["div_etapa"].max()
    summary = []
    for label, d, firm_imp in (
        ("painel_pre_filtro_10178", prefilter_stage, prefilter_firm_imp),
        ("amostra_final_4787", final, final_firm_imp),
    ):
        row_share = float((d["div_etapa"] >= 1).sum() / len(d) * 100)
        firm_share = float((firm_imp >= 1).mean() * 100)
        print(f"{label}: {row_share:.2f}% linhas imputadas; {firm_share:.2f}% firmas com ao menos uma linha imputada.")
        summary.extend([
            {"rodada": "mediana", "amostra": label, "medida": "linhas_com_alguma_imputacao_pct",
             "valor": row_share},
            {"rodada": "mediana", "amostra": label, "medida": "firmas_com_alguma_linha_imputada_pct",
             "valor": firm_share},
        ])
    summary.append({"rodada": "mediana", "amostra": "SEC_extração",
                   "medida": "firmas_que_nunca_reportam_cp_lp_pct",
                   "valor": stage_panels["SEC_extração"][1] / stage_panels["SEC_extração"][0] * 100})
    write_table("u4_resumo_imputacao.csv", summary)
    vars_ = ["Tamanho", "Cash_Ratio", "Fluxo_Caixa_Op_Ativo"]
    select_rows = []
    for label, mask in (("debito_observado", final["div_etapa"] == 0),
                        ("sem_debito_observado", final["div_etapa"] >= 1)):
        for v in vars_:
            select_rows.append({"rodada": "mediana", "amostra": "painel_final",
                                "grupo": label, "variavel": v, "n": int(final.loc[mask, v].notna().sum()),
                                "media": final.loc[mask, v].mean()})
    write_table("u7_selecao_observada_imputada.csv", select_rows)
    return stage_df


def u7_power_task(prepared: dict[str, pd.DataFrame], estimates: list[dict]) -> pd.DataFrame:
    med = prepared["mediana"]
    obs = prepared["obs_adj"]
    x = spec_x("ctrl_int")
    rng = np.random.default_rng(20261024)
    pivot = med.pivot(index="CNPJ_CIA", columns="t", values="log_Ativo_Fixo")
    out_rows = []
    for h in (-4, -3, 2, 5):
        target_sample = base_h(obs, h, x, REQUIRED_COMMON, None)
        target_n = len(target_sample)
        d = base_h(med, h, x, REQUIRED_COMMON, pivot)
        firm_codes, firms = pd.factorize(d["CNPJ_CIA"], sort=False)
        time_codes, times = pd.factorize(d["t"], sort=True)
        blocks = list(d.groupby("CNPJ_CIA", sort=False).indices.values())
        x_all, y_all = d[x].to_numpy(), d["y"].to_numpy()
        n_firms_all = len(firms)
        n_times_all = len(times)
        draw_coefs = []
        draw_p = []
        draw_ns = []
        draw_n_firms = []
        extra_df = None
        for rep in range(500):
            order = rng.permutation(n_firms_all)
            chosen: list[np.ndarray] = []
            n_now = 0
            for firm_id in order:
                block = blocks[firm_id]
                if n_now + len(block) <= target_n:
                    chosen.append(block)
                    n_now += len(block)
                else:
                    needed = target_n - n_now
                    if needed > 0:
                        chosen.append(rng.choice(block, size=needed, replace=False))
                        n_now += needed
                    break
                if n_now == target_n:
                    break
            selected = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
            y = y_all[selected]
            X = x_all[selected]
            f = firm_codes[selected]
            tt = time_codes[selected]
            nobs = len(selected)
            nfirm = int(np.unique(f).size)
            counts = np.bincount(f, minlength=n_firms_all).astype(float)
            counts[counts == 0] = 1.0
            y_within = y - (np.bincount(f, weights=y, minlength=n_firms_all) / counts)[f]
            x_within = np.empty_like(X)
            for j in range(X.shape[1]):
                x_within[:, j] = X[:, j] - (
                    np.bincount(f, weights=X[:, j], minlength=n_firms_all) / counts
                )[f]
            xtx = x_within.T @ x_within
            xtx_inv = np.linalg.inv(xtx)
            beta = xtx_inv @ (x_within.T @ y_within)
            residual = y_within - x_within @ beta
            scores = x_within * residual[:, None]
            by_time = np.zeros((n_times_all, X.shape[1]))
            np.add.at(by_time, tt, scores)
            meat = by_time.T @ by_time
            bandwidth = n_lags(h, int(np.unique(tt).size))
            for lag in range(1, bandwidth + 1):
                cross = by_time[lag:].T @ by_time[:-lag]
                meat += (1 - lag / (bandwidth + 1)) * (cross + cross.T)
            if extra_df is None:
                sample_panel = d.iloc[selected].set_index(["CNPJ_CIA", "t"])
                check = PanelOLS(sample_panel["y"], sample_panel[x], entity_effects=True).fit(
                    cov_type="kernel", kernel="bartlett",
                    bandwidth=n_lags(h, int(np.unique(tt).size))
                )
                extra_df = int(check.df_model) - len(x)
                max_beta_error = float(np.max(np.abs(beta - check.params[x].to_numpy())))
                if max_beta_error > 1e-8:
                    raise AssertionError(f"Fast FE coefficients differ from PanelOLS by {max_beta_error:.3e}")
            scale = nobs / (nobs - extra_df - len(x))
            cov = xtx_inv @ meat @ xtx_inv * scale
            j_cash = x.index("Ch_Caixa")
            se = float(np.sqrt(max(cov[j_cash, j_cash], 0.0)))
            coef = float(beta[j_cash])
            pvalue = float(2 * student_t.sf(abs(coef / se), nobs - extra_df - len(x))) if se else np.nan
            if rep == 0:
                se_error = abs(se - float(check.std_errors["Ch_Caixa"]))
                if se_error > 1e-8:
                    raise AssertionError(f"Fast DK standard error differs from PanelOLS by {se_error:.3e}")
            draw_coefs.append(coef)
            draw_p.append(pvalue)
            draw_ns.append(nobs)
            draw_n_firms.append(nfirm)
        coefs = np.asarray(draw_coefs, dtype=float)
        ps = np.asarray(draw_p, dtype=float)
        finite = np.isfinite(coefs) & np.isfinite(ps)
        pre = h < 0
        out_rows.append({
            "rodada": "mediana", "amostra": "random_firm_cluster", "h": h,
            "target_n_obs_obs_adj": target_n, "n_media": float(np.mean(draw_ns)),
            "n_firmas_media": float(np.mean(draw_n_firms)),
            "coef_media": float(np.nanmean(coefs)), "coef_sd": float(np.nanstd(coefs, ddof=1)),
            "coef_p05": float(np.nanpercentile(coefs, 5)), "coef_p50": float(np.nanpercentile(coefs, 50)),
            "coef_p95": float(np.nanpercentile(coefs, 95)),
            "frac_pretrend_signif_5pct": float(np.mean(ps[finite] < .05)) if pre else np.nan,
            "frac_h2_nao_signif_5pct": float(np.mean(ps[finite] >= .05)) if h == 2 else np.nan,
            "replicas_validas": int(finite.sum()), "replicas": 500,
            "seed": 20261024, "metodo": "firmas aleatórias; corte aleatório no último cluster para N exato",
        })
        print(f"U7 h={h}: N alvo={target_n:,}; coef média={np.nanmean(coefs):.5f}; "
              f"sd={np.nanstd(coefs, ddof=1):.5f}; p<.05={np.nanmean(ps < .05):.1%}", flush=True)
    return write_table("u7_poder_subamostras.csv", out_rows)


def monte_carlo_task() -> None:
    script = ROOT / "monte_carlo_checklist.py"
    if not script.exists():
        print("U9: Monte Carlo ausente neste repositório.")
        return
    old_argv = sys.argv
    sys.argv = [str(script)]
    spec = importlib.util.spec_from_file_location("mc_checklist_revisao", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {script}")
    mc = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mc)
    finally:
        sys.argv = old_argv
    mc.args.R = 2000
    mc.args.B = 300
    mc.args.seed = 20261024
    old_cwd = Path.cwd()
    OUT.mkdir(exist_ok=True)
    try:
        os.chdir(OUT)
        mc.main()
    finally:
        os.chdir(old_cwd)
    result_path = OUT / "monte_carlo_resultados.csv"
    if result_path.exists():
        frame = pd.read_csv(result_path)
        frame.insert(0, "rodada", "monte_carlo_reproduzido")
        frame.insert(1, "amostra", "DGP_repositorio_T54")
        frame.to_csv(result_path, index=False)
        write_table("u9_monte_carlo_t54.csv", frame)

    # Clean, un-concentrated null based on the repository DGP, with 54 and 400 quarters.
    rng = np.random.default_rng(20261024)
    clean_rows = []
    for T in (54, 400):
        L, N, R = 4, mc.args.N, 2000
        sig = 0
        coefs = []
        for _ in range(R):
            shock = rng.standard_normal(T)
            c = rng.standard_normal((N, T))
            c = (c - c.mean(0)) / c.std(0)
            x = shock * c
            y = rng.standard_normal((N, 1)) + 2.0 * rng.standard_normal((N, T))
            xd, yd = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
            sxy, sxx = (xd * yd).sum(0), (xd * xd).sum(0)
            total = sxx.sum()
            b = sxy.sum() / total
            se = mc.dk_se(sxy - b * sxx, total, L)
            pval = mc.p_norm(b / se)
            coefs.append(b)
            sig += pval < .05
        clean_rows.append({"rodada": "monte_carlo_extensao", "amostra": "nulo_limpo_sem_concentracao",
                           "cenario": "Nulo limpo sem concentração", "T": T, "R": R,
                           "seed": 20261024, "rejeicao_DK_5pct_pct": 100 * sig / R,
                           "coef_sd": float(np.std(coefs, ddof=1))})
    write_table("u9_nulo_limpo_400q.csv", clean_rows)


def tag_task() -> pd.DataFrame:
    print("\nTags XBRL do ativo fixo: PP&E =", " | ".join(TAG_PPE))
    print("Intangíveis =", " | ".join(TAG_INTANGIBLES))
    print("Goodwill excluído: SIM; as duas tags de intangíveis são net excluding goodwill / finite-lived.")
    return write_table("u8_tags_xbrl.csv", [
        {"rodada": "pipeline", "amostra": "Ativo_Fixo", "componente": "PP&E", "tag": t,
         "incluida": True, "goodwill_excluido": True} for t in TAG_PPE
    ] + [
        {"rodada": "pipeline", "amostra": "Ativo_Fixo", "componente": "intangíveis", "tag": t,
         "incluida": True, "goodwill_excluido": True} for t in TAG_INTANGIBLES
    ])


def report_divergences() -> pd.DataFrame:
    report: list[dict] = []

    def add(claim: str, text: str, calculated: str, passed: bool) -> None:
        report.append({"afirmacao_do_texto": claim, "valor_no_texto": text,
                       "valor_calculado": calculated, "status": "✓" if passed else "✗"})

    def load(name: str) -> pd.DataFrame:
        path = OUT / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    fwl = load("u1_fwl_ano.csv")
    if not fwl.empty:
        med22 = fwl[(fwl.rodada == "mediana") & (fwl.ano == 2022) & fwl.h.between(2, 8)]
        shares = med22.set_index("h")["participacao_mesmo_sinal_pct"]
        ok_med = len(shares) == 7 and shares.between(33, 39).all()
        add("2022 responde por 33–39% em h=2–8", "33–39% (mediana)",
            "; ".join(f"h{h}: {v:.2f}%" for h, v in shares.items()), bool(ok_med))
        obs22 = fwl[(fwl.rodada == "obs_adj") & (fwl.ano == 2022)].set_index("h")
        obs_shares = {h: obs22.loc[h, "participacao_mesmo_sinal_pct"] for h in (2, 4)
                      if h in obs22.index}
        ok_obs = len(obs_shares) == 2 and abs(obs_shares[2] - 41) <= 0.5 and abs(obs_shares[4] - 47) <= 0.5
        add("Com débito observado, share de 2022", "41% em h=2; 47% em h=4",
            "; ".join(f"h{h}: {v:.2f}%" for h, v in obs_shares.items()), bool(ok_obs))
        obs_contrib = obs22["contrib_ano"]
        target_contrib = {0: .006, 4: -.006, 6: -.014, 8: -.016}
        act = {h: float(obs_contrib.loc[h]) for h in target_contrib if h in obs_contrib.index}
        ok_contrib = len(act) == 4 and all(abs(act[h] - target_contrib[h]) <= .00051 for h in act)
        add("Coeficiente anual de 2022 com débito observado", "h=0: 0,006; h=4: −0,006; h=6: −0,014; h=8: −0,016",
            "; ".join(f"h{h}: {v:.5f}" for h, v in act.items()), bool(ok_contrib))
        tops = {}
        for h in (6, 8):
            part = fwl[(fwl.rodada == "obs_adj") & (fwl.h == h)]
            if not part.empty:
                top = part.loc[part["participacao_mesmo_sinal_pct"].idxmax()]
                tops[h] = (int(top.ano), float(top.participacao_mesmo_sinal_pct))
        ok_top = len(tops) == 2 and all(y == 2016 and 44 <= pct <= 48 for y, pct in tops.values())
        add("Ano de maior peso com débito observado em h=6 e h=8", "2016; 44–48%",
            "; ".join(f"h{h}: {y} ({pct:.2f}%)" for h, (y, pct) in tops.items()), bool(ok_top))
        max_err = float(fwl["erro_decomp_panelols"].max())
        add("Erro FWL da decomposição", "aprox. 5e−17", f"{max_err:.3e}",
            max_err < 1e-12)
    else:
        add("Decomposição FWL", "coeficientes e participações por ano", "saída ausente", False)
    add("Decomposição por ano na rodada zero", "valores citados para zero", ZERO_NOTE, False)

    entrants = load("u2_entrantes.csv")
    targets = {
        "mediana v1: pooled com→sem entrantes": np.array([.0052, .0055, .0054, .0069]),
        "zero: pooled com→sem entrantes": np.array([.0051, .0051, .0051, .0063]),
    }
    if not entrants.empty:
        for claim, target in targets.items():
            if claim.startswith("zero"):
                add(claim, "h=2,4,6,8; com→sem entrantes", ZERO_NOTE, False)
                continue
            target_flat = np.column_stack([target, np.array([.0036, .0039, .0033, .0044])]).ravel()
            matches = []
            for round_name in ROUNDS:
                g = entrants[(entrants.rodada == round_name) & (entrants.amostra == "comum")
                             & entrants.subamostra.isin(
                                 ["pooled_com_entrantes", "pooled_sem_entrantes"])
                             & entrants.h.isin(COMMON_H)]
                before0 = g[g.subamostra == "pooled_com_entrantes"].set_index("h").reindex(COMMON_H)["coef"].to_numpy()
                after0 = g[g.subamostra == "pooled_sem_entrantes"].set_index("h").reindex(COMMON_H)["coef"].to_numpy()
                values0 = np.column_stack([before0, after0]).ravel()
                if len(values0) == 8 and np.isfinite(values0).all():
                    matches.append((float(np.max(np.abs(values0 - target_flat))), round_name,
                                    before0, after0))
            if not matches:
                add(claim, "com→sem: vetores h=2,4,6,8", "estimativas ausentes", False)
                continue
            matches.sort(key=lambda x: x[0])
            mae, best_round, before, after = matches[0]
            passed = best_round == "mediana" and mae <= .00051
            actual = "; ".join(f"h{h}: {before[i]:.5f}→{after[i]:.5f}"
                               for i, h in enumerate(COMMON_H))
            round_errors = "; ".join(f"{r}: max|Δ|={err:.5f}" for err, r, _, _ in matches)
            add(claim, "com→sem: 0,0052→0,0036; 0,0055→0,0039; 0,0054→0,0033; 0,0069→0,0044",
                f"melhor correspondência: {best_round}; {actual}; erros por rodada: {round_errors}",
                bool(passed))
        g22 = entrants[(entrants.rodada == "mediana") & (entrants.amostra == "comum")
                       & (entrants.h == 2) & entrants.subamostra.isin(
                           ["somente_2022_com_entrantes", "somente_2022_sem_entrantes"])]
        v22 = g22.set_index("subamostra")["coef"]
        v_before = v22.get("somente_2022_com_entrantes", np.nan)
        v_after = v22.get("somente_2022_sem_entrantes", np.nan)
        add("Somente 2022, h=2, mediana", "0,011→0,002",
            f"{v_before:.5f}→{v_after:.5f}", bool(abs(v_before - .011) < .00051 and abs(v_after - .002) < .00051))
        sem22 = entrants[(entrants.rodada == "mediana") & (entrants.amostra == "comum")
                         & (entrants.subamostra == "pooled_sem_2022")].set_index("h")
        exact = "; ".join(f"h{h}: {sem22.loc[h, 'coef']:.5f}" for h in COMMON_H if h in sem22.index)
        exact_values = sem22.reindex(COMMON_H)["coef"].to_numpy()
        approx_target = np.array([.004, .003, .001, .001])
        add("Pooled sem 2022 exato (mediana)", "aprox. 0,004; 0,003; 0,001; 0,001",
            exact, len(sem22) >= len(COMMON_H) and np.isfinite(exact_values).all()
            and np.max(np.abs(exact_values - approx_target)) <= .00051)
    else:
        add("Entrantes e exclusão de 2022", "valores por rodada", "saída ausente", False)
    add("Conjunto de entrantes da rodada zero", "0,0051→0,0038; 0,0051→0,0036; 0,0051→0,0031; 0,0063→0,0038",
        ZERO_NOTE, False)

    concentration = load("u3_concentracao_trimestre.csv")
    if not concentration.empty:
        ss = concentration.groupby("episodio")["share_desvios2_pct"].sum()
        effective = 1 / np.square(concentration["share_desvios2_pct"].to_numpy() / 100).sum()
        top = concentration.loc[concentration["share_desvios2_pct"].idxmax()]
        passed = abs(ss.get("A", np.nan) - 33.7) <= .1 and abs(ss.get("B", np.nan) - 33.7) <= .1
        add("Participação dos episódios A e B", "33,7% e 33,7%",
            f"A={ss.get('A', np.nan):.3f}%; B={ss.get('B', np.nan):.3f}%", bool(passed))
        add("Número efetivo de trimestres", "17,8 (40%)",
            f"{effective:.3f} ({100 * effective / len(concentration):.1f}%)",
            abs(effective - 17.8) <= .1)
        add("Maior trimestre", "2023Q2 (11,7%)",
            f"{top.trimestre} ({top.share_desvios2_pct:.2f}%)",
            top.trimestre == "2023Q2" and abs(top.share_desvios2_pct - 11.7) <= .1)
        removed = sorted(concentration.loc[concentration["removido_checklist"], "trimestre"].tolist())
        add("Trimestres removidos do checklist", "2011Q4 e 2021Q1", ", ".join(removed),
            removed == ["2011Q4", "2021Q1"])
        add("Lista de trimestres com choque", "45; checklist 43 (13 A, 7 B, 23 calm)",
            f"{len(concentration)}; checklist {int((~concentration.removido_checklist).sum())}",
            len(concentration) == 45 and int((~concentration.removido_checklist).sum()) == 43)

    denom = load("u4_denominadores.csv")
    if not denom.empty:
        expected = {"SEC_extração": 15021, "painel_pré_filtro": 10178, "final_4787": 4787}
        counts = denom.set_index("amostra")["firmas"]
        actual = "; ".join(f"{key}: {int(counts.get(key, -1))}" for key in expected)
        add("Denominadores de firmas da imputação", "15.021 / 10.178 / 4.787", actual,
            all(int(counts.get(k, -1)) == v for k, v in expected.items()))
        shares = load("u4_estagios_amostrais.csv")
        final_sh = shares[shares.amostra == "amostra_final_4787_painel"].set_index("estagio")["share_pct"]
        actual_sh = [float(final_sh.get(i, np.nan)) for i in range(4)]
        text_sh = [37.6, 13.2, 11.0, 38.2]
        add("Participação das linhas por estágio na amostra final", "37,6 / 13,2 / 11,0 / 38,2%",
            "final 4.787: " + " / ".join(f"{v:.2f}" for v in actual_sh),
            len(actual_sh) == 4 and all(abs(a - b) <= .2 for a, b in zip(actual_sh, text_sh)))
        pre_sh = shares[shares.amostra == "painel_pre_filtro_10178"].set_index("estagio")["share_pct"]
        pre_values = [float(pre_sh.get(i, np.nan)) for i in range(4)]
        add("Distribuição de estágios que corresponde aos valores do texto",
            "37,6 / 13,2 / 11,0 / 38,2%",
            "pré-filtro 10.178: " + " / ".join(f"{v:.2f}" for v in pre_values),
            len(pre_values) == 4 and all(abs(a - b) <= .2 for a, b in zip(pre_values, text_sh)))
        summary_df = load("u4_resumo_imputacao.csv")
        summary = summary_df.set_index(["amostra", "medida"])["valor"]
        line_pre = summary.get(("painel_pre_filtro_10178", "linhas_com_alguma_imputacao_pct"), np.nan)
        line_final = summary.get(("amostra_final_4787", "linhas_com_alguma_imputacao_pct"), np.nan)
        firm_final = summary.get(("amostra_final_4787", "firmas_com_alguma_linha_imputada_pct"), np.nan)
        sec_firms = summary.get(("SEC_extração", "firmas_que_nunca_reportam_cp_lp_pct"), np.nan)
        add("Linhas com algum componente imputado", "62,4%",
            f"pré-filtro 10.178: {line_pre:.2f}%; amostra final 4.787: {line_final:.2f}%",
            abs(line_pre - 62.4) <= .2)
        add("Firmas que nunca reportam CP nem LP (extração SEC)", "25,4%",
            f"{sec_firms:.2f}% na extração SEC; {firm_final:.2f}% das firmas finais têm alguma linha imputada",
            abs(sec_firms - 25.4) <= .2)

    est = load("estimativas_lp.csv")
    if not est.empty:
        med2 = est[(est.rodada == "mediana") & (est.amostra == "comum")
                   & (est.h == 2) & (est.spec == "ctrl_int")]
        med5 = est[(est.rodada == "mediana") & (est.amostra == "checklist")
                   & (est.h == 5) & (est.spec == "ctrl_int")]
        med2_checklist = est[(est.rodada == "mediana") & (est.amostra == "checklist")
                             & (est.h == 2) & (est.spec == "ctrl_int")]
        med5_common = est[(est.rodada == "mediana") & (est.amostra == "comum")
                          & (est.h == 5) & (est.spec == "ctrl_int")]
        for claim, text, frame, coef_target, p_target in (
            ("Mediana h=2 na amostra comum", "0,0048; p=0,03", med2, .0048, .03),
            ("Mediana h=2 no checklist", "estimativa por checklist", med2_checklist, .0048, .03),
            ("Mediana h=5 na amostra comum", "0,0065; p=0,05", med5_common, .0065, .05),
            ("Mediana h=5 no checklist", "estimativa por checklist", med5, .0065, .05),
        ):
            if frame.empty:
                add(claim, text, "estimativa ausente", False)
            else:
                r = frame.iloc[0]
                add(claim, text, f"coef={r.coef:.5f}; p={r.p:.4f}",
                    abs(r.coef - coef_target) <= .00051 and abs(r.p - p_target) <= .015)
        add("Extensão com lags na rodada zero", "0,0049; p=0,030 e 0,0062; p=0,069",
            ZERO_NOTE, False)
        fwl_errs = load("u1_fwl_ano.csv")["erro_decomp_panelols"]
        add("Erro numérico da decomposição FWL", "aprox. 5e−17",
            f"{fwl_errs.max():.3e}", fwl_errs.max() < 1e-12)

    u6 = load("u6_interacao_imputacao.csv")
    if not u6.empty:
        r = u6[(u6.modelo == "u6_binary") & u6.h.isin([-4, -3, -2, 2, 3, 4, 5, 6, 7, 8])]
        add("Teste direto do artefato de imputação (mediana)", "interações observadas/imputadas, DK",
            f"{len(r)} linhas; coeficientes e SE em u6_interacao_imputacao.csv", len(r) == 20)
    spec_table = load("u5_regressores.csv")
    if not spec_table.empty:
        checklist = spec_table[spec_table.amostra == "checklist"].iloc[0]
        common = spec_table[spec_table.amostra == "comum"].iloc[0]
        reqs = set(str(common.requisitos_amostra).split(";"))
        base_terms = set(str(common.regressores_ctrl_int).split(";"))
        extended_terms = set(str(common.regressores_ctrl_int_L).split(";"))
        lags_are_sample_only = ("Shock_L2" in reqs and "Ch_Caixa_L1" in reqs
                                and "Shock_L2" not in base_terms
                                and "Shock_L2" in extended_terms
                                and set(str(checklist.regressores_ctrl_int).split(";")) == base_terms)
        add("Amostra comum: restrição ou novos regressores?",
            "comum acrescenta Shock_L2 e interações defasadas",
            f"ctrl_int checklist=comum: {set(str(checklist.regressores_ctrl_int).split(';')) == base_terms}; "
            f"requisitos comuns={';'.join(sorted(reqs))}; extensão inclui lags={lags_are_sample_only}",
            bool(lags_are_sample_only))
    ns = load("u8_n_por_h.csv")
    if not ns.empty:
        desc = (ns.groupby(["rodada", "amostra"]).size().to_dict())
        add("N por horizonte e amostra", "N listado por h para checklist e comum",
            f"{len(ns)} linhas; {desc}; tabela u8_n_por_h.csv",
            len(ns) == len(ROUNDS) * 2 * 12)
    power = load("u7_poder_subamostras.csv")
    if not power.empty:
        vals = power.set_index("h")
        description = "; ".join(f"h{h}: média {vals.loc[h, 'coef_media']:.5f}, SD {vals.loc[h, 'coef_sd']:.5f}"
                                for h in (-4, -3, 2, 5))
        add("Poder vs. padrão, 500 sorteios", "N obs_adj; h=−4,−3,2,5",
            description, len(vals) == 4 and (vals["replicas_validas"] == 500).all())
    selection = load("u7_selecao_observada_imputada.csv")
    if not selection.empty:
        desc = "; ".join(f"{r.grupo}/{r.variavel}: {r.media:.5f} (n={int(r.n)})"
                         for r in selection.itertuples(index=False))
        add("Seleção: tamanho, caixa e fluxo de caixa operacional", "médias em linhas com débito observado vs. imputado",
            desc, len(selection) == 6)

    tags = load("u8_tags_xbrl.csv")
    expected_tags = set(TAG_PPE + TAG_INTANGIBLES)
    actual_tags = set(tags["tag"]) if not tags.empty else set()
    add("Tags de ativo fixo e goodwill", "PP&E + intangíveis; goodwill excluído",
        "goodwill excluído; " + "; ".join(sorted(actual_tags)),
        actual_tags == expected_tags and tags["goodwill_excluido"].all())

    mc = load("u9_monte_carlo_t54.csv")
    if not mc.empty:
        null = mc[mc.cenario == "Nulo (beta = 0)"]
        single = mc[mc.cenario == "Um único episódio"]
        if not null.empty and not single.empty:
            no = null.iloc[0]
            si = single.iloc[0]
            add("Opposite signs: Null e Single episode", "50,4% em cada",
                f"Null={no['sinais_opostos_%']:.2f}%; Single={si['sinais_opostos_%']:.2f}%",
                abs(no["sinais_opostos_%"] - 50.4) <= 2 and abs(si["sinais_opostos_%"] - 50.4) <= 2)
            add("Null: linhas condicionais com 19,7%", "19,7% e 19,7%",
                f"opostos|signif={no['sinais_opostos_|signif_%']:.2f}%; "
                f"rejeição igualdade bootstrap|signif={no['igualdade_rejeitada_boot_|signif_%']:.2f}%",
                abs(no["sinais_opostos_|signif_%"] - 19.7) <= 3
                and abs(no["igualdade_rejeitada_boot_|signif_%"] - 19.7) <= 3)
    clean = load("u9_nulo_limpo_400q.csv")
    if not clean.empty:
        clean54 = clean[clean["T"] == 54]
        clean400 = clean[clean["T"] == 400]
        if not clean54.empty and not clean400.empty:
            r54, r400 = clean54.iloc[0], clean400.iloc[0]
            add("Nulo limpo sem concentração", "aprox. 10%",
                f"T=54: {r54.rejeicao_DK_5pct_pct:.2f}%",
                abs(r54.rejeicao_DK_5pct_pct - 10) <= 3)
            add("Nulo limpo com 400 trimestres", "aprox. 6%",
                f"T=400: {r400.rejeicao_DK_5pct_pct:.2f}%",
                abs(r400.rejeicao_DK_5pct_pct - 6) <= 2.5)

    frame = pd.DataFrame(report)
    frame.to_csv(OUT / "relatorio_divergencias.csv", index=False)
    print("\nRELATORIO_DE_DIVERGENCIAS")
    print("| afirmação do texto | valor no texto | valor calculado | ✓/✗ |")
    print("|---|---|---|---|")
    for row in frame.itertuples(index=False):
        print(f"| {row.afirmacao_do_texto} | {row.valor_no_texto} | {row.valor_calculado} | {row.status} |")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-mc", action="store_true", help="skip the 2,000-replication Monte Carlo")
    parser.add_argument("--skip-power", action="store_true", help="skip 500 firm-cluster resamples")
    args = parser.parse_args()
    start = time.time()
    shock_raw = read_shock()
    concentration_task(shock_raw)
    estimates, outputs = lp_tasks()
    write_table("estimativas_lp.csv", estimates)
    imputation_task(outputs["prepared"])
    if not args.skip_power:
        u7_power_task(outputs["prepared"], estimates)
    tag_task()
    print("Rodada zero:", ZERO_NOTE)
    if not args.skip_mc:
        monte_carlo_task()

    report_divergences()
    print(f"\nConcluído em {time.time() - start:.1f}s. Saídas em {OUT}")


if __name__ == "__main__":
    main()
