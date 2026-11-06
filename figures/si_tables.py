"""LaTeX bodies of the Supporting-information tables, built from data/ only (no transcription by hand).

Writes figures/si_tables.tex: one `% === name ===` block per table (tabular environment only; captions are in the
paper's Supporting Information), for Tables S1-S10.

Usage: python figures/si_tables.py
"""
import sys
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from style import DATA, MODEL_LABEL, MODELS, OUT, ROOT

sys.path.insert(0, str(ROOT / "src"))
from caravan_io import ROOT as _R  # noqa: E402,F401
from rq2_parameter import benchmarks  # noqa: E402


def f(x, nd=2):
    """Round half away from zero on the decimal value. The summary CSVs store 6 decimals (since 2026-10-22), so
    this no longer rounds an already-rounded value."""
    if pd.isna(x):
        return "--"
    s = str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
    if float(s) == 0:   # no "-0.00"
        s = s.lstrip("-")
    return s.replace("-", "$-$") if s.startswith("-") else s


def pct(x):
    return "--" if pd.isna(x) else f"{100 * x:.0f}"


def n(x):
    return f"{int(x):,}"


def tabular(spec, header, rows):
    out = [f"\\begin{{tabular}}{{{spec}}}", "\\hline", " & ".join(header) + " \\\\", "\\hline"]
    out += [" & ".join(r) + " \\\\" for r in rows]
    return "\n".join(out + ["\\hline", "\\end{tabular}"])


blocks = {}
LEVEL_NAME = {"parameter": "Calibrated capacity", "state": "Simulated deficit", "control": "Storage-free deficit"}
M = [MODEL_LABEL[m] for m in MODELS]

# S1: benchmarks among themselves
cat = pd.read_csv(DATA / "catalog/catchments.csv", index_col="gauge_id", low_memory=False)
cat = cat.loc[pd.read_csv(DATA / "catalog/screened_catchments.csv").gauge_id]
b = benchmarks("screened_catchments", cat, 0.5)
cols = {"climate": "Climate", "climate_mod16": "MOD16 only", "climate_ssebop": "SSEBop only",
        "stocker": "Stocker", "zhao": "Zhao", "soil": "Soil"}
r = b[list(cols)].corr("spearman")
rows = [[cols[i]] + [f(r.loc[i, j]) if k < list(cols).index(i) else ("1" if i == j else "") for k, j in
                     enumerate(cols)] + [n(b[i].notna().sum())] for i in cols]
blocks["benchmark_correlations"] = tabular("l" + "c" * len(cols) + "r", ["Benchmark"] + list(cols.values()) + ["$n$"],
                                           rows)

# S2: unidentifiable share by screening and behavioural threshold and rule
k = pd.read_csv(DATA / "sensitivity/kge_classification_screened_catchments.csv")
combos = [(0.5, 0.5), (0.3, 0.5), (0.7, 0.5), (0.5, 0.3), (0.5, 0.7), (0.3, 0.3), (0.7, 0.7)]
rules = [("envelope", "Main"), ("fixed10", "10-fold"), ("halfprior", "Half")]
rows = []
for sc, be in combos:
    row = [f"{sc}", f"{be}"]
    for m in MODELS:
        g = k[(k.screen == sc) & (k.behavioural == be) & (k.model == m)].set_index("rule")
        row.append(n(g.loc["envelope", ["unidentifiable", "consistent", "inconsistent"]].sum()))
        row += [pct(g.loc[r_, "unidentifiable_share"]) for r_, _ in rules]
    rows.append(row)
hdr = ["KGE", "Behav."] + ["$n$", "Main", "10-f.", "Half"] * 3
top = (" & & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{m}}}" for m in M) + " \\\\\n"
       "\\cline{3-6} \\cline{7-10} \\cline{11-14}\n")
t = tabular("cc" + "rccc" * 3, hdr, rows)
blocks["unidentifiable_thresholds"] = t.replace("\\hline\n", "\\hline\n" + top, 1)

# S3: agreement sensitivity, Spearman with the climate benchmark
s = pd.read_csv(DATA / "rq3/state_vs_benchmarks_summary_screened_catchments.csv")
s = s[s.stratum == "all"]
ka = pd.read_csv(DATA / "sensitivity/kge_agreement_screened_catchments.csv")
variants = [("Main analysis", s, "climate"), ("Return period 10 years", s, "climate_T10"),
            ("Return period 40 years", s, "climate_T40"), ("Including $E > P$", s, "climate_incl_EgtP"),
            ("MOD16 only", s, "climate_mod16"), ("SSEBop only", s, "climate_ssebop")]
rows = []
for label, tab, bench in variants:
    row = [label]
    for lev in ("parameter", "state", "control"):
        for m in MODELS:
            v = tab[(tab.level == lev) & (tab.model == m) & (tab.benchmark == bench)].spearman
            row.append(f(v.iloc[0]) if len(v) else "--")
    rows.append(row)
for sc in (0.3, 0.7):
    row = [f"KGE threshold {sc}"]
    for lev in ("parameter", "state", "control"):
        for m in MODELS:
            v = ka[(ka.screen == sc) & (ka.behavioural == 0.5) & (ka.level == lev) & (ka.model == m)
                   & (ka.benchmark == "climate")].spearman
            row.append(f(v.iloc[0]))
    rows.append(row)
hdr = ["Variant"] + ["GR6J", "HBV", "SAC"] * 3
top = (" & \\multicolumn{3}{c}{Calibrated capacity} & \\multicolumn{3}{c}{Simulated deficit} & \\multicolumn{3}{c}{Storage-free deficit} \\\\\n"
       "\\cline{2-4} \\cline{5-7} \\cline{8-10}\n")
t = tabular("l" + "c" * 9, hdr, rows)
blocks["agreement_sensitivity"] = t.replace("\\hline\n", "\\hline\n" + top, 1)

# S4: partial correlations of the simulated and storage-free deficits
par = pd.read_csv(DATA / "rq3/coupling_partial_screened_catchments.csv")
# benchmarks in the paper's order within each model
par = par.assign(_m=par.model.map({m: i for i, m in enumerate(MODELS)}),
                 _b=par.benchmark.map({"climate": 0, "stocker": 1, "zhao": 2, "soil": 3})).sort_values(["_m", "_b"])
rows = []
for _, r_ in par.iterrows():
    rows.append([MODEL_LABEL[r_.model], r_.benchmark.capitalize(), n(r_.n), f(r_.state), f(r_.control),
                 f"{f(r_.state_given_control)} [{f(r_.state_given_control_lo)}, {f(r_.state_given_control_hi)}]",
                 f(r_.control_given_state)])
blocks["partials"] = tabular("llrcccc", ["Model", "Benchmark", "$n$", "Sim.", "Free", "Sim. $\\mid$ free [90\\%]",
                                         "Free $\\mid$ sim."], rows)

# S5-S7: prior-agreement tests (water-balance benchmark, groups, restricted range)
g = pd.read_csv(DATA / "prior_tests/gao_agreement_screened_catchments.csv")
g = g[g["sample"] == "common"]
rows = []
for lev in ("parameter", "state", "control"):
    row = [LEVEL_NAME[lev]]
    for m in MODELS:
        for bench in ("climate", "gao"):
            v = g[(g.level == lev) & (g.model == m) & (g.benchmark == bench)].spearman
            row.append(f(v.iloc[0]) if len(v) else "--")
    rows.append(row)
blocks["prior_gao"] = tabular("l" + "cc" * 3, ["Quantity"] + [x for m in ("GR6J", "HBV", "SAC") for x in
                                                         (f"{m} sat.", f"{m} $P{{-}}Q$")], rows)
cl = pd.read_csv(DATA / "prior_tests/cluster_agreement_screened_catchments.csv")
cl = cl[cl.benchmark == "climate"]
groups = [("catchment", "Catchments"), ("kmeans50", "50 groups"), ("kmeans10", "10 groups"), ("stratum", "Climate classes")]
rows = []
for gr, lab in groups:
    for lev in ("parameter", "state"):
        row = [lab, LEVEL_NAME[lev]]
        for m in MODELS:
            v = cl[(cl.grouping == gr) & (cl.level == lev) & (cl.model == m)]
            row += [n(v.n_groups.iloc[0]) if len(v) else "--", f(v.spearman.iloc[0]) if len(v) else "--"]
        rows.append(row)
blocks["prior_clusters"] = tabular("ll" + "rc" * 3, ["Grouping", "Quantity"] + [x for m in ("GR6J", "HBV", "SAC") for x in
                                                                           (f"{m} groups", "$\\rho$")], rows)
cp = pd.read_csv(DATA / "prior_tests/constrained_prior_summary_screened_catchments.csv")
cp = cp[cp["sample"] == "in_band"]
# Median change in calibration KGE, from the per-pair table (as the summary now also gives at 6 decimals).
cpc = pd.read_csv(DATA / "prior_tests/constrained_prior_screened_catchments.csv")
dkge = cpc[cpc.n_in > 0].groupby(["model", "band"]).dkge_cal.median()
rows = []
for band, lab in (("narrow", "$S_{R,10}$--$S_{R,40}$"), ("wide", "$S_{R,5}$--$S_{R,80}$")):
    for m in MODELS:
        x = cp[(cp.band == band) & (cp.model == m)].set_index("estimate")
        rows.append([lab, MODEL_LABEL[m], n(x.loc["free_opt", "n"]), f(x.loc["free_opt", "spearman"]),
                     f(x.loc["band_best", "spearman"]), f(x.loc["band_prior_median", "spearman"]),
                     f(dkge[(m, band)]), pct(x.iloc[0].share_band_best_passes)])
blocks["prior_constrained"] = tabular("llrccccc", ["Range", "Model", "$n$", "Unrestr.", "Restr.", "No calib.", "$\\Delta$KGE",
                                                   "Pass (\\%)"], rows)

# S8: classification counts under each rule
c = pd.read_csv(DATA / "classification/classification_summary_screened_catchments.csv")
c = c[c.group == "all"]
rule_lab = {"class": "Main criterion", "class_ci": "Lower CI of width", "class_nosoil": "Range without soil",
            "class_fixed10": "Fixed 10-fold", "class_halfprior": "Half sampling range"}
rows = []
for m in MODELS:
    for v, lab in rule_lab.items():
        x = c[(c.model == m) & (c.variant == v)].iloc[0]
        tot = x.unidentifiable + x.consistent + x.inconsistent
        rows.append([MODEL_LABEL[m] if v == "class" else "", lab, n(x.fails_screen), n(x.few_behavioural),
                     n(x.not_classifiable), n(x.unidentifiable), n(x.consistent), n(x.inconsistent),
                     pct(x.unidentifiable / tot)])
blocks["classification"] = tabular("llrrrrrrr", ["Model", "Criterion", "KGE $<0.5$", "$<10$ sets", "$<2$ bench.", "Unid.", "Cons.",
                                                 "Incons.", "Unid. (\\%)"], rows)

# S9-S10: robustness runs and re-analyses (robustness/, written by src/robustness_*.py)
REV = ROOT / "robustness"
per = pd.read_csv(REV / "AD_per_catchment.csv")
BENCH4 = ("climate", "stocker", "zhao", "soil")
CRIT = ("class", "class_ci", "class_nosoil", "class_fixed10", "class_halfprior")
CRIT_HDR = ["Main", "Lower CI", "No soil", "10-fold", "Half"]
A = pd.read_csv(REV / "A_hbv_multiobjective.csv").set_index("variant")
B = pd.read_csv(REV / "B_hbv_convergence.csv").set_index("variant")
C = pd.read_csv(REV / "C_gr6j_optimiser.csv").iloc[0]
D = pd.read_csv(REV / "D_gr4j.csv").set_index("variant")
S = pd.read_csv(REV / "S_sacsma_optimum.csv").iloc[0]
J = pd.read_csv(REV / "J_flex.csv").set_index("variant")
Jper = pd.read_csv(REV / "J_per_catchment.csv")


def flex_row(model, label, variant):
    p = Jper[Jper.variant == variant]
    passes = (p.kge_cal >= 0.5) & (p.kge_val >= 0.5)
    return [model, label, n(passes.sum()), f(p.kge_cal.median(), 3)] + [f(J.loc[variant, f"rho_param_{x}"]) for x in BENCH4]


def optimum_row(model, label, item, variant, rho):
    p = per[(per.item == item) & (per.variant == variant)]
    passes = (p.kge_cal >= 0.5) & (p.kge_val >= 0.5)
    return [model, label, n(passes.sum()), f(p.kge_cal.median(), 3)] + [f(rho(x)) for x in BENCH4]


rows = [optimum_row("GR6J", "Michel (main)", "C", "michel", lambda x: C[f"rho_michel_{x}"]),
        optimum_row("", "SCE-UA", "C", "sce", lambda x: C[f"rho_sce_{x}"]),
        optimum_row("GR4J", "Michel", "D", "GR4J", lambda x: D.loc["GR4J", f"rho_param_{x}"]),
        optimum_row("HBV", "SCE-UA (main)", "A", "main", lambda x: A.loc["main", f"rho_param_{x}"]),
        optimum_row("", "SCE-UA, full budget", "A", "so", lambda x: A.loc["so", f"rho_param_{x}"]),
        optimum_row("", "SCE-UA, two objectives", "A", "mo_both", lambda x: A.loc["mo_both", f"rho_param_{x}"]),
        optimum_row("SAC-SMA", "SCE-UA (main)", "S", "main", lambda x: S[f"rho_main_{x}"]),
        optimum_row("", "SCE-UA, full budget", "S", "so", lambda x: S[f"rho_so_{x}"]),
        flex_row("FLEX", "SCE-UA, full budget", "FLEX_so"),
        flex_row("", "SCE-UA, two objectives", "FLEX_mo"),
        flex_row("", "SCE-UA, $S_{u,\\mathrm{max}}$ 30--700~mm", "FLEX_liang")]
blocks["robustness_optima"] = tabular("llrccccc", ["Model", "Calibration", "Pass", "KGE", "Climate", "Stocker", "Zhao",
                                                 "Soil"], rows)


# S10: identifiability in three blocks (subset runs, SAC-SMA prior reweighting, equal behavioural sets)
E = pd.read_csv(REV / "E_sacsma_reweight_classes.csv").set_index(["variant", "prior"])
G = pd.read_csv(REV / "G_common_subset_classes.csv")
G = G[G.variant == "min"].set_index(["model", "criterion"])


def ident_row(model, label, t):
    return [model, label, n(t.assessed_class)] + [pct(t[f"unidentifiable_{c_}"]) for c_ in CRIT]


def group(title):
    return [f"\\multicolumn{{{3 + len(CRIT)}}}{{l}}{{\\textit{{{title}}}}}"]


rows = [group("Subset of 397 catchments"),
        ident_row("GR6J", "Main analysis", D.loc["GR6J"]),
        ident_row("GR4J", "100,000 samples", D.loc["GR4J"]),
        ident_row("HBV", "Main analysis", A.loc["main"]),
        ident_row("", "150,000 samples", B.loc["150k"]),
        ident_row("", "Two objectives, 150,000 samples", A.loc["mo_both_150k"]),
        ident_row("FLEX", "40,000 samples", J.loc["FLEX_so"]),
        ident_row("", "Two objectives, 40,000 samples", J.loc["FLEX_mo"]),
        ident_row("", "$S_{u,\\mathrm{max}}$ 30--700~mm", J.loc["FLEX_liang"]),
        group("Full sample")]
for prior, lab in (("unweighted", "Sampled (main)"), ("weighted", "Uniform prior of the sum")):
    rows.append(["SAC-SMA" if prior == "unweighted" else "", lab, n(E.loc[("class", prior), "assessed"])]
                + [pct(E.loc[(c_, prior), "unidentifiable_share"]) for c_ in CRIT])
rows.append(group("Common catchments"))
for m in MODELS:
    for col, lab in (("full_unidentifiable", "All sets"), ("subsampled_unidentifiable", "Equal number")):
        rows.append([MODEL_LABEL[m] if col == "full_unidentifiable" else "", lab,
                     n(G.loc[(m, "class"), "n_catchments"])] + [pct(G.loc[(m, c_), col]) for c_ in CRIT])
blocks["robustness_identifiability"] = tabular("llr" + "c" * len(CRIT), ["Model", "Variant", "$n$"] + CRIT_HDR, rows)

with open(OUT / "si_tables.tex", "w") as fh:
    for name, body in blocks.items():
        fh.write(f"% === {name} ===\n{body}\n\n")
print("wrote", len(blocks), "tables to", OUT / "si_tables.tex")
