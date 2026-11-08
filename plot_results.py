"""Résumé statistique, tolérances et figures à partir des SI-SDR par scène.

    eval_comparatif.csv          résumé par (modèle, erreur) : moyenne, médiane,
                                 moyenne plafonnée, % de scènes gagnées, IC 95 %
    tolerance.csv                tolérance e* (perte de 3 dB), sur la moyenne et la médiane
    asymetrie.csv                SI-SDR(-e) - SI-SDR(+e), apparié par scène, avec IC 95 %
    robustesse_amplitude.png     SI-SDR vs erreur, moyenne et médiane, selon l'amplitude du bruit
    forme_variance_egale.png     effet de la forme du bruit (uniforme vs gaussien) à variance égale
    compromis.png                SI-SDR à 0° vs tolérance : le compromis directivité / robustesse
    victoires.png                % de scènes où le modèle bat le beamformer cardioïde
    distribution_0deg.png        distribution des SI-SDR par scène à 0° (pourquoi moyenne et médiane divergent)
    fixe_vs_beamformer.png       modèle fixed vs beamformer, scènes immobiles
"""
import argparse
import os
import re
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CAP_DB = 30.0       # plafond pour la « moyenne plafonnée »
N_BOOT = 2000

# Identité visuelle : une couleur et un marqueur fixes par modèle.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "h"]
ORDRE = ["model_cond_none", "model_cond_Uniform", "model_cond_gaussian",
         "model_cond_unif8p7", "model_cond_gauss2p9", "model_cond_unif30", "model_cond_gauss10"]
BF_COLOR, HYPER_COLOR = "#52514c", "#9a9890"
INK, MUTED, GRID, AXIS = "#2b2b29", "#6b6a63", "#e4e3de", "#b5b3aa"

LABELS = {
    "model_cond_none": "sans bruit",
    "model_cond_Uniform": "U(±5°)",
    "model_cond_gaussian": "N(0, 5²)",
    "model_cond_unif8p7": "U(±8,7°)",
    "model_cond_gauss2p9": "N(0, 2,9²)",
    "model_cond_unif30": "U(±30°)",
    "model_cond_gauss10": "N(0, 10²)",
    "model_fixed_none": "AE fixed",
}
# Série d'amplitudes : un modèle par niveau de bruit (les comparaisons de forme sont à part)
AMPLITUDE = ["model_cond_none", "model_cond_Uniform", "model_cond_gaussian",
             "model_cond_gauss10", "model_cond_unif30"]
# Paires de même variance : (gaussien, uniforme)
PAIRES_FORME = [("model_cond_gauss2p9", "model_cond_Uniform", "écart-type 2,9°"),
                ("model_cond_unif8p7", "model_cond_gaussian", "écart-type 5°")]


def base_name(name):
    return re.sub(r"_s\d+$", "", name)


def style(name):
    b = base_name(name)
    i = ORDRE.index(b) if b in ORDRE else len(ORDRE)
    return SLOTS[i % len(SLOTS)], MARKERS[i % len(MARKERS)]


def label(name):
    b = base_name(name)
    return LABELS.get(b, b) + (name[len(b):] if b != name else "")


def _axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK)


# Statistiques
def _boot_mean_ci(x, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    return np.quantile(m, 0.025), np.quantile(m, 0.975)


def _per_scene(df):
    # Une valeur par scène : si des scènes miroir existent, on moyenne les deux versions
    keys = ["model_name", "mode", "az_error_deg", "scene"]
    cols = ["sisdr_input", "sisdr_beamformer", "sisdr_hypercardioid", "sisdr_model"]
    return df.groupby(keys, as_index=False)[cols].mean()


def summarize(scenes):
    # Résumé par (modèle, erreur). Les médianes et moyennes portent sur toutes les lignes
   #  (scènes et, s'il y en a, leurs miroirs) ; les IC rééchantillonnent les scènes.
    rows = []
    paired = _per_scene(scenes)
    for (nom, mode, err), g in scenes.groupby(["model_name", "mode", "az_error_deg"], sort=False):
        p = paired[(paired.model_name == nom) & (paired.az_error_deg == err)]
        row = {"model_name": nom, "mode": mode, "az_error_deg": err, "n_rows": len(g),
               "sisdr_input": g.sisdr_input.mean()}
        for col, short in [("sisdr_model", "model"), ("sisdr_beamformer", "beamformer"),
                           ("sisdr_hypercardioid", "hypercardioid")]:
            row[f"mean_{short}"] = g[col].mean()
            row[f"median_{short}"] = g[col].median()
            row[f"mean_cap30_{short}"] = g[col].clip(upper=CAP_DB).mean()
        row["mean_model_ci_low"], row["mean_model_ci_high"] = _boot_mean_ci(p.sisdr_model)
        d = p.sisdr_model - p.sisdr_beamformer
        row["model_minus_bf"] = d.mean()
        row["model_minus_bf_ci_low"], row["model_minus_bf_ci_high"] = _boot_mean_ci(d)
        row["win_rate_vs_bf"] = (g.sisdr_model > g.sisdr_beamformer).mean()
        row["win_rate_vs_hyper"] = (g.sisdr_model > g.sisdr_hypercardioid).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def tolerance(err, val, drop=3.0):
    # Plus grande erreur (>= 0) telle que val >= val(0) - drop, par interpolation linéaire.# 
    order = np.argsort(err)
    err, val = np.asarray(err, dtype=float)[order], np.asarray(val, dtype=float)[order]
    if 0.0 not in err:
        return np.nan
    i0 = int(np.where(err == 0.0)[0][0])
    thr = val[i0] - drop
    for i in range(i0 + 1, len(err)):
        if val[i] < thr:
            return err[i - 1] + (err[i] - err[i - 1]) * (val[i - 1] - thr) / (val[i - 1] - val[i])
    return np.inf


def _tol_both_sides(g, col):
    pos = g[g.az_error_deg >= 0]
    neg = g[g.az_error_deg <= 0]
    t_pos = tolerance(pos.az_error_deg, pos[col])
    t_neg = tolerance(-neg.az_error_deg, neg[col]) if len(neg) > 1 else np.nan
    return t_pos, t_neg


def tolerance_table(summary):
    rows = []
    cond = summary[summary["mode"] == "conditioned"]
    for nom, g in cond.groupby("model_name"):
        r = {"model_name": nom,
             "mean_0deg": float(g.loc[g.az_error_deg == 0, "mean_model"].iloc[0]),
             "median_0deg": float(g.loc[g.az_error_deg == 0, "median_model"].iloc[0])}
        r["tol_mean_pos"], r["tol_mean_neg"] = _tol_both_sides(g, "mean_model")
        r["tol_median_pos"], r["tol_median_neg"] = _tol_both_sides(g, "median_model")
        rows.append(r)
    if len(cond):
        g = cond[cond.model_name == cond.model_name.iloc[0]]
        for short, nom in [("beamformer", "beamformer_cardioide"), ("hypercardioid", "beamformer_hypercardioide")]:
            r = {"model_name": nom,
                 "mean_0deg": float(g.loc[g.az_error_deg == 0, f"mean_{short}"].iloc[0]),
                 "median_0deg": float(g.loc[g.az_error_deg == 0, f"median_{short}"].iloc[0])}
            r["tol_mean_pos"], r["tol_mean_neg"] = _tol_both_sides(g, f"mean_{short}")
            r["tol_median_pos"], r["tol_median_neg"] = _tol_both_sides(g, f"median_{short}")
            rows.append(r)
    return pd.DataFrame(rows)


def asymmetry_table(scenes):
    # SI-SDR(-e) - SI-SDR(+e) par scène. Avec les scènes miroir, l'asymétrie due aux
   #  scènes disparaît : ce qui reste est propre au modèle.
    p = _per_scene(scenes[scenes["mode"] == "conditioned"])
    rows = []
    errs = sorted(e for e in p.az_error_deg.unique() if e > 0 and -e in set(p.az_error_deg))
    sources = [(n, "sisdr_model") for n in p.model_name.unique()]
    ref = p.model_name.iloc[0]
    sources += [("beamformer_cardioide", "sisdr_beamformer"), ("beamformer_hypercardioide", "sisdr_hypercardioid")]
    for nom, col in sources:
        src = p[p.model_name == (ref if nom.startswith("beamformer") else nom)]
        for e in errs:
            a = src[src.az_error_deg == -e].set_index("scene")[col]
            b = src[src.az_error_deg == e].set_index("scene")[col]
            d = (a - b).dropna()
            lo, hi = _boot_mean_ci(d)
            rows.append({"model_name": nom, "erreur_deg": e, "asym_moins_plus": d.mean(),
                         "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


# Figures
def _present(names, df):
    have = set(df.model_name)
    return [n for n in names if n in have]


def plot_amplitude(summary, out):
    cond = summary[summary["mode"] == "conditioned"]
    names = _present(AMPLITUDE, cond)
    if not names:
        return
    ref = cond[cond.model_name == names[0]].sort_values("az_error_deg")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, stat, titre in [(axes[0], "mean", "Moyenne"), (axes[1], "median", "Médiane")]:
        ax.plot(ref.az_error_deg, ref[f"{stat}_beamformer"], color=BF_COLOR, ls="--", lw=2,
                marker="o", ms=6, label="Beamformer cardioïde")
        ax.plot(ref.az_error_deg, ref[f"{stat}_hypercardioid"], color=HYPER_COLOR, ls=":", lw=2,
                marker="o", ms=5, label="Beamformer hypercardioïde")
        for n in names:
            g = cond[cond.model_name == n].sort_values("az_error_deg")
            c, m = style(n)
            if stat == "mean":
                ax.fill_between(g.az_error_deg, g.mean_model_ci_low, g.mean_model_ci_high,
                                color=c, alpha=0.12, lw=0)
            ax.plot(g.az_error_deg, g[f"{stat}_model"], color=c, marker=m, ms=7, lw=2,
                    markeredgecolor="white", markeredgewidth=1, label=f"AE, {label(n)}")
        ax.axhline(ref.sisdr_input.iloc[0], color=AXIS, lw=1)
        _axes(ax, "Erreur de visée (°)", "SI-SDR (dB)" if stat == "mean" else "")
        ax.set_title(f"{titre} sur les scènes (ligne grise : mélange brut)", color=INK, fontsize=11, loc="left")
    fig.suptitle("Sources mobiles : SI-SDR selon l'erreur de visée et l'amplitude du bruit d'entraînement",
                 color=INK, fontweight="bold", x=0.01, ha="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)

# Écart apparié gaussien - uniforme à variance égale, avec IC 95 %.
def plot_shape(scenes, out):
    p = _per_scene(scenes[scenes["mode"] == "conditioned"])
    pairs = [(a, b, t) for a, b, t in PAIRES_FORME if {a, b} <= set(p.model_name)]
    if not pairs:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.axhline(0, color=BF_COLOR, lw=1.2)
    for a, b, titre in pairs:
        A = p[p.model_name == a].set_index(["az_error_deg", "scene"]).sisdr_model
        B = p[p.model_name == b].set_index(["az_error_deg", "scene"]).sisdr_model
        d = (A - B).dropna()
        errs = sorted(d.index.get_level_values(0).unique())
        mean = [d.loc[e].mean() for e in errs]
        ci = [_boot_mean_ci(d.loc[e]) for e in errs]
        c, m = style(a)
        ax.fill_between(errs, [x[0] for x in ci], [x[1] for x in ci], color=c, alpha=0.15, lw=0)
        ax.plot(errs, mean, color=c, marker=m, ms=7, lw=2, markeredgecolor="white",
                label=f"{label(a)} − {label(b)}  ({titre})")
    _axes(ax, "Erreur de visée (°)", "Écart apparié de SI-SDR (dB)")
    ax.set_title("Forme du bruit à variance égale : gaussien − uniforme (IC 95 %)",
                 color=INK, fontweight="bold", loc="left")
    ax.annotate("au-dessus de 0 : le gaussien fait mieux", xy=(0.99, 0.97), xycoords="axes fraction",
                ha="right", va="top", fontsize=8.5, color=MUTED)
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# décalage (dx, dy, alignement) des étiquettes du nuage « compromis », pour éviter les chevauchements
LABEL_OFFSET = {"model_cond_none": (-8, 6, "right"), "model_cond_Uniform": (8, 6, "left"),
                "model_cond_gauss2p9": (-8, -12, "right")}


def plot_tradeoff(tol, out, x_inf=34.0):
    t = tol[tol.model_name.str.startswith("model_cond")]
    if t.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for ax, stat, titre in [(axes[0], "mean", "Moyenne"), (axes[1], "median", "Médiane")]:
        for _, r in t.iterrows():
            # tolérance retenue : la plus petite des deux côtés (le pire cas)
            x = np.nanmin([r[f"tol_{stat}_pos"], r[f"tol_{stat}_neg"]])
            x_plot = x_inf if np.isinf(x) else x
            c, m = style(r.model_name)
            ax.plot(x_plot, r[f"{stat}_0deg"], marker=m, ms=10, color=c, markeredgecolor="white",
                    ls="none", clip_on=False)
            txt = label(r.model_name) + (" (> 30°)" if np.isinf(x) else "")
            dx, dy, ha = LABEL_OFFSET.get(base_name(r.model_name), (7, 4, "left"))
            ax.annotate(txt, xy=(x_plot, r[f"{stat}_0deg"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8.5, color=INK, ha=ha)
        _axes(ax, "Tolérance e* (°), pire des deux côtés", "SI-SDR à 0° d'erreur (dB)")
        ax.set_xlim(0, x_inf + 8)
        ax.margins(y=0.12)
        ax.set_title(f"{titre}", color=INK, fontsize=11, loc="left")
    fig.suptitle("Compromis directivité / robustesse : plus le bruit est large, plus la tolérance augmente "
                 "et plus le SI-SDR à 0° baisse", color=INK, fontweight="bold", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_wins(summary, out):
    cond = summary[summary["mode"] == "conditioned"]
    names = _present(AMPLITUDE, cond)
    if not names:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(50, color=BF_COLOR, lw=1.2)
    ax.annotate("50 %", xy=(0.01, 50), xycoords=("axes fraction", "data"),
                xytext=(0, 4), textcoords="offset points", fontsize=8.5, color=MUTED)
    for n in names:
        g = cond[cond.model_name == n].sort_values("az_error_deg")
        c, m = style(n)
        ax.plot(g.az_error_deg, 100 * g.win_rate_vs_bf, color=c, marker=m, ms=7, lw=2,
                markeredgecolor="white", label=label(n))
    _axes(ax, "Erreur de visée (°)", "Scènes où le modèle bat le cardioïde (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Part des scènes où l'autoencodeur bat le beamformer cardioïde",
                 color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, title="Bruit d'entraînement", title_fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_distribution(scenes, out, models=("model_cond_none", "model_cond_unif30")):
    c = scenes[(scenes["mode"] == "conditioned") & (scenes.az_error_deg == 0)]
    if c.empty:
        return
    ref = c[c.model_name == c.model_name.iloc[0]]
    fig, ax = plt.subplots(figsize=(10, 5))

    def ecdf(x, **kw):
        x = np.sort(np.asarray(x))
        ax.step(x, np.arange(1, len(x) + 1) / len(x) * 100, where="post", lw=2, **kw)

    ecdf(ref.sisdr_beamformer, color=BF_COLOR, ls="--", label="Beamformer cardioïde")
    ecdf(ref.sisdr_hypercardioid, color=HYPER_COLOR, ls=":", label="Beamformer hypercardioïde")
    for n in models:
        g = c[c.model_name == n]
        if len(g):
            ecdf(g.sisdr_model, color=style(n)[0], label=f"AE, {label(n)}")
    bf = ref.sisdr_beamformer
    ax.axvline(CAP_DB, color=AXIS, lw=1)
    ax.annotate(f"{(bf > 20).sum()} scènes > 20 dB pour le cardioïde\n(interférent dans son zéro arrière)",
                xy=(CAP_DB, 75), xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center")
    _axes(ax, "SI-SDR d'une scène (dB)", "Part des scènes en dessous (%)")
    ax.set_xlim(min(-5, np.floor(c.sisdr_model.min())), min(60, max(40, bf.max())))
    ax.set_ylim(0, 100)
    ax.set_title("Distribution des SI-SDR par scène, 0° d'erreur (sources mobiles)",
                 color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_fixed(summary, out):
    fx = summary[summary["mode"] == "fixed"]
    if fx.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for n in fx.model_name.unique():
        g = fx[fx.model_name == n].sort_values("az_error_deg")
        ax.plot(g.az_error_deg, g.mean_beamformer, color=BF_COLOR, ls="--", lw=2, marker="o", ms=6,
                label="Cardioïde visé à 0° + erreur, moyenne")
        ax.plot(g.az_error_deg, g.median_beamformer, color=BF_COLOR, ls=":", lw=2, marker="o", ms=5,
                mfc="white", label="Cardioïde visé à 0° + erreur, médiane")
        ax.plot(g.az_error_deg, g.mean_model, color=SLOTS[0], lw=2, marker="v", ms=7,
                markeredgecolor="white", label=f"{label(n)}, moyenne (n'utilise pas la consigne)")
        ax.plot(g.az_error_deg, g.median_model, color=SLOTS[0], lw=2, ls=":", marker="v", ms=6,
                mfc="white", label=f"{label(n)}, médiane")
        r0 = g[g.az_error_deg == 0].iloc[0]
        ax.annotate(f"à 0° : bat le cardioïde sur {100 * r0.win_rate_vs_bf:.0f} % des scènes",
                    xy=(0.01, 0.04), xycoords="axes fraction", fontsize=9, color=INK)
        break
    ax.axhline(fx.sisdr_input.iloc[0], color=AXIS, lw=1)
    _axes(ax, "Erreur de visée donnée au beamformer (°)", "SI-SDR (dB)")
    ax.set_ylim(bottom=min(-1, fx.sisdr_input.min() - 1))
    ax.set_title("Cible fixe à 0°, sources immobiles", color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def make_all(per_scene_csv, out_dir="results"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = pd.read_csv(per_scene_csv)
    if "mirror" in scenes and scenes.mirror.any():
        print("Scènes miroir présentes : statistiques calculées sur les scènes ET leurs miroirs.")

    summary = summarize(scenes)
    summary.to_csv(out_dir / "eval_comparatif.csv", index=False, float_format="%.4f")
    tol = tolerance_table(summary)
    tol.to_csv(out_dir / "tolerance.csv", index=False, float_format="%.2f")
    asym = asymmetry_table(scenes)
    asym.to_csv(out_dir / "asymetrie.csv", index=False, float_format="%.3f")

    plot_amplitude(summary, out_dir / "robustesse_amplitude.png")
    plot_shape(scenes, out_dir / "forme_variance_egale.png")
    plot_tradeoff(tol, out_dir / "compromis.png")
    plot_wins(summary, out_dir / "victoires.png")
    plot_distribution(scenes, out_dir / "distribution_0deg.png")
    plot_fixed(summary, out_dir / "fixe_vs_beamformer.png")

    fmt = lambda v: f"{v:.2f}"
    print("\nTolérances (°) :\n" + tol.to_string(index=False, float_format=fmt))
    cols = ["model_name", "az_error_deg", "mean_model", "median_model", "win_rate_vs_bf"]
    print("\nÀ 0° d'erreur :\n" + summary[summary.az_error_deg == 0][cols].to_string(index=False, float_format=fmt))
    print(f"\nRésultats écrits dans {out_dir}/")
    return summary, tol, asym


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per_scene", default="results/eval_par_scene.csv")
    p.add_argument("--out_dir", default="results")
    args = p.parse_args()
    make_all(args.per_scene, args.out_dir)


if __name__ == "__main__":
    main()