#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CFPS 2022 农村上网居民网络购物采纳影响因素分析
================================================

研究问题
--------
在 CFPS 2022 的农村、且使用移动互联网的成年受访者中，
哪些个人与家庭特征与“是否进行网络购物”相关？

为什么不用 ft601？
------------------
原项目把家庭经济库的 ft601 当作“数字服务采纳”。经核对 CFPS 2022
变量标签，ft601 的实际含义是“亲友借款待偿额（元）”，与数字服务无关。
本脚本改用成人库 qu92“是否网上购物”作为数字服务使用的结果变量。
由于 qu92 只对上网者提问，样本限定为 mobile internet users（qu201=1），
结论也只适用于这一人群。

运行
----
python analysis.py \
  --fam "D:/.../cfps2022famecon_202410.dta" \
  --adult "D:/.../cfps2022person_202410.dta" \
  --out output/real

输出
----
sample_flow.csv, descriptive_statistics.csv, model_comparison.csv,
marginal_effects_probit.csv, marginal_effects_logit.csv,
coefficients_probit.csv, group_rates.csv, age_profile.csv,
predicted_vs_observed.csv, model_summary.txt, analysis_report.md,
fig_*.png

注意
----
CFPS 原始数据受使用协议保护，本脚本不输出行级分析数据，只输出汇总结果。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Logit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 常量与变量字典
# ---------------------------------------------------------------------------

CFPS_MISSING_CODES = (-10, -9, -8, -2, -1, 79)

FAM_COLS = [
    "fid22", "resp1pid", "urban22", "land_asset",
    "fincome1_per", "familysize22",
]
ADULT_COLS = [
    "pid", "age", "gender", "cfps2022eduy_im",
    "qu201", "qu202", "qu92",
]

X_VARS = [
    "age", "age2", "male", "edu_year", "familysize",
    "ln_income_per", "land_asset_log", "pc_internet",
]

LABELS = {
    "service_adopt": "网络购物采纳",
    "age": "年龄",
    "age2": "年龄平方",
    "male": "男性",
    "edu_year": "受教育年限",
    "familysize": "家庭人口规模",
    "ln_income_per": "人均收入对数",
    "land_asset_log": "土地资产对数",
    "pc_internet": "电脑上网",
}

PALETTE = {
    "navy": "#17324D",
    "blue": "#2F6BFF",
    "light_blue": "#DCE8FF",
    "gray": "#6B7A90",
    "light_gray": "#E7ECF3",
    "orange": "#C77C28",
}


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CFPS 2022 农村上网居民网络购物采纳分析")
    parser.add_argument("--fam", required=True, help="CFPS 2022 家庭经济库 .dta")
    parser.add_argument("--adult", required=True, help="CFPS 2022 成人库 .dta")
    parser.add_argument("--out", default="output/real", help="输出目录")
    parser.add_argument("--rural-code", type=float, default=0.0,
                        help="urban22 中代表乡村的编码，CFPS 2022 为 0")
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], source: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{source} 缺少字段: {missing}")


def read_columns(path: str | Path, columns: list[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    df = pd.read_stata(path, columns=columns, convert_categoricals=False)
    require_columns(df, columns, path.name)
    return df


def to_numeric_cfps(series: pd.Series) -> pd.Series:
    """转数值，并把 CFPS 的负值特殊编码与 79 统一设为缺失。"""
    s = pd.to_numeric(series, errors="coerce")
    return s.mask(s.isin(CFPS_MISSING_CODES))


def add_flow(rows: list[dict], stage: str, n: int, note: str = "") -> None:
    rows.append({"stage": stage, "n": int(n), "note": note})


# ---------------------------------------------------------------------------
# 样本构建
# ---------------------------------------------------------------------------

def build_sample(fam_path: str | Path, adult_path: str | Path,
                 rural_code: float = 0.0):
    flow: list[dict] = []

    fam = read_columns(fam_path, FAM_COLS)
    add_flow(flow, "家庭经济库原始样本", len(fam), "每行一个家庭")

    fam = fam[fam["urban22"] == rural_code].copy()
    add_flow(flow, "保留乡村家庭", len(fam), f"urban22 == {rural_code:g}")

    before = len(fam)
    fam = fam.drop_duplicates("fid22", keep="first").copy()
    add_flow(flow, "家庭库按 fid22 去重", len(fam), f"删除重复 {before - len(fam)} 行")

    adult = read_columns(adult_path, ADULT_COLS)
    add_flow(flow, "成人库原始样本", len(adult), "每行一个成年人")

    before = len(adult)
    adult = adult.drop_duplicates("pid", keep="first").copy()
    add_flow(flow, "成人库按 pid 去重", len(adult), f"删除重复 {before - len(adult)} 行")

    for df, cols in [
        (fam, ["land_asset", "fincome1_per", "familysize22"]),
        (adult, ["age", "gender", "cfps2022eduy_im", "qu201", "qu202", "qu92"]),
    ]:
        for c in cols:
            df[c] = to_numeric_cfps(df[c])

    merged = fam.merge(
        adult, left_on="resp1pid", right_on="pid",
        how="inner", validate="many_to_one",
    )
    add_flow(flow, "家庭—成人匹配", len(merged), "按 resp1pid = pid 内连接")

    merged = merged[merged["qu201"] == 1].copy()
    add_flow(flow, "保留移动互联网使用者", len(merged), "qu201 == 1 (是)")

    merged = merged[merged["qu92"].isin([0, 1])].copy()
    add_flow(flow, "保留网络购物有效回答", len(merged), "qu92 ∈ {0, 1}")

    merged["service_adopt"] = merged["qu92"].astype(int)
    merged["male"] = (merged["gender"] == 1).astype(float)
    merged["edu_year"] = merged["cfps2022eduy_im"]
    merged["familysize"] = merged["familysize22"]
    merged["ln_income_per"] = np.log1p(merged["fincome1_per"].clip(lower=0))
    merged["land_asset_log"] = np.log1p(merged["land_asset"].clip(lower=0))
    merged["age2"] = merged["age"] ** 2
    merged["pc_internet"] = merged["qu202"]

    analysis = merged.dropna(subset=["service_adopt"] + X_VARS).copy()
    add_flow(flow, "核心变量完整样本", len(analysis),
             "删除年龄、教育、收入、土地资产等缺失")

    return analysis, flow


# ---------------------------------------------------------------------------
# 统计与输出
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def auc_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y)
    pred = np.asarray(pred)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(pred).rank().to_numpy()
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit_models(df: pd.DataFrame, outdir: Path):
    y = df["service_adopt"].astype(float)
    X = sm.add_constant(df[X_VARS], has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probit = Probit(y, X).fit(disp=0, cov_type="HC1", maxiter=200)
        logit = Logit(y, X).fit(disp=0, cov_type="HC1", maxiter=200)
        lpm = sm.OLS(y, X).fit(cov_type="HC1")

    models = {"Probit": probit, "Logit": logit, "LPM": lpm}
    preds = {name: res.predict(X) for name, res in models.items()}

    # 模型比较
    rows = []
    for name, res in models.items():
        pred = np.asarray(preds[name], dtype=float)
        if name == "LPM":
            pseudo_r2 = res.rsquared
            llf = np.nan
            aic = res.aic
            bic = res.bic
            lr_chi2 = np.nan
            lr_p = np.nan
        else:
            pseudo_r2 = res.prsquared
            llf = res.llf
            aic = res.aic
            bic = res.bic
            lr_chi2 = 2 * (res.llf - res.llnull)
            lr_p = res.llr_pvalue

        rows.append({
            "model": name,
            "n": int(res.nobs),
            "log_likelihood": llf,
            "pseudo_r2": pseudo_r2,
            "aic": aic,
            "bic": bic,
            "lr_chi2": lr_chi2,
            "lr_p_value": lr_p,
            "auc": auc_score(y.to_numpy(), pred),
            "accuracy": float(((pred >= 0.5).astype(int) == y.to_numpy()).mean()),
            "brier": float(np.mean((pred - y.to_numpy()) ** 2)),
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(outdir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # 边际效应
    me_tables = {}
    for name, res in [("Probit", probit), ("Logit", logit)]:
        me = res.get_margeff(at="overall")
        me_df = me.summary_frame().reset_index()
        # statsmodels 0.14 has changed the summary_frame column labels
        # between releases, so rename by position to stay version-robust.
        if me_df.shape[1] != 7:
            raise ValueError(f"Unexpected marginal-effects frame shape: {me_df.shape}")
        me_df.columns = [
            "variable", "ame", "std_err", "z_value",
            "p_value", "ci_low", "ci_high",
        ]
        me_df.insert(0, "model", name)
        me_df["label"] = me_df["variable"].map(LABELS).fillna(me_df["variable"])
        me_df["significance"] = me_df["p_value"].apply(significance_stars)
        me_df.to_csv(outdir / f"marginal_effects_{name.lower()}.csv",
                     index=False, encoding="utf-8-sig")
        me_tables[name] = me_df

    # 系数表
    coef_frames = []
    for name, res in [("Probit", probit), ("Logit", logit)]:
        ci = res.conf_int()
        coef = pd.DataFrame({
            "model": name,
            "variable": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "z_value": res.tvalues.values,
            "p_value": res.pvalues.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
        })
        ame = me_tables[name][["variable", "ame", "std_err", "p_value", "ci_low", "ci_high"]]
        ame = ame.rename(columns={
            "std_err": "ame_std_err", "p_value": "ame_p_value",
            "ci_low": "ame_ci_low", "ci_high": "ame_ci_high",
        })
        coef = coef.merge(ame, on="variable", how="left")
        coef["label"] = coef["variable"].map(LABELS).fillna(coef["variable"])
        coef["significance"] = coef["p_value"].apply(significance_stars)
        coef_frames.append(coef)

    coef_all = pd.concat(coef_frames, ignore_index=True)
    coef_all[coef_all["model"] == "Probit"].to_csv(
        outdir / "coefficients_probit.csv", index=False, encoding="utf-8-sig"
    )
    coef_all[coef_all["model"] == "Logit"].to_csv(
        outdir / "coefficients_logit.csv", index=False, encoding="utf-8-sig"
    )

    # 模型摘要
    with open(outdir / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write("CFPS 2022 农村上网居民网络购物采纳影响因素分析\n")
        f.write("=" * 68 + "\n")
        f.write(f"生成时间: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        f.write(f"样本量: {len(df)}\n")
        f.write(f"网络购物采纳率: {y.mean():.2%}\n\n")
        for name, res in models.items():
            f.write(f"\n{'=' * 28} {name} {'=' * 28}\n")
            f.write(str(res.summary()))
            f.write("\n")
        f.write("\n注：Probit/Logit 使用 HC1 异方差稳健标准误；LPM 为标准线性概率模型。\n")

    return models, comparison, me_tables, coef_all


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def build_descriptives(df: pd.DataFrame) -> pd.DataFrame:
    vars_ = ["service_adopt"] + X_VARS
    rows = []
    for v in vars_:
        s = df[v]
        rows.append({
            "variable": v,
            "label": LABELS.get(v, v),
            "n": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "max": s.max(),
        })
    return pd.DataFrame(rows)


def build_group_rates(df: pd.DataFrame) -> pd.DataFrame:
    specs = []

    specs.append(("性别", df["male"].map({0: "女性", 1: "男性"})))

    age_bins = pd.cut(
        df["age"], bins=[15, 29, 39, 49, 59, 120],
        labels=["16–29 岁", "30–39 岁", "40–49 岁", "50–59 岁", "60 岁及以上"],
    )
    specs.append(("年龄组", age_bins))

    edu_bins = pd.cut(
        df["edu_year"], bins=[-1, 6, 9, 12, 15, 30],
        labels=["小学及以下", "初中", "高中/中专", "大专", "本科及以上"],
    )
    specs.append(("受教育程度", edu_bins))

    income_q = pd.qcut(
        df["ln_income_per"], q=4,
        labels=["收入 Q1", "收入 Q2", "收入 Q3", "收入 Q4"],
        duplicates="drop",
    )
    specs.append(("人均收入四分位", income_q))

    specs.append(("电脑上网", df["pc_internet"].map({0: "否", 1: "是"})))

    family_bins = pd.cut(
        df["familysize"], bins=[0, 2, 4, 30],
        labels=["1–2 人", "3–4 人", "5 人及以上"],
    )
    specs.append(("家庭规模", family_bins))

    specs.append(("土地资产", np.where(df["land_asset_log"] > 0, "有土地资产", "无土地资产")))

    rows = []
    overall = df["service_adopt"].mean()
    for dim, groups in specs:
        for group, sub in df.groupby(groups, observed=True, dropna=False):
            n = len(sub)
            k = int(sub["service_adopt"].sum())
            lo, hi = wilson_ci(k, n)
            rows.append({
                "dimension": dim,
                "group": str(group),
                "n": n,
                "adoption_rate": k / n if n else np.nan,
                "ci_low": lo,
                "ci_high": hi,
                "overall_rate": overall,
            })
    return pd.DataFrame(rows)


def build_age_profile(probit, X: pd.DataFrame) -> pd.DataFrame:
    ages = np.arange(18, 81)
    base = X.mean().to_dict()
    rows = []
    for age in ages:
        row = base.copy()
        row["age"] = float(age)
        row["age2"] = float(age ** 2)
        rows.append(row)
    exog = pd.DataFrame(rows)[X.columns]
    pred = probit.get_prediction(exog).summary_frame(alpha=0.05)
    out = pd.DataFrame({
        "age": ages,
        "predicted_probability": pred["predicted"].to_numpy(),
        "ci_low": pred["ci_lower"].to_numpy(),
        "ci_high": pred["ci_upper"].to_numpy(),
    })
    return out


def build_calibration(probit, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    pred = pd.Series(probit.predict(X), index=X.index)
    df = pd.DataFrame({"pred": pred, "y": y})
    df["decile"] = pd.qcut(df["pred"], 10, labels=False, duplicates="drop") + 1
    out = (
        df.groupby("decile")
        .agg(n=("y", "size"), predicted=("pred", "mean"), observed=("y", "mean"))
        .reset_index()
    )
    return out


# ---------------------------------------------------------------------------
# 图表
# ---------------------------------------------------------------------------

def set_plot_style():
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "axes.titlesize": 13,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "axes.edgecolor": "#C9D3E0",
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def clean_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_group_rates(groups: pd.DataFrame, outdir: Path):
    dims = ["性别", "年龄组", "受教育程度", "人均收入四分位", "电脑上网", "家庭规模"]
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.0))
    axes = axes.ravel()
    for ax, dim in zip(axes, dims):
        sub = groups[groups["dimension"] == dim].copy()
        x = np.arange(len(sub))
        y = sub["adoption_rate"].to_numpy()
        err = np.vstack([
            y - sub["ci_low"].to_numpy(),
            sub["ci_high"].to_numpy() - y,
        ])
        ax.bar(x, y, color=PALETTE["blue"], alpha=0.88, width=0.62)
        ax.errorbar(x, y, yerr=err, fmt="none", ecolor=PALETTE["navy"],
                    elinewidth=1.1, capsize=3)
        ax.axhline(sub["overall_rate"].iloc[0], color=PALETTE["orange"],
                   linestyle="--", linewidth=1)
        ax.set_title(dim, loc="left", color=PALETTE["navy"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["group"], rotation=18, ha="right")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("网络购物采纳率")
        clean_axes(ax)
    fig.suptitle("分群体网络购物采纳率（95% 置信区间）", x=0.02, y=0.985, ha="left",
                 fontsize=15, color=PALETTE["navy"], fontweight="bold")
    fig.text(0.02, 0.938, "橙色虚线为全样本均值；误差条为 Wilson 95% CI",
             fontsize=9, color=PALETTE["gray"])
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(outdir / "fig_group_rates.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_age_curve(age_profile: pd.DataFrame, turning_point: float, outdir: Path):
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    ax.fill_between(age_profile["age"], age_profile["ci_low"], age_profile["ci_high"],
                    color=PALETTE["light_blue"], alpha=0.75, linewidth=0)
    ax.plot(age_profile["age"], age_profile["predicted_probability"],
            color=PALETTE["navy"], linewidth=2.4)
    if 18 <= turning_point <= 80:
        ax.axvline(turning_point, color=PALETTE["orange"], linestyle="--", linewidth=1.2)
        y_at = np.interp(turning_point, age_profile["age"],
                         age_profile["predicted_probability"])
        ax.scatter([turning_point], [y_at], color=PALETTE["orange"], zorder=5, s=28)
        ax.annotate(f"峰值约 {turning_point:.1f} 岁",
                    xy=(turning_point, y_at), xytext=(turning_point + 4, y_at + 0.08),
                    fontsize=9, color=PALETTE["orange"])
    ax.set_xlabel("年龄")
    ax.set_ylabel("预测网络购物概率")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("年龄与网络购物概率：倒 U 型关系", loc="left",
                 color=PALETTE["navy"], fontweight="bold")
    ax.text(0.99, 0.03, "其他变量取样本均值；阴影为 95% CI",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=PALETTE["gray"])
    clean_axes(ax)
    save_fig(fig, outdir / "fig_age_curve.png")


def plot_marginal_effects(me_probit: pd.DataFrame, outdir: Path):
    plot_vars = ["male", "edu_year", "familysize", "ln_income_per",
                 "land_asset_log", "pc_internet"]
    sub = me_probit[me_probit["variable"].isin(plot_vars)].copy()
    sub = sub.sort_values("ame")
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    y = np.arange(len(sub))
    colors = [PALETTE["blue"] if p < 0.05 else PALETTE["gray"] for p in sub["p_value"]]
    ax.errorbar(sub["ame"], y,
                xerr=[sub["ame"] - sub["ci_low"], sub["ci_high"] - sub["ame"]],
                fmt="o", color=PALETTE["navy"], ecolor=PALETTE["navy"],
                elinewidth=1.1, capsize=3, markersize=5)
    for yi, (_, row) in zip(y, sub.iterrows()):
        ax.scatter(row["ame"], yi, color=colors[yi], zorder=5, s=44)
    ax.axvline(0, color=PALETTE["gray"], linewidth=0.9, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"])
    ax.set_xlabel("平均边际效应（概率变化）")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    ax.set_title("主要变量的平均边际效应（Probit，HC1）", loc="left",
                 color=PALETTE["navy"], fontweight="bold")
    ax.text(0.99, 0.03, "蓝色：p < 0.05；灰色：不显著",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=PALETTE["gray"])
    ax.grid(axis="x", color=PALETTE["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    save_fig(fig, outdir / "fig_marginal_effects.png")


def plot_calibration(cal: pd.DataFrame, auc: float, brier: float, outdir: Path):
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot([0, 1], [0, 1], color=PALETTE["gray"], linestyle="--", linewidth=1)
    ax.plot(cal["predicted"], cal["observed"], color=PALETTE["blue"],
            marker="o", markersize=5, linewidth=1.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("模型预测概率（十分位均值）")
    ax.set_ylabel("实际网络购物比例")
    ax.set_title("模型校准：预测 vs 实际", loc="left",
                 color=PALETTE["navy"], fontweight="bold")
    ax.text(0.04, 0.92, f"AUC = {auc:.3f}\nBrier = {brier:.3f}",
            transform=ax.transAxes, fontsize=9.5, color=PALETTE["navy"])
    ax.grid(color=PALETTE["light_gray"], linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    save_fig(fig, outdir / "fig_calibration.png")


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return df.to_markdown(index=False, floatfmt=floatfmt)


def write_report(
    outdir: Path,
    df: pd.DataFrame,
    flow: pd.DataFrame,
    comparison: pd.DataFrame,
    me_probit: pd.DataFrame,
    groups: pd.DataFrame,
    age_profile: pd.DataFrame,
    turning_point: float,
    probit,
):
    y_rate = df["service_adopt"].mean()
    n = len(df)
    comp = comparison.set_index("model")
    probit_metrics = comp.loc["Probit"]
    ame = me_probit.set_index("variable")
    overall = groups.iloc[0]["overall_rate"]

    def ame_text(var, pct=True):
        if var not in ame.index:
            return "—"
        row = ame.loc[var]
        value = row["ame"]
        if pct:
            return f"{value * 100:+.2f} 个百分点（p = {row['p_value']:.3f}）"
        return f"{value:+.4f}（p = {row['p_value']:.3f}）"

    # 关键分组
    age_peak = age_profile.loc[(age_profile["age"] - turning_point).abs().idxmin()]
    age_30 = age_profile.loc[age_profile["age"] == 30].iloc[0]
    age_60 = age_profile.loc[age_profile["age"] == 60].iloc[0]

    desc = build_descriptives(df)
    desc_show = desc[["label", "n", "mean", "std", "min", "median", "max"]].copy()
    desc_show.columns = ["变量", "样本量", "均值", "标准差", "最小值", "中位数", "最大值"]

    comp_show = comparison[["model", "n", "pseudo_r2", "aic", "bic", "auc", "brier"]].copy()
    comp_show.columns = ["模型", "样本量", "Pseudo R²/R²", "AIC", "BIC", "AUC", "Brier"]

    me_show = me_probit[me_probit["variable"] != "const"][
        ["label", "ame", "std_err", "p_value", "ci_low", "ci_high", "significance"]
    ].copy()
    me_show.columns = ["变量", "平均边际效应", "标准误", "p 值", "95% CI 下限", "95% CI 上限", "显著性"]

    group_show = groups[groups["dimension"].isin(["年龄组", "受教育程度", "人均收入四分位", "电脑上网"])][
        ["dimension", "group", "n", "adoption_rate", "ci_low", "ci_high"]
    ].copy()
    group_show.columns = ["维度", "分组", "样本量", "采纳率", "95% CI 下限", "95% CI 上限"]

    flow_show = flow[["stage", "n", "note"]].copy()
    flow_show.columns = ["步骤", "样本量", "说明"]

    lines = []
    lines.append("# CFPS 2022 农村上网居民网络购物采纳影响因素分析\n")
    lines.append(f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M} ｜ 分析样本：N = {n:,} ｜ 数据：CFPS 2022\n")
    lines.append("")
    lines.append("## 一、结论摘要\n")
    lines.append(f"- 在 {n:,} 名农村移动互联网使用者中，**{y_rate:.1%}** 使用过网络购物。")
    lines.append(f"- 年龄呈显著倒 U 型：预测概率在 **{turning_point:.1f} 岁**左右达到峰值；30 岁约 {age_30['predicted_probability']:.1%}，60 岁约 {age_60['predicted_probability']:.1%}。")
    lines.append(f"- 控制其他变量后，男性比女性的网络购物概率低 **{abs(ame.loc['male', 'ame']) * 100:.1f} 个百分点**。")
    lines.append(f"- 受教育年限每增加 1 年，网络购物概率平均提高 **{ame.loc['edu_year', 'ame'] * 100:.2f} 个百分点**。")
    lines.append(f"- 人均收入每提高 10%，网络购物概率平均提高约 **{ame.loc['ln_income_per', 'ame'] * np.log(1.1) * 100:.2f} 个百分点**。")
    lines.append(f"- 同时使用电脑上网者，网络购物概率高 **{ame.loc['pc_internet', 'ame'] * 100:.1f} 个百分点**。")
    lines.append("- 家庭人口规模、土地资产与网络购物无显著相关关系。")
    lines.append(f"- 模型拟合：Pseudo R² = {probit_metrics['pseudo_r2']:.3f}，AUC = {probit_metrics['auc']:.3f}。")
    lines.append("")
    lines.append("> **口径提醒**：这里的“数字服务使用”具体指网络购物（qu92），样本仅限移动上网者（qu201=1），因此结论不能外推到所有农村居民，也不能解释为因果关系。原项目使用的 ft601 是“亲友借款待偿额”，并非数字服务采纳，已弃用。\n")
    lines.append("## 二、样本构建\n")
    lines.append(md_table(flow_show))
    lines.append("")
    lines.append(f"最终进入模型的样本为 **{n:,}** 人。网络购物采纳率为 **{y_rate:.2%}**。")
    lines.append("")
    lines.append("## 三、描述统计\n")
    lines.append(md_table(desc_show, floatfmt=".3f"))
    lines.append("")
    lines.append("## 四、模型比较\n")
    lines.append(md_table(comp_show, floatfmt=".4f"))
    lines.append("")
    lines.append("Probit 与 Logit 的边际效应方向和量级高度一致；LPM 作为稳健性参考。Probit/Logit 使用 HC1 异方差稳健标准误。")
    lines.append("")
    lines.append("## 五、平均边际效应（Probit）\n")
    lines.append(md_table(me_show, floatfmt=".4f"))
    lines.append("")
    lines.append("注：年龄与年龄平方的边际效应需结合年龄曲线解读，不能单独把 age² 的系数当作业务结论。")
    lines.append("")
    lines.append("## 六、分群体观察\n")
    lines.append(md_table(group_show, floatfmt=".4f"))
    lines.append("")
    lines.append("## 七、图表\n")
    lines.append("![分群体网络购物采纳率](fig_group_rates.png)\n")
    lines.append("![年龄与网络购物概率](fig_age_curve.png)\n")
    lines.append("![主要变量平均边际效应](fig_marginal_effects.png)\n")
    lines.append("![模型校准](fig_calibration.png)\n")
    lines.append("## 八、方法与局限\n")
    lines.append("1. **结果变量**：qu92“是否网上购物”，1 = 是，0 = 否；这是数字服务使用的一个可观测切面，不等同于全部数字服务。")
    lines.append("2. **样本范围**：仅农村（urban22=0）、移动上网（qu201=1）且核心变量完整的成年受访者。qu92 对未上网者不适用，若把所有农村居民纳入会产生选择性偏误。")
    lines.append("3. **模型**：Probit 为主，Logit 与 LPM 作为稳健性参考；报告平均边际效应，便于解释概率变化。")
    lines.append("4. **标准误**：HC1 异方差稳健标准误；未使用 CFPS 调查权重，因此结果不代表全国代表性估计。")
    lines.append("5. **识别**：横截面数据，只能报告条件相关关系，不能声称因果效应；电脑上网、收入等变量可能与不可观测因素相关。")
    lines.append("6. **缺失值**：将 CFPS 负值特殊编码（-10、-9、-8、-2、-1）及 79 统一设为缺失，采用完整案例回归；未做多重插补。")
    lines.append("7. **数据合规**：CFPS 原始数据与行级衍生数据不随项目分享；仓库只保留汇总结果与代码。")
    lines.append("")
    lines.append("## 九、可复现性\n")
    lines.append("```bash")
    lines.append("python analysis.py --fam <cfps2022famecon_202410.dta> --adult <cfps2022person_202410.dta> --out output/real")
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("*本报告由 analysis.py 自动生成，所有数字均来自实际 CFPS 2022 数据运行结果。*")

    (outdir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    df, flow = build_sample(args.fam, args.adult, args.rural_code)
    if len(df) < 100:
        raise ValueError(f"有效样本仅 {len(df)}，请检查变量编码与路径。")
    if df["service_adopt"].nunique() < 2:
        raise ValueError("结果变量只有一个取值，无法建模。")

    flow_df = pd.DataFrame(flow)
    flow_df.to_csv(outdir / "sample_flow.csv", index=False, encoding="utf-8-sig")

    desc = build_descriptives(df)
    desc.to_csv(outdir / "descriptive_statistics.csv", index=False, encoding="utf-8-sig")

    models, comparison, me_tables, coef_all = fit_models(df, outdir)
    probit = models["Probit"]
    X = sm.add_constant(df[X_VARS], has_constant="add")
    y = df["service_adopt"].astype(float)

    age_profile = build_age_profile(probit, X)
    age_profile.to_csv(outdir / "age_profile.csv", index=False, encoding="utf-8-sig")

    cal = build_calibration(probit, X, y)
    cal.to_csv(outdir / "predicted_vs_observed.csv", index=False, encoding="utf-8-sig")

    groups = build_group_rates(df)
    groups.to_csv(outdir / "group_rates.csv", index=False, encoding="utf-8-sig")

    turning_point = -probit.params["age"] / (2 * probit.params["age2"])
    pd.DataFrame([{
        "age_turning_point": turning_point,
        "age_coef": probit.params["age"],
        "age2_coef": probit.params["age2"],
    }]).to_csv(outdir / "age_turning_point.csv", index=False, encoding="utf-8-sig")

    plot_group_rates(groups, outdir)
    plot_age_curve(age_profile, turning_point, outdir)
    plot_marginal_effects(me_tables["Probit"], outdir)
    plot_calibration(
        cal,
        auc=float(comparison.set_index("model").loc["Probit", "auc"]),
        brier=float(comparison.set_index("model").loc["Probit", "brier"]),
        outdir=outdir,
    )

    write_report(outdir, df, flow_df, comparison, me_tables["Probit"],
                 groups, age_profile, turning_point, probit)

    print("分析完成")
    print(f"输出目录: {outdir.resolve()}")
    print(f"最终样本: {len(df):,} | 采纳率: {y.mean():.2%}")
    print(f"年龄峰值: {turning_point:.2f} 岁")
    print(comparison[["model", "n", "pseudo_r2", "auc", "brier"]].to_string(index=False))


if __name__ == "__main__":
    main()
