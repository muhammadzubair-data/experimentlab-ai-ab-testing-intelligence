"""ExperimentLab AI — A/B Testing, Conversion Uplift & Experimentation Intelligence (Streamlit).

Synthetic demonstration data by default. Run: streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import io
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from experimentlab.core import (MIN_SEGMENT_PER_ARM, TRUE_BASELINE, TRUE_RELATIVE_LIFT, compare, cumulative, generate,
                                generate_scenario, mde_for, power_for, prepare, results_table, sample_size_alloc,
                                segment_analysis, simulate_peeking, simulate_tests, variant_summary)

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
WEBSITE = "https://zubairlabs.com"
REPO = "https://github.com/muhammadzubair-data/experimentlab-ai-ab-testing-intelligence"
NAVY, TEAL, MINT, AMBER, CORAL, INDIGO, SLATE = "#081329", "#0EA5B7", "#34D399", "#F59E0B", "#F43F5E", "#6366F1", "#94A3B8"
ARM = [TEAL, AMBER, INDIGO, CORAL, MINT, "#A855F7", "#3B82F6", "#84CC16", "#EC4899", "#64748B"]
MAX_UPLOAD_MB = 50

st.set_page_config(page_title="ExperimentLab AI | ZubairLabs", page_icon=str(ASSETS / "icon.svg"), layout="wide",
                   menu_items={"About": "ExperimentLab AI by ZubairLabs — A/B test analysis on synthetic data. "
                                        "Results are statistical comparisons, not proof of causal lift on their own.",
                               "Report a bug": REPO + "/issues"})
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "icon.svg"), link=WEBSITE)

st.markdown(f"""<style>
.block-container {{max-width: 1400px; padding-top: 3.2rem; padding-bottom: 3rem;}}
.el-hero {{position: relative; overflow: hidden; padding: 24px 28px 26px 28px; border-radius: 20px; margin-bottom: 16px; color: #F8FAFC;
  background: radial-gradient(680px 260px at 0% 0%, rgba(14,165,183,.36), transparent 60%),
              radial-gradient(620px 260px at 100% 0%, rgba(52,211,153,.24), transparent 62%),
              radial-gradient(520px 220px at 70% 130%, rgba(245,158,11,.18), transparent 60%),
              linear-gradient(120deg, #050C1D 0%, {NAVY} 55%, #10264A 100%);
  border: 1px solid rgba(148,163,184,.22); box-shadow: 0 12px 32px rgba(2,8,23,.22);}}
.el-hero:after {{content: ""; position: absolute; right: 34px; top: 22px; width: 150px; height: 90px; opacity: .22;
  background: linear-gradient(90deg, {TEAL} 0 46%, transparent 46% 54%, {AMBER} 54% 100%); border-radius: 10px;
  clip-path: polygon(0 100%, 0 40%, 46% 40%, 46% 100%, 54% 100%, 54% 10%, 100% 10%, 100% 100%);}}
.el-hero .eyebrow {{letter-spacing: .2em; font-size: 11px; font-weight: 700; color: #5EEAD4; text-transform: uppercase;}}
.el-hero h1 {{font-size: 31px; margin: 6px 0 4px 0; padding: 0; color: #F8FAFC; font-weight: 800;}}
.el-hero p {{margin: 0; color: #CBD5E1; max-width: min(980px, calc(100% - 190px));}}
.pill {{display: inline-block; margin: 10px 6px 0 0; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600;
  background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.2); color: #E2E8F0;}}
.pill.warn {{background: rgba(245,158,11,.2); border-color: rgba(245,158,11,.55); color: #FDE68A;}}
.pill.ok {{background: rgba(52,211,153,.2); border-color: rgba(52,211,153,.55); color: #A7F3D0;}}
.pill.teal {{background: rgba(14,165,183,.22); border-color: rgba(14,165,183,.6); color: #A5F3FC;}}
.kpi-grid {{display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 6px 0 14px 0;}}
.kpi {{border-radius: 16px; padding: 14px 16px; border: 1px solid rgba(148,163,184,.3); position: relative; overflow: hidden;
  background: linear-gradient(160deg, rgba(14,165,183,.10), rgba(52,211,153,.05) 60%, rgba(245,158,11,.06));}}
.kpi:before {{content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 4px; background: linear-gradient(180deg, {TEAL}, {MINT});}}
.kpi .label {{font-size: 11.5px; font-weight: 700; opacity: .75; text-transform: uppercase; letter-spacing: .06em;}}
.kpi .value {{font-size: 25px; font-weight: 800; margin-top: 4px; line-height: 1.1;}}
.kpi .sub {{font-size: 12.5px; opacity: .75; margin-top: 4px;}}
.verdict {{border-radius: 16px; padding: 16px 20px; margin: 4px 0 14px 0; border: 1px solid rgba(148,163,184,.3);
  border-left: 6px solid {TEAL}; background: rgba(14,165,183,.08);}}
.verdict.ok {{border-left-color: {MINT}; background: rgba(52,211,153,.10);}}
.verdict.warn {{border-left-color: {AMBER}; background: rgba(245,158,11,.11);}}
.verdict .t {{font-size: 18px; font-weight: 800; margin-bottom: 4px;}}
.note {{border-left: 4px solid {TEAL}; padding: 10px 14px; border-radius: 10px; background: rgba(14,165,183,.09); margin: 6px 0 12px 0;}}
.note.warn {{border-left-color: {AMBER}; background: rgba(245,158,11,.11);}}
.note.ok {{border-left-color: {MINT}; background: rgba(52,211,153,.10);}}
.note.indigo {{border-left-color: {INDIGO}; background: rgba(99,102,241,.10);}}
.side-head {{font-weight: 800; font-size: 13px; letter-spacing: .08em; text-transform: uppercase; opacity: .78; margin: 2px 0 6px 0;}}
.footer {{opacity: .65; font-size: 12px; text-align: center; margin-top: 28px;}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def esc(s) -> str:
    return html.escape(str(s))


def kpi_row(items):
    st.markdown("<div class='kpi-grid'>" + "".join(
        f"<div class='kpi' role='group' aria-label='{esc(a)}'><div class='label'>{esc(a)}</div>"
        f"<div class='value'>{esc(b)}</div><div class='sub'>{esc(c)}</div></div>" for a, b, c in items) + "</div>",
        unsafe_allow_html=True)


def note(text, kind=""):
    st.markdown(f"<div class='note {kind}' role='note'>{text}</div>", unsafe_allow_html=True)


def chart(fig, height=380, caption: str | None = None):
    top = 50
    if fig.layout.legend.orientation == "h":
        fig.update_layout(legend=dict(y=1.0, yanchor="bottom", x=0, xanchor="left"))
        top = 82
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10), legend_title_text="",
                      title_font=dict(size=15), title=dict(y=.98, yanchor="top"), font=dict(size=13))
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
    if caption:
        st.caption(caption)


def csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object or str(out[c].dtype).startswith(("string", "str", "category")):
            out[c] = out[c].astype(str).map(lambda v: "'" + v if v[:1] in ("=", "+", "-", "@") else v)
    return out.to_csv(index=False).encode()


def pct(v, dp=2, sign=False) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:+.{dp}%}" if sign else f"{v:.{dp}%}"


def pfmt(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


# ------------------------------------------------------------------ data source
ss = st.session_state


@st.cache_data(show_spinner=False, max_entries=4)
def load_original() -> pd.DataFrame:
    return prepare(generate())[0]


@st.cache_data(show_spinner=False, max_entries=12)
def load_scenario(n_total, baseline, lift, share, seed) -> pd.DataFrame:
    return prepare(generate_scenario(seed, n_total, baseline, lift, share))[0]


SOURCES = ["Original demo (seed 42)", "Scenario builder"]
with st.sidebar:
    st.markdown("<div class='side-head'>Data source</div>", unsafe_allow_html=True)
    has_upload = "upload_df" in ss
    opts = SOURCES + ([f"Uploaded: {ss['upload_name']}"] if has_upload else [])
    if ss.get("source_choice") not in opts or (has_upload and ss.pop("_jump_to_upload", False)):
        ss["source_choice"] = opts[-1] if has_upload and ss.get("use_upload", True) else opts[0]
    src = st.radio("Data source", opts, key="source_choice", label_visibility="collapsed")
    ss["use_upload"] = src.startswith("Uploaded")
    truth = None
    if src == SOURCES[0]:
        data = load_original()
        source_label, is_demo = "Original demo · 5,000 users · seed 42", True
        truth = {"baseline": TRUE_BASELINE, "lift": TRUE_RELATIVE_LIFT, "share": .5}
    elif src == SOURCES[1]:
        sc_n = st.select_slider("Total users", [2_000, 5_000, 10_000, 20_000, 40_000, 80_000], value=20_000)
        sc_base = st.slider("True baseline rate", .01, .50, .10, .01, format="%.2f")
        sc_lift = st.slider("True relative lift", -.30, .50, .12, .01, format="%.2f",
                            help="The effect built into the synthetic data. Use 0 for an A/A test.")
        sc_share = st.slider("Actual share sent to A", .40, .60, .50, .01,
                             help="Anything other than the planned split creates a sample ratio mismatch.")
        sc_seed = st.number_input("Seed", 0, 9999, 7)
        data = load_scenario(int(sc_n), float(sc_base), float(sc_lift), float(sc_share), int(sc_seed))
        source_label = f"Scenario · {sc_n:,} users · lift {sc_lift:+.0%} · A share {sc_share:.0%} · seed {sc_seed}"
        is_demo = True
        truth = {"baseline": sc_base, "lift": sc_lift, "share": sc_share}
    else:
        data = ss["upload_df"]
        source_label, is_demo = ss["upload_name"], False
        st.caption("Analysing your uploaded file in this session only.")

    st.markdown("<div class='side-head'>Analysis settings</div>", unsafe_allow_html=True)
    variants = sorted(data.variant.unique())
    if ss.get("control") not in variants:
        ss["control"] = "A" if "A" in variants else ("control" if "control" in variants else variants[0])
    control = st.selectbox("Control", variants, key="control")
    others = [v for v in variants if v != control]
    if ss.get("treatment") not in others:
        ss["treatment"] = others[0]
    treatment = st.selectbox("Treatment", others, key="treatment")
    alpha = st.select_slider("Significance level (α)", [.01, .05, .10], value=.05)
    planned = st.number_input("Planned share for control", .05, .95, .50, .05,
                              help="Used by the sample ratio mismatch check. 0.50 means a 50/50 design.")
    power_t = st.select_slider("Target power", [.8, .85, .9, .95], value=.8, format_func=lambda v: f"{v:.0%}")
    with st.expander("About these results", icon="🧪"):
        st.markdown("- Data is **synthetic** unless you upload your own.\n"
                    "- A significant result supports an effect **only if** randomization, exposure and tracking were sound.\n"
                    "- Not significant **does not** mean no effect.\n"
                    "- Segment results are **exploratory**.")
    st.caption("Theme: ⋮ menu (top right) → Light, Dark or System")
    st.caption(f"[ZubairLabs]({WEBSITE}) · [Source code]({REPO})")


def session_cached(name: str, args: tuple, fn):
    store = ss.setdefault("upload_results", OrderedDict())
    key = (ss.get("upload_key"), name, args)
    if key not in store:
        store[key] = fn()
        while len(store) > 24:
            store.popitem(last=False)
    store.move_to_end(key)
    return store[key]


@st.cache_data(show_spinner=False, max_entries=64, ttl=3600)
def _demo_compare(key, control, treatment, alpha, planned, power):
    return compare(_demo_frame(key), control, treatment, alpha, planned, power)


def _demo_frame(key):
    return load_original() if key[0] == "orig" else load_scenario(*key[1:])


demo_key = None
if is_demo:
    demo_key = ("orig",) if src == SOURCES[0] else ("sc", int(sc_n), float(sc_base), float(sc_lift), float(sc_share), int(sc_seed))


def get_result():
    args = (control, treatment, float(alpha), float(planned), float(power_t))
    if is_demo:
        return _demo_compare(demo_key, *args)
    return session_cached("compare", args, lambda: compare(data, *args))


try:
    R = get_result()
except ValueError as exc:
    st.error(f"These variants cannot be compared: {exc}", icon="⚠️")
    st.stop()

pair = data[data.variant.isin([control, treatment])]
segments = [c for c in data.columns if c not in ("variant", "converted", "user_id", "timestamp")]


def hero(eyebrow, title, text):
    pills = ("<span class='pill warn'>Synthetic demonstration data</span>" if is_demo
             else "<span class='pill ok'>Your uploaded file · this session only</span>")
    pills += (f"<span class='pill teal'>{esc(control)} vs {esc(treatment)} · α = {alpha:g}</span>"
              "<span class='pill'>Statistical evidence, not automatic proof of causal lift</span>")
    st.markdown(f"<header class='el-hero'><div class='eyebrow'>{esc(eyebrow)}</div><h1>{esc(title)}</h1>"
                f"<p>{esc(text)}</p>{pills}</header>", unsafe_allow_html=True)


def footer():
    st.markdown(f"<div class='footer'>ExperimentLab AI · built by <a href='{WEBSITE}' target='_blank'>ZubairLabs</a> · "
                f"{esc(source_label)}</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ page 1: overview
def page_overview():
    hero("Experiment overview", "Experiment Overview",
         "Conversion rates, uplift with confidence intervals, significance tests and a Bayesian view of the same data "
         "— with a plain-language reading that never claims more than the test supports.")
    r = R
    rel = r["relative_ci"]
    kpi_row([(f"{r['control']} (control)", pct(r["rate_control"]), f"{r['conversions_control']:,} / {r['n_control']:,} units"),
             (f"{r['treatment']} (treatment)", pct(r["rate_treatment"]), f"{r['conversions_treatment']:,} / {r['n_treatment']:,} units"),
             ("Absolute uplift", pct(r["absolute_uplift"], sign=True),
              f"{1 - alpha:.0%} CI {pct(r['newcombe_ci'][0], sign=True)} to {pct(r['newcombe_ci'][1], sign=True)}"),
             ("Relative uplift", pct(r["relative_uplift"], 1, True) if r["relative_uplift"] is not None else "—",
              f"CI {pct(rel[0], 1, True)} to {pct(rel[1], 1, True)}" if rel else "needs conversions in both arms"),
             ("p-value", pfmt(r["primary_p"]), r["primary_test"]),
             ("P(treatment better)", pct(r["bayes"]["prob_treatment_better"], 1), "Bayesian, uniform prior")])
    v = r["verdict"]
    icon = {"ok": "✅", "warn": "⚠️", "info": "ℹ️"}[v["kind"]]
    st.markdown(f"<div class='verdict {v['kind']}' role='status'><div class='t'>{icon} {esc(v['title'])}</div>"
                f"{esc(v['text'])}</div>", unsafe_allow_html=True)
    if r["small_sample"]:
        note(f"Small expected counts (minimum {r['min_expected_count']:.1f}): the verdict uses Fisher's exact test "
             "instead of the normal approximation.", "warn")
    if truth is not None:
        true_abs = truth["baseline"] * truth["lift"]
        pw = power_for(truth["baseline"], truth["lift"], r["n_control"], r["n_treatment"], alpha) if truth["lift"] else alpha
        lo, hi = r["newcombe_ci"]
        covered = lo <= true_abs <= hi
        what = ("Rejecting here would be a false positive (there is no true effect)." if truth["lift"] == 0 else
                f"The chance this design detects the true effect is only about {pct(pw, 0)}." if pw < .8 else
                f"The design had about {pct(pw, 0)} power for the true effect.")
        note(f"<b>Synthetic check.</b> The generator's true effect is a {truth['lift']:+.0%} relative lift "
             f"({true_abs:+.2%} absolute) on a {truth['baseline']:.0%} baseline. The {1 - alpha:.0%} interval "
             f"{'contains' if covered else 'misses'} it. {what} Real experiments have no such answer key.",
             "ok" if covered else "warn")
    c1, c2 = st.columns(2)
    with c1:
        summ = variant_summary(pair, alpha).set_index("variant").loc[[control, treatment]].reset_index()
        fig = go.Figure(go.Bar(x=summ.variant, y=summ.rate, marker_color=[TEAL, AMBER],
                               text=[pct(x) for x in summ.rate], textposition="inside", insidetextanchor="middle",
                               textfont=dict(color="white", size=15),
                               error_y=dict(type="data", symmetric=False, array=summ.ci_high - summ.rate,
                                            arrayminus=summ.rate - summ.ci_low, color=SLATE, thickness=2)))
        fig.update_layout(title=f"Conversion rate with {1 - alpha:.0%} Wilson intervals", yaxis_tickformat=".1%",
                          yaxis_title="Conversion rate", showlegend=False,
                          yaxis_range=[0, float(summ.ci_high.max()) * 1.25])
        chart(fig, 380, "Bars show observed rates; whiskers show the uncertainty in each rate.")
    with c2:
        rows = [("Newcombe (recommended)", *r["newcombe_ci"]), ("Wald (original)", r["ci_low"], r["ci_high"]),
                ("Bayesian 95% credible", *r["bayes"]["diff_ci"])]
        fig = go.Figure()
        for i, (name, lo, hi) in enumerate(rows):
            fig.add_trace(go.Scatter(x=[lo, hi], y=[name, name], mode="lines", line=dict(color=[TEAL, SLATE, INDIGO][i], width=6),
                                     showlegend=False, hovertemplate=f"{name}: %{{x:.2%}}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[r["absolute_uplift"]] * 3, y=[x[0] for x in rows], mode="markers",
                                 marker=dict(color=AMBER, size=13, symbol="diamond",
                                             line=dict(color="white", width=1)), name="Observed difference"))
        fig.add_vline(x=0, line_dash="dash", line_color=CORAL, annotation_text="no difference")
        if truth is not None:
            fig.add_vline(x=truth["baseline"] * truth["lift"], line_dash="dot", line_color=MINT,
                          annotation_text="true (synthetic)", annotation_position="bottom right")
        fig.update_layout(title="Absolute difference (treatment − control)", xaxis_tickformat="+.1%",
                          legend=dict(orientation="h", y=1.1, x=0))
        chart(fig, 380, "An interval that crosses the dashed zero line means the data are compatible with no difference.")
    c3, c4 = st.columns([1.1, 1])
    with c3:
        h = go.Figure(go.Histogram(x=r["bayes"]["diff_draws"], nbinsx=60, marker_color=INDIGO, opacity=.8))
        h.add_vline(x=0, line_dash="dash", line_color=CORAL)
        h.update_layout(title="Posterior distribution of the difference", xaxis_tickformat="+.1%",
                        xaxis_title="Treatment − control", yaxis_title="Draws", showlegend=False)
        chart(h, 320, f"Probability the treatment rate is higher: {pct(r['bayes']['prob_treatment_better'], 1)}. "
                      f"Expected loss if you pick the treatment: {pct(r['bayes']['expected_loss_treatment'], 3)} "
                      "conversion rate.")
    with c4:
        st.markdown("##### All statistics")
        tab = results_table(r)
        st.dataframe(tab.astype({"value": str}), hide_index=True, width="stretch", height=300)
        d1, d2 = st.columns(2)
        d1.download_button("Results (CSV)", csv_bytes(tab.astype({"value": str})), "experimentlab_results.csv", "text/csv",
                           icon="⬇️", width="stretch")
        js = {k: (v if not isinstance(v, (np.floating, np.integer)) else v.item()) for k, v in tab.set_index("metric").value.items()}
        d2.download_button("Results (JSON)", json.dumps(js, indent=2, default=str), "experimentlab_results.json",
                           "application/json", icon="⬇️", width="stretch")
    footer()


# ------------------------------------------------------------------ page 2: diagnostics
def page_diagnostics():
    hero("Diagnostics", "Assignment & Data Quality",
         "Check the experiment before trusting the result: sample ratio mismatch, sample adequacy, the trend over time "
         "and why stopping early on a good day is risky.")
    r = R
    srm = r["srm"]
    kpi_row([("Observed control share", pct(srm["observed_control_share"], 2), f"planned {srm['expected_control_share']:.0%}"),
             ("SRM p-value", pfmt(srm["p_value"]), "flag below 0.001"),
             ("Smallest expected cell", f"{r['min_expected_count']:,.0f}", "Fisher's test used below 5"),
             (f"Detectable lift ({power_t:.0%} power)", pct(r["mde_relative"], 1, True) if np.isfinite(r["mde_relative"]) else "—",
              "relative, at this sample size")])
    if srm["warning"]:
        note(f"⚠️ <b>Sample ratio mismatch.</b> {r['n_control']:,} vs {r['n_treatment']:,} units is very unlikely under a "
             f"{srm['expected_control_share']:.0%} / {1 - srm['expected_control_share']:.0%} design (χ² = {srm['chi2']:.1f}). "
             "Common causes: redirects that drop users, bot filtering applied to one arm, logging bugs. Results are not "
             "interpretable until the cause is found.", "warn")
    else:
        note(f"✅ No sample ratio mismatch detected (χ² = {srm['chi2']:.2f}, p = {srm['p_value']:.3f}).", "ok")
    vs = variant_summary(data, alpha)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        fig = px.bar(vs, x="variant", y="units", color="variant", color_discrete_sequence=ARM, text="units",
                     title="Units assigned per variant", labels={"units": "Units", "variant": ""})
        fig.update_layout(showlegend=False)
        chart(fig, 340)
    with c2:
        st.markdown("##### Variant summary")
        st.dataframe(vs, hide_index=True, width="stretch",
                     column_config={"rate": st.column_config.NumberColumn("rate", format="percent"),
                                    "ci_low": st.column_config.NumberColumn(f"{1 - alpha:.0%} low", format="percent"),
                                    "ci_high": st.column_config.NumberColumn(f"{1 - alpha:.0%} high", format="percent"),
                                    "share": st.column_config.ProgressColumn("share", min_value=0, max_value=1, format="percent")})
        if len(vs) > 2:
            note(f"{len(vs)} variants are present. Comparing several treatments with the same control multiplies the "
                 "chance of a false positive; adjust α (for example Bonferroni: α ÷ number of comparisons).", "warn")
    st.markdown("#### Results over time")
    if "timestamp" in data.columns:
        cu = cumulative(data, control, treatment)
        if len(cu) >= 2:
            fig = go.Figure()
            fig.add_scatter(x=cu.period, y=cu.ci_high, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip")
            fig.add_scatter(x=cu.period, y=cu.ci_low, mode="lines", fill="tonexty", fillcolor="rgba(14,165,183,.18)",
                            line=dict(width=0), name="95% CI (unadjusted)")
            fig.add_scatter(x=cu.period, y=cu.absolute_uplift, mode="lines+markers", line=dict(color=TEAL, width=3),
                            name="Cumulative uplift")
            fig.add_hline(y=0, line_dash="dash", line_color=CORAL)
            fig.update_layout(title="Cumulative absolute uplift by day", yaxis_tickformat="+.1%",
                              legend=dict(orientation="h", y=1.1, x=0))
            c3, c4 = st.columns([1.4, 1])
            with c3:
                chart(fig, 360, "Early days swing widely; the interval narrows as data accumulates.")
            with c4:
                f2 = px.line(cu, x="period", y="p_value", markers=True, title="Nominal p-value if you peeked each day",
                             labels={"period": "", "p_value": "p-value"}, color_discrete_sequence=[INDIGO])
                f2.add_hline(y=.05, line_dash="dot", line_color=CORAL, annotation_text="0.05")
                chart(f2, 360)
    else:
        st.info("Add a timestamp column (assignment or event time) to see cumulative results over time.", icon="🕒")
    with st.expander("Why peeking inflates false positives — simulation", icon="🔁", expanded=False):
        base = float(np.clip(r["rate_control"], .01, .99))
        n = int(min(max(r["n_control"], 200), 200_000))
        pk = simulate_peeking(base, n, looks=10, alpha=.05, sims=4000)
        kpi_row([("Check once at the end", pct(pk["final_only"], 1), "false-positive rate in A/A tests"),
                 ("Check 10 times, stop when significant", pct(pk["any_look"], 1), "same A/A tests"),
                 ("Simulated A/A tests", f"{pk['sims']:,}", f"{n:,} units per arm, baseline {base:.1%}")])
        f3 = px.line(pk["by_look"], x="look", y="false_positive_so_far", markers=True, color_discrete_sequence=[CORAL],
                     title="Share of A/A tests already declared 'significant'", labels={"false_positive_so_far": "False-positive rate",
                                                                                         "look": "Interim look"})
        f3.add_hline(y=.05, line_dash="dot", line_color=SLATE, annotation_text="nominal 5%")
        f3.update_layout(yaxis_tickformat=".0%")
        chart(f3, 300, "Fix the sample size in advance, or use a sequential method designed for repeated looks.")
    footer()


# ------------------------------------------------------------------ page 3: segments
def page_segments():
    hero("Segment explorer", "Segment Explorer",
         "Compare the uplift across segments such as device or country. Exploratory only: p-values are shown raw and "
         "Holm-adjusted, with a test of whether segments genuinely differ.")
    if not segments:
        st.info("No segment column found. Add a categorical column such as device, country or plan (2–30 values) to "
                "explore segments.", icon="🧩")
        footer()
        return
    seg = st.selectbox("Segment by", segments, index=segments.index("device") if "device" in segments else 0)
    sa = segment_analysis(data, control, treatment, seg, alpha)
    t = sa["table"]
    if t.empty:
        st.info("No segment has units in both arms.", icon="🧩")
        footer()
        return
    n_sig_raw, n_sig_adj = int((t.p_value < alpha).sum()), int(t.significant_holm.sum())
    kpi_row([("Segments compared", f"{len(t)}", f"by {seg}"),
             ("Significant (raw)", f"{n_sig_raw}", f"p < {alpha:g}, unadjusted"),
             ("Significant (Holm)", f"{n_sig_adj}", "family-wise error controlled"),
             ("Do segments differ?", pfmt(sa["q_p"]) if np.isfinite(sa["q_p"]) else "—",
              f"Cochran's Q = {sa['q']:.2f}, df {sa['df']}" if np.isfinite(sa["q"]) else "needs 2+ segments")])
    if np.isfinite(sa["q_p"]) and sa["q_p"] >= .05:
        note(f"No evidence that the uplift differs between {esc(seg)} segments (heterogeneity p = {sa['q_p']:.2f}). "
             "Apparent differences between segments are consistent with noise.", "ok")
    elif np.isfinite(sa["q_p"]):
        note(f"The uplift appears to differ between {esc(seg)} segments (heterogeneity p = {sa['q_p']:.3f}). Treat this "
             "as a hypothesis for a follow-up test, not a finding.", "warn")
    if t.small.any():
        note(f"⚠️ Segments with fewer than {MIN_SEGMENT_PER_ARM} units in an arm are marked; their intervals are very wide.", "warn")
    fig = go.Figure()
    order = t.sort_values("absolute_uplift")
    for _, rw in order.iterrows():
        col = AMBER if rw.small else TEAL
        fig.add_trace(go.Scatter(x=[rw.ci_low, rw.ci_high], y=[rw.segment] * 2, mode="lines", line=dict(color=col, width=5),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=order.absolute_uplift, y=order.segment, mode="markers", name="Segment uplift",
                             marker=dict(size=order.n_control.add(order.n_treatment).pipe(lambda s: 8 + 18 * s / s.max()),
                                         color=[AMBER if s else TEAL for s in order.small], line=dict(color="white", width=1)),
                             customdata=np.c_[order.p_value, order.p_holm],
                             hovertemplate="%{y}: %{x:+.2%}<br>p raw %{customdata[0]:.3f} · Holm %{customdata[1]:.3f}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dash", line_color=CORAL)
    fig.add_vline(x=R["absolute_uplift"], line_dash="dot", line_color=INDIGO, annotation_text="overall",
                  annotation_position="bottom right")
    fig.update_layout(title=f"Absolute uplift by {seg} with {1 - alpha:.0%} intervals (forest plot)", xaxis_tickformat="+.1%",
                      showlegend=False,
                      height=max(320, 70 * len(t) + 120))
    chart(fig, max(320, 70 * len(t) + 120), "Marker size reflects segment size; amber marks small segments.")
    st.dataframe(t, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(c, format="percent") for c in
                                ["rate_control", "rate_treatment", "absolute_uplift", "ci_low", "ci_high"]}
                 | {"p_value": st.column_config.NumberColumn("p (raw)", format="%.4f"),
                    "p_holm": st.column_config.NumberColumn("p (Holm)", format="%.4f"),
                    "small": "small segment", "significant_holm": "significant after Holm"})
    st.download_button("Download segment table", csv_bytes(t), f"experimentlab_segments_{seg}.csv", "text/csv", icon="⬇️")
    note("<b>Why caution.</b> Testing many segments makes some look significant by chance. Holm's adjustment controls "
         "that. Segments were not part of the randomization plan here, and a result found by slicing should be "
         "confirmed in a new, pre-registered test before acting on it.", "indigo")
    footer()


# ------------------------------------------------------------------ page 4: planner
def page_planner():
    hero("Sample size planner", "Sample Size Planner",
         "Plan how many users a future test needs, how long it will take, and check the formula against thousands of "
         "simulated experiments.")
    c1, c2, c3, c4 = st.columns(4)
    base = c1.number_input("Baseline conversion rate (%)", .1, 99.0, round(max(.1, min(99.0, 100 * R["rate_control"])), 2),
                           step=.1) / 100
    mde = c2.number_input("Minimum detectable relative lift (%)", .5, 300.0, 10.0, step=.5) / 100
    ratio = c3.selectbox("Allocation (treatment : control)", [1.0, 1.5, 2.0, 3.0], format_func=lambda v: f"{v:g} : 1")
    traffic = c4.number_input("Eligible users per day", 10, 10_000_000, 2_000, step=100)
    try:
        n0, n1 = sample_size_alloc(base, mde, alpha, power_t, ratio)
    except ValueError as exc:
        st.error(str(exc), icon="⚠️")
        footer()
        return
    total = n0 + n1
    days = int(np.ceil(total / traffic))
    kpi_row([("Control users", f"{n0:,}", f"α = {alpha:g}, power {power_t:.0%}"), ("Treatment users", f"{n1:,}", f"ratio {ratio:g} : 1"),
             ("Total users", f"{total:,}", f"{base:.2%} → {base * (1 + mde):.2%}"),
             ("Estimated duration", f"{days:,} days", f"at {traffic:,} users/day" + (" · run whole weeks" if days < 7 else ""))])
    if days < 7:
        note("Run for at least one full week (ideally two) so weekday and weekend behaviour are both represented.", "warn")
    grid = np.linspace(max(.02, mde / 4), min(mde * 3, 1 / base - 1 - 1e-3, 3), 50)
    rows = []
    for g in grid:
        try:
            a, b = sample_size_alloc(base, g, alpha, power_t, ratio)
            rows.append((g, a + b))
        except ValueError:
            pass
    curve = pd.DataFrame(rows, columns=["mde", "total"])
    cc1, cc2 = st.columns([1.3, 1])
    with cc1:
        fig = px.line(curve, x="mde", y="total", color_discrete_sequence=[TEAL],
                      title="Total users needed vs minimum detectable lift", labels={"mde": "Relative lift", "total": "Total users"})
        fig.add_scatter(x=[mde], y=[total], mode="markers", marker=dict(size=13, color=AMBER, symbol="diamond"), name="Your plan")
        fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=",", showlegend=False)
        chart(fig, 360, "Halving the detectable lift roughly quadruples the sample size.")
    with cc2:
        st.markdown("##### Check by simulation")
        st.caption("20,000 simulated experiments at your plan's sample size.")
        sim_p = simulate_tests(base, mde, n0, n1, alpha)
        sim_0 = simulate_tests(base, 0.0, n0, n1, alpha)
        kpi_row([("Simulated power", pct(sim_p["rejection_rate"], 1), f"target {power_t:.0%} · ±{pct(1.96 * sim_p['mc_se'], 1)}"),
                 ("False-positive rate", pct(sim_0["rejection_rate"], 2), f"target {alpha:.0%} (A/A tests)")])
        ok = abs(sim_p["rejection_rate"] - power_t) < .03
        note("The formula matches the simulation." if ok else
             "The normal approximation is off here (small counts or extreme rates); prefer the simulated figure.",
             "ok" if ok else "warn")
    st.markdown("#### What could the current experiment detect?")
    m = R["mde_relative"]
    note(f"With {R['n_control']:,} control and {R['n_treatment']:,} treatment units at a {R['rate_control']:.2%} baseline, "
         f"the test has {power_t:.0%} power only for relative lifts of about <b>{pct(m, 1, True)}</b> or more. "
         "Smaller real effects would usually go undetected." if np.isfinite(m) else
         "The current data cannot support a detectable-effect estimate (baseline of 0% or 100%).")
    plan = pd.DataFrame([("baseline_rate", base), ("target_rate", base * (1 + mde)), ("relative_mde", mde), ("alpha", alpha),
                         ("power", power_t), ("allocation_ratio", ratio), ("users_control", n0), ("users_treatment", n1),
                         ("users_total", total), ("users_per_day", traffic), ("estimated_days", days),
                         ("simulated_power", sim_p["rejection_rate"]), ("simulated_false_positive_rate", sim_0["rejection_rate"])],
                        columns=["parameter", "value"])
    st.download_button("Download plan (CSV)", csv_bytes(plan), "experimentlab_plan.csv", "text/csv", icon="⬇️")
    footer()


# ------------------------------------------------------------------ page 5: data & exports
def page_data():
    hero("Data & exports", "Data Upload & Exports",
         "Upload an experiment export, see exactly how it was cleaned, explore the rows and download every result. "
         "Uploads stay in this browser session and are never stored.")
    up = st.file_uploader("Experiment CSV", type=["csv"], max_upload_size=MAX_UPLOAD_MB,
                          key=f"uploader_{ss.get('uploader_gen', 0)}",
                          help="One row per randomized unit: variant and converted (0/1, yes/no, true/false); optional user_id, timestamp and segment columns.")
    if up is not None:
        raw_bytes = up.getvalue()
        key = hashlib.sha256(raw_bytes).hexdigest()[:16]
        if ss.get("upload_key") != key:
            try:
                if len(raw_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
                    raise ValueError(f"The file is larger than {MAX_UPLOAD_MB} MB.")
                raw = pd.read_csv(io.BytesIO(raw_bytes), low_memory=False)
                if raw.empty:
                    raise ValueError("The file has no rows.")
                clean, rep = prepare(raw)
            except (ValueError, pd.errors.ParserError, UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
                st.error(f"This file could not be used: {exc}", icon="⚠️")
            else:
                ss.update(upload_df=clean, upload_report=rep, upload_name=up.name, upload_key=key, use_upload=True,
                          _jump_to_upload=True)
                ss.pop("upload_results", None)
                ss.pop("control", None)
                ss.pop("treatment", None)
                st.rerun()
    if "upload_df" in ss:
        c1, c2 = st.columns([3, 1])
        c1.success(f"**{esc(ss['upload_name'])}** is loaded for this session ({len(ss['upload_df']):,} units, "
                   f"{ss['upload_df'].variant.nunique()} variants). Choose it under Data source in the sidebar.", icon="✅")
        if c2.button("Remove upload", icon="🗑️", width="stretch"):
            for k in ("upload_df", "upload_report", "upload_name", "upload_key", "upload_results", "use_upload", "control",
                      "treatment", "source_choice"):
                ss.pop(k, None)
            ss["uploader_gen"] = ss.get("uploader_gen", 0) + 1
            st.rerun()
        with st.expander("Validation report", expanded=not is_demo, icon="🧾"):
            st.table(ss["upload_report"].as_frame().set_index("check"))
            for n_ in ss["upload_report"].notes:
                st.caption("• " + n_)
    with st.expander("Upload instructions, data contract and samples", icon="📄"):
        st.markdown(f"""
One row per **randomized unit** (usually a user).

| column | meaning | rules |
|:--|:--|:--|
| `variant` | assigned group | 2–10 groups; also recognised as `group`, `arm`, `bucket`, `variation` |
| `converted` | outcome | 1/0, yes/no or true/false; also `outcome`, `conversion`, `purchased` |
| `user_id` *(optional)* | unit identifier | repeated rows in one arm are collapsed (converted if any row converted); users seen in two arms are excluded |
| `timestamp` *(optional)* | assignment or event time | enables the results-over-time view |
| any other column *(optional)* | segment, such as device or country | used for segments when it has 2–30 distinct values |

- Blank variants and outcomes that are not binary are dropped and counted; exact duplicate rows are removed.
- Binary outcomes only: revenue, order value or counts are not supported.
- Up to 2,000,000 rows and {MAX_UPLOAD_MB} MB. Do not upload names, emails or other personal data.
""")
        s1, s2 = st.columns(2)
        sample = ROOT / "data" / "sample_experiment.csv"
        if sample.exists():
            s1.download_button("Clean sample (original, 5,000 rows)", sample.read_bytes(), "experimentlab_sample.csv",
                               "text/csv", icon="⬇️", width="stretch")
        messy = ROOT / "sample_data" / "messy_upload_example.csv"
        if messy.exists():
            s2.download_button("Messy sample (tests the cleaning)", messy.read_bytes(), "experimentlab_messy_sample.csv",
                               "text/csv", icon="⬇️", width="stretch")
    st.markdown(f"#### Explore · {esc(source_label)}")
    f1, f2 = st.columns(2)
    vsel = f1.multiselect("Variants", sorted(data.variant.unique()), placeholder="All variants")
    out = f2.selectbox("Outcome", ["All units", "Converted only", "Not converted only"])
    d = data[data.variant.isin(vsel)] if vsel else data
    if out != "All units":
        d = d[d.converted == (1 if out == "Converted only" else 0)]
    seg_filters = {}
    if segments:
        cols = st.columns(min(3, len(segments)))
        for i, s in enumerate(segments[:3]):
            seg_filters[s] = cols[i].multiselect(s, sorted(data[s].unique()), placeholder=f"All {s}")
        for s, vals in seg_filters.items():
            if vals:
                d = d[d[s].isin(vals)]
    if d.empty:
        st.info("No rows match these filters.", icon="🔎")
        footer()
        return
    kpi_row([("Units", f"{len(d):,}", f"{d.variant.nunique()} variants"), ("Conversions", f"{int(d.converted.sum()):,}",
                                                                             pct(d.converted.mean()) + " overall")])
    st.dataframe(d.head(1000), hide_index=True, width="stretch", height=300)
    st.caption(f"Showing the first {min(1000, len(d)):,} of {len(d):,} rows. Downloads include all of them.")
    st.markdown("#### Exports")
    e = st.columns(4)
    e[0].download_button("Filtered data", csv_bytes(d), "experimentlab_data.csv", "text/csv", icon="⬇️", width="stretch")
    e[1].download_button("Variant summary", csv_bytes(variant_summary(data, alpha)), "experimentlab_variant_summary.csv",
                         "text/csv", icon="⬇️", width="stretch")
    e[2].download_button("Test results", csv_bytes(results_table(R).astype({"value": str})), "experimentlab_results.csv",
                         "text/csv", icon="⬇️", width="stretch")
    if "upload_report" in ss and not is_demo:
        e[3].download_button("Validation report", csv_bytes(ss["upload_report"].as_frame()), "experimentlab_validation.csv",
                             "text/csv", icon="⬇️", width="stretch")
    else:
        e[3].button("Validation report", disabled=True, width="stretch", help="Available when an uploaded file is selected")
    footer()


# ------------------------------------------------------------------ page 6: methodology
def page_method():
    hero("Methodology & limitations", "Methodology & Limitations",
         "Exactly how every number is computed, how the calculations were verified, and what an A/B test can and "
         "cannot tell you.")
    st.image(str(ASSETS / "architecture.svg"), width="stretch",
             caption="Pipeline: upload or synthetic data → validation → tests, diagnostics, segments and planning → exports")
    st.markdown("""
#### 1 · Data
- **Original demo:** the v1 generator, unchanged — 2,500 users per arm, 10% baseline, a **true +12% relative lift**,
  device segments, seed 42. It reproduces `data/sample_experiment.csv` exactly.
- **Scenario builder:** the same idea with timestamps, device and country segments and an adjustable split, so you
  can create A/A tests, larger tests or a sample ratio mismatch. The lift is identical in every segment.

#### 2 · Tests and intervals
- **Rates:** conversions ÷ units, with Wilson score intervals.
- **Primary test:** two-sided pooled two-proportion z-test (as in v1). If any expected cell count is below 5,
  **Fisher's exact test** is used instead.
- **Absolute difference:** Newcombe hybrid score interval (recommended), alongside the original Wald interval, which
  can collapse to zero width with no conversions.
- **Relative uplift:** Katz log interval for the rate ratio.
- **Bayesian view:** Beta(1, 1) priors, 200,000 posterior draws: probability the treatment is better, expected loss
  and a 95% credible interval.

#### 3 · Diagnostics
- **Sample ratio mismatch:** chi-square goodness-of-fit (1 df) against the planned split you set; flagged below 0.001.
- **Detectable effect:** the smallest relative lift this sample could detect at the target power (normal approximation).
- **Peeking:** simulated A/A tests checked at 10 interim looks show how optional stopping inflates false positives.

#### 4 · Segments
- Per-segment differences with Newcombe intervals, raw and **Holm-adjusted** p-values, and Cochran's Q test of
  whether the uplift differs between segments. Exploratory only.

#### 5 · Planning
- Users per arm for a two-sided test with optional unequal allocation (v1 formula at 1 : 1), duration from daily
  traffic, and a 20,000-run Monte Carlo check of power and the false-positive rate.

#### 6 · Verification
The test suite checks the z-test against SciPy's chi-square test, the Wilson and Newcombe intervals against published
worked examples, interval coverage and false-positive rates by simulation, and the planner's power by simulation.
""")
    st.markdown("""
#### Limitations
- **Synthetic data only** for every bundled result.
- **Causal interpretation needs a sound experiment:** random assignment, units that do not affect each other, stable
  exposure, accurate tracking and a primary metric chosen in advance. The app cannot check these for you.
- **Binary outcomes only;** no revenue or continuous metrics, no variance reduction (CUPED), no clustering.
- **Fixed-horizon tests.** Nominal p-values are invalid if you stop the test when it looks good; no sequential
  method is implemented.
- **Multiple variants** are compared pairwise; adjust α yourself when comparing several treatments.
- **Segments are exploratory** and were not stratified in the randomization.
- Not validated for production decision-making.
""")
    footer()


PAGES = [st.Page(page_overview, title="Experiment Overview", icon="🧪", url_path="overview", default=True),
         st.Page(page_diagnostics, title="Assignment & Data Quality", icon="🩺", url_path="diagnostics"),
         st.Page(page_segments, title="Segment Explorer", icon="🧩", url_path="segments"),
         st.Page(page_planner, title="Sample Size Planner", icon="📐", url_path="planner"),
         st.Page(page_data, title="Data Upload & Exports", icon="🗂️", url_path="data"),
         st.Page(page_method, title="Methodology & Limitations", icon="📘", url_path="method")]
st.navigation(PAGES).run()
