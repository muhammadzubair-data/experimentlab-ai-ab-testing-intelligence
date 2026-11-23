"""ExperimentLab AI analytics core.

Two-arm binary-outcome experiment analysis: forgiving data preparation, frequentist and Bayesian comparisons, sample
ratio mismatch checks, exploratory segment analysis with multiplicity control, sample-size planning and Monte Carlo
checks of the planner. Results describe randomized comparisons only when randomization, exposure and tracking were
sound; nothing here establishes causal lift on its own.
"""
from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd
from scipy import stats

N = NormalDist()
SEED = 42
MAX_ROWS = 2_000_000
MAX_VARIANTS = 10
MAX_SEGMENT_LEVELS = 30
MIN_SEGMENT_PER_ARM = 100
SRM_THRESHOLD = 0.001

# The original demo's generator settings (synthetic ground truth).
TRUE_BASELINE = 0.10
TRUE_RELATIVE_LIFT = 0.12


# ------------------------------------------------------------------------------------------------ original API
REQUIRED = {'variant', 'converted'}


def validate(df):
    """Original strict validation (unchanged behaviour)."""
    if not REQUIRED.issubset(df.columns):
        raise ValueError('CSV requires variant and converted columns')
    d = df.copy()
    d['variant'] = d['variant'].astype(str).str.strip()
    if d['variant'].isna().any() or (d['variant'] == '').any():
        raise ValueError('Missing variant')
    if not d['converted'].isin([0, 1, True, False]).all():
        raise ValueError('converted must contain binary 0/1 values')
    d['converted'] = d['converted'].astype(int)
    if d['variant'].nunique() != 2:
        raise ValueError('Exactly two variants required')
    if 'user_id' in d.columns and d['user_id'].duplicated().any():
        raise ValueError('Duplicate user_id: one row per randomized unit required')
    return d


def wilson(k, n, alpha=.05):
    """Wilson score interval for one proportion."""
    if n <= 0:
        raise ValueError('n must be positive')
    z = N.inv_cdf(1 - alpha / 2)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo, hi = center - half, center + half
    return (0.0 if lo < 1e-12 else lo), (1.0 if hi > 1 - 1e-12 else hi)


def analyze(df, control='A', alpha=.05):
    """Original analysis: pooled two-proportion z-test, unpooled Wald CI, Wilson rate CIs, 50/50 SRM check."""
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    d = validate(df)
    labels = sorted(d.variant.unique())
    if control not in labels:
        raise ValueError('Control not found')
    treatment = next(x for x in labels if x != control)
    a = d[d.variant == control].converted
    b = d[d.variant == treatment].converted
    n0, n1 = len(a), len(b)
    k0, k1 = int(a.sum()), int(b.sum())
    return _core_result(control, treatment, k0, n0, k1, n1, alpha)


def _core_result(control, treatment, k0, n0, k1, n1, alpha, srm_share=.5):
    p0, p1 = k0 / n0, k1 / n1
    diff = p1 - p0
    pooled = (k0 + k1) / (n0 + n1)
    se0 = math.sqrt(pooled * (1 - pooled) * (1 / n0 + 1 / n1))
    z = diff / se0 if se0 else 0.
    pval = 2 * (1 - N.cdf(abs(z))) if se0 else 1.
    se = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    margin = N.inv_cdf(1 - alpha / 2) * se
    srm_p = srm_test(n0, n1, srm_share)['p_value']
    return dict(control=control, treatment=treatment, n_control=n0, n_treatment=n1,
                conversions_control=k0, conversions_treatment=k1, rate_control=p0, rate_treatment=p1,
                absolute_uplift=diff, relative_uplift=diff / p0 if p0 else None,
                ci_low=diff - margin, ci_high=diff + margin, p_value=pval, significant=pval < alpha,
                srm_p_value=srm_p, srm_warning=srm_p < SRM_THRESHOLD,
                control_ci=wilson(k0, n0, alpha), treatment_ci=wilson(k1, n1, alpha), z=z)


def sample_size(baseline, relative_mde, alpha=.05, power=.8):
    """Original planner: users per arm, equal allocation, two-sided test."""
    if not (0 < baseline < 1 and relative_mde > 0 and 0 < alpha < 1 and 0 < power < 1):
        raise ValueError('Invalid sample size inputs')
    p1 = baseline * (1 + relative_mde)
    if p1 >= 1:
        raise ValueError('Target conversion rate must be below 1')
    z_a = N.inv_cdf(1 - alpha / 2)
    z_b = N.inv_cdf(power)
    pbar = (baseline + p1) / 2
    numerator = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(baseline * (1 - baseline) + p1 * (1 - p1))) ** 2
    return math.ceil(numerator / (p1 - baseline) ** 2)


def generate(seed=42, n_per_variant=2500, baseline=.10, relative_lift=.12):
    """Original synthetic generator (unchanged): true relative lift 12% on a 10% baseline."""
    if n_per_variant < 1 or not 0 <= baseline <= 1 or not 0 <= baseline * (1 + relative_lift) <= 1:
        raise ValueError('Invalid generator parameters')
    rng = np.random.default_rng(seed)
    variants = np.repeat(['A', 'B'], n_per_variant)
    probabilities = np.where(variants == 'A', baseline, baseline * (1 + relative_lift))
    return pd.DataFrame({'user_id': [f'U{i:06d}' for i in range(len(variants))],
                         'variant': variants, 'converted': rng.binomial(1, probabilities),
                         'device': rng.choice(['Mobile', 'Desktop', 'Tablet'], len(variants), p=[.65, .30, .05])})


# ------------------------------------------------------------------------------------------------ scenario generator
def generate_scenario(seed=7, n_total=20_000, baseline=.10, relative_lift=.12, control_share=.5, days=14,
                      start='2026-01-05'):
    """Synthetic experiment with timestamps, device and country segments. ``control_share`` is the *actual* share
    sent to A, so values other than 0.5 create a sample ratio mismatch against a 50/50 design. The lift is the same
    in every segment, so any segment differences are noise."""
    if not (0 < baseline < 1 and 0 <= baseline * (1 + relative_lift) <= 1 and 0 < control_share < 1 and n_total >= 20):
        raise ValueError('Invalid scenario parameters')
    rng = np.random.default_rng(seed)
    v = np.where(rng.random(n_total) < control_share, 'A', 'B')
    p = np.where(v == 'A', baseline, baseline * (1 + relative_lift))
    ts = pd.Timestamp(start) + pd.to_timedelta(rng.uniform(0, days * 86400, n_total), unit='s')
    d = pd.DataFrame({'user_id': [f'S{i:07d}' for i in range(n_total)], 'variant': v,
                      'converted': rng.binomial(1, p),
                      'device': rng.choice(['Mobile', 'Desktop', 'Tablet'], n_total, p=[.62, .32, .06]),
                      'country': rng.choice(['US', 'UK', 'DE', 'IN', 'BR', 'Other'], n_total,
                                            p=[.34, .14, .12, .16, .09, .15]),
                      'timestamp': ts.floor('s')})
    return d.sort_values('timestamp').reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ data preparation
ALIASES = {
    'group': 'variant', 'arm': 'variant', 'bucket': 'variant', 'treatment_group': 'variant', 'experiment_group': 'variant',
    'test_group': 'variant', 'cohort': 'variant', 'version': 'variant', 'variation': 'variant',
    'conversion': 'converted', 'converted_flag': 'converted', 'is_converted': 'converted', 'outcome': 'converted',
    'success': 'converted', 'purchased': 'converted', 'clicked': 'converted', 'signup': 'converted', 'target': 'converted',
    'user': 'user_id', 'userid': 'user_id', 'uid': 'user_id', 'visitor_id': 'user_id', 'customer_id': 'user_id',
    'session_id': 'user_id', 'id': 'user_id',
    'date': 'timestamp', 'datetime': 'timestamp', 'time': 'timestamp', 'event_time': 'timestamp', 'assigned_at': 'timestamp',
    'exposure_time': 'timestamp', 'created_at': 'timestamp',
}
YES = {'1', '1.0', 'true', 't', 'yes', 'y', 'converted', 'conversion', 'success', 'purchase', 'purchased'}
NO = {'0', '0.0', 'false', 'f', 'no', 'n', 'not converted', 'none', 'fail', 'failure'}


@dataclass
class DataReport:
    rows_in: int = 0
    rows_out: int = 0
    renamed: dict = field(default_factory=dict)
    dropped_variant: int = 0
    dropped_outcome: int = 0
    exact_duplicates: int = 0
    repeated_users: int = 0
    rows_collapsed: int = 0
    cross_arm_users: int = 0
    cross_arm_rows: int = 0
    bad_timestamps: int = 0
    variants: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)
    skipped_segments: list = field(default_factory=list)
    has_user_id: bool = False
    has_timestamp: bool = False
    notes: list = field(default_factory=list)

    def as_frame(self) -> pd.DataFrame:
        rows = [('Rows in file', self.rows_in), ('Rows kept (one per unit)', self.rows_out),
                ('Variants', ', '.join(f'{k} ({v:,})' for k, v in self.variants.items())),
                ('Dropped: blank variant', self.dropped_variant),
                ('Dropped: outcome not 0/1 (or yes/no, true/false)', self.dropped_outcome),
                ('Removed: exact duplicate rows (only when a user ID or timestamp identifies rows)', self.exact_duplicates),
                ('Users with repeated rows in one variant (collapsed: converted if any row converted)', self.repeated_users),
                ('Users seen in more than one variant (excluded)', self.cross_arm_users),
                ('Rows removed with those users', self.cross_arm_rows),
                ('User ID column', 'provided' if self.has_user_id else 'not provided (each row treated as one unit)'),
                ('Timestamp column', 'provided' + (f', {self.bad_timestamps} unparseable set to blank' if self.bad_timestamps else '')
                 if self.has_timestamp else 'not provided'),
                ('Segment columns found', ', '.join(self.segments) or '—'),
                ('Columns not used as segments', ', '.join(self.skipped_segments) or '—'),
                ('Columns renamed', ', '.join(f'{a} → {b}' for a, b in self.renamed.items()) or '—')]
        return pd.DataFrame(rows, columns=['check', 'result']).astype({'result': str})


def _norm(c) -> str:
    return re.sub(r'[^0-9a-z]+', '_', str(c).strip().lower()).strip('_')


def prepare(raw: pd.DataFrame, max_rows: int = MAX_ROWS) -> tuple[pd.DataFrame, DataReport]:
    """Forgiving validation for uploads. Maps common header names, repairs or drops bad rows (and counts them),
    enforces one row per randomized unit, and finds optional segment and timestamp columns."""
    rep = DataReport(rows_in=len(raw))
    if len(raw) > max_rows:
        raise ValueError(f'The file has {len(raw):,} rows; the limit for this demo is {max_rows:,}.')
    cols, used = {}, set()
    for c in raw.columns:
        n = _norm(c)
        t = n if n in {'variant', 'converted', 'user_id', 'timestamp'} else ALIASES.get(n)
        if t and t not in used:
            cols[c] = t
            used.add(t)
            if str(c) != t:
                rep.renamed[str(c)] = t
        else:
            cols[c] = str(c).strip()
    missing = [c for c in ('variant', 'converted') if c not in used]
    if missing:
        raise ValueError('Missing required column(s): ' + ', '.join(missing) + '. Expected a variant (group) column '
                         'and a converted (0/1) column.')
    d = raw.rename(columns=cols).copy()
    d = d.loc[:, ~d.columns.duplicated()]

    v = d['variant'].astype('string').str.strip()
    bad = v.isna() | v.isin(['', 'nan', 'None', 'NaN', 'null'])
    rep.dropped_variant = int(bad.sum())
    d = d.assign(variant=v)[~bad]

    s = d['converted'].astype('string').str.strip().str.lower()
    num = pd.to_numeric(s, errors='coerce')
    conv = pd.Series(np.nan, index=d.index)
    conv[s.isin(YES)] = 1
    conv[s.isin(NO)] = 0
    conv[num.isin([0, 1])] = num[num.isin([0, 1])]
    bad = conv.isna()
    rep.dropped_outcome = int(bad.sum())
    if rep.dropped_outcome and rep.dropped_outcome == len(d):
        raise ValueError('The converted column has no 0/1 values. Binary outcomes only (revenue or counts are not supported).')
    d = d.assign(converted=conv.astype(float))[~bad]
    d['converted'] = d['converted'].astype(int)

    if 'user_id' in d or 'timestamp' in d:          # without an identifier, identical rows are separate units
        before = len(d)
        d = d.drop_duplicates()
        rep.exact_duplicates = before - len(d)
    else:
        rep.notes.append('No user_id or timestamp column: every row is treated as one unit and identical rows are kept.')

    if 'user_id' in d:
        rep.has_user_id = True
        d['user_id'] = d['user_id'].astype('string').str.strip()
        d = d[d.user_id.notna() & (d.user_id != '')]
        arms = d.groupby('user_id').variant.nunique()
        cross = arms[arms > 1].index
        rep.cross_arm_users = int(len(cross))
        rep.cross_arm_rows = int(d.user_id.isin(cross).sum())
        d = d[~d.user_id.isin(cross)]
        counts = d.groupby('user_id').size()
        rep.repeated_users = int((counts > 1).sum())
        if rep.repeated_users:
            agg = {c: 'first' for c in d.columns if c not in ('user_id', 'converted', 'timestamp')}
            agg['converted'] = 'max'
            if 'timestamp' in d:
                agg['timestamp'] = 'first'
            before = len(d)
            d = d.groupby('user_id', as_index=False, sort=False).agg(agg)
            rep.rows_collapsed = before - len(d)
    if 'timestamp' in d:
        rep.has_timestamp = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t = pd.to_datetime(d['timestamp'], errors='coerce', format='mixed')
        if isinstance(t.dtype, pd.DatetimeTZDtype):
            t = t.dt.tz_convert('UTC').dt.tz_localize(None)
        rep.bad_timestamps = int(t.isna().sum() - d['timestamp'].isna().sum())
        d['timestamp'] = t
        if t.notna().sum() == 0:
            rep.has_timestamp = False
            d = d.drop(columns='timestamp')
            rep.notes.append('No timestamp could be parsed, so the time-trend view is unavailable.')

    counts = d.variant.value_counts()
    if len(counts) < 2:
        raise ValueError('At least two variants with valid rows are needed.')
    if len(counts) > MAX_VARIANTS:
        raise ValueError(f'{len(counts)} variants found; the limit is {MAX_VARIANTS}. Check that the variant column is correct.')
    rep.variants = {k: int(v) for k, v in counts.sort_index().items()}
    for c in d.columns:
        if c in ('variant', 'converted', 'user_id', 'timestamp'):
            continue
        nun = d[c].nunique(dropna=True)
        if 2 <= nun <= MAX_SEGMENT_LEVELS:
            d[c] = d[c].astype('string').fillna('(missing)').astype(str)
            rep.segments.append(c)
        else:
            rep.skipped_segments.append(f'{c} ({nun} values)')
    keep = ['variant', 'converted'] + [c for c in ('user_id', 'timestamp') if c in d] + rep.segments
    d = d[keep].reset_index(drop=True)
    d['variant'] = d['variant'].astype(str)
    if 'user_id' in d:
        d['user_id'] = d['user_id'].astype(str)
    rep.rows_out = len(d)
    if (counts < 30).any():
        rep.notes.append('At least one variant has fewer than 30 units; intervals and tests will be very wide.')
    return d, rep


# ------------------------------------------------------------------------------------------------ statistics
def variant_summary(d: pd.DataFrame, alpha: float = .05) -> pd.DataFrame:
    g = d.groupby('variant').converted.agg(units='size', conversions='sum').reset_index()
    g['rate'] = g.conversions / g.units
    ci = [wilson(int(k), int(n), alpha) for k, n in zip(g.conversions, g.units)]
    g['ci_low'] = [c[0] for c in ci]
    g['ci_high'] = [c[1] for c in ci]
    g['share'] = g.units / g.units.sum()
    return g


def newcombe(k0, n0, k1, n1, alpha=.05):
    """Newcombe hybrid score interval for p1 − p0 (method 10). Stays sensible with small counts or rates near 0/1."""
    l0, u0 = wilson(k0, n0, alpha)
    l1, u1 = wilson(k1, n1, alpha)
    p0, p1 = k0 / n0, k1 / n1
    d = p1 - p0
    lo = d - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2)
    return lo, hi


def relative_ci(k0, n0, k1, n1, alpha=.05):
    """Katz log interval for the ratio p1/p0, returned as relative uplift (ratio − 1). None if a count is zero."""
    if min(k0, k1) == 0:
        return None
    rr = (k1 / n1) / (k0 / n0)
    se = math.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    z = N.inv_cdf(1 - alpha / 2)
    return math.exp(math.log(rr) - z * se) - 1, math.exp(math.log(rr) + z * se) - 1


def srm_test(n0: int, n1: int, expected_control_share: float = .5) -> dict:
    """Chi-square goodness-of-fit test of the observed split against the planned allocation (1 df)."""
    tot = n0 + n1
    e0, e1 = tot * expected_control_share, tot * (1 - expected_control_share)
    chi = (n0 - e0) ** 2 / e0 + (n1 - e1) ** 2 / e1
    p = float(stats.chi2.sf(chi, 1))
    return {'chi2': float(chi), 'p_value': p, 'observed_control_share': n0 / tot if tot else np.nan,
            'expected_control_share': expected_control_share, 'warning': p < SRM_THRESHOLD}


def sample_size_alloc(baseline, relative_mde, alpha=.05, power=.8, ratio=1.0):
    """Units per arm for a two-sided two-proportion test with allocation ``ratio`` = n_treatment / n_control.
    With ratio 1 this is identical to the original ``sample_size``."""
    if not (0 < baseline < 1 and relative_mde > 0 and 0 < alpha < 1 and 0 < power < 1 and ratio > 0):
        raise ValueError('Invalid sample size inputs')
    p0 = baseline
    p1 = baseline * (1 + relative_mde)
    if p1 >= 1:
        raise ValueError('Target conversion rate must be below 1')
    z_a, z_b = N.inv_cdf(1 - alpha / 2), N.inv_cdf(power)
    pbar = (p0 + ratio * p1) / (1 + ratio)
    num = (z_a * math.sqrt(pbar * (1 - pbar) * (1 + 1 / ratio)) + z_b * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1) / ratio)) ** 2
    n0 = math.ceil(num / (p1 - p0) ** 2)
    return n0, math.ceil(n0 * ratio)


def power_for(baseline, relative_lift, n0, n1, alpha=.05):
    """Approximate power of the two-sided pooled z-test for a given true lift and sample sizes (planning only)."""
    p0, p1 = baseline, baseline * (1 + relative_lift)
    if not (0 < p0 < 1 and 0 < p1 < 1) or min(n0, n1) < 1:
        return np.nan
    pbar = (n0 * p0 + n1 * p1) / (n0 + n1)
    se0 = math.sqrt(pbar * (1 - pbar) * (1 / n0 + 1 / n1))
    se1 = math.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    z = N.inv_cdf(1 - alpha / 2)
    dlt = abs(p1 - p0)
    return float(N.cdf((dlt - z * se0) / se1) + N.cdf((-dlt - z * se0) / se1))


def mde_for(baseline, n0, n1, alpha=.05, power=.8):
    """Smallest relative lift detectable with the given power at these sample sizes (bisection on the planner)."""
    if not 0 < baseline < 1 or min(n0, n1) < 2:
        return np.nan
    lo, hi = 1e-4, 1 / baseline - 1 - 1e-6
    if power_for(baseline, hi, n0, n1, alpha) < power:
        return np.nan
    for _ in range(80):
        mid = (lo + hi) / 2
        if power_for(baseline, mid, n0, n1, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def bayes(k0, n0, k1, n1, draws=200_000, seed=SEED, prior=(1, 1)):
    """Beta-Binomial comparison with a uniform prior: probability the treatment rate is higher, expected loss of
    choosing each arm, and a 95% credible interval for the absolute difference."""
    rng = np.random.default_rng(seed)
    a0 = rng.beta(prior[0] + k0, prior[1] + n0 - k0, draws)
    a1 = rng.beta(prior[0] + k1, prior[1] + n1 - k1, draws)
    diff = a1 - a0
    return {'prob_treatment_better': float((diff > 0).mean()),
            'expected_loss_treatment': float(np.maximum(a0 - a1, 0).mean()),
            'expected_loss_control': float(np.maximum(a1 - a0, 0).mean()),
            'diff_ci': (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))),
            'diff_draws': diff[:5000]}


def compare(d: pd.DataFrame, control: str, treatment: str, alpha: float = .05, expected_control_share: float = .5,
            power: float = .8) -> dict:
    """Full comparison of two arms. Keeps the original pooled z-test and Wald interval and adds Newcombe and
    relative intervals, Fisher's exact test, a Bayesian view, SRM against the planned split and the MDE this sample
    could detect."""
    if control == treatment:
        raise ValueError('Control and treatment must be different variants.')
    if not 0 < alpha < 1:
        raise ValueError('alpha must be between 0 and 1')
    a = d.loc[d.variant == control, 'converted']
    b = d.loc[d.variant == treatment, 'converted']
    if len(a) == 0 or len(b) == 0:
        raise ValueError('Both variants need at least one row.')
    n0, n1, k0, k1 = len(a), len(b), int(a.sum()), int(b.sum())
    r = _core_result(control, treatment, k0, n0, k1, n1, alpha, expected_control_share)
    r['newcombe_ci'] = newcombe(k0, n0, k1, n1, alpha)
    r['relative_ci'] = relative_ci(k0, n0, k1, n1, alpha)
    tab = np.array([[k1, n1 - k1], [k0, n0 - k0]])
    r['fisher_p'] = float(stats.fisher_exact(tab)[1])
    exp = np.outer(tab.sum(1), tab.sum(0)) / tab.sum()
    r['min_expected_count'] = float(exp.min())
    r['small_sample'] = bool(exp.min() < 5)
    r['srm'] = srm_test(n0, n1, expected_control_share)
    r['srm_p_value'], r['srm_warning'] = r['srm']['p_value'], r['srm']['warning']
    r['bayes'] = bayes(k0, n0, k1, n1)
    r['mde_relative'] = mde_for(r['rate_control'], n0, n1, alpha, power) if 0 < r['rate_control'] < 1 else np.nan
    r['alpha'], r['power_target'] = alpha, power
    r['primary_p'] = r['fisher_p'] if r['small_sample'] else r['p_value']
    r['primary_test'] = "Fisher's exact test" if r['small_sample'] else 'pooled two-proportion z-test'
    r['significant'] = r['primary_p'] < alpha
    r['verdict'] = verdict(r)
    return r


def verdict(r: dict) -> dict:
    """Plain-language reading that never claims more than the test supports."""
    a = r['alpha']
    lvl = f'{1 - a:.0%}'
    lo, hi = r['newcombe_ci']
    if r['srm_warning']:
        return {'kind': 'warn', 'title': 'Do not interpret yet: sample ratio mismatch',
                'text': f"The split between arms differs from the planned allocation (SRM p = {r['srm_p_value']:.2g}). "
                        "Assignment or tracking may be broken, which can bias any comparison. Fix the cause first."}
    if r['significant'] and r['absolute_uplift'] > 0:
        return {'kind': 'ok', 'title': f"{r['treatment']} converted higher than {r['control']}",
                'text': f"The difference is statistically significant at α = {a:g} ({r['primary_test']}, "
                        f"p = {r['primary_p']:.3g}). The {lvl} interval for the absolute difference is {lo:+.2%} to "
                        f"{hi:+.2%}. If randomization, exposure and tracking were sound, this supports a positive effect "
                        "of roughly that size; it is still one experiment."}
    if r['significant']:
        return {'kind': 'warn', 'title': f"{r['treatment']} converted lower than {r['control']}",
                'text': f"The difference is statistically significant at α = {a:g} (p = {r['primary_p']:.3g}); "
                        f"the {lvl} interval is {lo:+.2%} to {hi:+.2%}."}
    mde = r.get('mde_relative')
    under = ''
    if mde is not None and np.isfinite(mde):
        under = (f" With these sample sizes the test could reliably detect (at {r['power_target']:.0%} power) only "
                 f"relative changes of about {mde:+.1%} or larger.")
    return {'kind': 'info', 'title': 'No statistically significant difference',
            'text': f"p = {r['primary_p']:.3g} at α = {a:g}. The {lvl} interval ({lo:+.2%} to {hi:+.2%}) still includes "
                    f"effects that could matter, so this does not show there is no effect.{under}"}


def holm(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def segment_analysis(d: pd.DataFrame, control: str, treatment: str, segment: str, alpha: float = .05) -> dict:
    """Exploratory per-segment comparison with Holm-adjusted p-values and a test of whether the uplift differs
    between segments (Cochran's Q on the absolute differences)."""
    rows = []
    for level, g in d[d.variant.isin([control, treatment])].groupby(segment):
        a, b = g[g.variant == control].converted, g[g.variant == treatment].converted
        if len(a) == 0 or len(b) == 0:
            continue
        n0, n1, k0, k1 = len(a), len(b), int(a.sum()), int(b.sum())
        r = _core_result(control, treatment, k0, n0, k1, n1, alpha)
        lo, hi = newcombe(k0, n0, k1, n1, alpha)
        var = (k0 / n0) * (1 - k0 / n0) / n0 + (k1 / n1) * (1 - k1 / n1) / n1
        rows.append({'segment': str(level), 'n_control': n0, 'n_treatment': n1, 'rate_control': k0 / n0,
                     'rate_treatment': k1 / n1, 'absolute_uplift': r['absolute_uplift'], 'ci_low': lo, 'ci_high': hi,
                     'p_value': r['p_value'], 'variance': var, 'small': min(n0, n1) < MIN_SEGMENT_PER_ARM})
    t = pd.DataFrame(rows)
    if t.empty:
        return {'table': t, 'q': np.nan, 'q_p': np.nan, 'df': 0}
    t['p_holm'] = holm(t.p_value)
    t['significant_holm'] = t.p_holm < alpha
    ok = t[t.variance > 0]
    if len(ok) >= 2:
        w = 1 / ok.variance
        pooled = (w * ok.absolute_uplift).sum() / w.sum()
        q = float((w * (ok.absolute_uplift - pooled) ** 2).sum())
        q_p = float(stats.chi2.sf(q, len(ok) - 1))
    else:
        q, q_p = np.nan, np.nan
    return {'table': t.drop(columns='variance').sort_values('segment').reset_index(drop=True), 'q': q, 'q_p': q_p,
            'df': max(len(ok) - 1, 0)}


def cumulative(d: pd.DataFrame, control: str, treatment: str, freq: str = 'D') -> pd.DataFrame:
    """Cumulative rates, uplift and nominal p-value by day — to show why repeated peeking is risky."""
    g = d[d.variant.isin([control, treatment])].dropna(subset=['timestamp'])
    if g.empty:
        return pd.DataFrame()
    t = (g.assign(period=g.timestamp.dt.floor(freq)).groupby(['period', 'variant']).converted
         .agg(['size', 'sum']).unstack('variant').fillna(0).cumsum())
    out = []
    for per, r in t.iterrows():
        n0, k0 = r[('size', control)], r[('sum', control)]
        n1, k1 = r[('size', treatment)], r[('sum', treatment)]
        if min(n0, n1) == 0:
            continue
        res = _core_result(control, treatment, int(k0), int(n0), int(k1), int(n1), .05)
        out.append({'period': per, 'n_control': int(n0), 'n_treatment': int(n1), 'rate_control': res['rate_control'],
                    'rate_treatment': res['rate_treatment'], 'absolute_uplift': res['absolute_uplift'],
                    'ci_low': res['ci_low'], 'ci_high': res['ci_high'], 'p_value': res['p_value']})
    return pd.DataFrame(out)


# ------------------------------------------------------------------------------------------------ simulations
def simulate_tests(baseline, relative_lift, n0, n1, alpha=.05, sims=20_000, seed=SEED) -> dict:
    """Monte Carlo check: how often the pooled z-test rejects when the true lift is ``relative_lift``.
    With lift 0 this is the false-positive rate; otherwise it is the empirical power."""
    rng = np.random.default_rng(seed)
    p1 = baseline * (1 + relative_lift)
    k0 = rng.binomial(n0, baseline, sims)
    k1 = rng.binomial(n1, p1, sims)
    pool = (k0 + k1) / (n0 + n1)
    se = np.sqrt(pool * (1 - pool) * (1 / n0 + 1 / n1))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(se > 0, (k1 / n1 - k0 / n0) / se, 0.0)
    rej = np.abs(z) > N.inv_cdf(1 - alpha / 2)
    rate = float(rej.mean())
    return {'rejection_rate': rate, 'mc_se': math.sqrt(rate * (1 - rate) / sims), 'sims': sims,
            'theoretical': power_for(baseline, relative_lift, n0, n1, alpha) if relative_lift else alpha}


def simulate_peeking(baseline, n_per_arm, looks=10, alpha=.05, sims=5_000, seed=SEED) -> dict:
    """A/A tests (no true effect) checked at ``looks`` equally spaced interim analyses, stopping at the first
    nominally significant look. Shows how optional stopping inflates the false-positive rate."""
    rng = np.random.default_rng(seed)
    step = max(1, n_per_arm // looks)
    k0 = rng.binomial(step, baseline, (sims, looks)).cumsum(1)
    k1 = rng.binomial(step, baseline, (sims, looks)).cumsum(1)
    n = step * np.arange(1, looks + 1)
    pool = (k0 + k1) / (2 * n)
    se = np.sqrt(pool * (1 - pool) * (2 / n))
    with np.errstate(divide='ignore', invalid='ignore'):
        z = np.where(se > 0, (k1 - k0) / n / se, 0.0)
    sig = np.abs(z) > N.inv_cdf(1 - alpha / 2)
    by_look = sig.cumsum(1).astype(bool).mean(0)
    return {'final_only': float(sig[:, -1].mean()), 'any_look': float(sig.any(1).mean()),
            'by_look': pd.DataFrame({'look': np.arange(1, looks + 1), 'units_per_arm': n,
                                     'false_positive_so_far': by_look}), 'sims': sims, 'looks': looks}


def results_table(r: dict) -> pd.DataFrame:
    rel = r['relative_ci']
    rows = [('control', r['control']), ('treatment', r['treatment']), ('units_control', r['n_control']),
            ('units_treatment', r['n_treatment']), ('conversions_control', r['conversions_control']),
            ('conversions_treatment', r['conversions_treatment']), ('rate_control', r['rate_control']),
            ('rate_treatment', r['rate_treatment']), ('absolute_uplift', r['absolute_uplift']),
            ('relative_uplift', r['relative_uplift']), ('wald_ci_low', r['ci_low']), ('wald_ci_high', r['ci_high']),
            ('newcombe_ci_low', r['newcombe_ci'][0]), ('newcombe_ci_high', r['newcombe_ci'][1]),
            ('relative_ci_low', rel[0] if rel else None), ('relative_ci_high', rel[1] if rel else None),
            ('z_test_p_value', r['p_value']), ('fisher_exact_p_value', r['fisher_p']), ('primary_test', r['primary_test']),
            ('alpha', r['alpha']), ('significant', r['significant']), ('srm_p_value', r['srm_p_value']),
            ('srm_expected_control_share', r['srm']['expected_control_share']), ('srm_warning', r['srm_warning']),
            ('bayes_prob_treatment_better', r['bayes']['prob_treatment_better']),
            ('mde_relative_at_target_power', r['mde_relative']), ('verdict', r['verdict']['title'])]
    return pd.DataFrame(rows, columns=['metric', 'value'])
