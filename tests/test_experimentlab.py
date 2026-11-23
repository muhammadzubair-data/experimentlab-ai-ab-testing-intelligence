"""Tests for the v2 core: data preparation, statistics verified against references and simulation, segments, planning."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import core as legacy
from experimentlab.core import (TRUE_RELATIVE_LIFT, analyze, bayes, compare, cumulative, generate, generate_scenario,
                                holm, mde_for, newcombe, power_for, prepare, relative_ci, results_table, sample_size,
                                sample_size_alloc, segment_analysis, simulate_peeking, simulate_tests, srm_test,
                                variant_summary, wilson)
from scripts.make_messy_sample import build as messy


@pytest.fixture(scope="module")
def demo():
    return prepare(generate())[0]


@pytest.fixture(scope="module")
def r(demo):
    return compare(demo, "A", "B")


# ---------------------------------------------------------------- data
def test_bundled_csv_is_the_generator_output():
    pd.testing.assert_frame_equal(pd.read_csv("data/sample_experiment.csv"), generate())


def test_legacy_api_still_works():
    for name in ["generate", "validate", "analyze", "wilson", "sample_size"]:
        assert callable(getattr(legacy, name))


def test_prepare_clean_round_trip():
    c, rep = prepare(generate())
    assert len(c) == 5000 and rep.variants == {"A": 2500, "B": 2500} and rep.segments == ["device"]
    assert (c.converted.values == generate().converted.values).all()


def test_prepare_messy_upload_counts_every_problem():
    c, rep = prepare(messy())
    assert rep.rows_in == 3017 and rep.rows_out == 2987
    assert (rep.dropped_variant, rep.dropped_outcome, rep.exact_duplicates) == (4, 5, 7)
    assert (rep.repeated_users, rep.cross_arm_users, rep.cross_arm_rows, rep.bad_timestamps) == (6, 4, 8, 3)
    assert rep.renamed == {"visitor_id": "user_id", "Group": "variant", "Outcome": "converted", "Date": "timestamp"}
    assert rep.segments == ["Device", "country"] and rep.skipped_segments[0].startswith("session_note")
    assert c.user_id.is_unique and set(c.variant) == {"control", "new_checkout"}
    assert set(c.converted.unique()) <= {0, 1}


def test_prepare_collapses_repeated_user_to_converted():
    raw = pd.DataFrame({"user_id": ["u1", "u1", "u2", "u3"], "variant": ["A", "A", "B", "A"], "converted": [0, 1, 0, 0]})
    c, rep = prepare(raw)
    assert c.set_index("user_id").loc["u1", "converted"] == 1 and rep.repeated_users == 1 and len(c) == 3


def test_prepare_text_outcomes_and_aliases():
    raw = pd.DataFrame({"arm": ["ctrl", "test"] * 5, "Outcome": ["Yes", "no", "TRUE", "false", "1", "0", "y", "n", "t", "f"]})
    c, rep = prepare(raw)
    assert c.converted.tolist() == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


@pytest.mark.parametrize("raw, msg", [
    (pd.DataFrame({"x": [1, 2]}), "Missing required"),
    (pd.DataFrame({"variant": ["A", "B"], "converted": [10.5, 20.1]}), "no 0/1 values"),
    (pd.DataFrame({"variant": ["A", "A"], "converted": [0, 1]}), "two variants"),
    (pd.DataFrame({"variant": [str(i) for i in range(12)], "converted": 0}), "limit"),
])
def test_prepare_rejects_unusable_files(raw, msg):
    with pytest.raises(ValueError, match=msg):
        prepare(raw)


def test_prepare_more_than_two_variants_allowed():
    raw = pd.DataFrame({"variant": list("ABC") * 20, "converted": [0, 1] * 30})
    c, rep = prepare(raw)
    assert rep.variants == {"A": 20, "B": 20, "C": 20}
    assert compare(c, "A", "C")["n_treatment"] == 20


# ---------------------------------------------------------------- statistics vs references
def test_original_analysis_unchanged(r):
    o = analyze(generate())
    for k in ["rate_control", "rate_treatment", "absolute_uplift", "ci_low", "ci_high", "p_value"]:
        assert r[k] == pytest.approx(o[k])
    assert (o["n_control"], o["conversions_control"], o["conversions_treatment"]) == (2500, 256, 253)


def test_z_test_matches_scipy_chi_square(r):
    tab = [[r["conversions_treatment"], r["n_treatment"] - r["conversions_treatment"]],
           [r["conversions_control"], r["n_control"] - r["conversions_control"]]]
    assert r["p_value"] == pytest.approx(stats.chi2_contingency(tab, correction=False)[1], rel=1e-9)
    assert r["fisher_p"] == pytest.approx(stats.fisher_exact(tab)[1])


def test_wilson_matches_newcombe_1998_examples():
    assert wilson(81, 263) == pytest.approx((0.2553, 0.3662), abs=5e-5)
    assert wilson(15, 148) == pytest.approx((0.0624, 0.1605), abs=5e-5)
    assert wilson(0, 20)[0] == 0


def test_newcombe_matches_published_examples():
    assert newcombe(48, 80, 56, 70) == pytest.approx((0.0524, 0.3339), abs=5e-5)
    assert newcombe(3, 10, 9, 10) == pytest.approx((0.1705, 0.8090), abs=5e-5)


def test_newcombe_stays_sensible_with_zero_conversions():
    lo, hi = newcombe(0, 20, 0, 20)
    assert lo < 0 < hi and hi - lo > .2              # Wald would collapse to [0, 0]


def test_newcombe_coverage_by_simulation():
    rng = np.random.default_rng(0)
    p0, p1, n = .08, .12, 150
    k0, k1 = rng.binomial(n, p0, 3000), rng.binomial(n, p1, 3000)
    cover = np.mean([newcombe(a, n, b, n)[0] <= p1 - p0 <= newcombe(a, n, b, n)[1] for a, b in zip(k0, k1)])
    assert .935 <= cover <= .965


def test_relative_ci_contains_point_and_handles_zero():
    lo, hi = relative_ci(100, 1000, 120, 1000)
    assert lo < .2 < hi
    assert relative_ci(0, 100, 5, 100) is None


def test_srm_matches_scipy_and_custom_split():
    s = srm_test(5200, 4800)
    assert s["p_value"] == pytest.approx(stats.chisquare([5200, 4800]).pvalue)
    assert not srm_test(6000, 4000, .6)["warning"] and srm_test(6000, 4000, .5)["warning"]


def test_original_srm_flag():
    d = pd.DataFrame({"variant": ["A"] * 1000 + ["B"] * 100, "converted": [0] * 1100})
    assert analyze(d)["srm_warning"]


def test_fisher_used_for_small_samples():
    d = pd.DataFrame({"variant": ["A"] * 8 + ["B"] * 8, "converted": [0] * 8 + [1, 1, 1, 0, 0, 0, 0, 0]})
    r = compare(d, "A", "B")
    assert r["small_sample"] and r["primary_test"] == "Fisher's exact test" and r["primary_p"] == r["fisher_p"]


def test_bayes_symmetry_and_direction():
    assert bayes(100, 1000, 100, 1000)["prob_treatment_better"] == pytest.approx(.5, abs=.01)
    assert bayes(100, 1000, 150, 1000)["prob_treatment_better"] > .99


def test_holm_adjustment():
    assert holm([.01, .04, .03]).tolist() == pytest.approx([.03, .06, .06])
    assert holm([.5]).tolist() == [.5]


# ---------------------------------------------------------------- verdicts (no overclaiming)
def test_default_demo_is_an_underpowered_true_effect(r):
    assert not r["significant"] and r["verdict"]["title"] == "No statistically significant difference"
    assert "does not show there is no effect" in r["verdict"]["text"]
    assert r["mde_relative"] > 2 * TRUE_RELATIVE_LIFT
    assert power_for(.10, TRUE_RELATIVE_LIFT, 2500, 2500) < .35
    lo, hi = r["newcombe_ci"]
    assert lo <= .10 * TRUE_RELATIVE_LIFT <= hi


def test_larger_scenario_detects_the_effect():
    c, _ = prepare(generate_scenario())
    r = compare(c, "A", "B")
    assert r["significant"] and r["verdict"]["kind"] == "ok" and "If randomization" in r["verdict"]["text"]


def test_srm_overrides_verdict():
    c, _ = prepare(generate_scenario(control_share=.53))
    assert compare(c, "A", "B")["verdict"]["title"].startswith("Do not interpret")


def test_compare_rejects_same_arm(demo):
    with pytest.raises(ValueError):
        compare(demo, "A", "A")


def test_results_table_has_every_key_number(r):
    t = results_table(r).set_index("metric").value
    assert t["z_test_p_value"] == r["p_value"] and t["verdict"] == r["verdict"]["title"]


# ---------------------------------------------------------------- planning
def test_allocation_formula_reduces_to_original():
    for b, m in [(.1, .1), (.03, .2), (.4, .05)]:
        assert sample_size_alloc(b, m)[0] == sample_size(b, m)


def test_unequal_allocation_needs_more_users_in_total():
    a = sum(sample_size_alloc(.1, .1, ratio=1))
    b = sum(sample_size_alloc(.1, .1, ratio=2))
    assert b > a


def test_planner_power_verified_by_simulation():
    n = sample_size(.10, .12)
    sim = simulate_tests(.10, .12, n, n, sims=20_000)
    assert abs(sim["rejection_rate"] - .80) < .015


def test_false_positive_rate_by_simulation():
    sim = simulate_tests(.10, 0.0, 3000, 3000, sims=20_000)
    assert abs(sim["rejection_rate"] - .05) < .006


def test_mde_inverts_power():
    m = mde_for(.1, 5000, 5000)
    assert power_for(.1, m, 5000, 5000) == pytest.approx(.8, abs=.005)


def test_peeking_inflates_false_positives():
    pk = simulate_peeking(.1, 4000, looks=10, sims=4000)
    assert pk["final_only"] < .07 and pk["any_look"] > .12


def test_bad_sample_size_inputs():
    with pytest.raises(ValueError):
        sample_size_alloc(.9, .5)


# ---------------------------------------------------------------- segments and time
def test_segment_analysis_shapes_and_holm(demo):
    s = segment_analysis(demo, "A", "B", "device")
    t = s["table"]
    assert list(t.segment) == ["Desktop", "Mobile", "Tablet"] and (t.p_holm >= t.p_value).all()
    assert s["q_p"] > .05                          # same true lift in every segment


def test_segment_false_positive_rate_controlled_by_holm():
    c, _ = prepare(generate_scenario(seed=3, n_total=30_000, relative_lift=0.0))
    s = segment_analysis(c, "A", "B", "country")["table"]
    assert (s.p_holm >= s.p_value).all()


def test_cumulative_ends_at_full_result():
    c, _ = prepare(generate_scenario())
    cu = cumulative(c, "A", "B")
    r = compare(c, "A", "B")
    assert cu.iloc[-1].n_control == r["n_control"] and cu.iloc[-1].p_value == pytest.approx(r["p_value"])


def test_variant_summary(demo):
    v = variant_summary(demo)
    assert v.units.sum() == 5000 and (v.ci_low <= v.rate).all() and (v.rate <= v.ci_high).all()
