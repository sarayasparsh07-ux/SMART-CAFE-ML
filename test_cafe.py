"""Run with:  pytest -q"""
import numpy as np
import pytest

from agent import epsilon_at, floor_episode, policy_from_Q, train
from cafe import CafeSim, day_economics, reachable_states, run_month, state_features, totals

P = CafeSim().params()


def test_worked_example():
    r = day_economics(5, 8, 7, P)
    assert (r["available"], r["sold"], r["leftover"], r["spoiled"], r["carry"]) == (13, 7, 6, 2, 4)
    assert r["profit"] == pytest.approx(22.2)


def test_bankers_rounding_stockout_and_clip():
    assert day_economics(0, 2, 0, P)["spoiled"] == 0      # round(0.5) -> 0
    assert day_economics(0, 10, 0, P)["spoiled"] == 2     # round(2.5) -> 2
    r = day_economics(0, 4, 9, P)
    assert r["stockout"] == 5 and r["profit"] == pytest.approx(-14.0)
    r = day_economics(10, 99, 0, P)
    assert r["order"] == 15 and r["available"] == 15 and r["clipped_units"] == 10
    assert r["profit"] == pytest.approx(-25.5)
    assert day_economics(3, -4, 0, P)["order"] == 0


def test_invariants_and_daily_reconciliation():
    env, g = CafeSim(seed=1), np.random.default_rng(2)
    env.reset()
    for _ in range(3000):
        prof, s2, done, i = env.step(int(g.integers(0, 16)))
        assert i["sold"] <= i["available"] and i["sold"] <= i["demand"] and i["stockout"] >= 0
        assert i["spoiled"] + i["carry"] == i["leftover"] and 0 <= s2 <= 15
        assert prof == pytest.approx(i["revenue"] - (i["order_cost"] + i["holding_cost"]
                                                     + i["spoilage_cost"] + i["stockout_penalty"]))
        assert {"order", "demand", "sold", "spoiled", "stockout", "carry", "profit"} <= set(i)
        assert i["profit"] == round(prof, 2)
        if done:
            assert env.day == 30
            env.reset()


def test_seeding_and_month_totals():
    a, b = CafeSim(seed=3), CafeSim(seed=3)
    assert [a._demand() for _ in range(50)] == [b._demand() for _ in range(50)]
    assert (CafeSim().sample_demands(5, seed=9) == CafeSim().sample_demands(5, seed=9)).all()
    rows = run_month(lambda s, t: 8, CafeSim().sample_demands(1, seed=4)[0], P)
    t = totals(rows)
    assert len(rows) == 30
    assert t["profit"] == pytest.approx(t["revenue"] - t["order_cost"] - t["holding_cost"]
                                        - t["spoilage_cost"] - t["stockout_penalty"])


def test_train_contract_and_diagnostics():
    Q, c = train(CafeSim(), episodes=300, seed=5)
    Q2, c2, d = train(CafeSim(), episodes=300, seed=5, return_diagnostics=True)
    assert Q.shape == (16, 16) and len(c) == 300
    assert np.allclose(Q, Q2) and np.allclose(c, c2)          # diagnostics change nothing
    pi = policy_from_Q(Q)
    assert pi.shape == (16,) and pi.min() >= 0 and pi.max() <= 15
    assert ((d["random_branch"] + d["greedy_branch"]) == 30).all()
    assert (d["nongreedy"] <= d["random_branch"]).all()
    assert np.allclose(d["eps"], [epsilon_at(k) for k in range(300)], rtol=1e-9)
    # legacy call style (no seed) still works
    Q3, c3 = train(CafeSim(), episodes=20)
    assert Q3.shape == (16, 16) and len(c3) == 20


def test_epsilon_schedule():
    assert epsilon_at(12000) == pytest.approx(0.0907, abs=5e-4)
    assert epsilon_at(15000) == 0.05 and floor_episode() == 14978
    assert epsilon_at(14977) > 0.05 and epsilon_at(14978) == 0.05


def test_reachability_features_edges():
    assert reachable_states(P) == set(range(12))               # stocks 12-15 never occur
    f = state_features(4, CafeSim(mean_demand=0).params(), 6)
    assert np.isnan(f["base_coverage"]) and f["post_order_inventory"] == 10
    assert state_features(4, P)["expected_demand"] == pytest.approx(8.0)
    assert (CafeSim(mean_demand=0, busy_chance=0).sample_demands(2, seed=1) == 0).all()
    r = day_economics(0, 6, 2, CafeSim(spoil_frac=0).params())
    assert r["spoiled"] == 0 and r["carry"] == 4
