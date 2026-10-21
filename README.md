# The Smart Café · Café Quant Lab

Q-learning for a perishable daily order, with a Streamlit dashboard. All figures are simulated.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # expect: 7 passed
streamlit run app.py
```

Press **Train / retrain agent** in the sidebar (about 10–15 s), then work through tabs A–I.

Keep these together in one folder: `app.py`, `cafe.py`, `agent.py`, `test_cafe.py`,
`requirements.txt` and the `.streamlit/` folder (it holds the colour theme; the app still
runs without it, just with Streamlit's default red controls).
`app.py` is the only dashboard. It replaces the supplied one.

## Files

| File | What it is |
|---|---|
| `cafe.py` | Simulator. `CafeSim` keeps the assignment interface; adds an optional `seed`, extra ledger fields in `info`, and one shared `day_economics()` used everywhere. |
| `agent.py` | Q-learning. `train(env, ...) -> (Q, curve)` unchanged; `seed=` and `return_diagnostics=True` are optional. |
| `app.py` | Dashboard, tabs A–I. |
| `test_cafe.py` | 7 tests: worked example, rounding, invariants, reconciliation, seeding, ε schedule, edge cases. |
| `.streamlit/config.toml` | Palette theme. |

## Observed results (defaults, training seed 2026, 15,000 episodes)

Every policy evaluated on the same 500 held-out months (seed 20260):

| Manager | Avg monthly profit | Service | Spoilage KPI | Stockout rate |
|---|---:|---:|---:|---:|
| RL agent (greedy) | AED 730.10 | 96.7% | 13.9% | 3.3% |
| Generous, order 10 (guide) | AED 714.96 | 97.4% | 14.5% | 2.6% |
| Best constant order (= 10, found afterwards) | AED 714.96 | | | |
| Generous, order 12 (original dashboard) | AED 668.44 | 98.6% | 14.0% | 1.4% |
| Random 0–15 | AED 339.91 | 78.3% | 10.5% | 21.7% |
| Cautious, order 4 | −AED 201.14 | 51.6% | 0.6% | 48.4% |

- Learning curve: first 1,500 episodes average AED 416.80, last 1,500 average AED 712.46.
- Learned policy (stock 0→15): `12 14 9 14 9 10 4 10 5 3 0 5 0 0 0 0`. It brings shelves to
  10–15 units after ordering (median 13). It is not perfectly monotone.
- Stock levels 12–15 can never occur under the defaults (max carry-over is 11). Their "0"
  is an argmax default, not a learned choice. Stock 11 had only 81 training visits.
- ε-decay sensitivity (3 seeds each): 0.999 → AED 721 ± 21, 0.9995 → AED 718 ± 9,
  0.9998 (default) → AED 731 ± 12. The differences are within about one standard deviation,
  so there is no clear winner.

Experiments at 15,000 episodes (retrained policy vs the old policy under the new economics):

| Change | What the agent did | Profit effect of retraining |
|---|---|---:|
| Spoilage cost 1.5 → 3.0 | ordered slightly less; spoilage KPI 13.9% → 13.4% | −AED 7.6 (noise; see note) |
| Stockout penalty 6 → 12 | bigger buffer; service 96.7% → 97.5% | +AED 21.0 |
| Demand mean 7 → 9 | orders 281 → 340 units/month | +AED 46.0 |
| Spoil fraction 25% → 50% | holds less overnight (median post-order 13 → 11) | +AED 22.8 |

Note: in the spoilage-cost run the agent moved in the expected direction, but the retrained
policy earned slightly less than the old one. A single-seed Q-learning policy is not
guaranteed to be optimal. Report this honestly rather than smoothing it over.

Your numbers will match these exactly if you keep seed 2026 and the defaults.

## Decisions taken (and why)

- **Generous baseline:** the guide says order 10, the original dashboard used 12. Both are shown, labelled.
- **Training episodes:** default 15,000 (guide), not 12,000 (original dashboard). At 12,000, ε only reaches 0.0907.
- **Kept as specified:** order cost on units the fridge cap discards; holding cost on units that then spoil;
  Python's banker's rounding for spoilage (2.5 → 2); bootstrapping from Q on day 30.
- **You vs AI:** demand now comes from the simulator's own generator (the original hard-coded the
  busy-day chance) and the AI replays the identical 10 days from stock 5.
- **Stale model guard:** if any sidebar setting changes after training, RL results are hidden until you retrain.
- **Exploration RNG:** switched from Python `random` to a seeded NumPy generator. Same distributions;
  results are now reproducible.

## Known limitations

- Charts are static Matplotlib images; on a phone they are small but readable when zoomed.
- One training seed is a demonstration, not a statistically settled result (see tab I sensitivity).
- The stock-only state cannot learn end-of-month run-down; that would need day-of-month in the state.
