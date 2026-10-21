# The Smart Café · Café Quant Lab

Q-learning for a perishable daily order, with an interactive Streamlit dashboard and a Word evidence
pack. All figures are simulated.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                          # expect: 7 passed
streamlit run app.py
```

Press **Train / retrain agent** in the sidebar (about 10–15 s), then work through tabs A–I.
**Reset to assignment defaults** in the sidebar restores every slider.

Keep these together in one folder: `app.py`, `cafe.py`, `agent.py`, `test_cafe.py`,
`requirements.txt` and the `.streamlit/` folder (it holds the colour theme; the app still
runs without it, just with Streamlit's default red controls).

## Files

| File | What it is |
|---|---|
| `cafe.py` | Simulator. `CafeSim` keeps the assignment interface; adds an optional `seed`, extra ledger fields in `info`, and one shared `day_economics()` used everywhere. |
| `agent.py` | Q-learning. `train(env, ...) -> (Q, curve)` unchanged; `seed=` and `return_diagnostics=True` are optional. |
| `app.py` | Dashboard, tabs A–I, plus the Word evidence-pack builder. |
| `test_cafe.py` | 7 tests: worked example, rounding, invariants, reconciliation, seeding, ε schedule, edge cases. |
| `.streamlit/config.toml` | Palette theme. |

## What the dashboard does

| Tab | Contents |
|---|---|
| A · Command Centre | KPI cards, revenue-to-net-profit waterfall, distribution of monthly profit (risk view), computed insight |
| B · Learning Lab | Learning curve (raw + rolling mean, zoomable) and ε by episode, sharing one pinned episode; hyperparameters |
| C · Learned Strategy | Policy heatmap, **order-up-to view** (shelf after ordering), rule summary, diagnostics table |
| D · Ledger | Full 30-day ledger for any manager, reconciliation, daily/cumulative chart, CSV |
| E · Scoreboard | Monthly scoreboard on 500 identical months, constant-order sweep, 10-day challenge ranking |
| F · You vs AI | 10-day game; one verdict calculation feeds every card; **10-day scoreboard with you, the AI and every other manager** |
| G · Experiments | Four what-if scenarios, each retrained on **3 seeds**, with means, seed ranges and a theory check |
| H · Assignment | MDP table, P1–P7 evidence map, **Word evidence pack**, CSV exports, screenshot checklist |
| I · ML Diagnostics | Exploration vs exploitation (logged), state visitation, ε-decay sensitivity chart |

**Inspecting values.** Hover any point, bar or cell for exact values. Click to pin its details
below the chart; only that chart reruns, so clicking never retrains, moves the game or changes
demand. Every chart also has a selector to pin any point without the mouse.

**Evidence pack (tab H).** One click builds a Word document (about 5 pages) from your run: headline KPIs,
scoreboard and charts, learning curve, learned policy, your You-vs-AI result and 10-day ranking,
every experiment you ran, a draft owner recommendation (to rewrite in your own words) and an
evidence index for P1–P7. Rebuild it after playing the game or running experiments.

## Observed results (defaults, training seed 2026, 15,000 episodes)

Every policy evaluated on the same 500 held-out months (seed 20260):

| Manager | Avg monthly profit | Service | Spoilage KPI | Stockout rate |
|---|---:|---:|---:|---:|
| RL agent (greedy) | AED 730.10 | 96.7% | 13.9% | 3.3% |
| Generous, order 10 (guide) | AED 714.96 | 97.4% | 14.5% | 2.6% |
| Best constant order (= 10, found afterwards) | AED 714.96 | 97.4% | 14.5% | 2.6% |
| Generous, order 12 (original dashboard) | AED 668.44 | 98.6% | 14.0% | 1.4% |
| Random 0–15 | AED 339.91 | 78.3% | 10.5% | 21.7% |
| Cautious, order 4 | −AED 201.14 | 51.6% | 0.6% | 48.4% |

- Learning curve: first 1,500 episodes average AED 416.80, last 1,500 average AED 712.46.
- Learned policy (stock 0→15): `12 14 9 14 9 10 4 10 5 3 0 5 0 0 0 0`. It brings shelves to
  10–15 units after ordering (median 13); the order-up-to view in tab C shows this clearly.
- Stock levels 12–15 can never occur under the defaults (max carry-over is 11). Their "0"
  is an argmax default, not a learned choice. Stock 11 had only 81 training visits.
- ε-decay sensitivity (3 seeds each): 0.999 → AED 721 ± 21, 0.9995 → AED 718 ± 9,
  0.9998 (default) → AED 731 ± 12. Differences are within about one standard deviation,
  so there is no clear winner.

### Experiments (3 training seeds: 2026, 2027, 2028)

Baseline agents use the assignment defaults; each scenario changes one setting and retrains.
"Retraining gain" compares the retrained policy with keeping the old policy under the new economics.

| Change | What the agent did (mean of 3 seeds) | Theory check | Retraining gain |
|---|---|---|---:|
| Spoilage cost 1.5 → 3.0 | 280.9 → 276.4 units ordered/month; spoilage KPI 14.4% → 13.0% | fewer units in 2 of 3 seeds | −AED 5.67 (1 of 3 seeds positive) |
| Stockout penalty 6 → 12 | service 97.1% → 97.8%; 10.4 more units/month | higher service in 3 of 3 | +AED 7.87 (3 of 3) |
| Demand mean 7 → 9 | 280.9 → 328.5 units ordered/month | more units in 3 of 3 | +AED 38.32 (3 of 3) |
| Spoil fraction 25% → 50% | leftover held overnight 162.6 → 134.6 units/month | lower leftover in 3 of 3 | +AED 22.61 (3 of 3) |

Honest note on the spoilage-cost scenario: the agent leans the expected way (leaner stock, lower
spoilage KPI), but retraining does not reliably earn more. Doubling spoilage cost moves the best
order only slightly, so the gain is smaller than the seed-to-seed noise. Report it that way.

Your numbers will match these exactly if you keep seed 2026 and the defaults.

## Decisions taken (and why)

- **Generous baseline:** the guide says order 10, the original dashboard used 12. Both are shown, labelled.
- **Training episodes:** default 15,000 (guide), not 12,000 (original dashboard). At 12,000, ε only reaches 0.0907.
- **Kept as specified:** order cost on units the fridge cap discards; holding cost on units that then spoil;
  Python's banker's rounding for spoilage (2.5 → 2); bootstrapping from Q on day 30.
- **You vs AI:** demand comes from the simulator's own generator, settings are snapshotted when a
  challenge starts, and the AI replays the identical 10 days from stock 5. Totals, difference and winner
  are computed once, in whole cents, and every card, banner and chart reads that one result.
- **Stale model guard:** if any sidebar setting changes after training, RL results are hidden until you retrain.
- **Exploration RNG:** switched from Python `random` to a seeded NumPy generator. Same distributions;
  results are now reproducible.
- **Multi-seed experiments:** a single training seed can mislead (it did for the spoilage scenario), so
  experiments average 3 seeds and show the range.

## Known limitations

- One training seed drives the main model; tabs G and I show how much results move across seeds.
- The stock-only state cannot learn end-of-month run-down; that would need day-of-month in the state.
- The first experiment run trains 6 agents (about 1–2 minutes); later scenarios reuse the baseline
  agents and are faster, and everything is cached once computed.
