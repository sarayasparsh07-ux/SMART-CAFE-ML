"""The Smart Cafe - Cafe Quant Lab dashboard.

Launch:   streamlit run app.py
Needs:    cafe.py (CafeSim + helpers) and agent.py (train, policy_from_Q) in the same folder.
All figures are SIMULATED - not real cafe sales or market data.
pandas is used only for table display (it ships with Streamlit); all simulation uses numpy.
"""
from __future__ import annotations

import csv
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="The Smart Café · Café Quant Lab", page_icon="☕", layout="wide")

try:
    from agent import epsilon_at, floor_episode, policy_from_Q, train
    from cafe import CafeSim, evaluate, reachable_states, run_month, state_features, totals
except Exception as exc:  # clear, actionable message instead of a stack trace
    st.error("Could not import **cafe.py** / **agent.py**. Keep both files next to app.py.")
    st.exception(exc)
    st.stop()

# ---------------------------------------------------------------- design tokens
NAVY, CARD, TEAL, EMERALD = "#14243A", "#FFFDF9", "#1F9D8B", "#23856D"
AMBER, CORAL, BLUE, SLATE = "#D9A441", "#D96C5F", "#557DA8", "#627184"
EVAL_SEED, SENS_SEED, GAME_SEED = 20260, 31337, 4242
EVAL_MONTHS, GAME_DAYS, MIN_VISITS = 500, 10, 100
DEFAULTS = dict(spoilage_cost=1.5, stockout_penalty=6.0, mean_demand=7, spoil_frac=0.25)

CSS = f"""<style>
.stApp {{background: radial-gradient(circle at 8% 0%, rgba(217,164,65,.10), transparent 38%),
  radial-gradient(circle at 95% 10%, rgba(85,125,168,.08), transparent 35%),
  linear-gradient(180deg,#F6F3EC 0%,#F1ECE1 100%);}}
.stApp::before {{content:"";position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(98,113,132,.045) 1px,transparent 1px),
  linear-gradient(90deg,rgba(98,113,132,.045) 1px,transparent 1px);background-size:28px 28px;}}
.hero {{background:linear-gradient(120deg,{NAVY},#1d3352);color:#fff;border-radius:16px;
  padding:20px 24px 40px;border-left:6px solid {TEAL};position:relative;overflow:hidden;margin-bottom:6px}}
.hero h1, .hero p {{position:relative;z-index:1}}
.hero h1 {{color:#fff;margin:0;font-size:1.6rem;line-height:1.25}}
.hero p {{color:#dbe3ec;margin:.4rem 0 0;max-width:60rem}}
.hero::after {{content:"";position:absolute;right:0;bottom:0;width:40%;height:30px;opacity:.22;
  background:repeating-linear-gradient(90deg,{AMBER} 0 3px,transparent 3px 13px);
  clip-path:polygon(0 70%,10% 40%,20% 60%,30% 20%,40% 55%,50% 30%,60% 65%,70% 25%,80% 50%,90% 35%,100% 60%,100% 100%,0 100%);}}
.kpi {{background:{CARD};border-radius:12px;padding:10px 14px;box-shadow:0 2px 8px rgba(20,36,58,.08);
  border:1px solid #ece6da;border-top:4px solid var(--c);font-variant-numeric:tabular-nums;
  min-height:88px;margin-bottom:8px}}
.kpi .l {{font-size:.72rem;color:{SLATE};text-transform:uppercase;letter-spacing:.05em}}
.kpi .v {{font-size:1.35rem;font-weight:700;color:{NAVY};white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.kpi .d {{font-size:.72rem;color:{SLATE}}}
.chips {{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 10px}}
.chip {{background:{CARD};border:1px solid #e2dccf;border-radius:999px;padding:2px 10px;font-size:.76rem;color:{NAVY}}}
.sect {{font-size:.8rem;color:{SLATE};text-transform:uppercase;letter-spacing:.06em;margin:10px 0 4px}}
@media (max-width: 820px) {{ .hero h1 {{font-size:1.2rem}} .kpi .v {{font-size:1.1rem}} }}
</style>"""


def html(s: str, where=st) -> None:
    fn = getattr(where, "html", None)
    if fn:
        fn(s)
    else:  # very old Streamlit fallback
        where.markdown(s, unsafe_allow_html=True)


html(CSS)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": SLATE, "axes.labelcolor": NAVY,
                     "axes.titlecolor": NAVY, "axes.titleweight": "bold", "xtick.color": SLATE,
                     "ytick.color": SLATE, "axes.grid": True, "grid.alpha": .25,
                     "figure.facecolor": CARD, "axes.facecolor": CARD})


def show(fig) -> None:
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)  # avoid figure leaks across reruns


def aed(x: float) -> str:
    return f"−AED {abs(x):,.2f}" if x < 0 else f"AED {x:,.2f}"


def pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:.1f}%"


def cards(items) -> None:
    for col, (label, value, note, color) in zip(st.columns(len(items)), items):
        html(f"<div class='kpi' style='--c:{color}'><div class='l'>{label}</div>"
             f"<div class='v'>{value}</div><div class='d'>{note}</div></div>", col)


def items_of(d: dict) -> tuple:
    return tuple(sorted(d.items()))


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🏪 Café economics")
    spoil_cost = st.slider("Spoilage cost (AED / spoiled unit)", 0.0, 5.0, 1.5, 0.1, key="sb_spoil_cost")
    penalty = st.slider("Stockout penalty (AED / unmet unit)", 0.0, 12.0, 6.0, 0.5, key="sb_penalty")
    mean_d = st.slider("Typical daily demand (Poisson mean)", 3, 12, 7, key="sb_mean")
    spoil_frac = st.slider("Share of leftovers that spoil overnight", 0.0, 1.0, 0.25, 0.05, key="sb_frac")
    st.caption("Fixed: price AED 5, order AED 1, holding AED 0.30, busy day 20% (+Poisson 5), "
               "fridge 15, start stock 5, 30 days.")
    st.header("🧠 Learning")
    episodes = st.slider("Training episodes (months)", 2000, 20000, 15000, 1000, key="sb_episodes")
    alpha = st.slider("Learning rate α", 0.01, 0.50, 0.15, 0.01, key="sb_alpha")
    gamma = st.slider("Discount γ (value of the future)", 0.0, 0.99, 0.95, 0.01, key="sb_gamma")
    seed = int(st.number_input("Training seed", 0, 10**6, 2026, key="sb_seed"))
    st.caption("ε schedule (assignment default): start 1.0, ×0.9998 per episode, floor 0.05.")
    train_clicked = st.button("🚀 Train / retrain agent", type="primary", key="btn_train")

ENV = items_of(dict(spoilage_cost=spoil_cost, stockout_penalty=penalty,
                    mean_demand=mean_d, spoil_frac=spoil_frac))
CFG = dict(env=ENV, episodes=episodes, alpha=alpha, gamma=gamma, seed=seed)
P = CafeSim(**dict(ENV)).params()


# ---------------------------------------------------------------- cached computations
@st.cache_data(show_spinner="Training the Q-learning agent (≈10–15 s)…", max_entries=24)
def cached_train(env_items, episodes, alpha, gamma, seed, eps_decay=0.9998):
    Q, curve, diag = train(CafeSim(**dict(env_items)), episodes=episodes, alpha=alpha, gamma=gamma,
                           seed=seed, eps_decay=eps_decay, return_diagnostics=True)
    return Q, np.asarray(curve), diag


@st.cache_data(max_entries=32, show_spinner=False)
def demand_matrix(env_items, months, seed):
    return CafeSim(**dict(env_items)).sample_demands(months, seed=seed)


@st.cache_data(max_entries=64, show_spinner=False)
def eval_pi(env_items, pi, seed=EVAL_SEED):
    p = CafeSim(**dict(env_items)).params()
    return evaluate(lambda s, t: pi[s], demand_matrix(env_items, EVAL_MONTHS, seed), p)


@st.cache_data(max_entries=16, show_spinner="Evaluating every policy on the same demand months…")
def scoreboard(env_items, pi):
    p = CafeSim(**dict(env_items)).params()
    D = demand_matrix(env_items, EVAL_MONTHS, EVAL_SEED)
    g = np.random.default_rng(EVAL_SEED + 1)  # random manager's own stream; demand stays common
    pols = {"RL agent (greedy)": lambda s, t: pi[s],
            "Cautious (order 4)": lambda s, t: 4,
            "Generous (order 10 · guide)": lambda s, t: 10,
            "Generous (order 12 · original dashboard)": lambda s, t: 12,
            "Random (uniform 0–15)": lambda s, t: int(g.integers(0, int(p["max_stock"]) + 1))}
    res = {k: evaluate(f, D, p) for k, f in pols.items()}
    const = {k: evaluate(lambda s, t, k=k: k, D, p)["mean_profit"] for k in range(int(p["max_stock"]) + 1)}
    return res, const


if train_clicked:
    Q, curve, diag = cached_train(ENV, episodes, alpha, gamma, seed)
    st.session_state["model"] = dict(cfg=CFG, Q=Q, curve=curve, diag=diag)
MODEL = st.session_state.get("model")
STALE = MODEL is not None and MODEL["cfg"] != CFG
PI = tuple(int(a) for a in policy_from_Q(MODEL["Q"])) if MODEL else None


def need_model() -> bool:
    """Guard: block RL views when no model exists or it was trained on other settings."""
    if MODEL is None:
        st.info("👈 Press **Train / retrain agent** in the sidebar to populate this section.")
        return False
    if STALE:
        st.warning("Settings changed since training, so these results are hidden to avoid showing a "
                   "policy evaluated on economics it never learned. Press **Train / retrain agent**. "
                   f"Model was trained on: {dict(MODEL['cfg']['env'])}, episodes={MODEL['cfg']['episodes']}, "
                   f"α={MODEL['cfg']['alpha']}, γ={MODEL['cfg']['gamma']}, seed={MODEL['cfg']['seed']}.")
        return False
    return True


# ---------------------------------------------------------------- header
html("<div class='hero'><h1>☕ The Smart Café · Café Quant Lab</h1><p>Every morning the café makes one "
     "decision: <b>how many units of a perishable item to order</b>. Too many spoil; too few lose sales. "
     "A Q-learning agent learns the ordering rule from profit alone. All figures are <b>simulated</b>, "
     "not real café data.</p></div>")
html("<div class='chips'>" + "".join(f"<span class='chip'>{t}</span>" for t in [
    f"Fridge {P['max_stock']} units · start {P['start_stock']}", f"Price AED {P['price']:.2f}",
    f"Order AED {P['order_cost']:.2f}/u", f"Holding AED {P['holding_cost']:.2f}/u",
    f"Spoilage AED {P['spoilage_cost']:.2f}/u", f"Stockout AED {P['stockout_penalty']:.2f}/u",
    f"{P['spoil_frac']:.0%} of leftovers spoil",
    f"Demand Poisson({P['mean_demand']}) + {P['busy_chance']:.0%} busy days +Poisson({P['busy_extra']:g})",
    f"{P['days']}-day month"]) + "</div>")
if MODEL is not None:
    (st.warning if STALE else st.success)(
        "Model status: " + ("**stale** – sidebar settings differ from training. Retrain." if STALE else
                            f"trained on the current settings ({MODEL['cfg']['episodes']:,} episodes, seed {MODEL['cfg']['seed']})."))

tabs = st.tabs(["A · Command Centre", "B · Learning Lab", "C · Learned Strategy", "D · Ledger",
                "E · Scoreboard", "F · You vs AI", "G · Experiments", "H · Assignment",
                "I · ML Diagnostics"])

# ================================================================ A · Command Centre
with tabs[0]:
    if need_model():
        res, const = scoreboard(ENV, PI)
        r = res["RL agent (greedy)"]
        html("<div class='sect'>Headline KPIs · RL policy on "
             f"{EVAL_MONTHS} held-out simulated months</div>")
        cards([("Avg monthly net profit", aed(r["mean_profit"]), f"± {r['std_profit']:,.0f} std across months", EMERALD),
               ("Service level", pct(r["service"]), "units sold ÷ units demanded", TEAL),
               ("Spoilage KPI", pct(r["spoilage_kpi"]), "units spoiled ÷ units ordered", AMBER),
               ("Stockout rate", pct(r["stockout_rate"]), "unmet units ÷ units demanded", CORAL)])
        html("<div class='sect'>Monthly P&amp;L (average month)</div>")
        cards([("Revenue", aed(r["mean_revenue"]), f"{r['mean_sold']:.1f} units sold", TEAL),
               ("Order cost", aed(r["mean_order_cost"]), f"{r['mean_order']:.1f} units ordered", SLATE),
               ("Holding cost", aed(r["mean_holding_cost"]), f"{r['mean_leftover']:.1f} leftover units", SLATE),
               ("Spoilage cost", aed(r["mean_spoilage_cost"]), f"{r['mean_spoiled']:.1f} units spoiled", AMBER),
               ("Stockout penalty", aed(r["mean_stockout_penalty"]), f"{r['mean_stockout']:.1f} unmet units", CORAL)])
        cards([("Profit per day", aed(r["profit_per_day"]), "monthly profit ÷ 30", EMERALD),
               ("Net margin", pct(r["net_margin"]), "net profit ÷ revenue", TEAL),
               ("Loss-making months", pct(r["loss_share"]), "share of months with profit < 0", CORAL),
               ("5th-percentile month", aed(r["p5"]), "a bad (1-in-20) month", SLATE)])
        costs = {"order cost": r["mean_order_cost"], "holding": r["mean_holding_cost"],
                 "spoilage": r["mean_spoilage_cost"], "stockout penalty": r["mean_stockout_penalty"]}
        top = max(costs, key=costs.get)
        best_k = max(const, key=const.get)
        gap = r["mean_profit"] - const[best_k]
        pressure = ("waste pressure (spoilage cost > stockout penalty)" if r["mean_spoilage_cost"] > r["mean_stockout_penalty"]
                    else "stockout pressure (stockout penalty ≥ spoilage cost)")
        tie = abs(gap) <= 0.01 * max(abs(const[best_k]), 1)
        st.info(f"**Computed insight.** The largest cost line is **{top}** ({aed(costs[top])}/month), and the "
                f"policy currently faces more **{pressure}**. On identical demand the RL policy earned "
                f"{aed(r['mean_profit'])}/month vs {aed(const[best_k])} for the best constant order "
                f"({best_k}/day, found afterwards by brute force): difference {aed(gap)}"
                + (" — effectively a tie (within 1%)." if tie else "."))
        with st.expander("Methodology: how these numbers are produced"):
            st.markdown(
                f"- The greedy policy (no exploration) runs on **{EVAL_MONTHS} demand months generated with seed "
                f"{EVAL_SEED}**, separate from the training seed. Every policy in tab E sees the *same* months "
                "(common random numbers), so differences come from the policy, not luck.\n"
                "- Ratios are pooled: e.g. service = total sold ÷ total demand over all months.\n"
                "- **Spoilage KPI** (spoiled ÷ ordered) is not the overnight spoil fraction "
                "(spoiled ÷ leftover, a business parameter). Pooled spoiled ÷ leftover here = "
                f"{pct(r['spoil_of_leftover'])}.\n"
                "- Fixed parameters (price, costs) are inputs; everything else is a simulation estimate.")

# ================================================================ B · Learning Lab
with tabs[1]:
    if need_model():
        c, d = MODEL["curve"], MODEL["diag"]
        w = max(10, min(200, len(c) // 20))
        roll = np.convolve(c, np.ones(w) / w, mode="valid")
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5), sharex=True,
                                      gridspec_kw={"height_ratios": [2.2, 1]})
        ax.plot(c, color=BLUE, alpha=.18, lw=.6, label="Raw profit per training month")
        ax.plot(np.arange(w - 1, len(c)), roll, color=NAVY, lw=2, label=f"Rolling mean ({w} episodes)")
        ax.set(ylabel="Training profit (AED / month)", title="Learning curve — judge the trend, not single episodes")
        ax.legend(loc="lower right")
        ax2.plot(d["eps"], color=BLUE, lw=1.5, label="ε used in episode")
        ax2.axhline(MODEL["diag"]["config"]["eps_min"], color=AMBER, ls="--", label="Floor ε_min = 0.05")
        fe = floor_episode()
        if fe is not None and fe < len(c):
            ax2.axvline(fe, color=SLATE, ls=":", label=f"Floor first used: episode {fe:,}")
        ax2.set(xlabel="Episode (simulated month)", ylabel="ε", title="Exploration rate ε")
        ax2.legend(loc="upper right")
        show(fig)
        n = max(1, len(c) // 10)
        early, late = float(np.mean(c[:n])), float(np.mean(c[-n:]))
        k1, k2, k3 = st.columns(3)
        with k1:
            cards([(f"First {n:,} episodes", aed(early), "avg training profit", BLUE)])
        with k2:
            cards([(f"Last {n:,} episodes", aed(late), "avg training profit", EMERALD)])
        with k3:
            cards([("Change", aed(late - early), "late − early (includes exploration)", SLATE)])
        st.caption("Both panels share the episode axis. The curve rises as ε falls, but a shared trend is not "
                   "proof of cause: Q-values are also improving. Training profit still contains random "
                   "exploration, so it sits below the held-out evaluation in tabs A and E.")
        cfg = MODEL["diag"]["config"]
        st.dataframe(pd.DataFrame({
            "Hyperparameter": ["Episodes", "Episode length", "α (learning rate)", "γ (discount)",
                               "ε start", "ε decay (per episode)", "ε floor", "Seed"],
            "Value": [f"{cfg['episodes']:,}", "30 days", cfg["alpha"], cfg["gamma"], cfg["eps_start"],
                      cfg["eps_decay"], cfg["eps_min"], cfg["seed"]],
            "What it controls": ["Months of practice", "Steps per episode",
                                 "How far each update moves Q toward the new target",
                                 "How much tomorrow's value counts today",
                                 "Probability of the random branch at the start",
                                 "Multiplier applied to ε after every episode",
                                 "Minimum exploration that never switches off",
                                 "Makes training reproducible"]}).astype(str), hide_index=True)
        weak = [s for s, v in enumerate(d["state_visits"]) if v < MIN_VISITS]
        st.markdown("**How to read this tab.** *Training reward* (above) mixes learning with exploration. "
                    "*Held-out evaluation* (tabs A/E) runs the greedy policy on separate months. *Policy behaviour* "
                    f"(tab C) is the stock → order rule itself. Stock levels {weak} were visited fewer than "
                    f"{MIN_VISITS} times, so treat their actions with caution (details in tab I).")

# ================================================================ C · Learned Strategy
with tabs[2]:
    if need_model():
        visits = MODEL["diag"]["state_visits"]
        reach = reachable_states(P)
        fig, ax = plt.subplots(figsize=(10, 2.0))
        ax.imshow(np.array(PI)[None, :], cmap="cividis", aspect="auto", vmin=0, vmax=15)
        for s, a in enumerate(PI):
            lab = f"{a}" if s in reach else f"{a}\nn/r"
            ax.text(s, 0, lab, ha="center", va="center", color="white" if a < 9 else NAVY, fontsize=9,
                    fontweight="bold")
        ax.set(xticks=range(16), yticks=[], xlabel="Stock on hand this morning (state)",
               title="Learned policy: recommended order for each stock level  (n/r = unreachable, not learned)")
        ax.grid(False)
        show(fig)
        rows = []
        for s in range(16):
            f = state_features(s, P, PI[s])
            status = ("unreachable" if s not in reach else
                      "under-visited" if visits[s] < MIN_VISITS else "well visited")
            rows.append({"Stock": s, "Recommended order": PI[s], "Post-order inventory": f["post_order_inventory"],
                         "Training visits": int(visits[s]), "Evidence": status,
                         "Utilisation (diag.)": f["utilisation"],
                         "Coverage ÷ base mean (diag.)": f["base_coverage"],
                         "Coverage ÷ expected demand (diag.)": f["expected_coverage"],
                         "Remaining capacity (diag.)": f["remaining_capacity"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={
            "Utilisation (diag.)": st.column_config.NumberColumn(format="%.2f"),
            "Coverage ÷ base mean (diag.)": st.column_config.NumberColumn(format="%.2f"),
            "Coverage ÷ expected demand (diag.)": st.column_config.NumberColumn(format="%.2f")}, height=600)
        learned = [s for s in range(16) if s in reach and visits[s] >= MIN_VISITS]
        if learned:
            post = [min(s + PI[s], 15) for s in learned]
            mono = all(PI[a] >= PI[b] for a, b in zip(learned, learned[1:]))
            lo_stock, hi_stock = learned[:len(learned) // 2], learned[len(learned) // 2:]
            lo_avg = np.mean([PI[s] for s in lo_stock])
            hi_avg = np.mean([PI[s] for s in hi_stock])
            st.markdown(
                f"**Rule summary (generated from this policy).** Across well-visited stock levels "
                f"{learned[0]}–{learned[-1]}, the agent orders on average **{lo_avg:.1f}** units when stock is in the "
                f"lower half ({lo_stock[0]}–{lo_stock[-1]}) and **{hi_avg:.1f}** in the upper half "
                f"({hi_stock[0]}–{hi_stock[-1]}), bringing shelves to **{min(post)}–{max(post)} units** after "
                f"ordering (median {int(np.median(post))}). "
                + ("Orders fall as stock rises. " if lo_avg > hi_avg else "Orders do **not** fall as stock rises. ")
                + ("Orders never rise with stock, a clean order-up-to pattern." if mono else
                   "The rule is not perfectly monotone: neighbouring stock levels can differ by several units. "
                   "This is typical of tabular Q-learning with noisy rewards, where several orders have nearly "
                   "equal value; inspect the post-order column rather than single entries."))
        weak = [s for s in range(16) if s not in learned]
        st.caption(f"Caution: stock levels {weak} are unreachable or visited fewer than {MIN_VISITS} times, so their "
                   "orders are not reliable. Never-visited rows keep all-zero Q-values, and argmax then returns 0 "
                   "by tie-breaking — that '0' is a default, not a learned decision.")
        with st.expander("Feature-engineering rationale: why the state is stock only"):
            st.markdown(
                "- **Model input:** only *stock on hand this morning* (0–15). Demand is independent day to day, "
                "so stock captures everything the order decision can act on.\n"
                "- **Diagnostic columns (diag.)** are one-to-one transformations of stock: utilisation = stock ÷ 15, "
                "coverage = stock ÷ mean demand (or ÷ expected demand "
                f"{P['mean_demand'] + P['busy_chance'] * P['busy_extra']:.1f} = mean + busy chance × 5), remaining "
                "capacity = 15 − stock. They help a human read the policy but add **no new information** to the "
                "Q-table, so they are **not model inputs**.\n"
                "- **No leakage:** today's demand is drawn *after* the order is chosen; it is never part of the state.")

# ================================================================ D · Ledger
with tabs[3]:
    pol_names = ["RL agent (greedy)", "Cautious (order 4)", "Generous (order 10 · guide)",
                 "Generous (order 12 · original dashboard)"]
    fixed = {"Cautious (order 4)": 4, "Generous (order 10 · guide)": 10, "Generous (order 12 · original dashboard)": 12}
    c1, c2 = st.columns([2, 1])
    choice = c1.selectbox("Who runs the café this month?", pol_names, key="ledger_policy")
    month = int(c2.number_input(f"Evaluation month (0–{EVAL_MONTHS - 1})", 0, EVAL_MONTHS - 1, 0, key="ledger_month"))
    if choice in fixed or need_model():
        pol = (lambda s, t: fixed[choice]) if choice in fixed else (lambda s, t: PI[s])
        rows = run_month(pol, demand_matrix(ENV, EVAL_MONTHS, EVAL_SEED)[month], P)
        cols = ["day", "start_stock", "order", "available", "demand", "sold", "stockout", "leftover", "spoiled",
                "carry", "revenue", "order_cost", "holding_cost", "spoilage_cost", "stockout_penalty", "profit"]
        names = ["Day", "Start stock", "Order", "Available", "Demand", "Sold", "Unmet (stockout)", "Leftover",
                 "Spoiled", "Carry-over", "Revenue", "Order cost", "Holding cost", "Spoilage cost",
                 "Stockout penalty", "Net profit"]
        df = pd.DataFrame(rows)[cols]
        df.columns = names
        money = {c: st.column_config.NumberColumn(format="%.2f") for c in names[10:]}
        st.dataframe(df, hide_index=True, column_config=money, height=420)
        t = totals(rows)
        recon = t["revenue"] - (t["order_cost"] + t["holding_cost"] + t["spoilage_cost"] + t["stockout_penalty"])
        cards([("Revenue", aed(t["revenue"]), f"{int(t['sold'])} units sold", TEAL),
               ("Order cost", aed(t["order_cost"]), f"{int(t['order'])} units ordered", SLATE),
               ("Holding", aed(t["holding_cost"]), f"{int(t['leftover'])} leftover units", SLATE),
               ("Spoilage", aed(t["spoilage_cost"]), f"{int(t['spoiled'])} units spoiled", AMBER),
               ("Stockout penalty", aed(t["stockout_penalty"]), f"{int(t['stockout'])} unmet units", CORAL),
               ("Net profit", aed(t["profit"]), "sum of 30 daily profits", EMERALD)])
        ok = abs(recon - t["profit"]) < 1e-6
        st.markdown(f"**Reconciliation:** revenue − order − holding − spoilage − stockout = **{aed(recon)}**; "
                    f"sum of daily net profit = **{aed(t['profit'])}** → "
                    + ("✅ reconciles exactly." if ok else "⚠️ does not reconcile."))
        st.caption(f"Accounting conventions (as specified): order cost is charged on every ordered unit, including "
                   f"{int(t['clipped_units'])} unit(s) this month discarded by the 15-unit fridge cap; holding cost "
                   "applies to all leftovers, including those that then spoil; spoiled units use Python's round() "
                   "(ties go to the even number, e.g. 2.5 → 2).")
        prof = np.array([r["profit"] for r in rows])
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.bar(range(1, 31), prof, color=[EMERALD if v >= 0 else CORAL for v in prof], alpha=.55,
               label="Daily net profit")
        ax2 = ax.twinx()
        ax2.plot(range(1, 31), np.cumsum(prof), color=NAVY, marker="o", ms=3, lw=1.8, label="Cumulative profit")
        ax2.spines["right"].set_visible(True)
        ax.axhline(0, color=SLATE, lw=.8)
        ax.set(xlabel="Day", ylabel="Daily profit (AED)", title=f"{choice} · month {month}")
        ax2.set_ylabel("Cumulative profit (AED)")
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left")
        show(fig)
        buf = io.StringIO()
        wr = csv.DictWriter(buf, fieldnames=cols + ["clipped_units"], extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)
        st.download_button("⬇️ Download this ledger (CSV)", buf.getvalue().encode("utf-8"),
                           f"ledger_{'rl' if choice not in fixed else fixed[choice]}_month{month}.csv",
                           "text/csv", key="dl_ledger")

# ================================================================ E · Scoreboard
with tabs[4]:
    if need_model():
        res, const = scoreboard(ENV, PI)
        best_k = max(const, key=const.get)
        rl = res["RL agent (greedy)"]["mean_profit"]
        tbl = [{"Policy": k, "Profit / month": v["mean_profit"], "Std": v["std_profit"],
                "IQR": v["q75"] - v["q25"], "5th pct month": v["p5"], "Loss months": v["loss_share"],
                "Service": v["service"], "Stockout rate": v["stockout_rate"], "Spoilage KPI": v["spoilage_kpi"],
                "Δ vs RL (AED)": v["mean_profit"] - rl} for k, v in res.items()]
        v = eval_pi(ENV, tuple([best_k] * 16))  # same held-out months as every other row
        tbl.append({"Policy": f"Best constant order = {best_k} (diagnostic benchmark)",
                    "Profit / month": v["mean_profit"], "Std": v["std_profit"], "IQR": v["q75"] - v["q25"],
                    "5th pct month": v["p5"], "Loss months": v["loss_share"], "Service": v["service"],
                    "Stockout rate": v["stockout_rate"], "Spoilage KPI": v["spoilage_kpi"],
                    "Δ vs RL (AED)": v["mean_profit"] - rl})
        tbl.sort(key=lambda r: -r["Profit / month"])
        st.dataframe(pd.DataFrame(tbl), hide_index=True, column_config={
            **{c: st.column_config.NumberColumn(format="%.2f") for c in
               ["Profit / month", "Std", "IQR", "5th pct month", "Δ vs RL (AED)"]},
            **{c: st.column_config.NumberColumn(format="%.3f") for c in
               ["Loss months", "Service", "Stockout rate", "Spoilage KPI"]}})
        fig, ax = plt.subplots(figsize=(9, 3.4))
        names, vals = [r["Policy"] for r in tbl], [r["Profit / month"] for r in tbl]
        colors = [EMERALD if n.startswith("RL") else BLUE if n.startswith("Best") else
                  (SLATE if v >= 0 else CORAL) for n, v in zip(names, vals)]
        ax.barh(names, vals, color=colors, hatch=["//" if n.startswith("Best") else "" for n in names])
        span = max(vals) - min(min(vals), 0)
        for i, v in enumerate(vals):
            ax.text(v + (0.01 * span if v >= 0 else -0.01 * span), i, aed(v), va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color=SLATE, lw=.8)
        ax.set_xlim(min(min(vals), 0) - 0.25 * span, max(vals) + 0.25 * span)
        ax.invert_yaxis()
        ax.set(xlabel="Average monthly profit (AED)",
               title=f"Same {EVAL_MONTHS} demand months for every manager (RL = green, benchmark = hatched)")
        show(fig)
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ks, vs = list(const), list(const.values())
        ax.plot(ks, vs, marker="s", color=BLUE, label="Constant daily order")
        ax.axhline(rl, color=EMERALD, ls="--", label=f"RL agent ({aed(rl)})")
        ax.axhline(0, color=SLATE, lw=.8)
        ax.set(xlabel="Constant order every day (units)", ylabel="AED / month", xticks=ks,
               title="Diagnostic: profit of every constant order (the agent was never shown this)")
        ax.legend(loc="lower right")
        show(fig)
        cu = P["price"] - P["order_cost"] + P["stockout_penalty"]
        co = P["holding_cost"] + P["spoil_frac"] * (P["spoilage_cost"] + P["order_cost"])
        st.markdown(f"**Why under-ordering hurts.** One unit short costs about **{aed(cu)}** (lost margin "
                    f"AED {P['price'] - P['order_cost']:.2f} + penalty AED {P['stockout_penalty']:.2f}). One unit too many "
                    f"costs about **{aed(co)}** (holding + the expected share that spoils). With shortages ~"
                    f"{cu / max(co, 1e-9):.0f}× dearer, the newsvendor critical ratio is ≈ {cu / (cu + co):.2f}, which "
                    "points to stocking above average demand. This is a rough single-day guide; carry-over and the "
                    "fridge cap make the real optimum a simulation question.")
        g = st.session_state.get("game")
        if g and len(g["rows"]) >= GAME_DAYS and g["env"] == ENV:
            ai_rows = run_month(lambda s, t: PI[s], g["demands"], P)
            you, ai = totals(g["rows"])["profit"], totals(ai_rows)["profit"]
            st.markdown(f"**You vs AI (10-day challenge, tab F):** you {aed(you)}, AI {aed(ai)} on the same "
                        "10 days. This is a 10-day result, so it is not comparable to the monthly averages above.")

# ================================================================ F · You vs AI
with tabs[5]:
    st.markdown("You run the café for **10 days**. Each morning you see your stock and choose an order; demand "
                "is revealed only after you open the shop. Then the AI plays the **same 10 demand days** from the "
                "same starting stock with the same economics.")

    def new_game(rnd: int) -> None:
        st.session_state["game"] = dict(env=ENV, round=rnd, rows=[], stock=int(P["start_stock"]),
                                        demands=CafeSim(**dict(ENV)).sample_demands(
                                            1, GAME_DAYS, GAME_SEED + rnd)[0].tolist())

    g = st.session_state.get("game")
    if g is None or g["env"] != ENV:
        if g is not None:
            st.warning("Business settings changed, so the challenge was reset — both players must face the same economics.")
        new_game(0 if g is None else g["round"] + 1)
        g = st.session_state["game"]

    def play() -> None:
        gm = st.session_state["game"]
        if len(gm["rows"]) < GAME_DAYS:
            t = len(gm["rows"])
            row = run_month(lambda s, _t: int(st.session_state["game_order"]), [gm["demands"][t]], P, gm["stock"])[0]
            row["day"] = t + 1
            gm["rows"].append(row)
            gm["stock"] = int(row["carry"])

    day = len(g["rows"])
    if day < GAME_DAYS:
        cards([("Day", f"{day + 1} of {GAME_DAYS}", f"challenge #{g['round'] + 1}", BLUE),
               ("Stock this morning", f"{g['stock']} units", "carried over from yesterday", TEAL),
               ("Your profit so far", aed(sum(r['profit'] for r in g['rows'])), "net of all costs", EMERALD)])
        st.number_input("Your order this morning (0–15 units)", 0, 15, 8, key="game_order")
    else:
        st.success("All 10 days played — results below.")
    b1, b2, b3 = st.columns(3)
    b1.button("▶️ Open the shop", on_click=play, disabled=day >= GAME_DAYS, key="game_submit", type="primary")
    b2.button("🔁 Replay the same 10 days", on_click=new_game, args=(g["round"],), key="game_replay")
    b3.button("🎲 New 10-day challenge", on_click=new_game, args=(g["round"] + 1,), key="game_new")
    if g["rows"]:
        st.markdown("**Your days so far**")
        st.dataframe(pd.DataFrame(g["rows"])[["day", "start_stock", "order", "demand", "sold", "stockout",
                                               "spoiled", "carry", "profit"]].rename(columns={
            "day": "Day", "start_stock": "Start", "order": "Order", "demand": "Demand", "sold": "Sold",
            "stockout": "Unmet", "spoiled": "Spoiled", "carry": "Carry", "profit": "Profit (AED)"}),
            hide_index=True, column_config={"Profit (AED)": st.column_config.NumberColumn(format="%.2f")})
    if day >= GAME_DAYS and need_model():
        ai = run_month(lambda s, t: PI[s], g["demands"], P)
        h = [r["profit"] for r in g["rows"]]
        a = [r["profit"] for r in ai]
        st.markdown("**The AI on the same days**")
        st.dataframe(pd.DataFrame([{"Day": r["day"], "Start": r["start_stock"], "AI order": r["order"],
                                    "Your order": y["order"], "Demand": r["demand"], "AI profit": r["profit"],
                                    "Your profit": y["profit"]} for r, y in zip(ai, g["rows"])]),
                     hide_index=True, column_config={c: st.column_config.NumberColumn(format="%.2f")
                                                     for c in ["AI profit", "Your profit"]})
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.plot(range(1, 11), np.cumsum(h), marker="o", color=AMBER, label="You")
        ax.plot(range(1, 11), np.cumsum(a), marker="s", ls="--", color=EMERALD, label="AI (same demand, same start)")
        ax.axhline(0, color=SLATE, lw=.8)
        ax.set(xlabel="Day", ylabel="Cumulative profit (AED)", title="Cumulative profit, you vs AI")
        ax.legend()
        show(fig)
        diff = sum(h) - sum(a)
        cards([("Your 10-day profit", aed(sum(h)), "", AMBER), ("AI 10-day profit", aed(sum(a)), "", EMERALD),
               ("You − AI", aed(diff), "positive = you won", SLATE)])
        st.caption("Ten days is a small sample: one lucky or unlucky busy day can decide it. The scoreboard "
                   "(tab E) is the fair long-run comparison.")

# ================================================================ G · Experiments
with tabs[6]:
    PRESETS = {"Spoilage cost up (1.5 → 3.0 AED)": dict(spoilage_cost=3.0),
               "Stockout penalty up (6 → 12 AED)": dict(stockout_penalty=12.0),
               "Demand up (mean 7 → 9)": dict(mean_demand=9),
               "Spoil fraction up (25% → 50%)": dict(spoil_frac=0.5)}
    THEORY = {"Spoilage cost up (1.5 → 3.0 AED)": "Over-stocking gets dearer → expect leaner stock, a lower spoilage KPI, slightly more stockouts.",
              "Stockout penalty up (6 → 12 AED)": "Shortages get dearer → expect a bigger safety buffer, higher service, more waste.",
              "Demand up (mean 7 → 9)": "More customers → expect bigger orders and higher profit; the 15-unit fridge may start to bind.",
              "Spoil fraction up (25% → 50%)": "Carry-over is worth less → expect fewer units held overnight and higher spoilage cost."}
    st.markdown("Each experiment changes **one** business setting from the assignment defaults, **retrains** a fresh "
                "agent with the same seed and hyperparameters, and evaluates both on the same held-out months. "
                "This tab is independent of the sidebar business sliders.")
    name = st.selectbox("Scenario", list(PRESETS), key="exp_preset")
    base_items, scen_items = items_of(DEFAULTS), items_of({**DEFAULTS, **PRESETS[name]})
    st.markdown(f"**Baseline:** `{dict(base_items)}`  \n**Scenario:** `{dict(scen_items)}`  \n"
                f"**Training:** {episodes:,} episodes, α={alpha}, γ={gamma}, seed={seed} (from sidebar)")
    st.caption("Theory expectation (to check against the result, not to assume): " + THEORY[name])
    run_key = (name, episodes, alpha, gamma, seed)
    if st.button("🔁 Retrain baseline + scenario", key="btn_exp", type="primary"):
        st.session_state.setdefault("exp_done", set()).add(run_key)
    if run_key in st.session_state.get("exp_done", set()):
        Qb, _, db = cached_train(base_items, episodes, alpha, gamma, seed)
        Qs, _, ds = cached_train(scen_items, episodes, alpha, gamma, seed)
        pb = tuple(int(x) for x in policy_from_Q(Qb))
        ps = tuple(int(x) for x in policy_from_Q(Qs))
        reach_b, reach_s = reachable_states(dict(CafeSim(**dict(base_items)).params())), \
            reachable_states(dict(CafeSim(**dict(scen_items)).params()))
        fig, axes = plt.subplots(2, 1, figsize=(10, 2.9))
        for ax, pi, lab, rch in [(axes[0], pb, "Before (baseline)", reach_b), (axes[1], ps, "After (retrained)", reach_s)]:
            ax.imshow(np.array(pi)[None, :], cmap="cividis", aspect="auto", vmin=0, vmax=15)
            for s, a in enumerate(pi):
                ax.text(s, 0, f"{a}" + ("" if s in rch else "*"), ha="center", va="center",
                        color="white" if a < 9 else NAVY, fontsize=8, fontweight="bold")
            ax.set(yticks=[0], yticklabels=[lab], xticks=range(16))
            ax.grid(False)
        axes[1].set_xlabel("Stock on hand (state)   * = unreachable under that setting (not learned)")
        show(fig)
        out = []
        for lab, items, pi in [("① Baseline policy · baseline economics", base_items, pb),
                               ("② Baseline policy · scenario economics (not retrained)", scen_items, pb),
                               ("③ Retrained policy · scenario economics", scen_items, ps)]:
            e = eval_pi(items, pi)
            out.append({"Run": lab, "Profit / month": e["mean_profit"], "Service": e["service"],
                        "Spoilage KPI": e["spoilage_kpi"], "Stockout rate": e["stockout_rate"],
                        "Units ordered": e["mean_order"], "Order cost": e["mean_order_cost"],
                        "Holding": e["mean_holding_cost"], "Spoilage cost": e["mean_spoilage_cost"],
                        "Stockout penalty": e["mean_stockout_penalty"]})
        st.dataframe(pd.DataFrame(out), hide_index=True, column_config={
            **{c: st.column_config.NumberColumn(format="%.2f") for c in
               ["Profit / month", "Order cost", "Holding", "Spoilage cost", "Stockout penalty"]},
            **{c: st.column_config.NumberColumn(format="%.3f") for c in ["Service", "Spoilage KPI", "Stockout rate"]},
            "Units ordered": st.column_config.NumberColumn(format="%.1f")})
        d_order = out[2]["Units ordered"] - out[0]["Units ordered"]
        d_serv = out[2]["Service"] - out[0]["Service"]
        d_spoil = out[2]["Spoilage KPI"] - out[0]["Spoilage KPI"]
        gain = out[2]["Profit / month"] - out[1]["Profit / month"]
        st.info(f"**Observed shift (computed).** After retraining, the agent orders "
                f"**{abs(d_order):.1f} units/month {'more' if d_order >= 0 else 'fewer'}** than before; service "
                f"moved {d_serv * 100:+.1f} pts and the spoilage KPI {d_spoil * 100:+.1f} pts. Retraining is worth "
                f"**{aed(gain)}/month** compared with keeping the old policy under the new economics (③ vs ②). "
                "Compare this with the theory line above and report honestly where they differ.")
    else:
        st.info("Press **Retrain baseline + scenario** (≈20–30 s the first time; results are cached).")

# ================================================================ H · Assignment
with tabs[7]:
    st.markdown("#### MDP framing (Part 1)")
    st.dataframe(pd.DataFrame({
        "RL term": ["State", "Action", "Reward", "Episode", "Policy"],
        "In the café": ["The stock on hand each morning (0–15 units)",
                        "How many units to order today (0–15)",
                        "Today's profit = sales − order cost − holding − spoilage − stockout penalty",
                        "One full month of 30 days, starting with 5 units",
                        "The rule mapping each stock level to an order, read from the Q-table (argmax)"]}),
        hide_index=True)
    st.markdown("#### Reflection prompts and where the evidence is")
    st.dataframe(pd.DataFrame({
        "Prompt": ["P1", "P2", "P3", "P4", "P5", "P6", "P7"],
        "Question (from the guide)": [
            "Fill in state, action, reward, episode, policy.",
            "Order 15 every day vs 1 every day: what happens to spoilage, stockouts and profit?",
            "Which naive manager is best? Does any lose money? Why is too little worse than too much?",
            "What do α, γ and ε control? Why does ε start high and shrink?",
            "Describe the strategy the AI discovered. Does 'order more when stock is low' appear?",
            "Did the AI beat you? Where did your guesses go wrong? Did 'cautious' surprise you?",
            "Pick one experiment: how did the strategy change, and why?"],
        "Evidence in this app": ["Table above", "Tab E constant-order sweep (orders 1 and 15); tab D ledger",
                                 "Tab E scoreboard", "Tab B hyperparameter table + ε chart; tab I",
                                 "Tab C policy heatmap + rule summary", "Tab F (your own game — record your result)",
                                 "Tab G before/after"]}), hide_index=True)
    st.markdown("#### Exports")
    if MODEL and not STALE:
        res, const = scoreboard(ENV, PI)
        buf = io.StringIO()
        wr = csv.writer(buf)
        wr.writerow(["policy", "mean_profit", "std", "service", "stockout_rate", "spoilage_kpi"])
        for k, v in res.items():
            wr.writerow([k, round(v["mean_profit"], 2), round(v["std_profit"], 2), round(v["service"], 4),
                         round(v["stockout_rate"], 4), round(v["spoilage_kpi"], 4)])
        for k, v in const.items():
            wr.writerow([f"constant order {k}", round(v, 2), "", "", "", ""])
        e1, e2, e3 = st.columns(3)
        e1.download_button("⬇️ Scoreboard CSV", buf.getvalue().encode(), "scoreboard.csv", "text/csv", key="dl_score")
        e2.download_button("⬇️ Policy CSV", ("stock,order,training_visits\n" + "\n".join(
            f"{s},{a},{int(MODEL['diag']['state_visits'][s])}" for s, a in enumerate(PI))).encode(),
            "policy.csv", "text/csv", key="dl_policy")
        e3.download_button("⬇️ Learning curve CSV", ("episode,profit,epsilon\n" + "\n".join(
            f"{i},{v:.2f},{MODEL['diag']['eps'][i]:.6f}" for i, v in enumerate(MODEL["curve"]))).encode(),
            "learning_curve.csv", "text/csv", key="dl_curve")
        st.caption("The 30-day ledger CSV is in tab D.")
    else:
        st.info("Train the agent (on the current settings) to enable exports.")
    st.markdown("#### Screenshot checklist")
    for i, item in enumerate(["Learning curve with ε panel (tab B)", "Policy heatmap (tab C)",
                              "Scoreboard table + bar chart (tab E)", "Finished You-vs-AI result (tab F)"]):
        st.checkbox(item, key=f"shot_{i}")

# ================================================================ I · ML Diagnostics
with tabs[8]:
    st.markdown("**State** = stock on hand (0–15) · **Action** = order (0–15) · **Q-table** = 16 × 16 · "
                "**Reward** = daily profit · **Episode** = 30 days · update target = r + γ · max Q(s′, ·)")
    with st.expander("Feature audit: what the agent sees vs what the dashboard shows", expanded=False):
        st.dataframe(pd.DataFrame([
            ["Stock on hand (STATE)", "carry-over from yesterday", "the decision input", "simulator", "Yes — only input"],
            ["Utilisation", "stock ÷ 15", "how full the fridge is", "stock", "No (diagnostic)"],
            ["Base-demand coverage", "stock ÷ mean_demand (n/a if 0)", "days of typical demand on hand", "stock, setting", "No (diagnostic)"],
            ["Model-implied expected demand", "mean_demand + busy_chance × 5", "context (8.0 at defaults)", "settings", "No (diagnostic)"],
            ["Expected-demand coverage", "stock ÷ expected demand", "coverage incl. busy days", "stock, settings", "No (diagnostic)"],
            ["Remaining capacity", "15 − stock", "room to order", "stock", "No (diagnostic)"],
            ["Post-order inventory", "min(stock + order, 15)", "order-up-to check", "stock, policy", "No (diagnostic)"],
            ["Today's realised demand", "drawn after ordering", "outcome only", "simulator", "No — would be leakage"]],
            columns=["Feature", "Formula", "Purpose", "Source", "Used by the agent?"]), hide_index=True)
    fe = floor_episode()
    cards([("Epsilon after 12,000 episodes", f"{epsilon_at(12000):.4f}", "ε = 0.9998^12000", BLUE),
           ("Epsilon after 15,000 episodes", f"{epsilon_at(15000):.4f}", "ε floored at 0.05", BLUE),
           ("First floored episode", f"{fe:,}", "0-based index", SLATE)])
    if need_model():
        d = MODEL["diag"]
        win = 500 if len(d["eps"]) >= 2000 else 100
        k = len(d["eps"]) // win
        if k:
            rb = d["random_branch"][: k * win].reshape(k, win).mean(1) / 30
            ng = d["nongreedy"][: k * win].reshape(k, win).mean(1) / 30
            ex = d["eps"][: k * win].reshape(k, win).mean(1)
            fig, ax = plt.subplots(figsize=(9, 2.9))
            x = (np.arange(k) + 0.5) * win
            ax.plot(x, rb, color=BLUE, marker="o", ms=3, label="Logged: random branch taken")
            ax.plot(x, ng, color=CORAL, ls="--", marker="^", ms=3, label="Logged: random AND ≠ greedy action")
            ax.plot(x, ex, color=SLATE, ls=":", label="Expected share (mean ε in window)")
            ax.set(xlabel=f"Episode (windows of {win})", ylabel="Share of decision days",
                   title="Exploration vs exploitation, measured during training")
            ax.legend()
            show(fig)
            phases = np.array_split(np.arange(len(d["eps"])), 3)
            st.dataframe(pd.DataFrame([{"Phase": p_name, "Episodes": f"{ix[0]:,}–{ix[-1]:,}",
                                        "Random-branch days": int(d["random_branch"][ix].sum()),
                                        "Greedy-branch days": int(d["greedy_branch"][ix].sum()),
                                        "Random share": d["random_branch"][ix].sum() / (30 * len(ix)),
                                        "Random but = greedy": int((d["random_branch"][ix] - d["nongreedy"][ix]).sum())}
                                       for p_name, ix in zip(["Early", "Middle", "Late"], phases)]),
                         hide_index=True, column_config={"Random share": st.column_config.NumberColumn(format="%.3f")})
        cur = d["eps"][-1]
        st.markdown(f"**What ε = {cur:.4f} means now:** on each day the agent takes the random branch with probability "
                    f"{cur:.4f}, so it is **expected** to explore on about **{30 * cur:.1f} of 30 days** per month. A "
                    f"random pick lands on the greedy order 1 time in 16, so about {30 * cur * 15 / 16:.1f} of those days "
                    "are expected to be genuinely non-greedy. These are expectations, not counts — the logged counts "
                    "are in the chart and table above.")
        sv = d["state_visits"]
        reach = reachable_states(P)
        fig, ax = plt.subplots(figsize=(9, 2.6))
        bars = ax.bar(range(16), np.maximum(sv, 0.8), color=[BLUE if v >= MIN_VISITS else AMBER for v in sv])
        for s, b in enumerate(bars):
            if s not in reach:
                b.set_hatch("xx")
                b.set_color("#e6e0d4")
        ax.set_yscale("log")
        ax.set(xticks=range(16), xlabel="Stock on hand (state)", ylabel="Training visits (log)",
               title=f"State visitation (amber < {MIN_VISITS} visits; grey hatched = unreachable)")
        show(fig)
        sa = d["sa_visits"]
        tried = int((sa[sorted(reach)] > 0).sum())
        st.caption(f"{tried} of {len(reach) * 16} reachable stock–order pairs were tried at least once. "
                   "No confidence score is shown: visit counts are the honest measure of evidence here.")
        with st.expander("Optional: ε-decay sensitivity (only the decay changes; 3 seeds each; held-out months)"):
            st.markdown(f"Compares decay 0.999, 0.9995 and the assignment default 0.9998 with everything else fixed "
                        f"({episodes:,} episodes, α={alpha}, γ={gamma}, current business settings). Each greedy policy is "
                        f"scored on {EVAL_MONTHS} held-out months (seed {SENS_SEED}). Runs 9 trainings — about 1–2 minutes.")
            if st.button("Run sensitivity study", key="btn_sens"):
                out = []
                for dec in (0.999, 0.9995, 0.9998):
                    vals = [eval_pi(ENV, tuple(int(x) for x in policy_from_Q(
                        cached_train(ENV, episodes, alpha, gamma, sd, dec)[0])), SENS_SEED)["mean_profit"]
                        for sd in (11, 12, 13)]
                    out.append({"ε decay": dec, "Default?": "✓" if dec == 0.9998 else "",
                                "Mean profit / month": float(np.mean(vals)), "Std (3 seeds)": float(np.std(vals, ddof=1)),
                                "Min": float(min(vals)), "Max": float(max(vals)),
                                "Floor reached at episode": floor_episode(eps_decay=dec)})
                st.session_state["sens"] = dict(key=CFG, rows=out)
            sens = st.session_state.get("sens")
            if sens and sens["key"] == CFG:
                st.dataframe(pd.DataFrame(sens["rows"]), hide_index=True, column_config={
                    c: st.column_config.NumberColumn(format="%.2f") for c in
                    ["Mean profit / month", "Std (3 seeds)", "Min", "Max"]})
                st.caption("Three seeds is a small sample — differences smaller than about one Std are not evidence "
                           "that one schedule is better. The assignment default remains the primary model.")
