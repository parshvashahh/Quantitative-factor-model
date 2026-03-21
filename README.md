# 📈 Portfolio Performance Analytics Dashboard

A comprehensive **portfolio analytics engine** computing performance attribution, risk-adjusted metrics, drawdown analysis, and benchmark-relative statistics — used daily by portfolio analysts at AMCs and custodian banks.

## 🎯 Target Roles
Portfolio Analytics Analyst | Investment Analyst | Fund Performance Analyst

## 🏦 Relevant For
State Street Global Advisors | Kotak AMC | Mirae Asset | DSP Mutual Fund | Morningstar India

---

## 📌 What This Project Does

| Feature | Description |
|---------|-------------|
| Performance Metrics | Sharpe, Sortino, Calmar, Information Ratio |
| Risk Analysis | Max Drawdown, Tracking Error, Beta, Alpha |
| Attribution | Brinson-Hood-Beebower sector attribution |
| Rolling Analytics | 63-day rolling Sharpe, Beta, Alpha |
| Visualization | Monthly returns heatmap, drawdown chart, active return distribution |

---

## 📂 Project Structure

```
project2_portfolio_analytics/
├── portfolio_analytics.py     # Main analytics engine
├── portfolio_returns.csv      # Output: daily returns data
├── attribution_results.csv    # Output: BHB attribution table
├── performance_metrics.csv    # Output: all performance metrics
├── portfolio_dashboard.png    # Output: 6-panel dashboard
└── README.md
```

---

## 🚀 How To Run

```bash
python portfolio_analytics.py
```

---

## 📊 Key Metrics Explained

**Sharpe Ratio** — Excess return per unit of total risk. >1 is good, >2 is excellent.

**Sortino Ratio** — Like Sharpe but only penalizes downside volatility. More relevant for asymmetric return profiles.

**Information Ratio** — Active return divided by tracking error. Measures skill of active management. >0.5 is strong.

**Calmar Ratio** — Annual return divided by maximum drawdown. Measures return per unit of worst-case loss.

**Brinson Attribution** — Decomposes excess return into:
- Allocation Effect: Were you overweight in sectors that outperformed?
- Selection Effect: Did your stock picks beat the sector benchmark?

---

## 🧠 Interview Talking Points

1. **Why Sortino over Sharpe?** Investors care more about downside risk than upside volatility. Sortino captures this asymmetry.
2. **What is tracking error?** Annualised standard deviation of daily active returns. Low TE = index-hugging. High TE = active bets.
3. **Explain Brinson attribution.** Alpha comes from two sources: sector allocation decisions and stock selection within sectors.
4. **Why monthly heatmap?** Identifies seasonality patterns and consistency of returns across market cycles.

---

## 🛠️ Tech Stack
`Python` `Pandas` `NumPy` `SciPy` `Matplotlib`
