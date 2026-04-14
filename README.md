# Portfolio Performance Analytics Dashboard

A comprehensive **portfolio analytics engine** computing performance attribution, risk-adjusted metrics, drawdown analysis, and benchmark-relative statistics — used daily by portfolio analysts at AMCs and custodian banks.



---

#What This Project Does

| Feature | Description |
|---------|-------------|
| Performance Metrics | Sharpe, Sortino, Calmar, Information Ratio |
| Risk Analysis | Max Drawdown, Tracking Error, Beta, Alpha |
| Attribution | Brinson-Hood-Beebower sector attribution |
| Rolling Analytics | 63-day rolling Sharpe, Beta, Alpha |
| Visualization | Monthly returns heatmap, drawdown chart, active return distribution |

---

#Project Structure

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

##Key Metrics Explained

**Sharpe Ratio** — Excess return per unit of total risk. >1 is good, >2 is excellent.

**Sortino Ratio** — Like Sharpe but only penalizes downside volatility. More relevant for asymmetric return profiles.

**Information Ratio** — Active return divided by tracking error. Measures skill of active management. >0.5 is strong.

**Calmar Ratio** — Annual return divided by maximum drawdown. Measures return per unit of worst-case loss.

**Brinson Attribution** — Decomposes excess return into:
- Allocation Effect: Were you overweight in sectors that outperformed?
- Selection Effect: Did your stock picks beat the sector benchmark?

---


##  Tech Stack
`Python` `Pandas` `NumPy` `SciPy` `Matplotlib`
