"""
Portfolio Performance Analytics Dashboard
Author      : Parshva Shah
LinkedIn    : https://www.linkedin.com/in/parshva-shah-b40683193/
GitHub      : github.com/parshvashahh

Description:
    A comprehensive portfolio analytics engine that computes
    performance attribution, risk-adjusted metrics, drawdown analysis,
    and benchmark-relative statistics — used daily by portfolio analysts
    at asset managers and custodian banks.

Key Concepts Demonstrated:
    - Sharpe, Sortino, Calmar, Information Ratio
    - Maximum Drawdown and recovery analysis
    - Rolling alpha/beta vs benchmark
    - Brinson-Hood-Beebower performance attribution
    - Sector allocation vs benchmark
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# GENERATE REALISTIC DATA
def generate_portfolio_data():
    """
    Simulate 3 years of daily portfolio and benchmark returns.
    Portfolio: Active equity fund (NIFTY 50 + active bets)
    Benchmark: NIFTY 50 Index
    """
    n_days = 756   # 3 years

    # Benchmark (NIFTY 50) — annual return 12%, vol 18%
    bm_annual_ret = 0.12
    bm_annual_vol = 0.18
    bm_daily_ret  = bm_annual_ret / 252
    bm_daily_vol  = bm_annual_vol / np.sqrt(252)

    benchmark_returns = np.random.normal(bm_daily_ret, bm_daily_vol, n_days)

    # Portfolio — slightly higher return with higher vol (active fund)
    # Correlated with benchmark (beta ~1.05) + alpha ~2% annually
    alpha_daily = 0.02 / 252
    beta        = 1.05
    port_noise  = np.random.normal(0, 0.004, n_days)
    port_returns = alpha_daily + beta * benchmark_returns + port_noise

    dates = pd.bdate_range(start='2021-01-01', periods=n_days)

    df = pd.DataFrame({
        'Portfolio': port_returns,
        'Benchmark': benchmark_returns,
    }, index=dates)

    # Add sector returns for attribution
    sectors = ['IT', 'Financials', 'Energy', 'FMCG', 'Healthcare']
    for sector in sectors:
        sector_vol = np.random.uniform(0.015, 0.025)
        df[f'Sector_{sector}'] = np.random.normal(bm_daily_ret, sector_vol, n_days)

    return df

# RISK-ADJUSTED PERFORMANCE METRICS
def compute_performance_metrics(port_returns, bench_returns, risk_free_rate=0.065):
    """
    Compute all standard performance metrics used by portfolio analysts.
    risk_free_rate: RBI repo rate proxy (annualised)
    """
    rf_daily = risk_free_rate / 252

    # Return metrics
    total_return   = (1 + port_returns).prod() - 1
    annual_return  = (1 + total_return) ** (252 / len(port_returns)) - 1
    annual_vol     = port_returns.std() * np.sqrt(252)

    # Sharpe Ratio
    excess_returns = port_returns - rf_daily
    sharpe         = (excess_returns.mean() / port_returns.std()) * np.sqrt(252)

    # Sortino Ratio — downside deviation only
    downside_returns = port_returns[port_returns < rf_daily]
    downside_vol     = downside_returns.std() * np.sqrt(252)
    sortino          = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    # Maximum Drawdown
    cumulative   = (1 + port_returns).cumprod()
    rolling_max  = cumulative.cummax()
    drawdown     = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Calmar Ratio — return / max drawdown
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

    # Beta and Alpha vs benchmark
    covariance   = np.cov(port_returns, bench_returns)[0, 1]
    bench_var    = bench_returns.var()
    beta         = covariance / bench_var if bench_var > 0 else 1
    bench_annual = (1 + bench_returns).prod() ** (252 / len(bench_returns)) - 1
    alpha        = annual_return - (risk_free_rate + beta * (bench_annual - risk_free_rate))

    # Information Ratio
    active_returns = port_returns - bench_returns
    tracking_error = active_returns.std() * np.sqrt(252)
    info_ratio     = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0

    # Win rate
    win_rate = (port_returns > 0).mean()

    # Benchmark metrics
    bench_total  = (1 + bench_returns).prod() - 1
    bench_annual_r = (1 + bench_total) ** (252 / len(bench_returns)) - 1
    bench_vol    = bench_returns.std() * np.sqrt(252)

    return {
        'Total Return (Portfolio)': f"{total_return*100:.2f}%",
        'Total Return (Benchmark)': f"{bench_total*100:.2f}%",
        'Annual Return (Portfolio)': f"{annual_return*100:.2f}%",
        'Annual Return (Benchmark)': f"{bench_annual_r*100:.2f}%",
        'Annualised Volatility': f"{annual_vol*100:.2f}%",
        'Benchmark Volatility': f"{bench_vol*100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.3f}",
        'Sortino Ratio': f"{sortino:.3f}",
        'Calmar Ratio': f"{calmar:.3f}",
        'Maximum Drawdown': f"{max_drawdown*100:.2f}%",
        'Beta': f"{beta:.3f}",
        'Alpha (Annual)': f"{alpha*100:.2f}%",
        'Information Ratio': f"{info_ratio:.3f}",
        'Tracking Error': f"{tracking_error*100:.2f}%",
        'Win Rate (Daily)': f"{win_rate*100:.1f}%",
    }, drawdown, cumulative

# DRAWDOWN ANALYSIS
def drawdown_analysis(drawdown, port_returns):
    """Find top 5 drawdown periods with recovery times."""
    in_drawdown = False
    drawdown_start = None
    drawdowns = []

    for date, val in drawdown.items():
        if val < -0.03 and not in_drawdown:  # >3% drawdown
            in_drawdown = True
            drawdown_start = date
            peak_dd = val
        elif in_drawdown:
            peak_dd = min(peak_dd, val)
            if val >= -0.005:  # recovery within 0.5%
                in_drawdown = False
                drawdowns.append({
                    'Start': drawdown_start,
                    'End': date,
                    'Max Drawdown': f"{peak_dd*100:.2f}%",
                    'Duration (Days)': (date - drawdown_start).days
                })

    return pd.DataFrame(drawdowns).sort_values('Duration (Days)',
                                               ascending=False).head(5)

# ROLLING METRICS
def compute_rolling_metrics(port_returns, bench_returns, window=63):
    """Compute 63-day (quarterly) rolling Sharpe, beta, alpha."""
    rf_daily = 0.065 / 252
    rolling = pd.DataFrame(index=port_returns.index)

    rolling['Sharpe'] = (
        port_returns.rolling(window).mean() - rf_daily
    ) / port_returns.rolling(window).std() * np.sqrt(252)

    # Rolling beta
    def rolling_beta(p, b):
        cov = np.cov(p, b)[0, 1]
        return cov / np.var(b) if np.var(b) > 0 else 1

    rolling['Beta'] = [
        rolling_beta(port_returns.iloc[max(0, i-window):i].values,
                     bench_returns.iloc[max(0, i-window):i].values)
        if i >= window else np.nan
        for i in range(len(port_returns))
    ]

    rolling['Active_Return'] = (
        port_returns.rolling(window).mean() -
        bench_returns.rolling(window).mean()
    ) * 252

    return rolling

# BRINSON PERFORMANCE ATTRIBUTION
def performance_attribution(df):
    """
    Simplified Brinson-Hood-Beebower attribution.
    Decomposes excess return into:
    - Allocation effect (sector weights)
    - Selection effect (stock picking within sector)
    """
    sectors = ['IT', 'Financials', 'Energy', 'FMCG', 'Healthcare']

    # Simulated weights: portfolio vs benchmark
    np.random.seed(10)
    port_weights = np.array([0.28, 0.30, 0.15, 0.15, 0.12])
    bench_weights = np.array([0.22, 0.32, 0.18, 0.16, 0.12])

    results = []
    for i, sector in enumerate(sectors):
        sector_ret   = df[f'Sector_{sector}'].mean() * 252
        bench_ret    = df['Benchmark'].mean() * 252
        port_w       = port_weights[i]
        bench_w      = bench_weights[i]

        # Allocation = (port_w - bench_w) * (sector_ret - bench_ret)
        allocation  = (port_w - bench_w) * (sector_ret - bench_ret)
        # Selection = bench_w * (port_sector_ret - sector_ret)
        selection   = bench_w * (sector_ret * np.random.uniform(0.95, 1.05) - sector_ret)
        total       = allocation + selection

        results.append({
            'Sector': sector,
            'Port Weight': f"{port_w*100:.1f}%",
            'Bench Weight': f"{bench_w*100:.1f}%",
            'Active Weight': f"{(port_w-bench_w)*100:+.1f}%",
            'Allocation Effect': f"{allocation*100:+.3f}%",
            'Selection Effect': f"{selection*100:+.3f}%",
            'Total Effect': f"{total*100:+.3f}%"
        })

    return pd.DataFrame(results)

# VISUALIZATION
def plot_portfolio_dashboard(df, metrics, drawdown, cumulative, rolling):
    """6-panel portfolio analytics dashboard."""
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Portfolio Performance Analytics Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    bench_cumulative = (1 + df['Benchmark']).cumprod()

    # Panel 1: Cumulative returns
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(cumulative.index, cumulative.values * 100 - 100,
             color='steelblue', linewidth=1.5, label='Portfolio')
    ax1.plot(bench_cumulative.index, bench_cumulative.values * 100 - 100,
             color='orange', linewidth=1.5, label='Benchmark (NIFTY 50)')
    ax1.set_title('Cumulative Return (%)', fontweight='bold')
    ax1.set_ylabel('Return (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                     color='red', alpha=0.5)
    ax2.plot(drawdown.index, drawdown.values * 100, color='darkred', linewidth=1)
    ax2.set_title('Portfolio Drawdown (%)', fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Rolling Sharpe
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rolling.index, rolling['Sharpe'], color='steelblue', linewidth=1.2)
    ax3.axhline(1, color='green', linestyle='--', linewidth=1, label='Sharpe = 1')
    ax3.axhline(0, color='red', linestyle='--', linewidth=1, label='Sharpe = 0')
    ax3.fill_between(rolling.index, rolling['Sharpe'], 0,
                     where=rolling['Sharpe'] > 0, alpha=0.2, color='green')
    ax3.fill_between(rolling.index, rolling['Sharpe'], 0,
                     where=rolling['Sharpe'] < 0, alpha=0.2, color='red')
    ax3.set_title('Rolling 63-Day Sharpe Ratio', fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Rolling Beta
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(rolling.index, rolling['Beta'], color='purple', linewidth=1.2)
    ax4.axhline(1.0, color='black', linestyle='--', linewidth=1, label='Beta = 1')
    ax4.fill_between(rolling.index, rolling['Beta'], 1.0,
                     where=rolling['Beta'] > 1, alpha=0.2, color='red')
    ax4.fill_between(rolling.index, rolling['Beta'], 1.0,
                     where=rolling['Beta'] < 1, alpha=0.2, color='green')
    ax4.set_title('Rolling 63-Day Beta vs Benchmark', fontweight='bold')
    ax4.set_ylabel('Beta')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Panel 5: Monthly returns heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    monthly = df['Portfolio'].resample('ME').apply(lambda x: (1 + x).prod() - 1) * 100
    years   = monthly.index.year.unique()
    months  = list(range(1, 13))
    month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec']
    heatmap_data = np.full((len(years), 12), np.nan)
    for i, year in enumerate(years):
        for j, month in enumerate(months):
            mask = (monthly.index.year == year) & (monthly.index.month == month)
            if mask.any():
                heatmap_data[i, j] = monthly[mask].values[0]
    im = ax5.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    ax5.set_xticks(range(12))
    ax5.set_xticklabels(month_names, fontsize=8)
    ax5.set_yticks(range(len(years)))
    ax5.set_yticklabels(years)
    ax5.set_title('Monthly Returns Heatmap (%)', fontweight='bold')
    plt.colorbar(im, ax=ax5)
    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(heatmap_data[i, j]):
                ax5.text(j, i, f'{heatmap_data[i, j]:.1f}',
                        ha='center', va='center', fontsize=6)

    # Panel 6: Active return distribution
    ax6 = fig.add_subplot(gs[2, 1])
    active_ret = (df['Portfolio'] - df['Benchmark']) * 100
    ax6.hist(active_ret, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax6.axvline(active_ret.mean(), color='red', linewidth=2,
                label=f'Mean: {active_ret.mean():.3f}%')
    ax6.axvline(0, color='black', linewidth=1, linestyle='--')
    ax6.set_title('Daily Active Return Distribution', fontweight='bold')
    ax6.set_xlabel('Active Return (%)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    plt.savefig('portfolio_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Dashboard saved as portfolio_dashboard.png")


# MAIN
if __name__ == '__main__':
    print("Loading portfolio data...")
    df = generate_portfolio_data()

    print("Computing performance metrics...")
    metrics, drawdown, cumulative = compute_performance_metrics(
        df['Portfolio'], df['Benchmark'])

    print("\n" + "="*60)
    print("     PORTFOLIO PERFORMANCE ANALYTICS REPORT")
    print("="*60)
    for k, v in metrics.items():
        print(f"  {k:<35}: {v}")

    print("\n── Top Drawdown Periods ──────────────────────────────")
    dd_analysis = drawdown_analysis(drawdown, df['Portfolio'])
    print(dd_analysis.to_string(index=False))

    print("\n── Brinson Performance Attribution ──────────────────")
    attribution = performance_attribution(df)
    print(attribution.to_string(index=False))

    print("\nComputing rolling metrics...")
    rolling = compute_rolling_metrics(df['Portfolio'], df['Benchmark'])

    print("Generating dashboard...")
    plot_portfolio_dashboard(df, metrics, drawdown, cumulative, rolling)

    # Export
    df.to_csv('portfolio_returns.csv')
    attribution.to_csv('attribution_results.csv', index=False)
    pd.DataFrame([metrics]).to_csv('performance_metrics.csv', index=False)
    print("\nAll outputs exported.")
    print("Project 2 Complete!")
