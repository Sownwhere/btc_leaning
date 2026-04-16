import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download("BTC-USD", start="2020-01-01")

# yfinance 在新版本里可能返回多层列；这里统一提取一维 OHLC 序列，避免后续列对齐报错
def as_series(column_name: str) -> pd.Series:
    col = df[column_name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col

close = as_series("Close")
high = as_series("High")
low = as_series("Low")

# ret: 日收益率，公式为 Close_t / Close_{t-1} - 1；首行因无前值通常为 NaN
df["ret"] = close.pct_change()

# === 参数区：可以按需调整 ===
SMA_WINDOW = 200
ATR_WINDOW = 14
ATR_FILTER_WINDOW = 100
FEE_RATE = 0.001  # 单边手续费0.1%

# signal: 长周期均线（默认SMA200），减少短期震荡噪音
df["signal"] = close.rolling(SMA_WINDOW).mean()
df["trend_on"] = close > df["signal"]

# ATR 波动率过滤器：ATR占价格比例高于其历史中位数时，认为趋势更“有力”
prev_close = close.shift(1)
tr = pd.concat(
    [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
    axis=1,
).max(axis=1)
df["atr"] = tr.rolling(ATR_WINDOW).mean()
df["atr_pct"] = df["atr"] / close
df["atr_thresh"] = df["atr_pct"].rolling(ATR_FILTER_WINDOW).median()
df["vol_on"] = df["atr_pct"] > df["atr_thresh"]

# position: 仅当“价格在长均线之上”且“波动率过滤通过”时持仓
df["position"] = (df["trend_on"] & df["vol_on"]).astype(int)


# strategy: 使用前一日仓位(position.shift(1))乘以当日收益，避免用“今天信号赚今天收益”的前视偏差
df["gross_strategy"] = df["position"].shift(1) * df["ret"]
# 交易成本：仓位变化即发生交易；long/flat 策略中，进场和离场都要扣一次单边手续费
df["turnover"] = df["position"].diff().abs().fillna(df["position"])
df["cost"] = df["turnover"] * FEE_RATE
df["strategy"] = df["gross_strategy"] - df["cost"]
# equity: 把每日策略收益转为资金净值曲线，(1 + strategy) 逐日连乘体现复利增长
df["equity"] = (1 + df["strategy"].fillna(0)).cumprod()

# buy_hold: 买入并持有净值，用于和策略做对比
df["buy_hold"] = (1 + df["ret"].fillna(0)).cumprod()

# 绩效指标：总收益、年化收益、最大回撤
def perf_stats(equity: pd.Series) -> dict:
    equity = equity.dropna()
    total_return = equity.iloc[-1] - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    cagr = equity.iloc[-1] ** (1 / years) - 1
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_dd,
    }

strategy_stats = perf_stats(df["equity"])
buy_hold_stats = perf_stats(df["buy_hold"])

print("=== Strategy (SMA200 + ATR filter + fees) ===")
print(
    f"Total Return: {strategy_stats['total_return']:.2%}, "
    f"CAGR: {strategy_stats['cagr']:.2%}, "
    f"Max Drawdown: {strategy_stats['max_drawdown']:.2%}"
)
print("=== Buy & Hold ===")
print(
    f"Total Return: {buy_hold_stats['total_return']:.2%}, "
    f"CAGR: {buy_hold_stats['cagr']:.2%}, "
    f"Max Drawdown: {buy_hold_stats['max_drawdown']:.2%}"
)

# 画三张图：价格+均线、ATR过滤状态、净值对比
fig, (ax1, ax_mid, ax2) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1.plot(df.index, close, label="BTC Close", linewidth=1.2)
ax1.plot(df.index, df["signal"], label=f"SMA{SMA_WINDOW}", linewidth=1.2)
ax1.fill_between(df.index, close.min(), close.max(), where=df["position"] > 0, alpha=0.08, label="In Position")
ax1.set_title("BTC Price and Long-term Trend Signal")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(alpha=0.3)

ax_mid.plot(df.index, df["atr_pct"], label="ATR% (14)", linewidth=1.1)
ax_mid.plot(df.index, df["atr_thresh"], label=f"ATR% Threshold ({ATR_FILTER_WINDOW}d median)", linewidth=1.1, linestyle="--")
ax_mid.set_title("Volatility Filter (ATR-based)")
ax_mid.set_ylabel("ATR / Price")
ax_mid.legend()
ax_mid.grid(alpha=0.3)

ax2.plot(df.index, df["equity"], label="Strategy Equity", linewidth=1.4)
ax2.plot(df.index, df["buy_hold"], label="Buy & Hold Equity", linewidth=1.2, linestyle="--")
ax2.set_title("Equity Curve Comparison")
ax2.set_ylabel("Net Value")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
