import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

df = yf.download("BTC-USD", start="2020-01-01")

# yfinance 在新版本里可能返回多层列；这里统一提取一维 Close 序列，避免后续列对齐报错
close = df["Close"]
if isinstance(close, pd.DataFrame):
    close = close.iloc[:, 0]

# ret: 日收益率，公式为 Close_t / Close_{t-1} - 1；首行因无前值通常为 NaN
df["ret"] = close.pct_change()
# signal: 20日简单移动均线(SMA)，表示最近20天平均价格，前19行会因窗口不足为 NaN
df["signal"] = close.rolling(20).mean()

# position: 当日收盘价高于均线时记为1(持仓)，否则为0(空仓)；True/False 通过 astype(int) 转为 1/0
df["position"] = (close > df["signal"]).astype(int)


# strategy: 使用前一日仓位(position.shift(1))乘以当日收益，避免用“今天信号赚今天收益”的前视偏差
df["strategy"] = df["position"].shift(1) * df["ret"]
# equity: 把每日策略收益转为资金净值曲线，(1 + strategy) 逐日连乘体现复利增长
df["equity"] = (1 + df["strategy"].fillna(0)).cumprod()

# buy_hold: 买入并持有净值，用于和策略做对比
df["buy_hold"] = (1 + df["ret"].fillna(0)).cumprod()

# 画两张图：上图看价格和均线；下图看策略净值 vs 买入持有
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax1.plot(df.index, close, label="BTC Close", linewidth=1.2)
ax1.plot(df.index, df["signal"], label="SMA20", linewidth=1.2)
ax1.set_title("BTC Price and 20-day Moving Average")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(df.index, df["equity"], label="Strategy Equity", linewidth=1.4)
ax2.plot(df.index, df["buy_hold"], label="Buy & Hold Equity", linewidth=1.2, linestyle="--")
ax2.set_title("Equity Curve Comparison")
ax2.set_ylabel("Net Value")
ax2.set_xlabel("Date")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
