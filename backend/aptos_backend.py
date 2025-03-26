import os
import uuid
import time
from flask import Flask, request, jsonify
from flask_cors import CORS  # Added CORS support
import ccxt
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import pandas_ta as ta
import numpy as np
import logging
from datetime import datetime
import pytz
import matplotlib.pyplot as plt
import io, base64
import scipy.signal as signal
import pywt
import asyncio

# For interactive plots using Plotly
import plotly.graph_objects as go

# Set timezone for India region
india_tz = pytz.timezone('Asia/Kolkata')

# Configure logging to file
logging.basicConfig(filename='trades.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# ----- OOP Classes -----
class User:
    def __init__(self, username, rsi_period=14, buy_threshold=30, sell_threshold=70, 
                 trade_type='long', strategy='RSI', ma_period=20):
        self.username = username
        self.rsi_period = rsi_period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.trade_type = trade_type.lower()      # "long" or "short"
        self.strategy = strategy.upper()            # "RSI", "MA", "RSI_MA", etc.
        self.ma_period = ma_period

class TradeLogger:
    @staticmethod
    def log_trade(trade_details):
        logging.info(trade_details)

class TradingStrategy:
    def __init__(self, user: User, exchange_id: str = 'binance'):
        self.user = user
        # Dynamically initialize the exchange using CCXT
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({
            'options': {
                'defaultType': 'spot'
            }
        })

    def safe_profit_pct(self, entry_price, exit_price, trade_type):
        if entry_price == 0:
            return 0
        if trade_type == 'long':
            return ((exit_price - entry_price) / entry_price) * 100
        else:
            return ((entry_price - exit_price) / entry_price) * 100

    def calculate_novel_stochastic(self, data, period=14):
        low_min = data['low'].rolling(window=period, min_periods=1).min()
        high_max = data['high'].rolling(window=period, min_periods=1).max()
        range_val = high_max - low_min
        range_val = range_val.replace(0, np.nan)
        stoch = ((data['close'] - low_min) / range_val * 100).fillna(50)
        return stoch

    def calculate_trade_summary(self, trades):
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.get('profit_pct', 0) > 0)
        total_profit_pct = sum(t.get('profit_pct', 0) for t in trades)
        avg_profit_per_trade = total_profit_pct / total_trades if total_trades > 0 else 0
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "total_profit_pct": total_profit_pct,
            "avg_profit_per_trade": avg_profit_per_trade
        }

    def fetch_data(self, symbol, timeframe='1m', limit=1000, use_csv=False):
        if use_csv:
            try:
                data = pd.read_csv("minute_data.csv")
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')\
                                    .dt.tz_localize('UTC').dt.tz_convert(india_tz)
                data.set_index('datetime', inplace=True)
                return data
            except Exception as e:
                print(f"Error loading CSV data: {e}")
                return None
        else:
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')\
                                    .dt.tz_localize('UTC').dt.tz_convert(india_tz)
                data.set_index('datetime', inplace=True)
                return data
            except Exception as e:
                print(f"Error fetching data for {symbol} from {self.exchange.id}: {e}")
                return None

    def calculate_rsi(self, data):
        data['RSI'] = ta.rsi(data['close'], length=self.user.rsi_period, mamode='wilder')
        data = data.iloc[self.user.rsi_period:]
        return data

    def calculate_rsi_scratch(self, data):
        delta = data['close'].diff()[1:]
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        period = self.user.rsi_period
        initial_avg_gain = up.iloc[:period].mean()
        initial_avg_loss = down.iloc[:period].mean()
        gains = [initial_avg_gain]
        losses = [initial_avg_loss]
        for i in range(period, len(up)):
            current_gain = up.iloc[i]
            current_loss = down.iloc[i]
            avg_gain = ((period - 1) * gains[-1] + current_gain) / period
            avg_loss = ((period - 1) * losses[-1] + current_loss) / period
            gains.append(avg_gain)
            losses.append(avg_loss)
        smoothed_gains = pd.Series(gains, index=data.index[period+1:])
        smoothed_losses = pd.Series(losses, index=data.index[period+1:])
        rs = smoothed_gains / smoothed_losses.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        data['RSI_Scratch'] = np.nan
        data.loc[rsi.index, 'RSI_Scratch'] = rsi
        return data.iloc[period+1:]

    def generate_plots(self, data, trades):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(data.index, data['close'], label='Close Price', color='blue')
        ax1.set_ylabel("Price", color='blue')
        title = f"{self.user.strategy} Strategy - {self.user.trade_type.capitalize()} Trades"
        if self.user.strategy in ["RSI", "RSI_MA", "MA"]:
            title += f" (RSI Period: {self.user.rsi_period}"
            if self.user.strategy in ["MA", "RSI_MA"]:
                title += f", MA Period: {self.user.ma_period}"
            title += ")"
        ax1.set_title(title)
        for trade in trades:
            try:
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                entry_price = trade['entry_price']
                exit_price = trade['exit_price']
                if self.user.trade_type == 'long':
                    entry_marker, exit_marker = '^', 'v'
                    entry_color, exit_color = 'green', 'red'
                else:
                    entry_marker, exit_marker = 'v', '^'
                    entry_color, exit_color = 'red', 'green'
                ax1.plot(entry_time, entry_price, marker=entry_marker, markersize=10, color=entry_color)
                ax1.plot(exit_time, exit_price, marker=exit_marker, markersize=10, color=exit_color)
                ax1.plot([entry_time, exit_time], [entry_price, exit_price], 'k--', alpha=0.3)
                pnl = self.safe_profit_pct(entry_price, exit_price, self.user.trade_type)
                midpoint_time = entry_time + (exit_time - entry_time) / 2
                midpoint_price = (entry_price + exit_price) / 2
                ax1.annotate(f"{pnl:.2f}%", (midpoint_time, midpoint_price), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))
            except Exception as e:
                print(f"Error plotting trade: {e}")
                continue

        indicator_plotted = False
        if 'RSI' in data.columns and data['RSI'].notna().any():
            if 'RSI_Scratch' in data.columns and data['RSI_Scratch'].notna().any():
                ax2.plot(data.index, data['RSI'], label='RSI (pandas-ta)', color='orange')
                ax2.plot(data.index, data['RSI_Scratch'], label='RSI (scratch)', color='purple', linestyle='--')
            else:
                ax2.plot(data.index, data['RSI'], label='RSI', color='orange')
            ax2.axhline(y=self.user.buy_threshold, color='green', linestyle='--', label=f'Buy Threshold ({self.user.buy_threshold})')
            ax2.axhline(y=self.user.sell_threshold, color='red', linestyle='--', label=f'Sell Threshold ({self.user.sell_threshold})')
            ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.3)
            ax2.set_ylabel("RSI")
            ax2.set_ylim(0, 100)
            indicator_plotted = True
        else:
            if self.user.strategy == "KAGE" and 'vol' in data.columns:
                ax2.plot(data.index, data['vol'], label='Volatility', color='orange')
                ax2.axhline(y=data['vol'].mean() * 1.5, color='red', linestyle='--', label='Volatility Threshold')
                ax2.set_ylabel("Volatility")
                indicator_plotted = True
            elif self.user.strategy == "KITSUNE":
                window = 20
                z_scores = []
                for i in range(window, len(data)):
                    window_prices = data['close'].iloc[i-window:i]
                    mean_price = window_prices.mean()
                    std_price = window_prices.std()
                    if std_price == 0:
                        std_price = 1e-8
                    z = (data['close'].iloc[i] - mean_price) / std_price
                    z_scores.append(z)
                data.loc[data.index[window:], 'kitsune_z'] = z_scores
                ax2.plot(data.index[window:], data.loc[data.index[window:], 'kitsune_z'], label=f'Kitsune Z-Score (window={window})', color='orange')
                ax2.axhline(y=-1, color='green', linestyle='--', label='Long Entry Threshold (-1)')
                ax2.axhline(y=1, color='red', linestyle='--', label='Long Exit Threshold (1)')
                ax2.set_ylabel("Kitsune Z-Score")
                indicator_plotted = True
            elif self.user.strategy == "RYU":
                window = 50
                z_returns = []
                for i in range(window, len(data)):
                    current_return = data['return'].iloc[i]
                    window_returns = data['return'].iloc[i-window:i]
                    mean_r = window_returns.mean()
                    std_r = window_returns.std()
                    if std_r == 0:
                        std_r = 1e-8
                    z_r = (current_return - mean_r) / std_r
                    z_returns.append(z_r)
                data.loc[data.index[window:], 'ryu_z'] = z_returns
                ax2.plot(data.index[window:], data.loc[data.index[window:], 'ryu_z'], label=f'RYU Return Z-Score (window={window})', color='orange')
                ax2.axhline(y=-1, color='green', linestyle='--', label='Long Entry Threshold (-1)')
                ax2.axhline(y=1, color='red', linestyle='--', label='Long Exit Threshold (1)')
                ax2.set_ylabel("RYU Z-Score")
                indicator_plotted = True
            elif self.user.strategy == "HIKARI" and 'return' in data.columns:
                window = 30
                data['momentum'] = data['return'].rolling(window=window).mean()
                ax2.plot(data.index, data['momentum'], label=f'Momentum (window={window})', color='orange')
                ax2.set_ylabel("Momentum")
                indicator_plotted = True
            elif self.user.strategy == "TENSHI":
                data['diff'] = data['close'].diff()
                ax2.plot(data.index, data['diff'], label='Price Diff', color='orange')
                ax2.set_ylabel("Price Difference")
                indicator_plotted = True
            elif self.user.strategy == "ZEN":
                ax2.plot(data.index, data['normalized_phase'], label='Normalized Phase', color='orange', linestyle=':')
                ax2.plot(data.index, data['momentum'], label='Momentum', color='purple', linestyle='--')
                ax2.axhline(y=0.3, color='green', linestyle='--', label='Phase Lower Threshold (0.3)')
                ax2.axhline(y=0.7, color='red', linestyle='--', label='Phase Upper Threshold (0.7)')
                ax2.set_ylabel("ZEN Indicators")
                indicator_plotted = True

        if 'novel_stoch' in data.columns and data['novel_stoch'].notna().any():
            ax2.plot(data.index, data['novel_stoch'], label='Novel Stochastic', color='blue', linestyle=':')
            ax2.axhline(y=20, color='green', linestyle='--', label='Stoch Buy Threshold (20)')
            ax2.axhline(y=80, color='red', linestyle='--', label='Stoch Sell Threshold (80)')

        if not indicator_plotted:
            ax2.text(0.5, 0.5, 'No indicator available', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
            ax2.set_ylabel("Indicator")

        ax2.set_xlabel("Date")
        ax2.legend()
        if self.user.strategy in ["MA", "RSI_MA"] and 'MA' in data.columns:
            ax1.plot(data.index, data['MA'], label=f"MA ({self.user.ma_period})", color='magenta', linestyle='--')
            ax1.legend()
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        return img_base64

    def execute_kage(self, symbol, use_csv):
        data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        data['novel_stoch'] = self.calculate_novel_stochastic(data, period=14)
        data['return'] = np.log(data['close'] / data['close'].shift(1))
        data.dropna(inplace=True)
        window = 30
        data['vol'] = data['return'].rolling(window=window).std()
        threshold_vol = data['vol'].mean() * 1.5
        trades = []
        open_positions = []
        for i in range(window, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_vol = data['vol'].iloc[i]
            stoch_val = data['novel_stoch'].iloc[i]
            if self.user.trade_type == 'long':
                if current_vol < threshold_vol and stoch_val < 20 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif current_vol > threshold_vol and stoch_val > 80 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'long')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'long',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"KAGE Long trade for {symbol}: Buy at {pos['entry_time']} (price: {pos['entry_price']}) | Sell at {current_time} (price: {current_price}) | P/L: {profit:.2f}%")
            else:
                if current_vol < threshold_vol and stoch_val > 80 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif current_vol > threshold_vol and stoch_val < 20 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'short')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'short',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"KAGE Short trade for {symbol}: Sell at {pos['entry_time']} (price: {pos['entry_price']}) | Cover at {current_time} (price: {current_price}) | P/L: {profit:.2f}%")
        summary = self.calculate_trade_summary(trades)
        plot_image = self.generate_plots(data, trades)
        return {"trades": trades, "plot": plot_image, "summary": summary}

    def execute_kitsune(self, symbol, use_csv):
        data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        data['novel_stoch'] = self.calculate_novel_stochastic(data, period=14)
        window = 20
        trades = []
        open_positions = []
        for i in range(window, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            window_prices = data['close'].iloc[i-window:i]
            mean_price = window_prices.mean()
            std_price = window_prices.std()
            if std_price == 0:
                std_price = 1e-8
            z_score = (current_price - mean_price) / std_price
            stoch_val = data['novel_stoch'].iloc[i]
            if self.user.trade_type == 'long':
                if z_score < -1.0 and stoch_val < 20 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif z_score > 1.0 and stoch_val > 80 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'long')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'long',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"KITSUNE Long trade for {symbol}: Buy at {pos['entry_time']} | Sell at {current_time} | P/L: {profit:.2f}%")
            else:
                if z_score > 1.0 and stoch_val > 80 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif z_score < -1.0 and stoch_val < 20 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'short')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'short',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"KITSUNE Short trade for {symbol}: Sell at {pos['entry_time']} | Cover at {current_price} | P/L: {profit:.2f}%")
        summary = self.calculate_trade_summary(trades)
        plot_image = self.generate_plots(data, trades)
        return {"trades": trades, "plot": plot_image, "summary": summary}

    def execute_ryu(self, symbol, use_csv):
        data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        data['novel_stoch'] = self.calculate_novel_stochastic(data, period=14)
        data['return'] = np.log(data['close'] / data['close'].shift(1))
        data.dropna(inplace=True)
        window = 50
        trades = []
        open_positions = []
        for i in range(window, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            current_return = data['return'].iloc[i]
            window_returns = data['return'].iloc[i-window:i]
            mean_r = window_returns.mean()
            std_r = window_returns.std()
            if std_r == 0:
                std_r = 1e-8
            z_r = (current_return - mean_r) / std_r
            stoch_val = data['novel_stoch'].iloc[i]
            if self.user.trade_type == 'long':
                if z_r < -1 and stoch_val < 20 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif z_r > 1 and stoch_val > 80 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'long')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'long',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"RYU Long trade for {symbol}: Buy at {pos['entry_time']} | Sell at {current_time} | P/L: {profit:.2f}%")
            else:
                if z_r > 1 and stoch_val > 80 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif z_r < -1 and stoch_val < 20 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'short')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'short',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"RYU Short trade for {symbol}: Sell at {pos['entry_time']} | Cover at {current_time} | P/L: {profit:.2f}%")
        summary = self.calculate_trade_summary(trades)
        plot_image = self.generate_plots(data, trades)
        return {"trades": trades, "plot": plot_image, "summary": summary}

    def execute_sakura(self, symbol, use_csv):
        data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        data['novel_stoch'] = self.calculate_novel_stochastic(data, period=14)
        trades = []
        open_positions = []
        window = 50
        for i in range(window, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            pivot = data['close'].iloc[i-window:i].median()
            up_segment = data['close'].iloc[i-window:i][data['close'].iloc[i-window:i] > pivot]
            down_segment = data['close'].iloc[i-window:i][data['close'].iloc[i-window:i] <= pivot]
            stoch_val = data['novel_stoch'].iloc[i]
            if len(up_segment) >= 2 and len(down_segment) >= 2:
                t_vals = np.arange(len(up_segment))
                coeffs_up = np.polyfit(t_vals, up_segment.values, 1)
                t_vals_down = np.arange(len(down_segment))
                coeffs_down = np.polyfit(t_vals_down, down_segment.values, 1)
                slope = (coeffs_up[0] - coeffs_down[0]) / 2
                mirror_price = (up_segment.values[-1] + down_segment.values[-1]) / 2 + slope
                deviation = abs(current_price - mirror_price)
                threshold = current_price * 0.003
                if self.user.trade_type == 'long':
                    if deviation < threshold and stoch_val < 20 and not open_positions:
                        open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                    elif deviation > threshold and stoch_val > 80 and open_positions:
                        pos = open_positions.pop(0)
                        profit = self.safe_profit_pct(pos['entry_price'], current_price, 'long')
                        trade = {
                            'symbol': symbol,
                            'entry_time': str(pos['entry_time']),
                            'entry_price': pos['entry_price'],
                            'exit_time': str(current_time),
                            'exit_price': current_price,
                            'trade_type': 'long',
                            'profit_pct': profit
                        }
                        trades.append(trade)
                        TradeLogger.log_trade(f"SAKURA Long trade for {symbol}: Buy at {pos['entry_time']} | Sell at {current_time} | P/L: {profit:.2f}%")
                else:
                    if deviation < threshold and stoch_val > 80 and not open_positions:
                        open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                    elif deviation > threshold and stoch_val < 20 and open_positions:
                        pos = open_positions.pop(0)
                        profit = self.safe_profit_pct(pos['entry_price'], current_price, 'short')
                        trade = {
                            'symbol': symbol,
                            'entry_time': str(pos['entry_time']),
                            'entry_price': pos['entry_price'],
                            'exit_time': str(current_time),
                            'exit_price': current_price,
                            'trade_type': 'short',
                            'profit_pct': profit
                        }
                        trades.append(trade)
                        TradeLogger.log_trade(f"SAKURA Short trade for {symbol}: Sell at {pos['entry_time']} | Cover at {current_time} | P/L: {profit:.2f}%")
        summary = self.calculate_trade_summary(trades)
        plot_image = self.generate_plots(data, trades)
        return {"trades": trades, "plot": plot_image, "summary": summary}

    def execute_hikari(self, symbol, use_csv):
        data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        data['novel_stoch'] = self.calculate_novel_stochastic(data, period=14)
        data['return'] = np.log(data['close'] / data['close'].shift(1))
        data.dropna(inplace=True)
        window = 30
        trades = []
        open_positions = []
        for i in range(window, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            window_returns = data['return'].iloc[i-window:i].values.reshape(-1, 1)
            cov_matrix = np.cov(window_returns.T)
            if cov_matrix.ndim < 2 or np.isscalar(cov_matrix):
                momentum = data['return'].iloc[i]
            else:
                eigvals, eigvecs = np.linalg.eig(cov_matrix)
                principal_vec = eigvecs[:, np.argmax(eigvals)]
                momentum = data['return'].iloc[i] * principal_vec[0]
            stoch_val = data['novel_stoch'].iloc[i]
            if self.user.trade_type == 'long':
                if momentum > 0.0005 and stoch_val < 20 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif momentum < 0 and stoch_val > 80 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'long')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'long',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"HIKARI Long trade for {symbol}: Buy at {pos['entry_time']} | Sell at {current_time} | P/L: {profit:.2f}%")
            else:
                if momentum < -0.0005 and stoch_val > 80 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif momentum > 0 and stoch_val < 20 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'short')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'short',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"HIKARI Short trade for {symbol}: Sell at {pos['entry_time']} | Cover at {current_time} | P/L: {profit:.2f}%")
        summary = self.calculate_trade_summary(trades)
        plot_image = self.generate_plots(data, trades)
        return {"trades": trades, "plot": plot_image, "summary": summary}

    def execute_tenshi(self, symbol, use_csv):
        data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        data['novel_stoch'] = self.calculate_novel_stochastic(data, period=14)
        trades = []
        open_positions = []
        prices = data['close'].values
        diff = np.diff(prices)
        sign = np.sign(diff)
        extrema = (np.diff(sign) != 0).nonzero()[0] + 1
        for idx in extrema:
            current_time = data.index[idx]
            current_price = prices[idx]
            stoch_val = data['novel_stoch'].iloc[idx]
            if idx > 0 and idx < len(prices)-1:
                if prices[idx] < prices[idx-1] and prices[idx] < prices[idx+1]:
                    if self.user.trade_type == 'long' and stoch_val < 20 and not open_positions:
                        open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif prices[idx] > prices[idx-1] and prices[idx] > prices[idx+1]:
                    if self.user.trade_type == 'long' and stoch_val > 80 and open_positions:
                        pos = open_positions.pop(0)
                        profit = self.safe_profit_pct(pos['entry_price'], current_price, 'long')
                        trade = {
                            'symbol': symbol,
                            'entry_time': str(pos['entry_time']),
                            'entry_price': pos['entry_price'],
                            'exit_time': str(current_time),
                            'exit_price': current_price,
                            'trade_type': 'long',
                            'profit_pct': profit
                        }
                        trades.append(trade)
                        TradeLogger.log_trade(f"TENSHI Long trade for {symbol}: Buy at {pos['entry_time']} | Sell at {current_time} | P/L: {profit:.2f}%")
        summary = self.calculate_trade_summary(trades)
        plot_image = self.generate_plots(data, trades)
        return {"trades": trades, "plot": plot_image, "summary": summary}

    def execute_zen(self, symbol, use_csv):
        data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        data['novel_stoch'] = self.calculate_novel_stochastic(data, period=14)
        window = 20
        data['SMA'] = data['close'].rolling(window=window, min_periods=1).mean()
        data['std'] = data['close'].rolling(window=window, min_periods=1).std()
        k = 2
        data['upper_band'] = data['SMA'] + k * data['std']
        data['lower_band'] = data['SMA'] - k * data['std']
        def compute_phase(row):
            band_width = row['upper_band'] - row['lower_band']
            if band_width == 0:
                return 0.5
            return (row['close'] - row['lower_band']) / band_width
        data['normalized_phase'] = data.apply(compute_phase, axis=1)
        data['momentum'] = data['close'] - data['close'].shift(1)
        data['momentum'] = data['momentum'].fillna(0)
        trades = []
        open_positions = []
        for i in range(window, len(data)):
            current_time = data.index[i]
            current_price = data['close'].iloc[i]
            phase = data['normalized_phase'].iloc[i]
            stoch_val = data['novel_stoch'].iloc[i]
            momentum = data['momentum'].iloc[i]
            if self.user.trade_type == 'long':
                if phase < 0.3 and stoch_val < 20 and momentum > 0 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif phase > 0.7 and stoch_val > 80 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'long')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'long',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"ZEN New Long trade for {symbol}: Buy at {pos['entry_time']} (price: {pos['entry_price']}) | Sell at {current_time} (price: {current_price}) | P/L: {profit:.2f}%")
            else:
                if phase > 0.7 and stoch_val > 80 and momentum < 0 and not open_positions:
                    open_positions.append({'entry_time': current_time, 'entry_price': current_price})
                elif phase < 0.3 and stoch_val < 20 and open_positions:
                    pos = open_positions.pop(0)
                    profit = self.safe_profit_pct(pos['entry_price'], current_price, 'short')
                    trade = {
                        'symbol': symbol,
                        'entry_time': str(pos['entry_time']),
                        'entry_price': pos['entry_price'],
                        'exit_time': str(current_time),
                        'exit_price': current_price,
                        'trade_type': 'short',
                        'profit_pct': profit
                    }
                    trades.append(trade)
                    TradeLogger.log_trade(f"ZEN New Short trade for {symbol}: Sell at {pos['entry_time']} (price: {pos['entry_price']}) | Cover at {current_time} (price: {current_price}) | P/L: {profit:.2f}%")
        summary = self.calculate_trade_summary(trades)
        plot_image = self.generate_plots(data, trades)
        return {"trades": trades, "plot": plot_image, "summary": summary}

    def execute_strategy(self, symbol, use_scratch_rsi=False, use_csv=False):
        strat = self.user.strategy
        if strat in ["RSI", "MA", "RSI_MA"]:
            data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
            if data is None or data.empty:
                return {"error": f"No data fetched for {symbol}"}
            data['novel_stoch'] = self.calculate_novel_stochastic(data, period=14)
            data = self.calculate_rsi(data)
            if use_scratch_rsi:
                data = self.calculate_rsi_scratch(data)
            if strat in ["MA", "RSI_MA"]:
                data['MA'] = data['close'].rolling(window=self.user.ma_period).mean()
                data = data.iloc[self.user.ma_period-1:]
            trades = []
            open_positions = []
            for i in range(1, len(data)):
                prev_rsi = data['RSI'].iloc[i-1]
                curr_rsi = data['RSI'].iloc[i]
                stoch_val = data['novel_stoch'].iloc[i]
                current_time = data.index[i]
                curr_price = data['close'].iloc[i]
                if self.user.trade_type == 'long':
                    if prev_rsi < self.user.buy_threshold and curr_rsi >= self.user.buy_threshold and stoch_val < 20:
                        if not open_positions:
                            open_positions.append({'entry_time': current_time, 'entry_price': curr_price, 'entry_rsi': curr_rsi})
                    elif prev_rsi > self.user.sell_threshold and curr_rsi <= self.user.sell_threshold and stoch_val > 80:
                        for pos in open_positions[:]:
                            profit = self.safe_profit_pct(pos['entry_price'], curr_price, 'long')
                            trade = {
                                'symbol': symbol,
                                'entry_time': str(pos['entry_time']),
                                'entry_price': pos['entry_price'],
                                'entry_rsi': pos['entry_rsi'],
                                'exit_time': str(current_time),
                                'exit_price': curr_price,
                                'exit_rsi': curr_rsi,
                                'trade_type': 'long',
                                'profit_pct': profit
                            }
                            trades.append(trade)
                            open_positions.remove(pos)
                            TradeLogger.log_trade(f"Long RSI trade for {symbol}: Buy at {pos['entry_time']} (price: {pos['entry_price']}, RSI: {pos['entry_rsi']}) | Sell at {current_time} (price: {curr_price}, RSI: {curr_rsi}) | P/L: {profit:.2f}%")
                else:
                    if prev_rsi > self.user.sell_threshold and curr_rsi <= self.user.sell_threshold and stoch_val > 80:
                        if not open_positions:
                            open_positions.append({'entry_time': current_time, 'entry_price': curr_price, 'entry_rsi': curr_rsi})
                    elif prev_rsi < self.user.buy_threshold and curr_rsi >= self.user.buy_threshold and stoch_val < 20:
                        for pos in open_positions[:]:
                            profit = self.safe_profit_pct(pos['entry_price'], curr_price, 'short')
                            trade = {
                                'symbol': symbol,
                                'entry_time': str(pos['entry_time']),
                                'entry_price': pos['entry_price'],
                                'entry_rsi': pos['entry_rsi'],
                                'exit_time': str(current_time),
                                'exit_price': curr_price,
                                'exit_rsi': curr_rsi,
                                'trade_type': 'short',
                                'profit_pct': profit
                            }
                            trades.append(trade)
                            open_positions.remove(pos)
                            TradeLogger.log_trade(f"Short RSI trade for {symbol}: Sell at {pos['entry_time']} (price: {pos['entry_price']}, RSI: {pos['entry_rsi']}) | Cover at {current_time} (price: {curr_price}, RSI: {curr_rsi}) | P/L: {profit:.2f}%")
            df = data.reset_index()
            df['datetime'] = df['datetime'].astype(str)
            data_json = df.to_dict('records')
            plot_image = self.generate_plots(data, trades)
            trade_summary = self.calculate_trade_summary(trades)
            return {"trades": trades, "data": data_json, "plot": plot_image, "summary": trade_summary}
        else:
            if strat == "KAGE":
                return self.execute_kage(symbol, use_csv)
            elif strat == "KITSUNE":
                return self.execute_kitsune(symbol, use_csv)
            elif strat == "RYU":
                return self.execute_ryu(symbol, use_csv)
            elif strat == "SAKURA":
                return self.execute_sakura(symbol, use_csv)
            elif strat == "HIKARI":
                return self.execute_hikari(symbol, use_csv)
            elif strat == "TENSHI":
                return self.execute_tenshi(symbol, use_csv)
            elif strat == "ZEN":
                return self.execute_zen(symbol, use_csv)
            else:
                return {"error": "Unknown strategy specified."}
# --------------------- Redesigned DefAIAgent ---------------------
class DefAIAgent:
    """
    Redesigned defAI agent strategy:
      - Computes rolling yield (average return) and risk (volatility).
      - Calculates a momentum indicator (difference between close price and its moving average).
      - Derives a stability index = yield / (risk + epsilon).
      - Determines an action based on thresholds:
           * If stability > thresh_high and momentum positive => yield_harvest.
           * If stability < thresh_low and momentum negative => rebalance.
           * Else => hold.
      - Simulates a transaction (using fake Aptos data) if action is not 'hold'.
      - Logs the trade decision to trades.log.
      - Returns an interactive Plotly plot and summary metrics.
    """
    def __init__(self, user: User, exchange_id: str = 'binance'):
        self.user = user
        self.simulate = True  # Always simulate Aptos transactions.
        try:
            self.exchange = getattr(ccxt, exchange_id)({
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            self.exchange.load_markets()
        except Exception as e:
            raise ValueError(f"Exchange error: {str(e)}")

    def fetch_data(self, symbol, timeframe='1m', limit=1000, use_csv=False):
        try:
            if use_csv:
                data = pd.read_csv("minute_data.csv")
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')\
                                    .dt.tz_localize('UTC').dt.tz_convert(india_tz)
                data.set_index('datetime', inplace=True)
                return data
            else:
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                except Exception as e:
                    print(f"Error fetching data from exchange: {e}")
                    ohlcv = []
                if not ohlcv:
                    # Generate dummy data if live fetch fails.
                    now = pd.Timestamp.now(tz=india_tz)
                    times = pd.date_range(end=now, periods=limit, freq='1min')
                    dummy_data = {
                        'timestamp': [int(ts.timestamp()*1000) for ts in times],
                        'open': np.random.uniform(100, 200, limit),
                        'high': np.random.uniform(100, 200, limit),
                        'low': np.random.uniform(100, 200, limit),
                        'close': np.random.uniform(100, 200, limit),
                        'volume': np.random.uniform(100, 1000, limit)
                    }
                    data = pd.DataFrame(dummy_data, index=times)
                    data['datetime'] = data.index
                    data.set_index('datetime', inplace=True)
                    return data
                else:
                    data = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')\
                                        .dt.tz_localize('UTC').dt.tz_convert(india_tz)
                    data.set_index('datetime', inplace=True)
                    return data
        except Exception as e:
            print(f"Error in fetch_data (defAI): {e}")
            return None

    def calculate_yield_score(self, data, window=30):
        returns = data['close'].pct_change()
        avg_return = returns.rolling(window=window).mean() * 100  # percentage
        return avg_return

    def calculate_risk_index(self, data, window=30):
        returns = data['close'].pct_change()
        risk = returns.rolling(window=window).std() * 100  # percentage volatility
        return risk

    def calculate_momentum(self, data, window=30):
        ma = data['close'].rolling(window=window).mean()
        momentum = data['close'] - ma
        return momentum

    def calculate_stability_index(self, yield_series, risk_series, epsilon=1e-8):
        latest_yield = yield_series.iloc[-1]
        latest_risk = risk_series.iloc[-1]
        stability = latest_yield / (latest_risk + epsilon)
        return stability

    def determine_action(self, stability, momentum, thresh_high=1.5, thresh_low=0.8):
        if stability > thresh_high and momentum > 0:
            return "yield_harvest"
        elif stability < thresh_low and momentum < 0:
            return "rebalance"
        else:
            return "hold"

    def generate_defai_plot(self, data, yield_series, risk_series, momentum_series, stability):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=yield_series, mode='lines', name='Yield Score'))
        fig.add_trace(go.Scatter(x=data.index, y=risk_series, mode='lines', name='Risk Index'))
        fig.add_trace(go.Scatter(x=data.index, y=momentum_series, mode='lines', name='Momentum'))
        stability_line = [stability] * len(data.index)
        fig.add_trace(go.Scatter(x=data.index, y=stability_line, mode='lines', name='Stability Index'))
        fig.update_layout(title="defAI Metrics Over Time",
                          xaxis_title="Time",
                          yaxis_title="Metric Values",
                          template="plotly_white")
        return fig.to_json()

    def perform_aptos_transaction(self, transaction_type):
        time.sleep(1)  # Simulate network delay.
        synthetic_hash = "0x" + uuid.uuid4().hex
        synthetic_account = "0x" + uuid.uuid4().hex[:40]
        return {
            "transaction_hash": synthetic_hash,
            "account": synthetic_account,
            "message": f"Simulated {transaction_type} transaction using fake data."
        }

    def execute_defai(self, symbol, use_csv=False):
        data = self.fetch_data(symbol, timeframe='1m', limit=1000, use_csv=use_csv)
        if data is None or data.empty:
            return {"error": f"No data fetched for {symbol}"}
        yield_series = self.calculate_yield_score(data)
        risk_series = self.calculate_risk_index(data)
        momentum_series = self.calculate_momentum(data)
        stability = self.calculate_stability_index(yield_series, risk_series)
        latest_momentum = momentum_series.iloc[-1]
        action = self.determine_action(stability, latest_momentum)
        interactive_plot = self.generate_defai_plot(data, yield_series, risk_series, momentum_series, stability)
        summary = {
            "latest_yield_score": yield_series.iloc[-1],
            "latest_risk_index": risk_series.iloc[-1],
            "latest_momentum": latest_momentum,
            "stability_index": stability,
            "action": action
        }
        trades = []
        if action in ["yield_harvest", "rebalance"]:
            aptos_result = self.perform_aptos_transaction(action)
            trade_details = {
                "action": action,
                "timestamp": str(data.index[-1]),
                "aptos_transaction": aptos_result,
                "details": summary
            }
            trades.append(trade_details)
            # Log the defAI trade decision
            TradeLogger.log_trade(
                f"defAI {action} for {symbol}: Stability Index = {stability:.2f}, "
                f"Momentum = {latest_momentum:.2f} | Aptos Txn = {aptos_result['transaction_hash']}"
            )
        return {"trades": trades, "plot": interactive_plot, "summary": summary}

@app.route('/trade', methods=['POST'])
def trade():
    content = request.json
    if not content:
        return jsonify({"error": "No input data provided"}), 400  # Early error check
    exchange_id = content.get('exchange', 'binance')
    symbol = content.get('symbol')
    username = content.get('username', 'default_user')
    rsi_period = int(content.get('rsi_period', 14))
    buy_threshold = float(content.get('buy_threshold', 30))
    sell_threshold = float(content.get('sell_threshold', 70))
    trade_type = content.get('trade_type', 'long')
    strategy = content.get('strategy', 'RSI')
    ma_period = int(content.get('ma_period', 20))
    use_scratch_rsi = content.get('use_scratch_rsi', False)
    use_csv = content.get('use_csv', False)
    
    user = User(username, rsi_period, buy_threshold, sell_threshold, trade_type, strategy, ma_period)
    if strategy.upper() == "DEFAI":
        agent = DefAIAgent(user, exchange_id)
        result = agent.execute_defai(symbol, use_csv)
    else:
        strategy_obj = TradingStrategy(user, exchange_id)
        result = strategy_obj.execute_strategy(symbol, use_scratch_rsi, use_csv)
    
    return jsonify(result)

@app.route('/exchanges', methods=['GET'])
def get_exchanges():
    exchanges = ccxt.exchanges
    return jsonify({"exchanges": exchanges})

@app.route('/symbols', methods=['GET'])
def get_symbols():
    exchange_id = request.args.get('exchange')
    if not exchange_id:
        return jsonify({"error": "Please provide an exchange id parameter."}), 400
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        markets = exchange.load_markets()
        symbols = list(markets.keys())
        return jsonify({"exchange": exchange_id, "symbols": symbols})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=False)
