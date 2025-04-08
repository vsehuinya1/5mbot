import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import requests
import os

# Nairobi timezone
nairobi_zone = pytz.timezone('Africa/Nairobi')

# Telegram settings from GitHub secrets
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

# UT Bot Parameters
H1_ATR_PERIOD = 14
H1_KEY_VALUE = 1.5
M5_ATR_PERIOD = 10
M5_KEY_VALUE = 1.0

# Volatility Filter Parameters
ATR_THRESHOLD_PCT = 0.015  # 1.5% of price (changed from 1.75%)
RANGE_PERIOD = 10          # Last 10 H1 candles
RANGE_THRESHOLD_PCT = 0.02 # 2% of price (changed from 3%)

# Pairs to scan
symbols = ['WIF/USDT', 'FARTCOIN/USDT', 'PI/USDT', 'HYPE/USDT', 'DOGE/USDT', 'XRP/USDT', 'AI16Z/USDT', 'BERA/USDT', 'S/USDT', 'MOODENG/USDT', 'ACT/USDT', 'BTC/USDT', 'APE/USDT', 'PEPE/USDT', 'ADA/USDT', 'LTC/USDT', 'SOL/USDT', 'DOT/USDT', 'APT/USDT', 'POPCAT/USDT', 'JUP/USDT', 'AVAX/USDT', 'RENDER/USDT', 'PENGU/USDT']

exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=payload)
        sent_time = datetime.now(nairobi_zone).strftime('%d-%b %H:%M:%S')
        print(f"Telegram sent at {sent_time}: {response.status_code}")
        response.raise_for_status()
    except Exception as e:
        print(f"Telegram error: {e}")

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def ut_bot_alerts(df, atr_period=14, key_value=1.0):
    df = df.copy()
    df['ATR'] = calculate_atr(df, atr_period)
    df['TrailingStop'] = np.nan
    df['Trend'] = 0
    df['Signal'] = 0
    for i in range(1, len(df)):
        prev_close = df['Close'].iloc[i - 1]
        prev_trailing = df['TrailingStop'].iloc[i - 1] if not pd.isna(df['TrailingStop'].iloc[i - 1]) else df['Close'].iloc[0]
        atr_val = df['ATR'].iloc[i] * key_value
        if df['Trend'].iloc[i - 1] == 1:
            new_stop = max(prev_trailing, prev_close - atr_val)
        else:
            new_stop = min(prev_trailing, prev_close + atr_val)
        df.loc[df.index[i], 'TrailingStop'] = new_stop
        close = df['Close'].iloc[i]
        if close > new_stop and prev_close <= prev_trailing:
            df.loc[df.index[i], 'Trend'] = 1
            df.loc[df.index[i], 'Signal'] = 1
        elif close < new_stop and prev_close >= prev_trailing:
            df.loc[df.index[i], 'Trend'] = -1
            df.loc[df.index[i], 'Signal'] = -1
        else:
            df.loc[df.index[i], 'Trend'] = df['Trend'].iloc[i - 1]
    return df

run_time = datetime.now(nairobi_zone)
print(f"Starting script at {run_time.strftime('%d-%b %H:%M:%S')}")

# Track latest signal per pair
latest_signals = {}

for symbol in symbols:
    try:
        # Fetch H1
        h1_fetch_time = datetime.now(nairobi_zone)
        h1_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        h1_df = pd.DataFrame(h1_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        h1_df['Timestamp'] = pd.to_datetime(h1_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
        h1_df = ut_bot_alerts(h1_df, atr_period=H1_ATR_PERIOD, key_value=H1_KEY_VALUE)
        
        # Filter to complete H1 candles only (before current hour)
        h1_complete = h1_df[h1_df['Timestamp'] < run_time.replace(minute=0, second=0, microsecond=0)]
        last_h1 = h1_complete[h1_complete['Signal'] != 0].iloc[-1] if not h1_complete[h1_complete['Signal'] != 0].empty else None
        print(f"{symbol} H1 fetched at {h1_fetch_time.strftime('%d-%b %H:%M:%S')}, latest signal at {last_h1['Timestamp'] if last_h1 is not None else 'None'}")

        if last_h1 is None:
            latest_signals[symbol] = (symbol.replace('/USDT', ''), 'No H1 Signal', run_time, run_time)
            continue

        # Volatility filters
        h1_atr = h1_df['ATR'].iloc[-1]
        atr_threshold = h1_df['Close'].iloc[-1] * ATR_THRESHOLD_PCT
        range_10h = h1_df['High'].tail(RANGE_PERIOD).max() - h1_df['Low'].tail(RANGE_PERIOD).min()
        range_threshold = h1_df['Close'].iloc[-1] * RANGE_THRESHOLD_PCT

        h1_signal = last_h1['Signal']
        h1_time = last_h1['Timestamp']

        # Fetch M5 regardless of volatility filter
        m5_fetch_time = datetime.now(nairobi_zone)
        m5_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
        m5_df = pd.DataFrame(m5_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        m5_df['Timestamp'] = pd.to_datetime(m5_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
        m5_df = ut_bot_alerts(m5_df, atr_period=M5_ATR_PERIOD, key_value=M5_KEY_VALUE)

        # Check volatility filter but still process M5
        if h1_atr < atr_threshold or range_10h < range_threshold:
            print(f"{symbol}: Market ranging - (ATR: {h1_atr:.6f} < {atr_threshold:.6f}, Range: {range_10h:.6f} < {range_threshold:.6f})")
            latest_signals[symbol] = (symbol.replace('/USDT', ''), 'Ranging', h1_time, run_time)
        else:
            # Latest M5 signal after H1, before run_time
            m5_signals = m5_df[(m5_df['Signal'] == h1_signal) & (m5_df['Timestamp'] > h1_time) & (m5_df['Timestamp'] <= run_time)]
            if not m5_signals.empty:
                m5_entry = m5_signals.iloc[-1]['Timestamp']  # Latest M5 entry
                signal_type = 'Buy' if h1_signal == 1 else 'Sell'
                latest_signals[symbol] = (symbol.replace('/USDT', ''), signal_type, h1_time, m5_entry)
                print(f"{symbol} M5 fetched at {m5_fetch_time.strftime('%d-%b %H:%M:%S')}, latest entry at {m5_entry}")
            else:
                latest_signals[symbol] = (symbol.replace('/USDT', ''), 'No M5 Signal', h1_time, run_time)

    except Exception as e:
        error_msg = f"Error for {symbol}: {str(e)}"
        print(error_msg)
        latest_signals[symbol] = (symbol.replace('/USDT', ''), f"Error: {str(e)}", run_time, run_time)

# Convert to list, include all signals
final_signals = list(latest_signals.values())
final_signals = sorted(final_signals, key=lambda x: x[3], reverse=True)

# Send summary with all pairs
if final_signals:
    message = f"🔔 Latest UT Bot M5 Entries (H1: KV={H1_KEY_VALUE}, ATR={H1_ATR_PERIOD} | M5: KV={M5_KEY_VALUE}, ATR={M5_ATR_PERIOD})\nRun at {run_time.strftime('%d-%b %H:%M:%S')}\n\n"
    message += f"{'Pair':<10} | {'Signal':<10} | {'H1 Time':<16} | {'M5 Entry'}\n"
    message += "-" * 60 + "\n"
    for pair, signal, h1_time, m5_time in final_signals:
        h1_str = h1_time.strftime('%d-%b %H:%M') if isinstance(h1_time, pd.Timestamp) else 'N/A'
        m5_str = m5_time.strftime('%d-%b %H:%M') if isinstance(m5_time, pd.Timestamp) else 'N/A'
        message += f"{pair:<10} | {signal:<10} | {h1_str:<16} | {m5_str}\n"
    print(message)
    send_telegram_message(message)
else:
    print("No signals detected.")
    send_telegram_message(f"🔔 No UT Bot signals detected.\nRun at {run_time.strftime('%d-%b %H:%M:%S')}")

print(f"Script ended at {datetime.now(nairobi_zone).strftime('%d-%b %H:%M:%S')}")
