import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import requests

# Nairobi timezone
nairobi_zone = pytz.timezone('Africa/Nairobi')

# Telegram settings
TELEGRAM_TOKEN = '7207679435:AAFDD8_voRSiyT5XgQOHNhIKOBAhxIZ7YaA'
CHAT_ID = '6707099301'

# UT Bot Parameters
H1_ATR_PERIOD = 14
H1_KEY_VALUE = 1.5
M5_ATR_PERIOD = 10
M5_KEY_VALUE = 1.0

# Pairs to scan
symbols = ['WIF/USDT', 'DOGE/USDT', 'XRP/USDT', 'AI16Z/USDT', 'BERA/USDT', 'S/USDT', 'MOODENG/USDT', 'ACT/USDT', 'BTC/USDT', 'APE/USDT', 'PEPE/USDT', 'ADA/USDT', 'LTC/USDT', 'SOL/USDT', 'DOT/USDT', 'APT/USDT', 'POPCAT/USDT', 'JUP/USDT', 'AVAX/USDT', 'RENDER/USDT', 'PENGU/USDT']

exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})
exchange.load_markets()

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

# Track signals and active positions
final_signals = []
active_positions = {}

for symbol in symbols:
    try:
        # Fetch H1
        h1_fetch_time = datetime.now(nairobi_zone)
        h1_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
        h1_df = pd.DataFrame(h1_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        h1_df['Timestamp'] = pd.to_datetime(h1_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
        h1_df = ut_bot_alerts(h1_df, atr_period=H1_ATR_PERIOD, key_value=H1_KEY_VALUE)
        last_h1 = h1_df[h1_df['Signal'] != 0].iloc[-1] if not h1_df[h1_df['Signal'] != 0].empty else None
        print(f"{symbol} H1 fetched at {h1_fetch_time.strftime('%d-%b %H:%M:%S')}, latest signal at {last_h1['Timestamp'] if last_h1 is not None else 'None'}")

        if last_h1 is not None:
            h1_signal = last_h1['Signal']
            h1_time = last_h1['Timestamp']

            # Fetch M5
            m5_fetch_time = datetime.now(nairobi_zone)
            m5_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=200)  # Wider for 12-hour window
            m5_df = pd.DataFrame(m5_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            m5_df['Timestamp'] = pd.to_datetime(m5_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
            m5_df = ut_bot_alerts(m5_df, atr_period=M5_ATR_PERIOD, key_value=M5_KEY_VALUE)

            # Latest M5 entry after H1
            m5_signals = m5_df[(m5_df['Signal'] == h1_signal) & (m5_df['Timestamp'] > h1_time) & (m5_df['Timestamp'] <= run_time)]
            if not m5_signals.empty:
                latest_m5 = m5_signals.iloc[-1]
                direction = 'Buy' if h1_signal == 1 else 'Sell'
                active_positions[symbol] = {'time': latest_m5['Timestamp'], 'price': latest_m5['Close'], 'direction': direction}
                final_signals.append((symbol.replace('/USDT', ''), f"Enter {direction}", h1_time, latest_m5['Timestamp'], latest_m5['Close']))
                print(f"{symbol} M5 entry at {latest_m5['Timestamp']}")

            # Check H1 for exit if position exists
            if symbol in active_positions:
                active_entry = active_positions[symbol]
                h1_rows = h1_df[(h1_df['Timestamp'] <= run_time) & (h1_df['Timestamp'] >= active_entry['time'])]
                for j, h1_row in h1_rows.iterrows():
                    if (active_entry['direction'] == 'Buy' and h1_row['Trend'] == -1) or (active_entry['direction'] == 'Sell' and h1_row['Trend'] == 1):
                        profit = (h1_row['Close'] - active_entry['price']) / active_entry['price'] if active_entry['direction'] == 'Buy' else (active_entry['price'] - h1_row['Close']) / active_entry['price']
                        final_signals.append((symbol.replace('/USDT', ''), f"Exit {active_entry['direction']}", h1_time, h1_row['Timestamp'], active_entry['price'], h1_row['Close'], profit))
                        del active_positions[symbol]
                        print(f"{symbol} H1 exit at {h1_row['Timestamp']}")
                        break

    except Exception as e:
        error_msg = f"Error for {symbol}: {str(e)}"
        print(error_msg)
        final_signals.append((symbol.replace('/USDT', ''), f"Error: {str(e)}", run_time, run_time))

# Sort and filter last 12 hours
final_signals = sorted(final_signals, key=lambda x: x[3], reverse=True)
recent_signals = [sig for sig in final_signals if (run_time - sig[3]).total_seconds() <= 43200]  # 12 hours

# Send summary of last 20 signals
if recent_signals:
    message = f"🔔 UT Bot H1-M5 Signals (H1: KV={H1_KEY_VALUE}, ATR={H1_ATR_PERIOD} | M5: KV={M5_KEY_VALUE}, ATR={M5_ATR_PERIOD})\nRun at {run_time.strftime('%d-%b %H:%M:%S')}\n\n"
    message += f"{'Pair':<10} | {'Signal':<10} | {'H1 Time':<16} | {'M5 Time':<16} | {'Details'}\n"
    message += "-" * 80 + "\n"
    for sig in recent_signals[:20]:
        pair = sig[0]
        signal = sig[1]
        h1_time = sig[2].strftime('%d-%b %H:%M') if isinstance(sig[2], pd.Timestamp) else 'N/A'
        m5_time = sig[3].strftime('%d-%b %H:%M') if isinstance(sig[3], pd.Timestamp) else 'N/A'
        if 'Exit' in signal:
            entry_price, exit_price, profit = sig[4], sig[5], sig[6]
            details = f"Entry: {entry_price:.6f}, Exit: {exit_price:.6f}, Profit: {profit:.2%}"
        elif 'Enter' in signal:
            entry_price = sig[4]
            details = f"Price: {entry_price:.6f}"
        else:
            details = ''
        message += f"{pair:<10} | {signal:<10} | {h1_time:<16} | {m5_time:<16} | {details}\n"
    print(message)
    send_telegram_message(message)
else:
    print("No recent signals in the last 12 hours.")
    send_telegram_message(f"🔔 No UT Bot signals in the last 12 hours.\nRun at {run_time.strftime('%d-%b %H:%M:%S')}")

print(f"Script ended at {datetime.now(nairobi_zone).strftime('%d-%b %H:%M:%S')}")
