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
M15_ATR_PERIOD = 14
M15_KEY_VALUE = 1.5
M5_ATR_PERIOD = 10
M5_KEY_VALUE = 1.0

# Pairs to scan
symbols = ['WIF/USDT', 'DOGE/USDT', 'XRP/USDT', 'AI16Z/USDT', 'BERA/USDT', 'S/USDT', 'MOODENG/USDT']

exchange = ccxt.mexc({'enableRateLimit': True, 'timeout': 30000})

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=payload)
        print(f"Telegram response: {response.status_code}")
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

print(f"Starting script at {datetime.now(nairobi_zone).strftime('%d-%b %H:%M')}")

all_signals = []

for symbol in symbols:
    try:
        m15_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=100)
        m15_df = pd.DataFrame(m15_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        m15_df['Timestamp'] = pd.to_datetime(m15_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
        m15_df = ut_bot_alerts(m15_df, atr_period=M15_ATR_PERIOD, key_value=M15_KEY_VALUE)

        m5_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=100)
        m5_df = pd.DataFrame(m5_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        m5_df['Timestamp'] = pd.to_datetime(m5_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
        m5_df = ut_bot_alerts(m5_df, atr_period=M5_ATR_PERIOD, key_value=M5_KEY_VALUE)

        combined_df = m5_df[['Timestamp', 'Signal']].rename(columns={'Signal': 'M5_Signal'})
        m15_df_resampled = m15_df.set_index('Timestamp').resample('5min').ffill().reset_index()
        combined_df = combined_df.merge(m15_df_resampled[['Timestamp', 'Signal']], on='Timestamp', how='left')
        combined_df = combined_df.rename(columns={'Signal': 'M15_Signal'}).ffill()

        position = None
        symbol_signals = []  # Track signals per symbol

        for i in range(1, len(combined_df)):
            m15_signal = combined_df['M15_Signal'].iloc[i]
            m5_signal = combined_df['M5_Signal'].iloc[i]
            timestamp = combined_df['Timestamp'].iloc[i]

            if position is None:
                if m15_signal == 1 and m5_signal == 1:
                    position = 'Long'
                    signal = (symbol.replace('/USDT', ''), 'Enter Long', timestamp)
                    symbol_signals.append(signal)
                    send_telegram_message(f"{signal[0]} | {signal[1]} | {timestamp.strftime('%d-%b %H:%M')}")
                elif m15_signal == -1 and m5_signal == -1:
                    position = 'Short'
                    signal = (symbol.replace('/USDT', ''), 'Enter Short', timestamp)
                    symbol_signals.append(signal)
                    send_telegram_message(f"{signal[0]} | {signal[1]} | {timestamp.strftime('%d-%b %H:%M')}")
            elif position == 'Long' and m15_signal == -1:
                position = None
                signal = (symbol.replace('/USDT', ''), 'Exit Long', timestamp)
                symbol_signals.append(signal)
                send_telegram_message(f"{signal[0]} | {signal[1]} | {timestamp.strftime('%d-%b %H:%M')}")
            elif position == 'Short' and m15_signal == 1:
                position = None
                signal = (symbol.replace('/USDT', ''), 'Exit Short', timestamp)
                symbol_signals.append(signal)
                send_telegram_message(f"{signal[0]} | {signal[1]} | {timestamp.strftime('%d-%b %H:%M')}")

        # Keep last 7 signals for this symbol
        all_signals.extend(symbol_signals[-7:])

    except Exception as e:
        error_msg = f"Error for {symbol}: {str(e)}"
        print(error_msg)
        send_telegram_message(error_msg)

# Sort all signals by timestamp (youngest first)
all_signals = sorted(all_signals, key=lambda x: x[2], reverse=True)

# Send summary of last 7 signals if there are any new signals
if all_signals:
    message = f"🔔 Last 7 UT Bot Signals (M15: KV={M15_KEY_VALUE}, ATR={M15_ATR_PERIOD} | M5: KV={M5_KEY_VALUE}, ATR={M5_ATR_PERIOD})\n\n"
    message += f"{'Pair':<10} | {'Action':<12} | {'Time'}\n"
    message += "-" * 40 + "\n"
    for pair, action, time in all_signals[:7]:
        time_str = time.strftime('%d-%b %H:%M')
        message += f"{pair:<10} | {action:<12} | {time_str}\n"
    print(message)
    send_telegram_message(message)

print(f"Script ended at {datetime.now(nairobi_zone).strftime('%d-%b %H:%M')}")
