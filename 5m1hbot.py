import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import requests

TELEGRAM_TOKEN = '7207679435:AAFDD8_voRSiyT5XgQOHNhIKOBAhxIZ7YaA'
CHAT_ID = '6707099301'
nairobi_zone = pytz.timezone('Africa/Nairobi')
ATR_PERIOD = 10
KEY_VALUE = 1.2

# Full list of USDT pairs from your original script
symbols = [
    'SOL/USDT', 'BTC/USDT', 'BERA/USDT', 'VINE/USDT', 'AI16Z/USDT', 'HYPE/USDT',
    'PENGU/USDT', 'XRP/USDT', 'DOGE/USDT', 'GOAT/USDT', 'ACT/USDT', 'JUP/USDT',
    'MOODENG/USDT', 'FLOKI/USDT', 'PEPE/USDT', 'POPCAT/USDT', 'WIF/USDT',
    'BANANA/USDT', 'DOT/USDT', 'APE/USDT', 'SUI/USDT', 'AVAX/USDT', 'S/USDT',
    'LTC/USDT', 'JUP/USDT', 'ADA/USDT', 'TRX/USDT', 'TON/USDT', 'APT/USDT',
    'RENDER/USDT', 'LINK/USDT'
]

exchange = ccxt.mexc({
    'enableRateLimit': True,
    'timeout': 30000,
})

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': CHAT_ID, 'text': message, 'parse_mode': 'Markdown'}
    try:
        response = requests.post(url, data=payload)
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

final_signals = []
for symbol in symbols:
    try:
        h1_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=500)
        h1_df = pd.DataFrame(h1_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        h1_df['Timestamp'] = pd.to_datetime(h1_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
        h1_df = ut_bot_alerts(h1_df, atr_period=ATR_PERIOD, key_value=KEY_VALUE)
        last_h1 = h1_df[h1_df['Signal'] != 0].iloc[-1] if not h1_df[h1_df['Signal'] != 0].empty else None

        if last_h1 is not None:
            h1_signal = last_h1['Signal']
            h1_time = last_h1['Timestamp']
            m5_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=1000)
            m5_df = pd.DataFrame(m5_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            m5_df['Timestamp'] = pd.to_datetime(m5_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
            m5_df = ut_bot_alerts(m5_df, atr_period=ATR_PERIOD, key_value=KEY_VALUE)
            m5_signals = m5_df[(m5_df['Signal'] == h1_signal) & (m5_df['Timestamp'] > h1_time)]

            if not m5_signals.empty:
                m5_entry = m5_signals.iloc[0]['Timestamp']
                signal_type = 'Buy' if h1_signal == 1 else 'Sell'
                final_signals.append((symbol.replace('/USDT', ''), signal_type, h1_time, m5_entry))
    except Exception as e:
        full_error = f"Error for {symbol}: {type(e).__name__} - {str(e)}"
        print(full_error)
        final_signals.append((symbol.replace('/USDT', ''), full_error, None, None))

final_signals = sorted(final_signals, key=lambda x: x[3] if isinstance(x[3], pd.Timestamp) else datetime(1970, 1, 1, tzinfo=nairobi_zone), reverse=True)

message = f"🔔 *Confirmed UT Bot M5 Entries (KV={KEY_VALUE}, ATR={ATR_PERIOD})*\n\n"
message += f"{'Pair':<10} | {'Signal':<6} | {'H1 Time':<16} | {'M5 Entry Time'}\n"
message += "-" * 60 + "\n"
for pair, signal, h1_time, m5_time in final_signals:
    h1_str = h1_time.strftime('%d-%b %H:%M') if isinstance(h1_time, pd.Timestamp) else 'N/A'
    m5_str = m5_time.strftime('%d-%b %H:%M') if isinstance(m5_time, pd.Timestamp) else 'N/A'
    message += f"{pair:<10} | {signal:<6} | {h1_str:<16} | {m5_str}\n"

send_telegram_message(message)
print(message)
