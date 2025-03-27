import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# Nairobi timezone
nairobi_zone = pytz.timezone('Africa/Nairobi')

# UT Bot Parameters for H1 and M5
H1_ATR_PERIOD = 14
H1_KEY_VALUE = 1.5
M5_ATR_PERIOD = 10
M5_KEY_VALUE = 1.0

# USDT pairs to scan
symbols = [
    'SOL/USDT', 'BTC/USDT', 'BERA/USDT', 'VINE/USDT', 'AI16Z/USDT',
    'PENGU/USDT', 'XRP/USDT', 'DOGE/USDT', 'GOAT/USDT', 'ACT/USDT', 'JUP/USDT',
    'MOODENG/USDT', 'FLOKI/USDT', 'PEPE/USDT', 'POPCAT/USDT', 'WIF/USDT',
    'BANANA/USDT', 'DOT/USDT', 'APE/USDT', 'SUI/USDT', 'AVAX/USDT', 'S/USDT',
    'LTC/USDT', 'JUP/USDT', 'ADA/USDT', 'TRX/USDT', 'TON/USDT', 'APT/USDT',
    'RENDER/USDT', 'LINK/USDT'
]

# Initialize exchange (using MEXC to avoid Binance geo-restrictions)
exchange = ccxt.mexc({
    'enableRateLimit': True,
    'timeout': 30000,
})

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

print(f"Starting script at {datetime.now(nairobi_zone).strftime('%d-%b %H:%M')}")

for symbol in symbols:
    try:
        # Fetch H1 data with H1 parameters
        h1_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', limit=500)
        h1_df = pd.DataFrame(h1_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        h1_df['Timestamp'] = pd.to_datetime(h1_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
        h1_df = ut_bot_alerts(h1_df, atr_period=H1_ATR_PERIOD, key_value=H1_KEY_VALUE)
        last_h1 = h1_df[h1_df['Signal'] != 0].iloc[-1] if not h1_df[h1_df['Signal'] != 0].empty else None

        if last_h1 is not None:
            h1_signal = last_h1['Signal']
            h1_time = last_h1['Timestamp']

            # Fetch M5 data with M5 parameters
            m5_ohlcv = exchange.fetch_ohlcv(symbol, timeframe='5m', limit=1000)
            m5_df = pd.DataFrame(m5_ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            m5_df['Timestamp'] = pd.to_datetime(m5_df['Timestamp'], unit='ms', utc=True).dt.tz_convert(nairobi_zone)
            m5_df = ut_bot_alerts(m5_df, atr_period=M5_ATR_PERIOD, key_value=M5_KEY_VALUE)

            # Filter M5 signals after the H1 signal time
            m5_signals = m5_df[(m5_df['Signal'] == h1_signal) & (m5_df['Timestamp'] > h1_time)]

            if not m5_signals.empty:
                m5_entry = m5_signals.iloc[0]['Timestamp']
                signal_type = 'Buy' if h1_signal == 1 else 'Sell'
                final_signals.append((symbol.replace('/USDT', ''), signal_type, h1_time, m5_entry))
            else:
                print(f"No M5 confirmation for {symbol}")

        else:
            print(f"No H1 signal for {symbol}")

    except Exception as e:
        final_signals.append((symbol.replace('/USDT', ''), f"Error: {str(e)}", None, None))

# Sort by M5 entry timestamp (youngest at top)
final_signals = sorted(final_signals, key=lambda x: x[3] if isinstance(x[3], pd.Timestamp) else datetime(1970, 1, 1, tzinfo=nairobi_zone), reverse=True)

# Print formatted output
print(f"\n🔔 Confirmed UT Bot M5 Entries (H1: KV={H1_KEY_VALUE}, ATR={H1_ATR_PERIOD} | M5: KV={M5_KEY_VALUE}, ATR={M5_ATR_PERIOD})\n")
print(f"{'Pair':<10} | {'Signal':<6} | {'H1 Time':<16} | {'M5 Entry Time'}")
print("-" * 60)
for pair, signal, h1_time, m5_time in final_signals:
    h1_str = h1_time.strftime('%d-%b %H:%M') if isinstance(h1_time, pd.Timestamp) else 'N/A'
    m5_str = m5_time.strftime('%d-%b %H:%M') if isinstance(m5_time, pd.Timestamp) else 'N/A'
    print(f"{pair:<10} | {signal:<6} | {h1_str:<16} | {m5_str}")

print(f"Script ended at {datetime.now(nairobi_zone).strftime('%d-%b %H:%M')}")
