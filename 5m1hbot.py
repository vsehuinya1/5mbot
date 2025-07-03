{\rtf1\ansi\ansicpg1251\cocoartf2639
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;\f1\fnil\fcharset0 AppleColorEmoji;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import asyncio\
import aiohttp\
import pandas as pd\
import numpy as np\
from datetime import datetime, timedelta\
import logging\
import os\
import pytz\
import time\
import signal\
from telegram import Bot\
from telegram.request import HTTPXRequest\
import ssl\
import certifi\
from typing import Dict, Optional\
\
# --- ================================================================= --\
# --- CERBERUS TOP GAINERS - LIVE RUNNER (v2.1 - Let Winners Run)        --\
# --- Description: The final, stable version implementing the            --\
# --- validated "Let Winners Run" trailing stop exit strategy.           --\
# --- ================================================================= --\
\
# --- Basic Setup ---\
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',\
                    handlers=[logging.FileHandler("top_gainers_live.log", mode='w'), logging.StreamHandler()])\
logger = logging.getLogger("TopGainerLive")\
\
# --- CONFIGURATION ---\
QUALITY_WATCHLIST_SIZE = 25\
SCAN_INTERVAL = 300\
WATCHLIST_REFRESH_CYCLES = 12\
SYMBOL_COOLDOWN_HOURS = 4\
\
# --- Constants ---\
BINANCE_API_BASE, KLINES_ENDPOINT, TICKER_24H_ENDPOINT = "https://fapi.binance.com", "/fapi/v1/klines", "/fapi/v1/ticker/24hr"\
EXCHANGE_INFO_ENDPOINT = "/fapi/v1/exchangeInfo"\
LOOKBACK_15M_CANDLES, LOOKBACK_1H_CANDLES, MAX_SIGNAL_STALENESS_MINUTES = 200, 55, 15\
\
# --- Telegram Credentials ---\
TELEGRAM_TOKEN = "7207679435:AAFDD8_voRSiyT5XgQOHNhIKOBAhxIZ7YaA"\
TELEGRAM_CHAT_ID = "6707099301"\
\
# --- EXIT LOGIC (The "Let Winners Run" version from v1.7) ---\
class BreakoutExitLogic:\
    def __init__(self, risk_reduction_target_r=1.5, breakeven_target_r=2.5, trailing_stop_r_distance=1.5):\
        self.risk_reduction_target_r=risk_reduction_target_r; self.breakeven_target_r=breakeven_target_r; self.trailing_stop_r_distance=trailing_stop_r_distance\
        logger.info(f"Let Winners Run Logic Initialized: De-risk at \{risk_reduction_target_r\}R, Breakeven at \{breakeven_target_r\}R, Trail by \{trailing_stop_r_distance\}R.")\
    def _calculate_current_r_multiple(self, trade: Dict, current_price: float) -> float:\
        entry_price=trade['entry']; initial_risk=abs(entry_price-trade['initial_stop_loss'])\
        if initial_risk<=0: return 0\
        return (current_price-entry_price)/initial_risk\
    def manage_exit(self, trade: Dict, current_candle: pd.Series) -> Optional[Dict]:\
        entry_price,initial_sl,current_sl=trade['entry'],trade['initial_stop_loss'],trade['current_stop_loss']\
        initial_risk=abs(entry_price-initial_sl)\
        if initial_risk<=0: return None\
        if current_candle['low']<=current_sl:\
            reason="STOP LOSS"\
            if trade.get('is_trailing',False): reason="TRAILING STOP"\
            elif trade.get('is_breakeven',False): reason="BREAKEVEN STOP"\
            return self._close_trade(trade,current_sl,reason)\
        current_high=current_candle['high']; current_r=self._calculate_current_r_multiple(trade,current_high)\
        trade['highest_price']=max(trade.get('highest_price',entry_price),current_high)\
        if trade.get('is_breakeven',False):\
            new_sl=trade['highest_price']-(initial_risk*self.trailing_stop_r_distance)\
            if new_sl>current_sl: trade['current_stop_loss'],trade['is_trailing']=new_sl,True; return\{'status':'trailing_update','new_stop_loss':new_sl,'current_r':current_r\}\
        elif not trade.get('is_breakeven',False) and current_r>=self.breakeven_target_r:\
            new_sl=entry_price\
            if new_sl>current_sl: trade['current_stop_loss'],trade['is_breakeven']=new_sl,True; return\{'status':'breakeven_update','new_stop_loss':new_sl,'current_r':current_r\}\
        elif not trade.get('is_derisked',False) and current_r>=self.risk_reduction_target_r:\
            new_sl=entry_price-(initial_risk*0.25)\
            if new_sl>current_sl: trade['current_stop_loss'],trade['is_derisked']=new_sl,True; return\{'status':'derisk_update','new_stop_loss':new_sl,'current_r':current_r\}\
        return None\
    def _close_trade(self, trade: Dict, close_price: float, reason: str) -> Dict:\
        final_r=self._calculate_current_r_multiple(trade,close_price)\
        outcome="WIN" if final_r>0.05 else "LOSS" if final_r<-0.05 else "BREAKEVEN"\
        return \{**trade,'Result':outcome,'Close Price':close_price,'Final R Multiple':final_r,'Close Reason':reason\}\
\
# --- Signal Module (Unchanged) ---\
class RangeBreakoutModule:\
    def __init__(self, create_signal_callback): self._create_signal=create_signal_callback; self.lookback=50\
    def find_signal(self, df_15m, context):\
        if len(df_15m)<self.lookback+2: return None\
        retest_signal=self._find_retest(df_15m, context)\
        if retest_signal: return retest_signal\
        return self._find_breakout(df_15m)\
    def _check_breakout_confirmation(self,c,d): cr=c['high']-c['low']; return((c['close']-c['low'])/(cr or 1))>=0.7 if d=='long' else False\
    def _find_breakout(self,df):\
        breakout_candle=df.iloc[-2]; entry_candle=df.iloc[-1]\
        ldf=df.iloc[-(self.lookback+2):-2]\
        rh=ldf['high'].max(); volume_threshold=np.percentile(ldf['volume'],80)*1.5\
        if breakout_candle['close']>rh and breakout_candle['volume']>volume_threshold and self._check_breakout_confirmation(breakout_candle,'long'):\
            return self._create_signal(type="Range Breakout (Long)",entry=entry_candle['open'],stop_loss=breakout_candle['low'],candle_time=entry_candle.name,range_high=rh)\
        return None\
    def _find_retest(self,df,context):\
        if not context: return None\
        last,prev,atr=df.iloc[-1],df.iloc[-2],df.iloc[-1]['atr']; retest_zone_size=atr*0.3\
        if context['direction']=='long' and(range_high:=context.get('range_high')):\
            if range_high-retest_zone_size<=last['low']<=range_high+retest_zone_size and last['macd_hist']>0 and prev['macd_hist']<=0:\
                return self._create_signal(type="Breakout-Retest (Long)",entry=last['close'],stop_loss=last['low']-atr*0.1,candle_time=last.name)\
        return None\
\
# --- Strategy Class (Unchanged) ---\
class TopGainerStrategy:\
    def __init__(self):\
        self.name = "Cerberus Top Gainer"\
        self.active_trades, self.active_context = \{\}, \{\}\
        self.cooldown_until = \{\}\
        self.range_module = RangeBreakoutModule(lambda **kwargs: kwargs)\
        self.exit_logic = BreakoutExitLogic()\
\
    def process_new_signal(self, symbol, df_15m, df_1h, current_timestamp):\
        if symbol in self.cooldown_until and current_timestamp < self.cooldown_until[symbol]: return None\
        if symbol in self.active_trades: return None\
        if df_1h.empty: return None\
        latest_1h = df_1h.iloc[-1]\
        if latest_1h['close'] < latest_1h['ema_50']: return None\
        context = self.active_context.get(symbol)\
        signal = self.range_module.find_signal(df_15m, context)\
        if signal:\
            if 'Short' in signal.get('type', ''): return None\
            if signal['type'].startswith("Range Breakout"): self.active_context[symbol]=\{'timestamp':df_15m.index[-1],'direction':'long','range_high':signal.get('range_high')\}\
            elif signal['type'].startswith("Breakout-Retest") and symbol in self.active_context: del self.active_context[symbol]\
            return signal\
        return None\
\
    def manage_trade(self, symbol, df_15m, current_timestamp):\
        if symbol not in self.active_trades: return None\
        trade = self.active_trades[symbol]\
        exit_decision = self.exit_logic.manage_exit(trade, df_15m.iloc[-1])\
        if exit_decision:\
            if 'Result' in exit_decision:\
                del self.active_trades[symbol]\
                if symbol in self.active_context: del self.active_context[symbol]\
                self.cooldown_until[symbol] = current_timestamp + timedelta(hours=SYMBOL_COOLDOWN_HOURS)\
                return exit_decision\
            return exit_decision\
        return None\
\
# --- Main Live Scanner Class ---\
class MasterScanner:\
    def __init__(self):\
        self.session, self.all_pairs, self.watched_pairs, self.sent_alerts = None, set(), set(), \{\}\
        self.strategy = TopGainerStrategy()\
        self.telegram_semaphore = asyncio.Semaphore(5)\
        self.bot = None\
        if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:\
            self.bot = Bot(token=TELEGRAM_TOKEN, request=HTTPXRequest(pool_timeout=20.0, connect_timeout=15.0))\
            logger.info(f"Telegram Bot initialized for chat ID: \{TELEGRAM_CHAT_ID\}")\
        else:\
            logger.warning("Telegram credentials not found. Alerts will be disabled.")\
\
    async def initialize_session(self):\
        self.session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl.create_default_context(cafile=certifi.where())))\
        await self.fetch_all_futures_pairs()\
\
    async def close_session(self):\
        if self.session and not self.session.closed: await self.session.close()\
\
    async def _managed_request(self, url, params=None):\
        try:\
            async with self.session.get(url, params=params) as response:\
                response.raise_for_status(); return await response.json()\
        except Exception as e: logger.error(f"Request for \{url\} failed: \{e\}"); return None\
\
    async def fetch_all_futures_pairs(self):\
        data=await self._managed_request(f"\{BINANCE_API_BASE\}\{EXCHANGE_INFO_ENDPOINT\}")\
        if data and 'symbols' in data: self.all_pairs=\{s['symbol'] for s in data['symbols'] if s['contractType']=='PERPETUAL' and s['symbol'].endswith('USDT')\}\
        logger.info(f"Fetched \{len(self.all_pairs)\} USDT perpetual pairs.")\
\
    async def fetch_klines(self, symbol, interval, num_candles):\
        params = \{'symbol': symbol, 'interval': interval, 'limit': num_candles\}\
        try:\
            url = f"\{BINANCE_API_BASE\}\{KLINES_ENDPOINT\}"\
            async with self.session.get(url, params=params) as response:\
                response.raise_for_status(); data = await response.json()\
                if not data: logger.warning(f"No \{interval\} data for \{symbol\}."); return None\
                df = pd.DataFrame(data,columns=['ts','o','h','l','c','v','ct','qav','not','tbbav','tbqav','ig'])[['ts','o','h','l','c','v']]\
                df.rename(columns=\{'ts':'timestamp','o':'open','h':'high','l':'low','c':'close','v':'volume'\},inplace=True)\
                df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms',utc=True); df.set_index('timestamp',inplace=True)\
                for col in df.columns:df[col]=pd.to_numeric(df[col])\
                return df\
        except Exception as e: logger.error(f"Failed to fetch \{interval\} data for \{symbol\}: \{e\}"); return None\
\
    async def scan_for_significant_moves(self):\
        logger.info("Scanning for high-quality gainer setups...")\
        try:\
            tickers = await self._managed_request(f"\{BINANCE_API_BASE\}\{TICKER_24H_ENDPOINT\}")\
            if not tickers: return\
            df = pd.DataFrame([t for t in tickers if t.get('symbol') in self.all_pairs])\
            cols = ['quoteVolume', 'priceChangePercent', 'highPrice', 'lastPrice']\
            for col in cols: df[col] = pd.to_numeric(df[col], errors='coerce')\
            df.dropna(subset=cols, inplace=True)\
            gainers_df = df[df['priceChangePercent'] > 5].copy()\
            gainers_df['proximity_to_high'] = gainers_df['lastPrice'] / gainers_df['highPrice']\
            gainers_df['changeRank'] = gainers_df['priceChangePercent'].rank(pct=True)\
            gainers_df['volumeRank'] = gainers_df['quoteVolume'].rank(pct=True)\
            gainers_df['proximityRank'] = gainers_df['proximity_to_high'].rank(pct=True)\
            gainers_df['quality_score'] = (gainers_df['changeRank']*0.4 + gainers_df['volumeRank']*0.3 + gainers_df['proximityRank']*0.3)\
            top_quality_setups = gainers_df.sort_values('quality_score', ascending=False).head(QUALITY_WATCHLIST_SIZE)\
            self.watched_pairs = set(top_quality_setups['symbol'])\
            watched_list_str = ", ".join(sorted(list(self.watched_pairs)))\
            logger.info(f"Watchlist updated. Monitoring \{len(self.watched_pairs)\} symbols: \{watched_list_str\}")\
        except Exception as e: logger.error(f"Error scanning for quality moves: \{e\}", exc_info=True)\
\
    def escape_markdown_v2(self, text: str) -> str: return "".join([f'\\\\\{c\}' if c in r'_*[]()~`>#+-=|\{\}.!' else c for c in str(text)])\
\
    def calculate_margin_required(self, entry_price, stop_loss, risk_usd=10.0):\
        if not all([entry_price, stop_loss]): return "N/A"\
        price_difference = abs(entry_price - stop_loss)\
        if price_difference == 0: return "N/A"\
        margin = (risk_usd / price_difference) * entry_price\
        return f"$\{margin:.2f\}"\
\
    async def send_telegram_message(self, message, alert_id=None):\
        now=datetime.now(pytz.utc)\
        if not self.bot or (alert_id and (now - self.sent_alerts.get(alert_id,now-timedelta(hours=1))).total_seconds()<300):return\
        if alert_id:self.sent_alerts[alert_id]=now\
        async with self.telegram_semaphore:\
            try:await self.bot.send_message(chat_id=TELEGRAM_CHAT_ID,text=message,parse_mode='MarkdownV2');await asyncio.sleep(0.2)\
            except Exception as e:logger.error(f"Telegram send failed: \{e\}")\
\
    async def process_single_pair(self, symbol):\
        try:\
            df_15m = await self.fetch_klines(symbol, '15m', LOOKBACK_15M_CANDLES + 2)\
            df_1h = await self.fetch_klines(symbol, '1h', LOOKBACK_1H_CANDLES)\
            if df_15m is None or df_1h is None or len(df_15m) < LOOKBACK_15M_CANDLES + 2 or len(df_1h) < LOOKBACK_1H_CANDLES: return\
            \
            df_15m['atr'] = pd.concat([df_15m['high']-df_15m['low'], abs(df_15m['high']-df_15m['close'].shift()), abs(df_15m['low']-df_15m['close'].shift())], axis=1).max(axis=1).ewm(com=13, adjust=False).mean()\
            df_15m['macd_hist'] = df_15m['close'].ewm(span=12, adjust=False).mean() - df_15m['close'].ewm(span=26, adjust=False).mean()\
            df_1h['ema_50'] = df_1h['close'].ewm(span=50, adjust=False).mean()\
            \
            now_utc = datetime.now(pytz.utc)\
            trade_status = self.strategy.manage_trade(symbol, df_15m, now_utc)\
            if trade_status:\
                if 'Result' in trade_status:\
                    msg=(f"
\f1 \uc0\u55356 \u57263 
\f0  *TRADE CLOSED* \\\\- \{self.escape_markdown_v2(trade_status['Close Reason'])\}\\n\\n"\
                         f"*Symbol:* \\\\#\{self.escape_markdown_v2(symbol.replace('USDT',''))\}\\n"\
                         f"*Result:* *\{self.escape_markdown_v2(trade_status['Result'])\}* `\\\\(\{trade_status['Final R Multiple']:.2f\}R\\\\)`")\
                    await self.send_telegram_message(msg)\
                elif trade_status.get('status') == 'derisk_update':\
                    alert_id=f"derisk_\{symbol\}_\{trade_status['new_stop_loss']\}"\
                    msg=(f"
\f1 \uc0\u55357 \u57057 \u65039 
\f0  *RISK REDUCED* \\\\(to \\\\-0\\\\.25R\\\\)\\n\\n"\
                         f"*Symbol:* \\\\#\{self.escape_markdown_v2(symbol.replace('USDT',''))\}\\n"\
                         f"*New Stop Loss:* `\{trade_status['new_stop_loss']:.4f\}`")\
                    await self.send_telegram_message(msg,alert_id)\
                elif trade_status.get('status') == 'breakeven_update':\
                    alert_id=f"be_\{symbol\}_\{trade_status['new_stop_loss']\}"\
                    msg=(f"
\f1 \uc0\u9989 
\f0  *RISK FREE* \\\\(Moved to Breakeven\\\\)\\n\\n"\
                         f"*Symbol:* \\\\#\{self.escape_markdown_v2(symbol.replace('USDT',''))\}\\n"\
                         f"*New Stop Loss:* `\{trade_status['new_stop_loss']:.4f\}`")\
                    await self.send_telegram_message(msg,alert_id)\
                elif trade_status.get('status') == 'trailing_update':\
                     alert_id=f"tsl_\{symbol\}_\{trade_status['new_stop_loss']\}"\
                     msg=(f"
\f1 \uc0\u55357 \u56594 
\f0  *PROFIT LOCKED* \\\\(Trailing\\\\)\\n\\n"\
                          f"*Symbol:* \\\\#\{self.escape_markdown_v2(symbol.replace('USDT',''))\}\\n"\
                          f"*New Stop Loss:* `\{trade_status['new_stop_loss']:.4f\}`")\
                     await self.send_telegram_message(msg,alert_id)\
\
            signal = self.strategy.process_new_signal(symbol, df_15m, df_1h, now_utc)\
            if signal:\
                await self.format_and_send_entry_alert(symbol, signal)\
                trade=\{'direction': 'long', **signal,\
                       'initial_stop_loss': signal['stop_loss'], 'current_stop_loss': signal['stop_loss'],\
                       'is_derisked': False, 'is_breakeven': False, 'is_trailing': False,\
                       'highest_price': signal['entry']\}\
                self.strategy.active_trades[symbol] = trade\
        except Exception as e:logger.error(f"CRITICAL error processing \{symbol\}: \{e\}", exc_info=True)\
\
    async def format_and_send_entry_alert(self, symbol, signal):\
        now_utc = datetime.now(pytz.utc)\
        if (now_utc - signal['candle_time'].to_pydatetime().replace(tzinfo=pytz.utc)) > timedelta(minutes=MAX_SIGNAL_STALENESS_MINUTES):\
            logger.info(f"Stale signal for \{symbol\}. Ignoring."); return\
        alert_id = f"entry_\{symbol\}_\{signal['type']\}_\{signal['candle_time'].strftime('%Y%m%d%H%M')\}".replace(" ", "")\
        margin_required_str = self.calculate_margin_required(signal.get('entry'), signal.get('stop_loss'))\
        msg=(f"
\f1 \uc0\u55357 \u57000 
\f0  *NEW SIGNAL: \{self.escape_markdown_v2(self.strategy.name)\}*\\n\\n"\
             f"*Symbol:* \\\\#\{self.escape_markdown_v2(symbol.replace('USDT',''))\}\\n"\
             f"*Strategy:* \{self.escape_markdown_v2(signal['type'])\}\\n"\
             f"*Entry:* `\{signal.get('entry',0):.4f\}`\\n"\
             f"*Stop Loss:* `\{signal.get('stop_loss',0):.4f\}`\\n"\
             f"*MR \\\\($10 Risk\\\\):* `\{self.escape_markdown_v2(margin_required_str)\}`")\
        await self.send_telegram_message(msg,alert_id)\
\
    async def run(self):\
        await self.initialize_session()\
        await self.scan_for_significant_moves()\
        count=0\
        while True:\
            active_trade_symbols = list(self.strategy.active_trades.keys())\
            watched_symbols_str = ", ".join(sorted(list(self.watched_pairs)))\
            logger.info(f"--- Scan Cycle \{count\} | Watching \{len(self.watched_pairs)\} pairs: [\{watched_symbols_str\}] | Active Trades: \{len(active_trade_symbols)\} \{active_trade_symbols\} ---")\
            \
            start=time.time()\
            if count > 0 and count % WATCHLIST_REFRESH_CYCLES == 0: await self.scan_for_significant_moves()\
            await asyncio.gather(*(self.process_single_pair(s) for s in self.watched_pairs))\
            duration=time.time()-start\
            logger.info(f"Cycle \{count\} finished in \{duration:.2f\} seconds.")\
            await asyncio.sleep(max(0,SCAN_INTERVAL-duration))\
            count+=1\
\
async def main():\
    scanner=MasterScanner()\
    loop=asyncio.get_running_loop()\
    if os.name=='posix':\
        for sig in (signal.SIGINT,signal.SIGTERM):loop.add_signal_handler(sig,lambda:asyncio.create_task(shutdown(loop,scanner)))\
    try:await scanner.run()\
    except asyncio.CancelledError:pass\
\
async def shutdown(loop,scanner):\
    logger.info("Shutdown requested...");await scanner.close_session()\
    tasks=[t for t in asyncio.all_tasks() if t is not asyncio.current_task()]\
    [task.cancel() for task in tasks];await asyncio.gather(*tasks,return_exceptions=True)\
    loop.stop()\
\
if __name__=="__main__":\
    try:asyncio.run(main())\
    except(KeyboardInterrupt,SystemExit):logger.info("Application terminated.")}
