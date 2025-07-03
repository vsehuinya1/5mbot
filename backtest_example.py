import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import pytz
import csv
from typing import Dict, Optional

# --- ================================================================= --
# --- CERBERUS ENHANCED BACKTESTER v2.1 (Let Winners Run)                --
# --- Description: Backtests the new enhanced signal generation logic    --
# --- with the "Let Winners Run" trailing stop exit strategy.            --
# --- ================================================================= --

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("cerberus_enhanced_backtest.log", mode='w'), logging.StreamHandler()])
logger = logging.getLogger("CerberusEnhancedBacktester")

# --- CONFIGURATION ---
BACKTEST_START_DATE_STR = "2025-07-02 00:00:00"
HOURS_TO_BACKTEST = 24
SYMBOLS_TO_TEST = [
    'SPXUSDT', 'COOKIEUSDT', 'VIRTUALUSDT', 'BANANAUSDT', 'HIFIUSDT', 
    'GRAIUSDT', 'MOODENGUSDT', 'BIDUSDT', 'POPCATUSDT', 'AIXBTUSDT', 
    '1000000MOGUSDT', 'ILVUSDT', 'SUPERUSDT', 'CHILLGUYUSDT', 'HUSDT', 
    'ARPAUSDT', 'CHESSUSDT', 'PENGUUSDT', 'FIDAUSDT', 'TIAUSDT', 
    'BUSDT', 'NEARUSDT', 'JELLYUSDT', 'ADAUSDT', 'WLDUSDT', 'PLUMEUSDT'
]
SYMBOL_COOLDOWN_HOURS = 4

# --- Constants ---
BINANCE_API_BASE, KLINES_ENDPOINT = "https://fapi.binance.com", "/fapi/v1/klines"
LOOKBACK_15M_CANDLES, LOOKBACK_1H_CANDLES = 200, 55

# --- ================================================================= --
# --- ALL LOGIC CLASSES (Ported from your cerberus_enhanced.py)          --
# --- ================================================================= --

class RangeBreakoutModule:
    def __init__(self, create_signal_callback):
        self._create_signal = create_signal_callback
        self.lookback = 50  # Fixed lookback like original v2.1
        logger.info("🔧 BASELINE v2.1 Signal Module: 70% breakout + 1.5x volume")
    
    def find_signal(self, df_15m, context):
        if len(df_15m) < self.lookback + 2: return None
        retest_signal = self._find_retest(df_15m, context)
        if retest_signal: return retest_signal
        return self._find_breakout(df_15m)
    
    def _find_breakout(self, df):
        """Original simple breakout logic from baseline v2.1"""
        breakout_candle = df.iloc[-2]
        entry_candle = df.iloc[-1]
        
        # Get range data (fixed lookback)
        lookback_data = df.iloc[-(self.lookback + 2):-2]
        range_high = lookback_data['high'].max()
        
        # Original volume threshold (80th percentile * 1.5x)
        volume_threshold = np.percentile(lookback_data['volume'], 80) * 1.5
        
        # Simple breakout conditions:
        # 1. Close above range high
        # 2. Volume above threshold  
        # 3. 70% breakout confirmation
        if (breakout_candle['close'] > range_high and 
            breakout_candle['volume'] > volume_threshold and 
            self._check_breakout_confirmation(breakout_candle, 'long')):
            
            return self._create_signal(
                type="Range Breakout (Long)",
                entry=entry_candle['open'],  # Simple: next candle open
                stop_loss=breakout_candle['low'],  # Simple: breakout candle low
                candle_time=entry_candle.name,
                range_high=range_high
            )
        
        return None
    
    def _check_breakout_confirmation(self, candle, direction):
        """Original 70% breakout confirmation"""
        candle_range = candle['high'] - candle['low']
        if candle_range <= 0:
            return False
        
        if direction == 'long':
            close_position = (candle['close'] - candle['low']) / candle_range
            return close_position >= 0.7  # Original 70% threshold
        return False
    
    def _calculate_optimal_lookback(self,df):
        volatility=(df['atr'].iloc[-10:].mean())/(df['close'].iloc[-1]);
        if volatility>0.04: return self.lookback_min
        elif volatility<0.02: return self.lookback_max
        else: ratio=(0.04-volatility)/0.02; return int(self.lookback_min+ratio*(self.lookback_max-self.lookback_min))
    
    def _identify_range(self,df,lookback):
        range_data=df.iloc[-(lookback+2):-2]; resistance=range_data['high'].max(); support=range_data['low'].min()
        range_size=resistance-support; avg_atr=range_data['atr'].mean()
        if range_size<avg_atr*1.5: return None
        resistance_tests=sum(1 for high in range_data['high'] if abs(high-resistance)<=avg_atr*0.1)
        if resistance_tests<2: return None
        return {'resistance':resistance,'support':support,'range_size':range_size,'avg_atr':avg_atr,'tests':resistance_tests}
    
    def _validate_strong_breakout(self,candle,range_data):
        if candle['close']<=range_data['resistance']: return False
        candle_range=candle['high']-candle['low']
        if candle_range<=0: return False
        close_position=(candle['close']-candle['low'])/candle_range
        if close_position<self.STRONG_BREAKOUT_THRESHOLD: return False
        if candle['close']-range_data['resistance']<range_data['avg_atr']*0.1: return False
        return True
    
    def _validate_enhanced_volume(self,df,lookback):
        breakout_candle=df.iloc[-2]; volume_history=df.iloc[-(lookback+2):-2]['volume']; avg_volume=volume_history.mean()
        if avg_volume > 0 and breakout_candle['volume']/avg_volume<self.VOLUME_SURGE_MULTIPLIER: return False
        if len(df) > 16 and df.iloc[-8:-2]['volume'].mean()<=df.iloc[-16:-8]['volume'].mean(): return False
        if breakout_candle['volume']<np.percentile(volume_history,80)*1.3: return False
        return True
    
    def _validate_momentum_strength(self,df):
        if len(df) < 6: return False
        recent_macd=df.iloc[-5:]['macd_hist'];
        if recent_macd.iloc[-1]<=recent_macd.iloc[-3]: return False
        for i in range(1,self.MOMENTUM_PERIODS+1):
            if df.iloc[-i]['close']<=df.iloc[-(i+1)]['close']: return False
        if df.iloc[-1]['atr']<df.iloc[-15:-1]['atr'].mean()*1.05: return False
        return True
    
    def _detect_false_breakout(self,df,range_data):
        recent_data=df.iloc[-15:-2]; failed_breaks=0; resistance=range_data['resistance']
        for i in range(len(recent_data)-1):
            if recent_data.iloc[i]['high']>resistance and recent_data.iloc[i+1]['close']<resistance: failed_breaks+=1
        if failed_breaks>=2: return True
        breakout_candle=df.iloc[-2]; upper_wick=breakout_candle['high']-breakout_candle['close']; body_size=abs(breakout_candle['close']-breakout_candle['open'])
        if body_size > 0 and upper_wick>body_size*1.5: return True
        return False
    
    def _calculate_smart_levels(self,breakout_candle,entry_candle,range_data):
        resistance=range_data['resistance']; entry_buffer=range_data['avg_atr']*0.05; optimal_entry=resistance+entry_buffer
        if entry_candle['open']>resistance+(range_data['avg_atr']*0.25): optimal_entry=entry_candle['open']
        stop_loss=min(breakout_candle['low'],range_data['support'])-(range_data['avg_atr']*0.1)
        return optimal_entry,stop_loss
    
    def _calculate_confidence(self,df,range_data):
        score=0; volume_ratio=df.iloc[-2]['volume']/(df.iloc[-20:-2]['volume'].mean()or 1); score+=min(25,volume_ratio*8)
        score+=min(1.0,range_data['tests']/4)*25
        macd_recent=df.iloc[-5:]['macd_hist']
        if len(macd_recent)>=5: score+=min(1.0,max(0,(macd_recent.iloc[-1]-macd_recent.iloc[-5])/(abs(macd_recent.iloc[-5])or 1)))*25
        breakout_candle=df.iloc[-2]; candle_range=breakout_candle['high']-breakout_candle['low']
        if candle_range>0: score+=((breakout_candle['close']-breakout_candle['low'])/candle_range)*25
        return int(min(100,score))
    
    def _find_retest(self, df, context):
        """Original simple retest logic"""
        if not context: return None
        last = df.iloc[-1]; prev = df.iloc[-2]; atr = df.iloc[-1]['atr']; retest_zone_size = atr * 0.3
        if context['direction'] == 'long':
            range_high = context.get('range_high')
            if range_high:
                if (range_high - retest_zone_size <= last['low'] <= range_high + retest_zone_size):
                    if last['macd_hist'] > 0 and prev['macd_hist'] <= 0:
                        return self._create_signal(type="Breakout-Retest (Long)", entry=last['close'], stop_loss=last['low'] - atr * 0.1, candle_time=last.name)
        return None

class BreakoutExitLogic:
    def __init__(self, risk_reduction_target_r=1.5, breakeven_target_r=2.5, trailing_stop_r_distance=1.5):
        self.risk_reduction_target_r=risk_reduction_target_r; self.breakeven_target_r=breakeven_target_r; self.trailing_stop_r_distance=trailing_stop_r_distance
        logger.info(f"Let Winners Run Logic: De-risk at {risk_reduction_target_r}R, Breakeven at {breakeven_target_r}R, Trail by {trailing_stop_r_distance}R")
    def _calculate_current_r_multiple(self,trade,current_price):
        entry_price=trade['entry']; initial_risk=abs(entry_price-trade['initial_stop_loss'])
        return (current_price-entry_price)/initial_risk if initial_risk>0 else 0
    def manage_exit(self,trade,current_candle):
        entry_price,initial_sl,current_sl=trade['entry'],trade['initial_stop_loss'],trade['current_stop_loss']
        initial_risk=abs(entry_price-initial_sl)
        if initial_risk<=0: return None
        if current_candle['low']<=current_sl:
            reason="STOP LOSS"
            if trade.get('is_trailing',False): reason="TRAILING STOP"
            elif trade.get('is_breakeven',False): reason="BREAKEVEN STOP"
            return self._close_trade(trade,current_sl,reason)
        current_high=current_candle['high']; current_r=self._calculate_current_r_multiple(trade,current_high)
        trade['highest_price']=max(trade.get('highest_price',entry_price),current_high)
        if trade.get('is_breakeven',False):
            new_sl=trade['highest_price']-(initial_risk*self.trailing_stop_r_distance)
            if new_sl>current_sl: trade['current_stop_loss'],trade['is_trailing']=new_sl,True; return{'status':'trailing_update','new_stop_loss':new_sl,'current_r':current_r}
        elif not trade.get('is_breakeven',False) and current_r>=self.breakeven_target_r:
            new_sl=entry_price
            if new_sl>current_sl: trade['current_stop_loss'],trade['is_breakeven']=new_sl,True; return{'status':'breakeven_update','new_stop_loss':new_sl,'current_r':current_r}
        elif not trade.get('is_derisked',False) and current_r>=self.risk_reduction_target_r:
            new_sl=entry_price-(initial_risk*0.25)
            if new_sl>current_sl: trade['current_stop_loss'],trade['is_derisked']=new_sl,True; return{'status':'derisk_update','new_stop_loss':new_sl,'current_r':current_r}
        return None
    def _close_trade(self,trade,close_price,reason):
        final_r=self._calculate_current_r_multiple(trade,close_price)
        outcome="WIN" if final_r>0.05 else "LOSS" if final_r<-0.05 else "BREAKEVEN"
        return {**trade,'Result':outcome,'Close Price':close_price,'Final R Multiple':final_r,'Close Reason':reason}

class TopGainerStrategy:
    def __init__(self):
        self.name="Cerberus Top Gainer"; self.active_trades={}; self.active_context={}; self.cooldown_until={}
        self.range_module=RangeBreakoutModule(lambda **kwargs:kwargs)
        self.exit_logic=BreakoutExitLogic()
        logger.info("🚀 BASELINE v2.1 strategy initialized")
    def process_new_signal(self,symbol,df_15m,df_1h,current_timestamp):
        if symbol in self.cooldown_until and current_timestamp<self.cooldown_until[symbol]: return None
        if symbol in self.active_trades: return None
        if df_1h.empty: return None
        if not self._validate_higher_timeframe(df_1h): return None
        context=self.active_context.get(symbol); signal=self.range_module.find_signal(df_15m,context)
        if signal:
            if 'Short' in signal.get('type',''): return None
            logger.info(f"✅ ENHANCED ENTRY FOUND for {symbol} at {signal['entry']:.4f} (Confidence: {signal.get('confidence', 'N/A')}%)")
            if signal['type'].startswith("Enhanced Range Breakout"): self.active_context[symbol]={'timestamp':df_15m.index[-1],'direction':'long','range_high':signal.get('range_high')}
            elif signal['type'].startswith("Enhanced Breakout-Retest") and symbol in self.active_context: del self.active_context[symbol]
            return signal
        return None
    def _validate_higher_timeframe(self,df_1h):
        if len(df_1h)<10: return False
        latest=df_1h.iloc[-1]
        if latest['close']<latest['ema_50']: return False
        if len(df_1h) > 3 and (latest['ema_50']-df_1h.iloc[-3]['ema_50'])/3<=0: return False
        if len(df_1h) > 2 and (latest['close']-df_1h.iloc[-2]['close'])/df_1h.iloc[-2]['close']<-0.015: return False
        return True
    def manage_trade(self,symbol,df_15m,current_timestamp):
        if symbol not in self.active_trades: return None
        trade=self.active_trades[symbol]; exit_decision=self.exit_logic.manage_exit(trade,df_15m.iloc[-1])
        if exit_decision:
            if 'Result' in exit_decision:
                del self.active_trades[symbol]
                if symbol in self.active_context: del self.active_context[symbol]
                self.cooldown_until[symbol]=current_timestamp+timedelta(hours=SYMBOL_COOLDOWN_HOURS)
                logger.info(f"🧊 PLACING {symbol} on cooldown until {self.cooldown_until[symbol]}")
                return exit_decision
            if exit_decision.get('status')=='derisk_update': logger.info(f"🛡️ RISK REDUCED on {symbol} at {exit_decision['current_r']:.2f}R. New SL: {exit_decision['new_stop_loss']:.4f}")
            elif exit_decision.get('status')=='breakeven_update': logger.info(f"✅ RISK FREE on {symbol} at {exit_decision['current_r']:.2f}R. New SL: {exit_decision['new_stop_loss']:.4f}")
            elif exit_decision.get('status')=='trailing_update': logger.info(f"🔒 PROFIT LOCKED on {symbol} at {exit_decision['current_r']:.2f}R. New SL: {exit_decision['new_stop_loss']:.4f}")
        return None

# --- Main Backtester Class ---
class EnhancedBacktester:
    def __init__(self, start_date_str, hours_to_backtest, symbols_to_test):
        self.start_date=pd.to_datetime(start_date_str,utc=True); self.end_date=self.start_date+timedelta(hours=hours_to_backtest)
        self.symbols_to_test=symbols_to_test; self.historical_data={}; self.trade_log=[]
        logger.info(f"Backtester initialized for period: {self.start_date} to {self.end_date}")

    async def setup_data(self):
        logger.info("Fetching historical data for backtest...")
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*(self._fetch_symbol_data(symbol,session) for symbol in self.symbols_to_test))
        logger.info("Data setup and indicator calculation complete.")

    async def _fetch_symbol_data(self,symbol,session):
        self.historical_data[symbol]={}
        fetch_start_time_15m=self.start_date-timedelta(hours=(LOOKBACK_15M_CANDLES//4)+20)
        fetch_start_time_1h=self.start_date-timedelta(hours=LOOKBACK_1H_CANDLES+20)
        df_15m=await self._fetch_klines(session,symbol,'15m',fetch_start_time_15m)
        if df_15m is not None:
            df_15m['atr']=pd.concat([df_15m['high']-df_15m['low'],abs(df_15m['high']-df_15m['close'].shift()),abs(df_15m['low']-df_15m['close'].shift())],axis=1).max(axis=1).ewm(com=13,adjust=False).mean()
            df_15m['macd_hist']=df_15m['close'].ewm(span=12,adjust=False).mean()-df_15m['close'].ewm(span=26,adjust=False).mean()
            self.historical_data[symbol]['15m']=df_15m.dropna()
        df_1h=await self._fetch_klines(session,symbol,'1h',fetch_start_time_1h)
        if df_1h is not None:
            df_1h['ema_50']=df_1h['close'].ewm(span=50,adjust=False).mean()
            self.historical_data[symbol]['1h']=df_1h.dropna()

    async def _fetch_klines(self,session,symbol,interval,start_time):
        params={'symbol':symbol,'interval':interval,'startTime':int(start_time.timestamp()*1000),'limit':1500}
        try:
            url=f"{BINANCE_API_BASE}{KLINES_ENDPOINT}"
            async with session.get(url,params=params) as response:
                response.raise_for_status();data=await response.json()
                if not data: logger.warning(f"No {interval} data for {symbol}."); return None
                df=pd.DataFrame(data,columns=['ts','o','h','l','c','v','ct','qav','not','tbbav','tbqav','ig'])[['ts','o','h','l','c','v']]
                df.rename(columns={'ts':'timestamp','o':'open','h':'high','l':'low','c':'close','v':'volume'},inplace=True)
                df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms',utc=True);df.set_index('timestamp',inplace=True)
                for col in df.columns:df[col]=pd.to_numeric(df[col])
                return df
        except Exception as e: logger.error(f"Failed to fetch {interval} data for {symbol}: {e}"); return None

    def run(self):
        logger.info("--- Starting BASELINE v2.1 Backtest Simulation ---")
        strategy=TopGainerStrategy()
        time_index=pd.date_range(start=self.start_date,end=self.end_date,freq='15min',tz='UTC')
        for timestamp in time_index:
            for symbol in self.symbols_to_test:
                if '15m' not in self.historical_data.get(symbol,{}) or '1h' not in self.historical_data.get(symbol,{}): continue
                df_15m_all=self.historical_data[symbol]['15m'];df_1h_all=self.historical_data[symbol]['1h']
                if timestamp not in df_15m_all.index: continue
                df_15m_slice=df_15m_all.loc[:timestamp];df_1h_slice=df_1h_all.loc[:timestamp]
                if len(df_15m_slice)<LOOKBACK_15M_CANDLES or len(df_1h_slice)<LOOKBACK_1H_CANDLES: continue
                
                closed_trade=strategy.manage_trade(symbol,df_15m_slice,timestamp)
                if closed_trade: self.trade_log.append({**closed_trade,'symbol':symbol})

                new_signal=strategy.process_new_signal(symbol,df_15m_slice,df_1h_slice,timestamp)
                if new_signal:
                    trade={'direction':'long',**new_signal,
                             'initial_stop_loss':new_signal['stop_loss'],
                             'current_stop_loss':new_signal['stop_loss'],
                             'is_derisked':False,'is_breakeven':False,
                             'is_trailing':False,'highest_price':new_signal['entry']}
                    strategy.active_trades[symbol]=trade
        logger.info("--- Backtest Simulation Finished ---")
        self.generate_report()

    def generate_report(self):
        if not self.trade_log: logger.info("\n--- BACKTEST REPORT ---\nNo trades were executed."); return
        report=["\n--- BACKTEST REPORT ---"]; df_log=pd.DataFrame(self.trade_log)
        total_trades=len(df_log); wins=len(df_log[df_log['Result']=='WIN']); losses=len(df_log[df_log['Result']=='LOSS']); breakeven=total_trades-wins-losses
        win_rate=(wins/(wins+losses)*100) if (wins+losses)>0 else 0
        total_r=df_log['Final R Multiple'].sum(); avg_r=df_log['Final R Multiple'].mean() if total_trades>0 else 0
        report.extend([f"Period: {self.start_date} to {self.end_date}",f"Symbols Tested: {len(self.symbols_to_test)}",
                       "\n--- Overall Performance ---",f"Total Trades: {total_trades}",f"Wins: {wins} | Losses: {losses} | Breakeven: {breakeven}",
                       f"Win Rate: {win_rate:.2f}%",f"Total R-Multiple: {total_r:.2f}R",f"Average R per Trade: {avg_r:.3f}R",
                       "\n--- Trade Log ---"]); print("\n".join(report))
        if not df_log.empty:
            df_display=df_log[['symbol','candle_time','type','Result','Final R Multiple','Close Reason']]
            print(df_display.to_string())
            df_log.to_csv('cerberus_enhanced_trade_log.csv',index=False)
            logger.info("Detailed trade log saved to cerberus_enhanced_trade_log.csv")

async def main():
    backtester=EnhancedBacktester(BACKTEST_START_DATE_STR,HOURS_TO_BACKTEST,SYMBOLS_TO_TEST)
    await backtester.setup_data()
    backtester.run()

if __name__=="__main__":
    asyncio.run(main())
