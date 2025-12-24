import requests
from datetime import datetime, timezone, timedelta
import csv
import os
import time
import numpy as np
import threading
from queue import Queue
import pandas as pd

API_TICKER_ENDPOINT = "https://api.bitget.com/api/v2/mix/market/tickers"
API_CANDLE_ENDPOINT = "https://api.bitget.com/api/v2/mix/market/history-candles"
NOTIFICATION_WEBHOOK = "https://api.primeautomation.ai/webhook/ChartPrime/52ff03d4-f744-4d9a-8f77-1e9791bb1731"

TRADING_PAIR = "UNIUSDT"
CONTRACT_TYPE = "USDT-FUTURES"
HISTORY_FILE = "uni_blofin_history.csv"

STARTING_CAPITAL = 10000
MULTIPLIER = 3
TIMEFRAME = "1H"
TRANSACTION_FEE = 0.0006
ATR_PERIOD = 14
FS_PERIOD = 10
RSI_PERIOD = 14

LONG_FS_THRESHOLD = 1.1
LONG_RSI_THRESHOLD = 28.0
LONG_SL_MULTIPLIER = 2.9
LONG_TP_MULTIPLIER = 1.1
LONG_TRAILING_SL_MULTIPLIER = 0.1

SHORT_FS_THRESHOLD = 4.7
SHORT_RSI_THRESHOLD = 1.0
SHORT_SL_MULTIPLIER = 2.9
SHORT_TP_MULTIPLIER = 2.0
SHORT_TRAILING_SL_MULTIPLIER = 0.1

class TradingEngine:
    def __init__(self):
        self.enable_notifications = True
        self.account_balance = STARTING_CAPITAL
        self.is_active = True
        self.execution_lock = threading.Lock()
        self.quote_lock = threading.Lock()
        self.ohlc_data = []
        self.fisher_series = []
        self.trigger_line = []
        self.trade_counter = 1
        self.trade_history = []
        self.latest_quote = None
        self.quote_queue = Queue()
        self.quote_update_event = threading.Event()
        self.active_long = None
        self.active_short = None
        self.atr_current = 0.0
        self.volume_osc = 0.0
        self.rsi_current = 0.0
        self.launch_date = datetime(2025, 12, 1, tzinfo=timezone.utc)
        
        self.latest_quote = self.fetch_latest_quote()
        if self.latest_quote is None:
            print("Initial price from REST: None")
        else:
            print(f"Initial price from REST: ${self.latest_quote:.4f}")
        
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as file_handle:
                    csv_reader = csv.DictReader(file_handle)
                    for record in csv_reader:
                        order_data = {
                            'entry_time': record['entry_time'],
                            'exit_time': record['exit_time'],
                            'action': record['action'],
                            'entry_price': float(record['entry_price']),
                            'exit_price': float(record['exit_price']) if record['exit_price'].strip() else None,
                            'size': float(record['size']),
                            'status': record['status'],
                            'pnl': float(record['pnl']) if record['pnl'] else 0.0,
                            'fee': float(record.get('fee', 0.0)),
                            'ideal_pnl': float(record.get('ideal_pnl', 0.0)),
                            'reason': record.get('reason', ''),
                            'stop_loss': float(record.get('stop_loss', 0.0)),
                            'take_profit': float(record.get('take_profit', 0.0)),
                            'max_profit_price': float(record.get('max_profit_price', 0.0)),
                            'trailing_stop_active': record.get('trailing_stop_active', 'False') == 'True',
                            'half_exit_done': record.get('half_exit_done', 'False') == 'True',
                            'original_size': float(record.get('original_size', float(record['size']))),
                        }
                        self.trade_history.append(order_data)
                        self.trade_counter = len(self.trade_history)
                        
                        if order_data['status'] == 'open':
                            if order_data['action'] == 'long':
                                self.active_long = order_data
                            elif order_data['action'] == 'short':
                                self.active_short = order_data
            except Exception as e:
                print(f"Error loading trades from CSV: {e}")

        threading.Thread(target=self.main_loop, daemon=True).start()
        threading.Thread(target=self.watch_positions, daemon=True).start()

    def fetch_latest_quote(self):
        try:
            request_params = {
                "symbol": TRADING_PAIR,
                "productType": CONTRACT_TYPE,
            }
            api_response = requests.get(API_TICKER_ENDPOINT, params=request_params, timeout=10)
            response_data = api_response.json()
            
            if response_data['code'] != '00000':
                print(f"API Error: {response_data['msg']}")
                return None
                
            for item in response_data['data']:
                if item['symbol'] == TRADING_PAIR:
                    quote = float(item['lastPr'])
                    print(f"Fetched price: ${quote:.4f}")
                    return quote
                    
            print(f"Current USDT symbol not found in response")
            return None
            
        except Exception as e:
            print(f"Price fetch error: {e}")
            return None

    def retrieve_ohlc(self, timeframe: str, limit: int):
        
        request_params = {
            "symbol": TRADING_PAIR,
            "productType": CONTRACT_TYPE,
            "granularity": timeframe,
            "limit": limit
        }
        try:
            api_response = requests.get(API_CANDLE_ENDPOINT, params=request_params, timeout=10)
            api_response.raise_for_status()
            response_data = api_response.json()
            
            if response_data['code'] != '00000':
                print(f"API Error: {response_data['msg']}")
                return None
                
            response_data = response_data.get("data", [])
            ohlc_list = []
            for candle_entry in response_data:
                candle = {
                    "timestamp": datetime.fromtimestamp(int(candle_entry[0]) / 1000, timezone.utc),
                    "open": float(candle_entry[1]),
                    "high": float(candle_entry[2]),
                    "low": float(candle_entry[3]),
                    "close": float(candle_entry[4]),
                    "volume": float(candle_entry[5]),
                    "quote_volume": float(candle_entry[6])
                }
                ohlc_list.append(candle)
            print(f"Fetched {len(ohlc_list)} {timeframe} candles")
            return ohlc_list
        except Exception as e:
            print(f"Error fetching candles: {e}")
            return None
        
    def compute_atr(self, dataframe, period=14):
        high_price = dataframe['high']
        low_price = dataframe['low']
        close_price = dataframe['close']
        tr1 = high_price - low_price
        tr2 = (high_price - close_price.shift(1)).abs()
        tr3 = (low_price - close_price.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        dataframe['atr'] = atr.fillna(0)
        return dataframe
    
    def compute_volume_osc(self, dataframe):
        dataframe = pd.DataFrame(dataframe)
        vol_ema5 = dataframe['volume'].ewm(span=5, adjust=False).mean()
        vol_ema10 = dataframe['volume'].ewm(span=10, adjust=False).mean()
        vol_osc = (vol_ema5 - vol_ema10) / vol_ema10 * 100
        dataframe['vol_os'] = vol_osc.fillna(0)
        return dataframe
    
    def compute_rsi(self, dataframe, period=RSI_PERIOD):
        price_change = dataframe['close'].diff()
        positive_change = (price_change.where(price_change > 0, 0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        negative_change = (-price_change.where(price_change < 0, 0)).ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        relative_strength = positive_change / negative_change
        dataframe['rsi'] = 100 - (100 / (1 + relative_strength))
        dataframe['rsi'] = dataframe['rsi'].fillna(50)
        return dataframe
    
    def compute_fisher(self, dataframe, period=FS_PERIOD):
        high_price = dataframe['high'].values
        low_price = dataframe['low'].values
        mid_price = (high_price + low_price) / 2
        fisher_value = np.zeros(len(mid_price))
        fs = np.zeros(len(mid_price))
        tr = np.zeros(len(mid_price))
        
        fisher_value[0] = 0
        fs[0] = 0
        tr[0] = 0
        
        for i in range(1, len(mid_price)):
            if i < period:
                lookback_window = mid_price[:i+1]
                window_high = np.max(lookback_window)
                window_low = np.min(lookback_window)
            else:
                lookback_window = mid_price[i-period+1:i+1]
                window_high = np.max(lookback_window)
                window_low = np.min(lookback_window)
            
            if window_high != window_low:
                computed_value = 0.33 * 2 * ((mid_price[i] - window_low)/(window_high - window_low) - 0.5) + 0.67 * fisher_value[i-1]
                computed_value = np.clip(computed_value, -0.999, 0.999)
                fisher_value[i] = computed_value
                fs[i] = 0.5 * np.log((1 + computed_value)/(1 - computed_value)) + 0.5 * fs[i-1]
                tr[i] = fs[i-1]
            else:
                fisher_value[i] = 0
                fs[i] = fs[i-1]
                tr[i] = tr[i-1]
        
        dataframe['fs'] = fs
        dataframe['tr'] = tr
        return dataframe
    
    def adjust_trailing_sl(self, current_quote, holding):
        if not holding:
            return
        
        target_price = holding['take_profit']

        if 'max_profit_price' not in holding:
            holding['max_profit_price'] = holding['entry_price']
            holding['trailing_stop_active'] = False

        was_updated = False

        if holding['action'] == 'long':
            if current_quote > holding['max_profit_price']:
                holding['max_profit_price'] = current_quote
                was_updated = True
                print(f"New max profit price for LONG: ${holding['max_profit_price']:.4f}")

            if not holding.get('half_exit_done', False) and current_quote >= target_price:
                self.partial_close(holding, current_quote)
                holding['trailing_stop_active'] = True
                was_updated = True

            if holding['trailing_stop_active']:
                trailing_sl = holding['max_profit_price'] - (self.atr_current * LONG_TRAILING_SL_MULTIPLIER)
                if trailing_sl > holding['stop_loss']:
                    holding['stop_loss'] = trailing_sl
                    was_updated = True
                    print(f"Updated stop loss for LONG: ${holding['stop_loss']:.4f}")

        else:
            if current_quote < holding['max_profit_price']:
                holding['max_profit_price'] = current_quote
                was_updated = True
                print(f"New max profit price for SHORT: ${holding['max_profit_price']:.4f}")

            if not holding.get('half_exit_done', False) and current_quote <= target_price:
                self.partial_close(holding, current_quote)
                holding['trailing_stop_active'] = True
                was_updated = True

            if holding['trailing_stop_active']:
                trailing_sl = holding['max_profit_price'] + (self.atr_current * SHORT_TRAILING_SL_MULTIPLIER)
                if trailing_sl < holding['stop_loss']:
                    holding['stop_loss'] = trailing_sl
                    was_updated = True
                    print(f"Updated stop loss for SHORT: ${holding['stop_loss']:.4f}")

        if was_updated:
            self.persist_trades()

    def partial_close(self, holding, current_quote):
        with self.execution_lock:
            if holding.get('half_exit_done', False):
                print(f"Half exit already executed for {holding['action'].upper()} position")
                return

            if 'original_size' not in holding:
                holding['original_size'] = holding['size']

            partial_quantity = holding['size'] / 2
            remaining_quantity = holding['size'] - partial_quantity

            if holding['action'] == 'long':
                profit_loss = (current_quote - holding['entry_price']) * partial_quantity
            else:
                profit_loss = (holding['entry_price'] - current_quote) * partial_quantity

            commission = (current_quote + holding['entry_price']) * TRANSACTION_FEE * partial_quantity
            net_profit = profit_loss - commission

            self.account_balance += net_profit

            partial_order = {
                'entry_time': holding['entry_time'],
                'exit_time': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                'action': holding['action'],
                'entry_price': holding['entry_price'],
                'exit_price': current_quote,
                'size': partial_quantity,
                'status': 'closed',
                'pnl': net_profit,
                'fee': commission,
                'ideal_pnl': profit_loss,
                'reason': 'half_exit_take_profit',
                'stop_loss': holding['stop_loss'],
                'take_profit': holding['take_profit'],
                'max_profit_price': holding.get('max_profit_price', holding['entry_price']),
                'trailing_stop_active': False,
                'half_exit_done': True,
                'original_size': holding['original_size']
            }

            holding['size'] = remaining_quantity
            holding['half_exit_done'] = True
            holding['trailing_stop_active'] = True

            self.trade_history.append(partial_order)

            self.notify_webhook(f'exit_{holding["action"]}', current_quote, partial_quantity, 50)

            print(f"Executed half exit for {holding['action']} position")

            self.persist_trades()
            self.broadcast_performance()

    def refresh_indicators(self):
        fresh_ohlc = self.retrieve_ohlc(TIMEFRAME, 200)
        if not fresh_ohlc:
            print("No new candles fetched. Using existing data.")
            return
            
        self.ohlc_data = sorted(fresh_ohlc, key=lambda x: x['timestamp'])
        print(f"Updating with {len(self.ohlc_data)} candles")

        dataframe = pd.DataFrame(self.ohlc_data)

        dataframe = self.compute_fisher(dataframe, FS_PERIOD)
        dataframe = self.compute_volume_osc(dataframe)
        dataframe = self.compute_atr(dataframe, ATR_PERIOD)
        dataframe = self.compute_rsi(dataframe, RSI_PERIOD)

        self.fisher_series = dataframe['fs'].tolist()
        self.trigger_line = dataframe['tr'].tolist()
        self.volume_osc = dataframe['vol_os'].iloc[-1]
        self.atr_current = dataframe['atr'].iloc[-1]
        self.rsi_current = dataframe['rsi'].iloc[-1]

        print(f"Updated FS: {self.fisher_series[-1]:.4f}, TR: {self.trigger_line[-1]:.4f}, VOL_OS: {self.volume_osc:.4f}, ATR_VAL: {self.atr_current:.4f}, RSI_VALUE: {self.rsi_current:.4f}")

    def evaluate_signals(self, timestamp: str):
        if len(self.fisher_series) < 3:
            print("Not enough data to check signals")
            return
        
        fisher_current = self.fisher_series[-1]
        trigger_current = self.trigger_line[-1]
        print(f"1H indicator values now fs:{fisher_current:.4f}, tr:{trigger_current:.4f}")
        
        fisher_previous = self.fisher_series[-2]
        trigger_previous = self.trigger_line[-2]
        print(f"1H indicator values prev fs:{fisher_previous:.4f}, tr:{trigger_previous:.4f}")

        bullish_cross = (fisher_previous < trigger_previous) and (fisher_current > trigger_current)
        bearish_cross = (fisher_previous > trigger_previous) and (fisher_current < trigger_current)

        current_quote = self.fetch_latest_quote()
        if current_quote is None:
            print("Cannot get current price, skipping signal check")
            return

        if bullish_cross:
            print(f"UP cross is detected!")
        if bearish_cross:
            print(f"DOWN cross is detected!")
        
        if bullish_cross:
            entry_condition = abs(self.rsi_current - 50) < LONG_RSI_THRESHOLD
            fs_condition = max(abs(trigger_current), abs(fisher_current)) < LONG_FS_THRESHOLD
            volume_condition = self.volume_osc > 0
            
            print(f"LONG Conditions - RSI: {entry_condition}, FS: {fs_condition}, VOL: {volume_condition}")
            
            if entry_condition and fs_condition and volume_condition:
                if self.active_long:
                    if current_quote < self.active_long['entry_price']:
                        self.exit_position('long', "Replace")
                        self.enter_position('long')
                    else:
                        pass
                else:
                    self.enter_position('long')
                    
        elif bearish_cross:
            entry_condition = abs(self.rsi_current - 50) > SHORT_RSI_THRESHOLD
            fs_condition = max(abs(trigger_current), abs(fisher_current)) < SHORT_FS_THRESHOLD
            volume_condition = self.volume_osc > 0
            
            print(f"SHORT Conditions - RSI: {entry_condition}, FS: {fs_condition}, VOL: {volume_condition}")
            
            if entry_condition and fs_condition and volume_condition:
                if self.active_short:
                    if current_quote > self.active_short['entry_price']:
                        self.exit_position('short', "Replace")
                        self.enter_position('short')
                    else:
                        pass
                else:
                    self.enter_position('short')

    def enter_position(self, side):
        with self.execution_lock:
            if side == 'long' and self.active_long:
                print("Cannot open LONG: an open LONG position already exists")
                return
            if side == 'short' and self.active_short:
                print("Cannot open SHORT: an open SHORT position already exists")
                return
        
            quote = self.fetch_latest_quote()
            if quote is None:
                print("Failed to fetch current price for open position")
                return
            
            quantity = round((6000 * MULTIPLIER) / quote, 4)

            timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

            if side == 'long':
                sl_multiplier = LONG_SL_MULTIPLIER
                tp_multiplier = LONG_TP_MULTIPLIER
            else:
                sl_multiplier = SHORT_SL_MULTIPLIER
                tp_multiplier = SHORT_TP_MULTIPLIER

            risk_amount = self.atr_current * sl_multiplier
            profit_target = self.atr_current * tp_multiplier

            if side == 'long':
                stop_price = quote - risk_amount
                target_price = quote + profit_target 

            else:
                stop_price = quote + risk_amount
                target_price = quote - profit_target

            order = {
                'entry_time': timestamp,
                'exit_time': None,
                'action': side,
                'entry_price': quote,
                'exit_price': None,
                'size': quantity,
                'status': 'open',
                'pnl': 0.0,
                'fee': 0.0,
                'ideal_pnl': 0.0,
                'reason': '',
                'stop_loss': stop_price,
                'take_profit': target_price,
                'max_profit_price': quote,
                'trailing_stop_active': False,
                'half_exit_done': False,
                'original_size': quantity
            }
            
            self.notify_webhook(side, quote, quantity)

            if side == 'long':
                self.active_long = order
            else:
                self.active_short = order

            self.trade_history.append(order)
            self.persist_trades()
            print(f"Opened {side.upper()} position at ${quote:.4f}, size: {quantity:.4f}, SL: ${stop_price:.4f}")

    def finalize_trade(self, order, current_quote, reason):
        with self.execution_lock:
            order['exit_time'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            order['exit_price'] = current_quote
            
            if order['action'] == 'long':
                order['pnl'] = (current_quote - order['entry_price']) * order['size']
            else:
                order['pnl'] = (order['entry_price'] - current_quote) * order['size']
                
            order['fee'] = (current_quote + order['entry_price']) * TRANSACTION_FEE * order['size']
            order['ideal_pnl'] = order['pnl']
            order['pnl'] -= order['fee']
            self.account_balance += order['pnl']
            order['status'] = 'closed'
            order['reason'] = reason
            
            if order['action'] == 'long':
                self.active_long = None
            else:
                self.active_short = None

            self.notify_webhook(f'exit_{order["action"]}', current_quote, order['size'], 100)
            self.persist_trades()
            
            print(f"Closed {order['action']} position")
            
            self.broadcast_performance()

    def exit_position(self, side, reason="signal"):
        if side == 'long' and self.active_long:
            holding = self.active_long
        elif side == 'short' and self.active_short:
            holding = self.active_short
        else:
            print(f"No open {side.upper()} positions to close 🤷‍♂️")
            return
            
        current_quote = self.fetch_latest_quote()
        if current_quote is None:
            print("Failed to fetch current price for closing positions")
            return

        self.finalize_trade(holding, current_quote, reason)

    def partial_exit(self, side):
        if side == 'long' and self.active_long:
            holding = self.active_long
        elif side == 'short' and self.active_short:
            holding = self.active_short
        else:
            print(f"No open {side.upper()} positions to close half")
            return

        if holding.get('half_exit_done', False):
            print(f"Half exit already executed for {side.upper()} position")
            return

        current_quote = self.fetch_latest_quote()
        if current_quote is None:
            print("Failed to fetch current price")
            return

        self.partial_close(holding, current_quote)

    def watch_positions(self):
        print("Starting real-time position monitoring (every 1s)...")
        while self.is_active:
            try:
                current_quote = self.fetch_latest_quote()
                if current_quote:
                    self.evaluate_risk(current_quote)
                else:
                    print("Skipping — price unavailable")
                
                time.sleep(1)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)

    def evaluate_risk(self, current_quote):
        
        if self.active_long:
            self.adjust_trailing_sl(current_quote, self.active_long)
        if self.active_short:
            self.adjust_trailing_sl(current_quote, self.active_short)

        if self.active_long:
            holding = self.active_long
            if current_quote <= holding['stop_loss']:
                print(f"Stop loss hit for LONG: ${current_quote:.4f} <= ${holding['stop_loss']:.4f}")
                self.finalize_trade(holding, current_quote, "stop_loss")
                return

        if self.active_short:
            holding = self.active_short
            if current_quote >= holding['stop_loss']:
                print(f"Stop loss hit for SHORT: ${current_quote:.4f} >= ${holding['stop_loss']:.4f}")
                self.finalize_trade(holding, current_quote, "stop_loss")
                return

    def broadcast_performance(self):
        if not self.trade_history:
            return

        dataframe = pd.DataFrame(self.trade_history)

        completed_trades = dataframe[dataframe['status'] == 'closed'].copy()

        last_position_size = completed_trades['size'].iloc[-1] if not completed_trades.empty else 0.0
        last_position_entry_price = completed_trades['entry_price'].iloc[-1] if not completed_trades.empty else 0.0
        last_position_usdt = last_position_size * last_position_entry_price

        if completed_trades.empty:
            return

        completed_trades['exit_time'] = pd.to_datetime(completed_trades['exit_time'], errors='coerce')
        completed_trades['date'] = completed_trades['exit_time'].dt.strftime('%m-%d')

        completed_trades['pnl'] = completed_trades['pnl'].astype(float)
        daily_performance = completed_trades.groupby('date')['pnl'].sum().reset_index()

        daily_performance = daily_performance.sort_values('date', ascending=False)

        today_str = datetime.now(timezone.utc).strftime('%m-%d')
        today_pnl = daily_performance[daily_performance['date'] == today_str]['pnl'].sum() if today_str in daily_performance['date'].values else 0.0

        trade_count = len(completed_trades)
        winning_trades = (completed_trades['pnl'] > 0).sum()
        win_rate = (winning_trades / trade_count) * 100 if trade_count > 0 else 0

        completed_trades_sorted = completed_trades.sort_values('exit_time')
        cumulative_profit = completed_trades_sorted['pnl'].cumsum()
        account_equity = STARTING_CAPITAL + cumulative_profit
        
        total_profit = completed_trades['pnl'].sum()
        total_return_pct = (total_profit / STARTING_CAPITAL) * 100 if STARTING_CAPITAL != 0 else 0.0

        peak_equity = account_equity.expanding().max()
        
        dd_percentage = (account_equity - peak_equity) / peak_equity * 100
        
        max_dd_pct = dd_percentage.min() if not dd_percentage.empty else 0

        daily_list = [f"{row.date}:{round(row.pnl,2)}" for row in daily_performance.itertuples()]
        daily_string = "[" + ", ".join(daily_list) + "]"

        last_profit = completed_trades['pnl'].iloc[-1]

        webhook_data = {
            "ticker": "UNI-USDT",
            "action": "stats",
            "time": datetime.now(timezone.utc).isoformat(),
            "PNL": str(round(total_return_pct, 2)),
            "LastPNL": str(round(last_profit, 2)),
            "MaxDraw": str(round(abs(max_dd_pct), 2)),
            "WinRate": str(round(win_rate, 2)),
            "DailyPNL": str(round(today_pnl, 2)),
            "Strategy": "20251201",
            "StartDate": self.launch_date.strftime("%b %d, %Y"),
            "historyPNL": daily_string,
            "TotalTrades": str(trade_count)
        }

        print(f"Sending daily PNL report: {webhook_data}")

        if self.enable_notifications:
            try:
                api_response = requests.post(NOTIFICATION_WEBHOOK, json=webhook_data, timeout=5)
                if api_response.ok:
                    print(f"Sent successfully ({api_response.status_code})")
                else:
                    print(f"Failed: {api_response.status_code} {api_response.reason}")
            except Exception as e:
                print(f"Error sending report: {e}")

    def notify_webhook(self, signal_type, quote, quantity=None, percentage=None):
        if signal_type in ["short", "long"]:
            webhook_data = {
                "time": datetime.now(timezone.utc).isoformat(),
                "ticker": TRADING_PAIR,
                "action": "sell" if signal_type == "short" else "buy",
                "price": str(round(quote, 5)),
                "OrderType": "market",
                "size": str(round(quantity, 4)) if quantity else "",
                "TotalTrades": str(self.trade_counter)
            }

            self.trade_counter += 1

        elif signal_type in ["exit_short", "exit_long"]:
            webhook_data = {
                "time": datetime.now(timezone.utc).isoformat(),
                "ticker": TRADING_PAIR,
                "action": "exit_sell" if signal_type == "exit_short" else "exit_buy",
                "price": str(round(quote, 5)),
                "OrderType": "market",
                "per": str(round(percentage, 2)) if percentage else "",
                "size": str(round(quantity, 4)) if quantity else "",
            }
            
        else:
            return
        
        print(f"Sending: {webhook_data}")

        if self.enable_notifications:
            try:
                api_response = requests.post(NOTIFICATION_WEBHOOK, json=webhook_data, timeout=5)
                if api_response.ok:
                    print(f"Success: {api_response.status_code}")
                else:
                    print(f"Failed: {api_response.status_code} {api_response.reason}")
            except Exception as err:
                print(f"Error: {err}")
            

    def persist_trades(self):
        try:
            with open(HISTORY_FILE, 'w', newline='') as file_handle:
                csv_writer = csv.writer(file_handle)
                csv_writer.writerow(['entry_time', 'exit_time', 'action', 'entry_price', 'exit_price', 
                               'size', 'status', 'fee', 'ideal_pnl', 'pnl', 'reason', 'stop_loss', 
                               'take_profit', 'max_profit_price', 'trailing_stop_active', 'half_exit_done', 'original_size'])
                for order in self.trade_history:
                    csv_writer.writerow([
                        order['entry_time'],
                        order['exit_time'],
                        order['action'],
                        round(order['entry_price'], 5),
                        round(order['exit_price'], 5) if order['exit_price'] else '',
                        round(order['size'], 2),
                        order['status'],
                        round(order['fee'], 2),
                        round(order['ideal_pnl'], 2),
                        round(order.get('pnl', 0), 2),
                        order['reason'],
                        round(order['stop_loss'], 5),
                        round(order['take_profit'], 5),
                        round(order.get('max_profit_price', 0), 5),
                        str(order.get('trailing_stop_active', False)),
                        str(order.get('half_exit_done', False)),
                        round(order.get('original_size', order['size']), 2)
                    ])
            print(f"Saved {len(self.trade_history)} trades to {HISTORY_FILE}")
        except Exception as e:
            print(f"Error saving trades: {e}")

    def handle_commands(self):
        print('Bot is running...')
        while self.is_active:
            try:
                command = input(">> ").lower().strip()

                if command == 'force open long':
                    self.enter_position('long')
                elif command == 'force open short':
                    self.enter_position('short')
                elif command == 'force close long':
                    self.exit_position('long')
                elif command == 'force close short':
                    self.exit_position('short')
                elif command == 'force half close long':
                    self.partial_exit('long')
                elif command == 'force half close short':
                    self.partial_exit('short')
                elif command == 'balance':
                    print(f"Current balance: ${self.account_balance:.4f}")
                elif command == 'price':
                    print(f"Current price: ${self.fetch_latest_quote():.4f}")
                elif command == 'exit':
                    self.is_active = False
                    print("Shutting down...")
                    break
                else:
                    print("Invalid command")
            except KeyboardInterrupt:
                self.is_active = False
                print("Shutting down...")
                break
            except Exception as e:
                print(f"Input error: {e}")

    def schedule_next_cycle(self):
        current_time = datetime.now(timezone.utc)
        hour_start = current_time.replace(minute=0, second=0, microsecond=0)

        next_cycle = hour_start + timedelta(hours=1, seconds=1)

        if next_cycle <= current_time:
            next_cycle += timedelta(hours=1)

        wait_duration = (next_cycle - current_time).total_seconds()
        print(f"Sleeping for {wait_duration:.4f} seconds until {next_cycle.strftime('%H:%M:%S')} UTC")
        time.sleep(wait_duration)

    def main_loop(self):
        print("Starting strategy loop")
        while self.is_active:
            try:
                timestamp = datetime.now(timezone.utc)
                print(f"\nRunning strategy check at {timestamp.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                self.refresh_indicators()
                self.evaluate_signals(timestamp.strftime('%Y-%m-%d %H:%M:%S'))
                self.schedule_next_cycle()
            except Exception as e:
                print(f"Error in strategy loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = TradingEngine()
    bot.handle_commands()