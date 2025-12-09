import requests
from datetime import datetime, timezone, timedelta
import csv
import os
import time
import numpy as np
import threading
from queue import Queue
import pandas as pd

# ========== Constants ==========
BITGET_TICKERS_URL = "https://api.bitget.com/api/v2/mix/market/tickers"
BITGET_CANDLE_URL = "https://api.bitget.com/api/v2/mix/market/history-candles"
WEBHOOK_URL = "https://api.primeautomation.ai/webhook/ChartPrime/52ff03d4-f744-4d9a-8f77-1e9791bb1731"

UNIUSDT_SYMBOL = "UNIUSDT"
PRODUCT_TYPE = "USDT-FUTURES"
CSV_FILE = "uni_blofin_history.csv"

# ========== Trading Configuration ==========
INITIAL_BALANCE = 10000
LEVERAGE = 3
# POSITION_SIZE_RATIO = 0.4
INTERVAL = "1H"
FEE_PERCENT = 0.0006
ATR_LENGTH = 14
FS_LENGTH = 10
RSI_LENGTH = 14

LONG_FS_ENTRY_LEVEL = 1.1
LONG_RSI_ENTRY_LEVEL = 28.0
LONG_STOP_LOSS_LEVEL = 2.9
LONG_TAKE_PROFIT_LEVEL = 1.1
LONG_SECOND_SL_LEVEL = 0.1

SHORT_FS_ENTRY_LEVEL = 4.7
SHORT_RSI_ENTRY_LEVEL = 1.0
SHORT_STOP_LOSS_LEVEL = 2.9
SHORT_TAKE_PROFIT_LEVEL = 2.0
SHORT_SECOND_SL_LEVEL = 0.1

class AutoTradeBot:
    def __init__(self):
        self.flag_webhook_sent = True
        self.balance = INITIAL_BALANCE
        self.running = True
        self.trade_lock = threading.Lock()
        self.price_lock = threading.Lock()
        self.candles = []
        self.fs = []
        self.tr = []
        self.total_trades = 1
        self.trades = []
        self.current_price = None
        self.price_queue = Queue()
        self.price_update_event = threading.Event()
        self.current_long_position = None
        self.current_short_position = None
        self.current_atr_value = 0.0
        self.current_vol_os = 0.0
        self.current_rsi_value = 0.0
        self.start_date = datetime(2025, 12, 1, tzinfo=timezone.utc)
        
        # Initialize current price via REST API
        self.current_price = self.get_current_price()
        if self.current_price is None:
            print("[INIT] Initial price from REST: None")
        else:
            print(f"[INIT] Initial price from REST: ${self.current_price:.4f}")
        
        # Load existing trades from CSV
        if os.path.exists(CSV_FILE):
            try:
                with open(CSV_FILE, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        trade_data = {
                            'entry_time': row['entry_time'],
                            'exit_time': row['exit_time'],
                            'action': row['action'],
                            'entry_price': float(row['entry_price']),
                            'exit_price': float(row['exit_price']) if row['exit_price'].strip() else None,
                            'size': float(row['size']),
                            'status': row['status'],
                            'pnl': float(row['pnl']) if row['pnl'] else 0.0,
                            'fee': float(row.get('fee', 0.0)),
                            'ideal_pnl': float(row.get('ideal_pnl', 0.0)),
                            'reason': row.get('reason', ''),
                            'stop_loss': float(row.get('stop_loss', 0.0)),
                            'take_profit': float(row.get('take_profit', 0.0)),
                            'max_profit_price': float(row.get('max_profit_price', 0.0)),
                            'trailing_stop_active': row.get('trailing_stop_active', 'False') == 'True',
                            'half_exit_done': row.get('half_exit_done', 'False') == 'True',
                            'original_size': float(row.get('original_size', float(row['size']))),
                        }
                        self.trades.append(trade_data)
                        self.total_trades = len(self.trades)
                        
                        # Restore open positions
                        if trade_data['status'] == 'open':
                            if trade_data['action'] == 'long':
                                self.current_long_position = trade_data
                            elif trade_data['action'] == 'short':
                                self.current_short_position = trade_data
            except Exception as e:
                print(f"[INIT] Error loading trades from CSV: {e}")

        # Start background threads
        threading.Thread(target=self.strategy_loop, daemon=True).start()
        threading.Thread(target=self.monitor_positions, daemon=True).start()

    # ========== Data Fetching & Indicators ==========
    def get_current_price(self):
        try:
            params = {
                "symbol": UNIUSDT_SYMBOL,
                "productType": PRODUCT_TYPE,
            }
            response = requests.get(BITGET_TICKERS_URL, params=params, timeout=10)
            data = response.json()
            
            if data['code'] != '00000':
                print(f"[REST] API Error: {data['msg']}")
                return None
                
            # Find SOLUSDT ticker
            for ticker in data['data']:
                if ticker['symbol'] == UNIUSDT_SYMBOL:
                    price = float(ticker['lastPr'])
                    print(f"[REST] Fetched price: ${price:.4f}")
                    return price
                    
            print(f"[REST] SOLUSDT symbol not found in response")
            return None
            
        except Exception as e:
            print(f"[REST] Price fetch error: {e}")
            return None

    def fetch_candles(self, interval: str, limit: int):
        
        params = {
            "symbol": UNIUSDT_SYMBOL,
            "productType": PRODUCT_TYPE,
            "granularity": interval,
            "limit": limit
        }
        try:
            response = requests.get(BITGET_CANDLE_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] != '00000':
                print(f"[DATA] API Error: {data['msg']}")
                return None
                
            data = data.get("data", [])
            candles = []
            for entry in data:
                candle = {
                    "timestamp": datetime.fromtimestamp(int(entry[0]) / 1000, timezone.utc),
                    "open": float(entry[1]),
                    "high": float(entry[2]),
                    "low": float(entry[3]),
                    "close": float(entry[4]),
                    "volume": float(entry[5]),
                    "quote_volume": float(entry[6])
                }
                candles.append(candle)
            print(f"[DATA] Fetched {len(candles)} {interval} candles")
            return candles
        except Exception as e:
            print(f"[DATA] Error fetching candles: {e}")
            return None
        
    def calculate_atr(self, df, atr_length=14):
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        # Wilder's moving average (RMA) uses ewm with alpha = 1/period
        atr = tr.ewm(alpha=1/atr_length, adjust=False, min_periods=atr_length).mean()
        df['atr'] = atr.fillna(0)
        return df
    
    def calculate_volume_oscillator(self, df):
        df = pd.DataFrame(df)
        vol_ema5 = df['volume'].ewm(span=5, adjust=False).mean()
        vol_ema10 = df['volume'].ewm(span=10, adjust=False).mean()
        vol_osc = (vol_ema5 - vol_ema10) / vol_ema10 * 100
        df['vol_os'] = vol_osc.fillna(0)
        return df
    
    # ========== RSI Calculation ==========
    def calculate_rsi(self, df, rsi_length=RSI_LENGTH):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/rsi_length, adjust=False, min_periods=rsi_length).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].fillna(50)
        return df
    
    def calculate_fs(self, df, fs_length=FS_LENGTH):
        high = df['high'].values
        low = df['low'].values
        median = (high + low) / 2
        value = np.zeros(len(median))
        fs = np.zeros(len(median))
        tr = np.zeros(len(median))
        
        # Initialize first values
        value[0] = 0
        fs[0] = 0
        tr[0] = 0
        
        for i in range(1, len(median)):
            if i < fs_length:
                # Not enough data for full window, use simple calculation
                window = median[:i+1]
                max_h = np.max(window)
                min_l = np.min(window)
            else:
                window = median[i-fs_length+1:i+1]
                max_h = np.max(window)
                min_l = np.min(window)
            
            if max_h != min_l:
                val = 0.33 * 2 * ((median[i] - min_l)/(max_h - min_l) - 0.5) + 0.67 * value[i-1]
                val = np.clip(val, -0.999, 0.999)
                value[i] = val
                fs[i] = 0.5 * np.log((1 + val)/(1 - val)) + 0.5 * fs[i-1]
                tr[i] = fs[i-1]
            else:
                value[i] = 0
                fs[i] = fs[i-1]
                tr[i] = tr[i-1]
        
        df['fs'] = fs
        df['tr'] = tr
        return df
    
    def update_trailing_stop(self, current_price, position):
        if not position:
            return
        
        take_profit = position['take_profit']

        if 'max_profit_price' not in position:
            position['max_profit_price'] = position['entry_price']
            position['trailing_stop_active'] = False

        changed = False  # track if we update anything

        if position['action'] == 'long':
            if current_price > position['max_profit_price']:
                position['max_profit_price'] = current_price
                changed = True
                print(f"[TRAILING STOP] New max profit price for LONG: ${position['max_profit_price']:.4f}")

            if not position.get('half_exit_done', False) and current_price >= take_profit:
                self.execute_half_exit(position, current_price)
                position['trailing_stop_active'] = True
                changed = True

            if position['trailing_stop_active']:
                trailing_stop_price = position['max_profit_price'] - (self.current_atr_value * LONG_SECOND_SL_LEVEL)
                if trailing_stop_price > position['stop_loss']:
                    position['stop_loss'] = trailing_stop_price
                    changed = True
                    print(f"[TRAILING STOP] Updated stop loss for LONG: ${position['stop_loss']:.4f}")

        else:  # short
            if current_price < position['max_profit_price']:
                position['max_profit_price'] = current_price
                changed = True
                print(f"[TRAILING STOP] New max profit price for SHORT: ${position['max_profit_price']:.4f}")

            if not position.get('half_exit_done', False) and current_price <= take_profit:
                self.execute_half_exit(position, current_price)
                position['trailing_stop_active'] = True
                changed = True

            if position['trailing_stop_active']:
                trailing_stop_price = position['max_profit_price'] + (self.current_atr_value * SHORT_SECOND_SL_LEVEL)
                if trailing_stop_price < position['stop_loss']:
                    position['stop_loss'] = trailing_stop_price
                    changed = True
                    print(f"[TRAILING STOP] Updated stop loss for SHORT: ${position['stop_loss']:.4f}")

        if changed:
            self.save_trades()

    def execute_half_exit(self, position, current_price):
        """Execute half position exit and update position size"""
        with self.trade_lock:
            if position.get('half_exit_done', False):
                print(f"[HALF EXIT] Half exit already executed for {position['action'].upper()} position")
                return

            # Store original size if not already stored
            if 'original_size' not in position:
                position['original_size'] = position['size']

            # Calculate half size to close
            half_size = position['size'] / 2
            remaining_size = position['size'] - half_size

            # Calculate PnL for the half position
            if position['action'] == 'long':
                pnl = (current_price - position['entry_price']) * half_size
            else:
                pnl = (position['entry_price'] - current_price) * half_size

            fee = (current_price + position['entry_price']) * FEE_PERCENT * half_size
            net_pnl = pnl - fee

            # Update balance
            self.balance += net_pnl

            # Create a closed trade record for the half exit
            half_trade = {
                'entry_time': position['entry_time'],
                'exit_time': datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                'action': position['action'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'size': half_size,
                'status': 'closed',
                'pnl': net_pnl,
                'fee': fee,
                'ideal_pnl': pnl,
                'reason': 'half_exit_take_profit',
                'stop_loss': position['stop_loss'],
                'take_profit': position['take_profit'],
                'max_profit_price': position.get('max_profit_price', position['entry_price']),
                'trailing_stop_active': False,
                'half_exit_done': True,
                'original_size': position['original_size']
            }

            # Update the original position
            position['size'] = remaining_size
            position['half_exit_done'] = True
            position['trailing_stop_active'] = True

            # Add the half exit trade to trades list
            self.trades.append(half_trade)

            # Send webhook for half exit
            self.send_webhook(f'exit_{position["action"]}', current_price, half_size, 50)

            print(f"[HALF EXIT] 🔵 Executed half exit for {position['action'].upper()} position | "
                  f"Exit Price: ${current_price:.4f} | "
                  f"Half Size: {half_size:.4f} | "
                  f"Remaining Size: {remaining_size:.4f} | "
                  f"PNL: ${net_pnl:.4f}")

            self.save_trades()
            self.send_daily_pnl_signal()

    def update_indicators(self):
        new_candles = self.fetch_candles(INTERVAL, 200)
        if not new_candles:
            print("[DATA] No new candles fetched. Using existing data.")
            return
            
        self.candles = sorted(new_candles, key=lambda x: x['timestamp'])
        print(f"[INDICATORS] Updating with {len(self.candles)} candles")

        df = pd.DataFrame(self.candles)

        df = self.calculate_fs(df, FS_LENGTH)
        df = self.calculate_volume_oscillator(df)
        df = self.calculate_atr(df, ATR_LENGTH)
        df = self.calculate_rsi(df, RSI_LENGTH)

        self.fs = df['fs'].tolist()
        self.tr = df['tr'].tolist()
        self.current_vol_os = df['vol_os'].iloc[-1]
        self.current_atr_value = df['atr'].iloc[-1]
        self.current_rsi_value = df['rsi'].iloc[-1]

        print(f"[INDICATORS] Updated FS: {self.fs[-1]:.4f}, TR: {self.tr[-1]:.4f}, "
            f"VOL_OS: {self.current_vol_os:.4f}, ATR_VAL: {self.current_atr_value:.4f}, "
            f"RSI_VALUE: {self.current_rsi_value:.4f}")

    # ========== Trading Logic ==========
    def check_signal(self, current_time: str):
        if len(self.fs) < 3:
            print("[SIGNAL] Not enough data to check signals")
            return
        
        fs_now = self.fs[-1]
        tr_now = self.tr[-1]
        print(f"[{current_time}] 1H indicator values now fs:{fs_now:.4f}, tr:{tr_now:.4f}")
        
        fs_prev = self.fs[-2]
        tr_prev = self.tr[-2]
        print(f"[{current_time}] 1H indicator values prev fs:{fs_prev:.4f}, tr:{tr_prev:.4f}")

        fs_cross_up = (fs_prev < tr_prev) and (fs_now > tr_now)
        fs_cross_down = (fs_prev > tr_prev) and (fs_now < tr_now)

        current_price = self.get_current_price()
        if current_price is None:
            print("[SIGNAL] Cannot get current price, skipping signal check")
            return

        if fs_cross_up:
            print(f"UP cross is detected!")
        if fs_cross_down:
            print(f"DOWN cross is detected!")
        
        if fs_cross_up:
            cond_entry = abs(self.current_rsi_value - 50) < LONG_RSI_ENTRY_LEVEL
            cond_entry_fs = max(abs(tr_now), abs(fs_now)) < LONG_FS_ENTRY_LEVEL
            cond_vol = self.current_vol_os > 0
            
            print(f"[SIGNAL] LONG Conditions - RSI: {cond_entry}, FS: {cond_entry_fs}, VOL: {cond_vol}")
            
            if cond_entry and cond_entry_fs and cond_vol:
                if self.current_long_position:
                    if current_price < self.current_long_position['entry_price']:
                        self.close_position('long', "Replace")
                        self.open_position('long')
                    else:
                        pass
                else:
                    self.open_position('long')
                    
        elif fs_cross_down:
            cond_entry = abs(self.current_rsi_value - 50) > SHORT_RSI_ENTRY_LEVEL
            cond_entry_fs = max(abs(tr_now), abs(fs_now)) < SHORT_FS_ENTRY_LEVEL
            cond_vol = self.current_vol_os > 0
            
            print(f"[SIGNAL] SHORT Conditions - RSI: {cond_entry}, FS: {cond_entry_fs}, VOL: {cond_vol}")
            
            if cond_entry and cond_entry_fs and cond_vol:
                if self.current_short_position:
                    if current_price > self.current_short_position['entry_price']:
                        self.close_position('short', "Replace")
                        self.open_position('short')
                    else:
                        pass
                else:
                    self.open_position('short')

    def open_position(self, direction):
        with self.trade_lock:
            if direction == 'long' and self.current_long_position:
                print("[TRADE] Cannot open LONG: an open LONG position already exists.")
                return
            if direction == 'short' and self.current_short_position:
                print("[TRADE] Cannot open SHORT: an open SHORT position already exists.")
                return
        
            price = self.get_current_price()
            if price is None:
                print("[TRADE] Failed to fetch current price for open position.")
                return
            
            size = round((3000 * LEVERAGE) / price, 4)

            current_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

            if direction == 'long':
                stop_loss_level = LONG_STOP_LOSS_LEVEL
                take_profit_level = LONG_TAKE_PROFIT_LEVEL
            else:
                stop_loss_level = SHORT_STOP_LOSS_LEVEL
                take_profit_level = SHORT_TAKE_PROFIT_LEVEL

            initial_risk = self.current_atr_value * stop_loss_level
            initial_profit = self.current_atr_value * take_profit_level

            if direction == 'long':
                stop_loss = price - initial_risk
                take_profit = price + initial_profit 

            else:
                stop_loss = price + initial_risk
                take_profit = price - initial_profit


            trade = {
                'entry_time': current_time,
                'exit_time': None,
                'action': direction,
                'entry_price': price,
                'exit_price': None,
                'size': size,
                'status': 'open',
                'pnl': 0.0,
                'fee': 0.0,
                'ideal_pnl': 0.0,
                'reason': '',
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'max_profit_price': price,
                'trailing_stop_active': False,
                'half_exit_done': False,
                'original_size': size
            }
            
            self.send_webhook(direction, price, size)

            if direction == 'long':
                self.current_long_position = trade
            else:
                self.current_short_position = trade

            self.trades.append(trade)
            self.save_trades()
            print(f"[TRADE] 🟢 Opened {direction.upper()} position at ${price:.4f}, size: {size:.4f}, SL: ${stop_loss:.4f}")

    def close_trade(self, trade, current_price, reason):
        with self.trade_lock:
            trade['exit_time'] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            trade['exit_price'] = current_price
            
            if trade['action'] == 'long':
                trade['pnl'] = (current_price - trade['entry_price']) * trade['size']
            else:
                trade['pnl'] = (trade['entry_price'] - current_price) * trade['size']
                
            trade['fee'] = (current_price + trade['entry_price']) * FEE_PERCENT * trade['size']
            trade['ideal_pnl'] = trade['pnl']
            trade['pnl'] -= trade['fee']
            self.balance += trade['pnl']
            trade['status'] = 'closed'
            trade['reason'] = reason
            
            if trade['action'] == 'long':
                self.current_long_position = None
            else:
                self.current_short_position = None

            self.send_webhook(f'exit_{trade["action"]}', current_price, trade['size'], 100)
            self.save_trades()
            
            print(f"[TRADE] 🔴 Closed {trade['action'].upper()} position | "
                  f"Entry: ${trade['entry_price']:.4f} | "
                  f"Exit: ${current_price:.4f} | "
                  f"PNL: ${trade['pnl']:.4f} (Fee: ${trade['fee']:.4f}) | "
                  f"Reason: {reason}")
            
            self.send_daily_pnl_signal()

    def close_position(self, direction, reason="signal"):
        if direction == 'long' and self.current_long_position:
            position = self.current_long_position
        elif direction == 'short' and self.current_short_position:
            position = self.current_short_position
        else:
            print(f"[TRADE] No open {direction.upper()} positions to close 🤷‍♂️")
            return
            
        current_price = self.get_current_price()
        if current_price is None:
            print("[TRADE] Failed to fetch current price for closing positions")
            return

        self.close_trade(position, current_price, reason)

    def close_half_position(self, direction):
        """Manually close half of the position"""
        if direction == 'long' and self.current_long_position:
            position = self.current_long_position
        elif direction == 'short' and self.current_short_position:
            position = self.current_short_position
        else:
            print(f"[HALF EXIT] No open {direction.upper()} positions to close half 🤷‍♂️")
            return

        if position.get('half_exit_done', False):
            print(f"[HALF EXIT] Half exit already executed for {direction.upper()} position")
            return

        current_price = self.get_current_price()
        if current_price is None:
            print("[HALF EXIT] Failed to fetch current price")
            return

        self.execute_half_exit(position, current_price)

    # ========== Risk Management ==========
    def monitor_positions(self):
        print("[MONITOR] Starting real-time position monitoring (every 1s)...")
        while self.running:
            try:
                current_price = self.get_current_price()
                if current_price:
                    self.check_risk(current_price)
                else:
                    print("[MONITOR] Skipping — price unavailable")
                
                # Sleep exactly 1 second before next check
                time.sleep(1)

            except Exception as e:
                print(f"[MONITOR] Error: {e}")
                time.sleep(1)

    def check_risk(self, current_price):
        
        if self.current_long_position:
            self.update_trailing_stop(current_price, self.current_long_position)
        if self.current_short_position:
            self.update_trailing_stop(current_price, self.current_short_position)

        # Check long position
        if self.current_long_position:
            position = self.current_long_position
            if current_price <= position['stop_loss']:
                print(f"[RISK] Stop loss hit for LONG: ${current_price:.4f} <= ${position['stop_loss']:.4f}")
                self.close_trade(position, current_price, "stop_loss")
                return

        # Check short position  
        if self.current_short_position:
            position = self.current_short_position
            if current_price >= position['stop_loss']:
                print(f"[RISK] Stop loss hit for SHORT: ${current_price:.4f} >= ${position['stop_loss']:.4f}")
                self.close_trade(position, current_price, "stop_loss")
                return

    def send_daily_pnl_signal(self):
        """Send a performance summary webhook each time a trade is closed"""
        if not self.trades:
            return

        df = pd.DataFrame(self.trades)

        # Use only closed trades
        closed = df[df['status'] == 'closed'].copy()

        last_position_size = closed['size'].iloc[-1] if not closed.empty else 0.0
        last_position_entry_price = closed['entry_price'].iloc[-1] if not closed.empty else 0.0
        last_position_usdt = last_position_size * last_position_entry_price

        if closed.empty:
            return

        # Convert timestamps
        closed['exit_time'] = pd.to_datetime(closed['exit_time'], errors='coerce')
        closed['date'] = closed['exit_time'].dt.strftime('%m-%d')

        # Calculate total PNLs
        closed['pnl'] = closed['pnl'].astype(float)
        daily_pnl = closed.groupby('date')['pnl'].sum().reset_index()

        daily_pnl = daily_pnl.sort_values('date', ascending=False)

        # Today's date (UTC)
        today_str = datetime.now(timezone.utc).strftime('%m-%d')
        today_pnl = daily_pnl[daily_pnl['date'] == today_str]['pnl'].sum() if today_str in daily_pnl['date'].values else 0.0

        # Compute win rate
        total_trades = len(closed)
        wins = (closed['pnl'] > 0).sum()
        winrate = (wins / total_trades) * 100 if total_trades > 0 else 0

        # CORRECTED: Max drawdown calculation
        # Calculate equity curve including initial balance
        closed_sorted = closed.sort_values('exit_time')
        cumulative_pnl = closed_sorted['pnl'].cumsum()
        equity_curve = INITIAL_BALANCE + cumulative_pnl
        
        total_pnl = closed['pnl'].sum()
        total_pnl_percent = (total_pnl / INITIAL_BALANCE) * 100 if INITIAL_BALANCE != 0 else 0.0

        # Calculate running maximum
        running_max = equity_curve.expanding().max()
        
        # Calculate drawdown
        drawdown = (equity_curve - running_max) / running_max * 100
        
        # Get maximum drawdown (most negative)
        max_drawdown_pct = drawdown.min() if not drawdown.empty else 0

        # History string
        history_list = [f"{row.date}:{round(row.pnl,2)}" for row in daily_pnl.itertuples()]
        history_str = "[" + ", ".join(history_list) + "]"

        last_pnl = closed['pnl'].iloc[-1]
        # last_pnl_percent = (last_pnl / last_position_usdt) * 100 * LEVERAGE if last_position_usdt != 0 else 0.0

        payload = {
            "ticker": "UNI-USDT",
            "action": "stats",
            "time": datetime.now(timezone.utc).isoformat(),
            "PNL": str(round(total_pnl_percent, 2)),
            "LastPNL": str(round(last_pnl, 2)),
            "MaxDraw": str(round(abs(max_drawdown_pct), 2)),
            "WinRate": str(round(winrate, 2)),
            "DailyPNL": str(round(today_pnl, 2)),
            "Strategy": "20251201",
            "StartDate": self.start_date.strftime("%b %d, %Y"),
            "historyPNL": history_str,
            "TotalTrades": str(total_trades)
        }

        print(f"[DAILY PNL🟡] Sending daily PNL report: {payload}")

        if self.flag_webhook_sent:
            try:
                response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
                if response.ok:
                    print(f"[DAILY PNL🟢] Sent successfully ({response.status_code})")
                else:
                    print(f"[DAILY PNL🔴] Failed: {response.status_code} {response.reason}")
            except Exception as e:
                print(f"[DAILY PNL🔴] Error sending report: {e}")

    # ========== Utility Methods ==========
    def send_webhook(self, action, price, size=None, per=None):
        if action in ["short", "long"]:
            payload = {
                "time": datetime.now(timezone.utc).isoformat(),
                "ticker": UNIUSDT_SYMBOL,
                "action": "sell" if action == "short" else "buy",
                "price": str(round(price, 4)),
                "size": str(round(size, 4)) if size else "",
                "TotalTrades": str(self.total_trades)
            }

            self.total_trades += 1

        elif action in ["exit_short", "exit_long"]:
            payload = {
                "time": datetime.now(timezone.utc).isoformat(),
                "ticker": UNIUSDT_SYMBOL,
                "action": "exit_sell" if action == "exit_short" else "exit_buy",
                "price": str(round(price, 4)),
                "per": str(round(per, 2)) if per else "",
                "size": str(round(size, 4)) if size else "",
            }
            
        else:
            return
        
        print(f"[Signal 🟡] Sending: {payload}")

        if self.flag_webhook_sent:
            try:
                response = requests.post(WEBHOOK_URL, json=payload, timeout=5)
                if response.ok:
                    print(f"[Signal 🟢] Success: {response.status_code}")
                else:
                    print(f"[Signal 🔴] Failed: {response.status_code} {response.reason}")
            except Exception as err:
                print(f"[Signal 🔴] Error: {err}")
            

    def save_trades(self):
        try:
            with open(CSV_FILE, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['entry_time', 'exit_time', 'action', 'entry_price', 'exit_price', 
                               'size', 'status', 'fee', 'ideal_pnl', 'pnl', 'reason', 'stop_loss', 
                               'take_profit', 'max_profit_price', 'trailing_stop_active', 'half_exit_done', 'original_size'])
                for trade in self.trades:
                    writer.writerow([
                        trade['entry_time'],
                        trade['exit_time'],
                        trade['action'],
                        round(trade['entry_price'], 4),
                        round(trade['exit_price'], 4) if trade['exit_price'] else '',
                        round(trade['size'], 4),
                        trade['status'],
                        round(trade['fee'], 4),
                        round(trade['ideal_pnl'], 4),
                        round(trade.get('pnl', 0), 4),
                        trade['reason'],
                        round(trade['stop_loss'], 4),
                        round(trade['take_profit'], 4),
                        round(trade.get('max_profit_price', 0), 4),
                        str(trade.get('trailing_stop_active', False)),
                        str(trade.get('half_exit_done', False)),
                        round(trade.get('original_size', trade['size']), 4)
                    ])
            print(f"[DATA] Saved {len(self.trades)} trades to {CSV_FILE}")
        except Exception as e:
            print(f"[DATA] Error saving trades: {e}")

    # ========== Command Interface ==========
    def listen_for_input(self):
        print('Bot is running... Type "help" for commands')
        while self.running:
            try:
                cmd = input(">> ").lower().strip()

                if cmd == 'help':
                    print("Commands: force open long/short, force close long/short, "
                          "force half close long/short, balance, positions, price, status, exit")
                
                elif cmd == 'force open long':
                    self.open_position('long')
                elif cmd == 'force open short':
                    self.open_position('short')
                elif cmd == 'force close long':
                    self.close_position('long')
                elif cmd == 'force close short':
                    self.close_position('short')
                elif cmd == 'force half close long':
                    self.close_half_position('long')
                elif cmd == 'force half close short':
                    self.close_half_position('short')
                elif cmd == 'balance':
                    print(f"Current balance: ${self.balance:.4f}")
                elif cmd == 'price':
                    print(f"Current price: ${self.get_current_price():.4f}")
                elif cmd == 'exit':
                    self.running = False
                    print("Shutting down...")
                    break
                else:
                    print("Invalid command. Type 'help' for available commands")
            except KeyboardInterrupt:
                self.running = False
                print("Shutting down...")
                break
            except Exception as e:
                print(f"Input error: {e}")

    def wait_until_next_hour_plus_5sec(self):
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        # Target is the next full hour + 5 seconds
        next_time = current_hour + timedelta(hours=1, seconds=5)

        # Just in case (if somehow equal/past), push to the following hour
        if next_time <= now:
            next_time += timedelta(hours=1)

        sleep_seconds = (next_time - now).total_seconds()
        print(f"Sleeping for {sleep_seconds:.4f} seconds until {next_time.strftime('%H:%M:%S')} UTC")
        time.sleep(sleep_seconds)

    # ========== Main Loop ==========
    def strategy_loop(self):
        print("[STRATEGY] Starting strategy loop")
        while self.running:
            try:
                current_time = datetime.now(timezone.utc)
                print(f"\n[STRATEGY] Running strategy check at {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                self.update_indicators()
                self.check_signal(current_time.strftime('%Y-%m-%d %H:%M:%S'))
                self.wait_until_next_hour_plus_5sec()
            except Exception as e:
                print(f"[STRATEGY] Error in strategy loop: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = AutoTradeBot()
    bot.listen_for_input()