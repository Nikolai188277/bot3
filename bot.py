import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from binance.client import Client
from ta.trend import MACD, SMAIndicator, ADXIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import json
import os
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import logging
import websocket


# === НАСТРОЙКИ ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


API_KEY = "kdqPPEpDDLVjfZaQkQhtxOQhADs7OIzdFDG8lvfzFKQg6yXn6XQt8YR1u0FYqNV5"
API_SECRET = "PZ0OE1MOOOvb82gMlWnnq1ZimCer3YnA9QOClUabPK30ZpjstjmYef9JASi2z6ps"


WEIGHTS_FILE = "prediction_weights.json"
TOP_COINS_FILE = "top_coins.json"
RECOMMENDED_TRADE_FILE = "recommended_trade.json"
SETTINGS_FILE = "settings_bot3.json"
TRAINING_SESSION_FILE = "training_session.json"


SEQ_LENGTH = 10
LSTM_EPOCHS = 10
BATCH_SIZE = 64
PROB_DIFF_THRESHOLD = 30
BAD_HOURS = {13, 16}
GOOD_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT']


# === ДИНАМИЧЕСКИЙ ГОРИЗОНТ ===
INTERVALS = [
   (60, Client.KLINE_INTERVAL_1MINUTE),
   (300, Client.KLINE_INTERVAL_5MINUTE),
   (900, Client.KLINE_INTERVAL_15MINUTE),
   (1800, Client.KLINE_INTERVAL_30MINUTE),
   (3600, Client.KLINE_INTERVAL_1HOUR),
   (14400, Client.KLINE_INTERVAL_4HOUR),
   (86400, Client.KLINE_INTERVAL_1DAY),
   (604800, Client.KLINE_INTERVAL_1WEEK),
   (2592000, Client.KLINE_INTERVAL_1MONTH),
]


def get_binance_interval(seconds):
   for sec, interval in INTERVALS:
       if seconds <= sec:
           return interval, max(1, int(seconds / sec))
   return Client.KLINE_INTERVAL_1MONTH, int(seconds / 2592000)


# === ИНИЦИАЛИЗАЦИЯ ===
client = Client(API_KEY, API_SECRET)
data_cache = {}
cache_lock = threading.Lock()


# === УТИЛИТЫ ===
def load_json(file):
   if os.path.exists(file):
       try:
           with open(file, 'r', encoding='utf-8') as f:
               return json.load(f)
       except UnicodeDecodeError:
           # fallback для старых файлов
           with open(file, 'r', encoding='cp1251') as f:
               return json.load(f)
       except Exception as e:
           logging.error(f"Ошибка чтения {file}: {e}")
           return {}
   return {}


def save_json(file, data):
    """Безопасное сохранение JSON"""
    def convert_for_json(obj):
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    try:
        temp_file = file + '.tmp'
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(convert_for_json(data), f, indent=2, ensure_ascii=False)
        os.replace(temp_file, file)
        logging.info(f"✅ Сохранено {file} — {len(data)} монет")
    except Exception as e:
        logging.error(f"❌ Ошибка сохранения {file}: {e}")


weights = load_json(WEIGHTS_FILE)

# === ИСПРАВЛЕНА ИНИЦИАЛИЗАЦИЯ ===

def initialize_all_weights(all_pairs):
    """Создаёт веса для ВСЕХ монет"""
    global weights
    created = 0
    default = {
        'up_weight': 1.0, 'down_weight': 1.0,
        'correct': 0, 'total': 0, 'streak': 0,
        'score': 0.0, 'accuracy': 50.0, 'history': []
    }
    for sym in all_pairs:
        if sym not in weights or not isinstance(weights.get(sym), dict):
            weights[sym] = default.copy()
            created += 1
    if created > 0:
        logging.info(f"✅ Инициализировано {created} новых весов (всего {len(weights)} монет)")
        save_json(WEIGHTS_FILE, weights)
    else:
        logging.info(f"Веса уже инициализированы ({len(weights)} монет)")


def load_training_session():
   data = load_json(TRAINING_SESSION_FILE)
   if not data:
       return None
   # Восстанавливаем datetime
   if 'predictions' in data:
       for p in data['predictions']:
           if 'prediction_time' in p and isinstance(p['prediction_time'], str):
               try:
                   p['prediction_time'] = datetime.fromisoformat(p['prediction_time'].replace('Z', '+00:00'))
               except:
                   pass
   return data




def save_training_session(session_data):
   # Глубокое копирование + очистка от несериализуемых объектов
   def clean_for_json(obj):
       if isinstance(obj, datetime):
           return obj.isoformat()
       if isinstance(obj, dict):
           return {k: clean_for_json(v) for k, v in obj.items()}
       if isinstance(obj, list):
           return [clean_for_json(i) for i in obj]
       if isinstance(obj, (int, float, str, bool, type(None))):
           return obj
       return str(obj)  # fallback


   try:
       with open(TRAINING_SESSION_FILE, 'w', encoding='utf-8') as f:
           json.dump(clean_for_json(session_data), f, indent=2, ensure_ascii=False)
       logging.info("Сессия обучения сохранена")
   except Exception as e:
       logging.error(f"Ошибка сохранения сессии: {e}")


# Исправляем history после загрузки
for sym, w in weights.items():
   if isinstance(w, dict) and 'history' in w:
       w['history'] = [1 if x else 0 for x in w['history']]


# === НАСТРОЙКИ ПОЛЬЗОВАТЕЛЯ ===
default_settings = {
   "auto_interval": 300,
   "weight_multiplier": 2.0,
   "horizon_seconds": 3600,
   "history_candles": 500,
   "tremor_enabled": False,
   "tremor_interval": 60,
   "dev_filter_enabled": False,
   "dev_threshold": 3.0,
   "recent_accuracy_window": 10,
   "rr_ratio": 2.0,
   "trade_size_usd": 20.0,
   "commission_rate": 0.00045
}


settings = load_json(SETTINGS_FILE)
for k, v in default_settings.items():
   if k not in settings:
       settings[k] = v


# ====================== УМНАЯ КОРРЕКТИРОВКА ВЕСОВ ======================
def update_weights_smart(symbol, is_correct, profit=0.0):
    """Обновляет веса монеты + сохраняет файл"""
    global weights
    if symbol not in weights:
        initialize_all_weights([symbol])  # на всякий случай

    w = weights[symbol]
    w['total'] += 1
    if is_correct:
        w['correct'] += 1
        w['streak'] = max(0, w.get('streak', 0) + 1)
        w['up_weight'] = min(5.0, w['up_weight'] * 1.08)   # усиливает
        w['down_weight'] = max(0.2, w['down_weight'] * 0.95)
    else:
        w['streak'] = min(0, w.get('streak', 0) - 1)
        w['up_weight'] = max(0.2, w['up_weight'] * 0.92)
        w['down_weight'] = min(5.0, w['down_weight'] * 1.08)

    # Пересчёт accuracy и score
    w['accuracy'] = (w['correct'] / w['total']) * 100 if w['total'] > 0 else 50.0
    w['score'] = (w['accuracy'] - 50) * (w['total'] / 50) + (profit * 100)

    if 'history' not in w:
        w['history'] = []
    w['history'].append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'correct': is_correct,
        'profit': profit
    })
    if len(w['history']) > 100:
        w['history'] = w['history'][-100:]

    save_json(WEIGHTS_FILE, weights)   # ← КРИТИЧНОЕ СОХРАНЕНИЕ
    logging.info(f"Обновлён вес {symbol}: correct={w['correct']}/{w['total']} | acc={w['accuracy']:.1f}%")

    update_weights_smart(symbol, is_correct, profit)


# === ДАННЫЕ ===
def get_futures_pairs():
   try:
       info = client.futures_exchange_info()
       pairs = [
           s['symbol'] for s in info['symbols']
           if s.get('contractType') == 'PERPETUAL' and s.get('status') == 'TRADING'
       ]
       logging.info(f"Получено {len(pairs)} активных пар")
       return pairs
   except Exception as e:
       logging.warning(f"Пары не получены: {e}")
       return GOOD_SYMBOLS


def get_historical_data(symbol, interval, limit=None):
   if limit is None:
       limit = settings.get("history_candles", 500)
   limit = max(50, min(1500, limit))


   key = f"{symbol}_{interval}_{limit}"
   with cache_lock:
       if key in data_cache and time.time() - data_cache[key][1] < 60:
           return data_cache[key][0].copy()


   try:
       interval_map = {
           Client.KLINE_INTERVAL_1MINUTE: "1500 minutes ago UTC",
           Client.KLINE_INTERVAL_5MINUTE: "7500 minutes ago UTC",
           Client.KLINE_INTERVAL_15MINUTE: "22500 minutes ago UTC",
           Client.KLINE_INTERVAL_30MINUTE: "45000 minutes ago UTC",
           Client.KLINE_INTERVAL_1HOUR: "1500 hours ago UTC",
           Client.KLINE_INTERVAL_4HOUR: "6000 hours ago UTC",
           Client.KLINE_INTERVAL_1DAY: "1500 days ago UTC",
           Client.KLINE_INTERVAL_1WEEK: "10500 days ago UTC",
           Client.KLINE_INTERVAL_1MONTH: "45000 days ago UTC",
       }
       start_str = interval_map.get(interval, "1500 hours ago UTC")


       klines = client.futures_historical_klines(symbol, interval, start_str, limit=limit)
       if not klines:
           return None


       df = pd.DataFrame(klines, columns=[
           'timestamp', 'open', 'high', 'low', 'close', 'volume',
           'close_time', 'quote_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignored'
       ])
       df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
       df = df.astype({'open': float, 'high': float, 'low': float, 'close': float,
                       'volume': float, 'taker_buy_base': float, 'taker_buy_quote': float, 'trades': int})


       df['long_volume'] = df['taker_buy_base']
       df['short_volume'] = df['volume'] - df['taker_buy_base']
       df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'long_volume', 'short_volume', 'trades']]


       with cache_lock:
           data_cache[key] = (df, time.time())
       return df
   except Exception as e:
       logging.error(f"Ошибка получения данных {symbol}: {e}")
       return None


# === ИСТОРИЧЕСКИЙ FUNDING RATE ===
def get_funding_rate_history(symbol, limit=500):
   try:
       funding = client.futures_funding_rate(symbol=symbol, limit=limit)
       if not funding:
           return pd.DataFrame()


       df_fund = pd.DataFrame(funding)
       df_fund['fundingTime'] = pd.to_datetime(df_fund['fundingTime'], unit='ms')
       df_fund['fundingRate'] = df_fund['fundingRate'].astype(float)
       df_fund = df_fund[['fundingTime', 'fundingRate']].rename(columns={'fundingTime': 'timestamp'})
       df_fund = df_fund.set_index('timestamp')
       return df_fund
   except Exception as e:
       logging.warning(f"Не удалось получить историю funding для {symbol}: {e}")
       return pd.DataFrame()




# === НОВАЯ ФУНКЦИЯ ДЛЯ ДОПОЛНИТЕЛЬНЫХ ДАННЫХ ===
def get_additional_futures_data(symbol):
   try:
       # Текущий Funding Rate (самый свежий из premiumIndex)
       premium = client.futures_premium_index(symbol=symbol)
       funding_rate = float(premium['lastFundingRate'])  # Например, 0.0001 = +0.01%


       # Global Long/Short Account Ratio (последний)
       ls_ratio = client.futures_global_longshort_ratio(symbol=symbol, period="5m", limit=1)
       if ls_ratio:
           long_short_ratio = float(ls_ratio[0]['longShortRatio'])  # >1 = больше лонгов
           long_account = float(ls_ratio[0]['longAccount'])
       else:
           long_short_ratio = 1.0
           long_account = 0.5


       # Open Interest (текущий)
       oi_info = client.futures_open_interest(symbol=symbol)
       open_interest = float(oi_info['openInterest'])


       return {
           'funding_rate': funding_rate,
           'long_short_ratio': long_short_ratio,
           'long_account_perc': long_account * 100,
           'open_interest': open_interest
       }
   except Exception as e:
       logging.warning(f"Доп данные не получены для {symbol}: {e}")
       return {'funding_rate': 0.0, 'long_short_ratio': 1.0, 'long_account_perc': 50.0, 'open_interest': 0.0}


# === ИНДИКАТОРЫ ===
def calculate_indicators(df, symbol=None):
   try:
       df['rsi'] = RSIIndicator(df['close'], 14).rsi()
       macd = MACD(df['close'])
       df['macd'] = macd.macd()
       df['macd_signal'] = macd.macd_signal()
       bb = BollingerBands(df['close'])
       df['bb_high'] = bb.bollinger_hband()
       df['bb_low'] = bb.bollinger_lband()
       df['sma20'] = SMAIndicator(df['close'], 20).sma_indicator()
       df['sma50'] = SMAIndicator(df['close'], 50).sma_indicator()
       df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
       df['adx'] = ADXIndicator(df['high'], df['low'], df['close'], 14).adx()
       df['ema20'] = EMAIndicator(df['close'], 20).ema_indicator()
       df['stoch_k'] = StochasticOscillator(df['high'], df['low'], df['close']).stoch()
       df['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], 14).average_true_range()
       df['vwap'] = VolumeWeightedAveragePrice(df['high'], df['low'], df['close'], df['volume'],
                                               14).volume_weighted_average_price()
       # Добавляем ratio объёмов лонг/шорт как индикатор
       total_volume = df['volume'].sum()
       if total_volume > 0:
           df['long_volume_ratio'] = df['long_volume'].sum() / total_volume
       else:
           df['long_volume_ratio'] = 0.5


       # УЛУЧШЕННЫЙ CVD (кумулятивная дельта)
       df['volume_delta'] = df['long_volume'] - df['short_volume']  # дельта на свечу
       df['cvd'] = df['volume_delta'].cumsum()  # кумулятивный CVD


       # Если есть symbol, добавим внешние данные (один ряд для всего df)
       if symbol:
           add_data = get_additional_futures_data(symbol)
           df['current_funding_rate'] = add_data['funding_rate']
           df['long_short_ratio'] = add_data['long_short_ratio']
           df['oi'] = add_data['open_interest']


           # === ИСТОРИЧЕСКИЙ FUNDING RATE ===
           df = df.set_index('timestamp')
           df_fund = get_funding_rate_history(symbol, limit=500)
           if not df_fund.empty:
               # Присоединяем funding rate по ближайшему левому времени (ffill)
               df = df.join(df_fund, how='left')
               df['funding_rate'] = df['fundingRate'].ffill().fillna(0)
               df['funding_rate_ma'] = df['funding_rate'].rolling(5).mean().fillna(0)
               df['funding_rate_extreme'] = ((df['funding_rate'] > 0.0005) | (df['funding_rate'] < -0.0005)).astype(int)
               df = df.drop(columns=['fundingRate'], errors='ignore')
           else:
               df['funding_rate'] = 0.0
               df['funding_rate_ma'] = 0.0
               df['funding_rate_extreme'] = 0
           df = df.reset_index()


       return df.dropna()
   except Exception as e:
       logging.error(f"Индикаторы: {e}")
       return None




# === РАСЧЁТ ОЖИДАЕМОЙ ПРИБЫЛИ (EV$) ===
def calculate_ev(acc_percent, rr=2.0, size=20.0, comm=0.00045):
   """Простой расчёт ожидаемой прибыли на одну сделку"""
   winrate = acc_percent / 100.0
   lossrate = 1 - winrate


   # Прибыль если выиграл
   profit_on_win = size * rr
   # Убыток если проиграл
   loss_on_loss = size * 1


   # Ожидаемая прибыль без комиссии
   ev = (winrate * profit_on_win) - (lossrate * loss_on_loss)


   # Вычитаем комиссию (вход + выход)
   commission_cost = size * comm * 2


   net_ev = ev - commission_cost
   return round(net_ev, 3)


# === РАСЧЁТ ТОЧНОСТИ ЗА ПОСЛЕДНИЕ N ПРОГНОЗОВ ===
def get_recent_accuracy(w, window=10):
   """Возвращает точность за последние N прогнозов"""
   history = w.get('history', [])
   if not history:
       return 50.0
   recent = history[-window:]          # берём последние N
   if not recent:
       return 50.0
   acc = (sum(recent) / len(recent)) * 100
   return round(acc, 1)


# === ПОДГОТОВКА ДАННЫХ ===
# Убрали 'long_short_ratio'
FEATURES = ['rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low', 'sma20', 'sma50',
           'obv', 'adx', 'ema20', 'stoch_k', 'atr', 'vwap', 'long_volume_ratio',
           'cvd', 'funding_rate', 'funding_rate_ma', 'funding_rate_extreme', 'long_short_ratio', 'oi']




def prepare_data(df, horizon, for_lstm=False):
   df = df.copy()
   df['price_change'] = df['close'].shift(-horizon) / df['close'] - 1
   df['target'] = (df['price_change'] > 0).astype(int)
   df = df.dropna()


   X = df[FEATURES]
   y_class = df['target']
   y_reg = df['price_change']
   if for_lstm:
       scaler = StandardScaler()
       X_scaled = scaler.fit_transform(X)
       X_seq, y_seq = [], []
       for i in range(len(X_scaled) - SEQ_LENGTH):
           X_seq.append(X_scaled[i:i + SEQ_LENGTH])
           y_seq.append(y_class.iloc[i + SEQ_LENGTH])
       return np.array(X_seq), np.array(y_seq), scaler
   return X, y_class, y_reg




# === МОДЕЛИ ===
class LSTMModel(nn.Module):
   def __init__(self, input_size):
       super().__init__()
       self.lstm = nn.LSTM(input_size, 50, 2, batch_first=True)
       self.fc = nn.Linear(50, 1)
       self.sigmoid = nn.Sigmoid()


   def forward(self, x):
       h0 = torch.zeros(2, x.size(0), 50).to(x.device)
       c0 = torch.zeros(2, x.size(0), 50).to(x.device)
       out, _ = self.lstm(x, (h0, c0))
       return self.sigmoid(self.fc(out[:, -1]))




def train_lstm(X, y):
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = LSTMModel(X.shape[2]).to(device)
   criterion = nn.BCELoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).unsqueeze(1))
   loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
   model.train()
   for _ in range(LSTM_EPOCHS):
       for bx, by in loader:
           bx, by = bx.to(device), by.to(device)
           optimizer.zero_grad()
           loss = criterion(model(bx), by)
           loss.backward()
           optimizer.step()
   return model, device




# === N CANDLES TREND PREDICTION ===
def predict_n_candles(symbol, horizon_seconds, n_candles, weights_dict=None):
   try:
       if n_candles < 2:
           logging.warning(f"N свечей мало: {n_candles}")
           return None


       interval_binance, _ = get_binance_interval(horizon_seconds)
       df = get_historical_data(symbol, interval_binance, limit=n_candles)
       if df is None or len(df) < n_candles:
           logging.warning(f"Не хватает свечей для {symbol}: {len(df) if df is not None else 0}/{n_candles}")
           return None


       df = calculate_indicators(df, symbol=symbol)


       low_old = df['low'].iloc[0]
       high_new = df['high'].iloc[-1]
       current_price = df['close'].iloc[-1]


       is_up = high_new > low_old
       decision = "Long" if is_up else "Short"
       up_prob = 100 if is_up else 0
       down_prob = 100 - up_prob
       prob_diff = 100
       change_percent = (high_new - low_old) / low_old * 100 if low_old != 0 else 0
       predicted_change = change_percent


       hour = datetime.now().hour
       if symbol in GOOD_SYMBOLS and hour not in BAD_HOURS:
           if prob_diff >= PROB_DIFF_THRESHOLD:
               decision = "Long" if up_prob > down_prob else "Short"
           else:
               decision = "Hold"


       w = weights_dict.get(symbol, {}) if weights_dict else {}
       accuracy = w.get('accuracy', 50.0)   # уже в процентах
       streak = w.get('streak', 0)


       # Добавляем объёмы лонг/шорт (агрегированные за период)
       long_vol = df['long_volume'].sum()
       short_vol = df['short_volume'].sum()


       prediction_time = datetime.now().isoformat()


       add_data = get_additional_futures_data(symbol)


       # ЛОГИКА УЧЁТА В РЕШЕНИИ (bias)
       if add_data['funding_rate'] > 0.0005:  # Слишком высокий положительный → перегрев лонгов → bias вниз
           up_prob *= 0.9
       elif add_data['funding_rate'] < -0.0005:  # Негативный → bias вверх
           up_prob *= 1.1


       if add_data['long_short_ratio'] > 2.0:  # Много лонгов → риск short squeeze? Или коррекции
           up_prob *= 0.95


       down_prob = 100 - up_prob
       prob_diff = abs(up_prob - down_prob)
       decision = "Long" if up_prob > down_prob else "Short"


       latest_row = df.iloc[-1]


       result = {
           'symbol': symbol,
           'up_prob': round(up_prob, 2),
           'down_prob': round(down_prob, 2),
           'prob_diff': round(prob_diff, 2),
           'current_price': round(current_price, 6),
           'predicted_change': round(predicted_change, 2),
           'decision': decision,
           'accuracy': round(accuracy, 1),
           'streak': streak,
           'n_candles_dir': decision,
           'n_candles_up_prob': up_prob,  # Для комбинации
           'long_volume': round(long_vol, 2),
           'short_volume': round(short_vol, 2),
           'prediction_time': prediction_time,
           'prediction_price': current_price,
           'funding_rate': round(latest_row.get('funding_rate', 0.0) * 10000, 2),  # в базисных пунктах, напр. 10.00
           'funding_rate_ma': round(latest_row.get('funding_rate_ma', 0.0) * 10000, 2),
           'current_funding_rate': round(latest_row.get('current_funding_rate', 0.0) * 10000, 2),
           'long_short_ratio': round(latest_row.get('long_short_ratio', 1.0), 2),
           'oi': round(latest_row.get('oi', 0), 2)
       }
       return result


   except Exception as e:
       logging.error(f"Ошибка N свечей для {symbol}: {e}")
       return None




# === ПРОГНОЗ (СТАРЫЙ RF + LSTM) ===
def predict_symbol(symbol, horizon_seconds, use_lstm=False, weights_dict=None, history_limit=None, use_n_candles=False,
                  n_candles=500, use_dev_forecast=False):
   if use_n_candles:
       return predict_n_candles(symbol, horizon_seconds, n_candles, weights_dict)


   try:
       interval_binance, horizon = get_binance_interval(horizon_seconds)
       df = get_historical_data(symbol, interval_binance, limit=history_limit)
       if df is None or len(df) < 100:
           return None


       df = calculate_indicators(df, symbol=symbol)
       if df is None or len(df) == 0:
           return None


       # Если включен dev_forecast, используем dev из top_coins.json для направления
       if use_dev_forecast:
           top_data = load_json(TOP_COINS_FILE)
           dev = 0.0
           if isinstance(top_data, list):
               for item in top_data:
                   pair = item.get("pair") or item.get("Пара")
                   if pair and pair.upper() == symbol.upper():
                       dev = item.get("dev", 0.0)
                       break
           if dev > 0:
               up_prob = 0.0
               down_prob = 100.0
               decision = "Short"
           elif dev < 0:
               up_prob = 100.0
               down_prob = 0.0
               decision = "Long"
           else:
               up_prob = 50.0
               down_prob = 50.0
               decision = "Hold"
           prob_diff = abs(up_prob - down_prob)
           predicted_change = 0.0  # Можно оставить 0 или рассчитать
       else:
           # Стандартная логика ML
           df_with_pos = prepare_data(df, horizon)


           X, y_class, y_reg = df_with_pos


           if len(X) == 0:
               return None


           split = int(len(X) * 0.8)
           X_train = X.iloc[:split]
           y_train_class = y_class.iloc[:split]
           y_train_reg = y_reg.iloc[:split]


           clf = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=42)
           reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)


           sample_weight = None
           if weights_dict and symbol in weights_dict:
               w = weights_dict[symbol]
               sample_weight = np.ones(len(y_train_class))
               sample_weight[y_train_class == 1] *= w.get('up_weight', 1.0)
               sample_weight[y_train_class == 0] *= w.get('down_weight', 1.0)


           clf.fit(X_train, y_train_class, sample_weight=sample_weight)
           reg.fit(X_train, y_train_reg)


           latest = df.tail(1)[FEATURES]


           up_prob = clf.predict_proba(latest)[0][1] * 100


           if use_lstm:
               X_lstm, y_lstm, scaler = prepare_data(df, horizon, for_lstm=True)
               if len(X_lstm) > 50:
                   split_l = int(len(X_lstm) * 0.8)
                   model, device = train_lstm(X_lstm[:split_l], y_lstm[:split_l])
                   model.eval()
                   seq_input = scaler.transform(latest.values.reshape(1, -1))
                   seq = torch.tensor(seq_input, dtype=torch.float32).unsqueeze(0).to(device)
                   if seq.shape[1] < SEQ_LENGTH:
                       padding = torch.zeros(1, SEQ_LENGTH - seq.shape[1], seq.shape[2]).to(device)
                       seq = torch.cat([padding, seq], dim=1)
                   else:
                       seq = seq[:, -SEQ_LENGTH:, :]
                   lstm_prob = model(seq).item() * 100
                   up_prob = (up_prob + lstm_prob) / 2


           add_data = get_additional_futures_data(symbol)


           # ЛОГИКА УЧЁТА В РЕШЕНИИ (bias)
           if add_data['funding_rate'] > 0.0005:  # Слишком высокий положительный → перегрев лонгов → bias вниз
               up_prob *= 0.9
           elif add_data['funding_rate'] < -0.0005:  # Негативный → bias вверх
               up_prob *= 1.1


           if add_data['long_short_ratio'] > 2.0:  # Много лонгов → риск short squeeze? Или коррекции
               up_prob *= 0.95


           down_prob = 100 - up_prob
           predicted_change = reg.predict(latest)[0] * 100
           hour = datetime.now().hour
           decision = "Hold"
           if symbol in GOOD_SYMBOLS and hour not in BAD_HOURS:
               if abs(up_prob - down_prob) >= PROB_DIFF_THRESHOLD:
                   decision = "Long" if up_prob > down_prob else "Short"


       current_price = df['close'].iloc[-1]


       w = weights_dict.get(symbol, {}) if weights_dict else {}
       accuracy = w.get('accuracy', 50.0)  # уже в процентах
       streak = w.get('streak', 0)


       long_vol = df['long_volume'].sum()
       short_vol = df['short_volume'].sum()


       prediction_time = datetime.now().isoformat()


       latest_row = df.iloc[-1]


       result = {
           'symbol': symbol,
           'up_prob': round(up_prob, 2),
           'down_prob': round(down_prob, 2),
           'prob_diff': round(abs(up_prob - down_prob), 2),
           'current_price': round(current_price, 6),
           'predicted_change': round(predicted_change, 2),
           'decision': decision,
           'accuracy': round(accuracy, 1),
           'streak': streak,
           'n_candles_dir': decision,  # Для совместимости
           'long_volume': round(long_vol, 2),
           'short_volume': round(short_vol, 2),
           'prediction_time': prediction_time,
           'prediction_price': current_price,
           'funding_rate': round(latest_row.get('funding_rate', 0.0) * 10000, 2),  # в базисных пунктах, напр. 10.00
           'funding_rate_ma': round(latest_row.get('funding_rate_ma', 0.0) * 10000, 2),
           'current_funding_rate': round(latest_row.get('current_funding_rate', 0.0) * 10000, 2),
           'long_short_ratio': round(latest_row.get('long_short_ratio', 1.0), 2),
           'oi': round(latest_row.get('oi', 0), 2)
       }
       return result


   except Exception as e:
       logging.error(f"Ошибка прогноза для {symbol}: {e}")
       return None




# === ДРОЖАНИЕ ЦЕНЫ (ОПТИМИЗИРОВАННАЯ ВЕРСИЯ) ===
def monitor_tremor(symbol, interval_seconds, n_samples=20, max_retries=3):
   """
   Оптимизированная версия:
   - Использует WebSocket для реал-тайм тикера (быстрее, стабильнее, без задержек sleep).
   - Собирает ровно n_samples точек.
   - Авто-ретрей при ошибках.
   - Если WS недоступен — fallback на REST с оптимизированными паузами.
   - Минимальный интервал между запросами 0.2 сек.
   """
   try:
       # Попытка через WebSocket (самый быстрый и надёжный способ)
       prices = []
       ws_url = f"wss://fstream.binance.com/ws/{symbol.lower()}@ticker"
       start_time = time.time()


       def on_message(ws, message):
           if len(prices) >= n_samples:
               return
           data = json.loads(message)
           if 'c' in data:  # current price
               price = float(data['c'])
               prices.append(price)
               logging.info(f"[TREMOR WS] {symbol} sample {len(prices)}: {price}")


       def on_error(ws, error):
           logging.error(f"[TREMOR WS ERROR] {symbol}: {error}")


       def on_close(ws, *args):
           pass


       def on_open(ws):
           logging.info(f"[TREMOR WS] Connected to {symbol}")


       ws = websocket.WebSocketApp(ws_url,
                                   on_open=on_open,
                                   on_message=on_message,
                                   on_error=on_error,
                                   on_close=on_close)


       # Запуск в отдельном потоке с таймаутом
       wst = threading.Thread(target=ws.run_forever, daemon=True)
       wst.start()
       time.sleep(1)  # Даем подключиться


       # Ждём данные или таймаут
       while len(prices) < n_samples and time.time() - start_time < interval_seconds + 10:
           time.sleep(0.2)


       ws.close()


       if len(prices) >= 3:
           diffs = np.diff(prices)
           up_moves = np.sum(diffs > 0)
           down_moves = np.sum(diffs < 0)
           total_moves = up_moves + down_moves
           if total_moves > 0:
               direction = "Long" if up_moves > down_moves else "Short"
               confidence = abs(up_moves - down_moves) / total_moves * 100
               return {
                   'direction': direction,
                   'confidence': round(confidence, 1),
                   'up_moves': up_moves,
                   'down_moves': down_moves,
                   'method': 'WebSocket',
                   'tremor_up_prob': 100 if direction == "Long" else 0
               }


   except Exception as e:
       logging.warning(f"[TREMOR] WebSocket failed for {symbol}: {e}, fallback to REST")


   # === FALLBACK: REST API с оптимизацией ===
   prices = []
   start_time = time.time()
   sample_interval = max(0.2, interval_seconds / n_samples)  # Минимум 0.2 сек


   for attempt in range(max_retries):
       try:
           while len(prices) < n_samples and time.time() - start_time < interval_seconds + 5:
               ticker = client.futures_symbol_ticker(symbol=symbol)  # Более точный для фьючерсов
               if isinstance(ticker, list):
                   ticker = ticker[0] if ticker else {}
               current_price = float(ticker.get('price', 0)) if isinstance(ticker, dict) else float(ticker)
               if current_price > 0:
                   prices.append(current_price)
                   logging.info(f"[TREMOR REST] {symbol} sample {len(prices)}: {current_price}")
               time.sleep(sample_interval)
           break  # Успех — выходим
       except Exception as e:
           logging.error(f"[TREMOR REST] Attempt {attempt + 1} failed for {symbol}: {e}")
           time.sleep(1)
           if attempt == max_retries - 1:
               return None


   if len(prices) < 3:
       logging.warning(f"[TREMOR] Недостаточно данных для {symbol}: {len(prices)}")
       return None


   diffs = np.diff(prices)
   up_moves = np.sum(diffs > 0)
   down_moves = np.sum(diffs < 0)
   total_moves = up_moves + down_moves


   if total_moves == 0:
       return None


   direction = "Long" if up_moves > down_moves else "Short"
   confidence = abs(up_moves - down_moves) / total_moves * 100


   return {
       'direction': direction,
       'confidence': round(confidence, 1),
       'up_moves': up_moves,
       'down_moves': down_moves,
       'method': 'REST',
       'tremor_up_prob': 100 if direction == "Long" else 0
   }


# === ТОП МОНЕТЫ (ОБНОВЛЁННАЯ ВЕРСИЯ ДЛЯ НОВОГО ФОРМАТА) ===
def get_top_symbols(dev_filter_enabled=False, dev_threshold=3.0):
   if not os.path.exists(TOP_COINS_FILE):
       logging.info("top_coins.json не найден → используем GOOD_SYMBOLS")
       return GOOD_SYMBOLS[:10]


   try:
       with open(TOP_COINS_FILE, 'r', encoding='utf-8') as f:
           data = json.load(f)


       extracted = []


       # === Новый формат: список объектов с полем "pair" ===
       if isinstance(data, list) and data and isinstance(data[0], dict):
           for item in data:
               pair = item.get("pair") or item.get("Пара")  # на случай смешанного формата
               dev = abs(item.get("dev", 0))
               if pair and isinstance(pair, str):
                   pair = pair.strip().upper()
                   if pair.endswith("USDT"):
                       if not dev_filter_enabled or dev > dev_threshold:
                           extracted.append(pair)
           logging.info(f"Успешно загружено {len(extracted)} пар из нового формата top_coins.json (после фильтра)")
           if len(extracted) == 0 and dev_filter_enabled:
               logging.info("Нет монет, прошедших фильтр dev")
           return extracted[:10]


       # === Старый формат: просто список строк ===
       if isinstance(data, list) and data and isinstance(data[0], str):
           old_list = [s.strip().upper() for s in data if isinstance(s, str) and s.strip().endswith("USDT")]
           logging.info(f"Загружен старый формат top_coins.json: {len(old_list)} пар")
           return old_list[:10]


       # === Очень старый формат с "Пара" внутри объекта (на всякий случай) ===
       if isinstance(data, list) and data and "Пара" in data[0]:
           old_list = [item["Пара"].strip().upper() for item in data if "Пара" in item]
           logging.warning("Обнаружен древний формат top_coins.json")
           return old_list[:10]


       logging.warning("Неизвестный формат top_coins.json — используем GOOD_SYMBOLS")
       return GOOD_SYMBOLS[:10]


   except Exception as e:
       logging.error(f"Ошибка чтения top_coins.json: {e}")
       return GOOD_SYMBOLS[:10]


# === РЕДАКТОР ВЕСОВ ===
class WeightsEditor:
   def __init__(self, parent, weights, all_pairs, save_callback):
       self.parent = parent
       self.weights = weights
       self.all_pairs = all_pairs
       self.save_callback = save_callback
       self.top = tk.Toplevel(parent)
       self.top.title("Редактирование весов")
       self.top.geometry("400x300")
       self.top.grab_set()


       tk.Label(self.top, text="Выберите пару:").pack(pady=5)
       self.symbol_var = tk.StringVar()
       self.symbol_cb = ttk.Combobox(self.top, textvariable=self.symbol_var, values=self.all_pairs)
       self.symbol_cb.pack()
       self.symbol_cb.bind('<<ComboboxSelected>>', self.load_weights)


       tk.Label(self.top, text="Up Weight:").pack(pady=5)
       self.up_var = tk.DoubleVar(value=1.0)
       tk.Entry(self.top, textvariable=self.up_var).pack()


       tk.Label(self.top, text="Down Weight:").pack(pady=5)
       self.down_var = tk.DoubleVar(value=1.0)
       tk.Entry(self.top, textvariable=self.down_var).pack()


       btns = tk.Frame(self.top)
       btns.pack(pady=10)
       tk.Button(btns, text="+ Up", command=lambda: self.adjust('up', 0.1)).pack(side='left', padx=5)
       tk.Button(btns, text="- Up", command=lambda: self.adjust('up', -0.1)).pack(side='left', padx=5)
       tk.Button(btns, text="+ Down", command=lambda: self.adjust('down', 0.1)).pack(side='left', padx=5)
       tk.Button(btns, text="- Down", command=lambda: self.adjust('down', -0.1)).pack(side='left', padx=5)
       tk.Button(self.top, text="Сохранить", command=self.save).pack(pady=10)


   def load_weights(self, event=None):
       sym = self.symbol_var.get()
       if sym in self.weights:
           self.up_var.set(self.weights[sym].get('up_weight', 1.0))
           self.down_var.set(self.weights[sym].get('down_weight', 1.0))
       else:
           self.up_var.set(1.0)
           self.down_var.set(1.0)


   def adjust(self, typ, delta):
       val = self.up_var.get() if typ == 'up' else self.down_var.get()
       new_val = max(0.1, val + delta)
       if typ == 'up':
           self.up_var.set(new_val)
       else:
           self.down_var.set(new_val)


   def save(self):
       sym = self.symbol_var.get()
       if not sym: return
       if sym not in self.weights:
           self.weights[sym] = {'up_weight': 1.0, 'down_weight': 1.0, 'correct': 0, 'total': 0, 'streak': 0}
       self.weights[sym]['up_weight'] = self.up_var.get()
       self.weights[sym]['down_weight'] = self.down_var.get()
       self.save_callback(self.weights)
       self.top.destroy()


# === GUI ===
class CryptoApp:
   def __init__(self, root):
       self.root = root
       self.root.title("Крипто Прогноз v5.7 (с временем и ценой проверки)")
       self.root.geometry("1750x720")


       self.stop_training = False
       self.auto_thread = None


       self.all_pairs = get_futures_pairs()

       # === КРИТИЧЕСКАЯ ИНИЦИАЛИЗАЦИЯ ВСЕХ ВЕСОВ ===
       initialize_all_weights(self.all_pairs)

       # === ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ДО UI ===
       self.tremor_enabled = tk.BooleanVar(value=settings.get("tremor_enabled", False))
       self.tremor_interval = tk.IntVar(value=settings.get("tremor_interval", 60))
       self.dev_filter_enabled = tk.BooleanVar(value=settings.get("dev_filter_enabled", False))
       self.dev_threshold = tk.DoubleVar(value=settings.get("dev_threshold", 3.0))
       self.dev_forecast_enabled = tk.BooleanVar(value=False)


       # === Авто одна монета ===
       self.auto_single_enabled = tk.BooleanVar(value=False)
       self.auto_single_interval = tk.IntVar(value=300)
       self.auto_single_thread = None


       # === Переменные для системы продолжения обучения ===
       self.current_session = None
       self.auto_retrain = tk.BooleanVar(value=True)
       self.manual_weights = tk.BooleanVar(value=False)
       self.disable_weights = tk.BooleanVar(value=False)
       self.exclude_perfect = tk.BooleanVar(value=False)


       self.resume_button = None  # ссылка на кнопку


       self.setup_ui()


       self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


   def resumed_training_loop(self):
       session = self.current_session
       if not session:
           return


       symbols = session.get('symbols', [])
       horizon = session.get('horizon', self.horizon_seconds)
       wait = session.get('wait', 60)
       alpha = 0.1
       multiplier = session.get('multiplier', self.weight_multiplier)


       # Если уже были сделаны прогнозы, но не было проверки
       if session.get('predictions') and not session.get('verified', False):
           self.root.after(0, lambda: self.update_training_status("Проверка сохранённых прогнозов...", "blue"))
           self.perform_verification(session['predictions'])
           session['verified'] = True
           save_training_session(session)


       # Продолжаем нормальный цикл
       while not self.stop_training and self.auto_retrain.get():
           self.single_training_cycle(symbols, horizon, wait, alpha, multiplier)
           # После каждого цикла сохраняем пустую сессию (чтобы не было "зависшей")
           if os.path.exists(TRAINING_SESSION_FILE):
               try:
                   os.remove(TRAINING_SESSION_FILE)
               except:
                   pass


   def perform_verification(self, predictions):
       """Отдельная функция для проверки прогнозов и обновления весов.
       Используется и в обычном обучении, и при возобновлении после отключения."""


       if not predictions:
           self.root.after(0, lambda: self.update_training_status("Нет прогнозов для проверки", "red"))
           return


       self.root.after(0, lambda: self.update_training_status("Проверка прогнозов...", "blue"))
       verification_time = datetime.now().isoformat()


       # Если включены ручные веса или веса отключены — пропускаем обновление
       if self.manual_weights.get() or self.disable_weights.get():
           self.root.after(0, lambda: self.display_results(predictions))
           self.root.after(0, lambda: self.update_training_status("Проверка завершена (веса отключены)", "green"))
           return


       updated_any = False


       for p in predictions:
           sym = p['symbol']


           # Пропускаем идеально точные, если включена такая опция
           if self.exclude_perfect.get() and weights.get(sym, {}).get('accuracy', 0) >= 1.0:
               continue


           try:
               # Получаем текущую реальную цену
               current_price_data = client.get_symbol_ticker(symbol=sym)
               current_price = float(current_price_data['price'])


               # Записываем время и цену проверки в результат
               p['verification_time'] = verification_time
               p['verification_price'] = current_price


               # Создаём запись весов, если её ещё нет
               if sym not in weights:
                   weights[sym] = {
                       'up_weight': 1.0,
                       'down_weight': 1.0,
                       'correct': 0,
                       'total': 0,
                       'streak': 0,
                       'score': 0.0,
                       'history': []
                   }


               ww = weights[sym]
               factor = self.weight_multiplier * 0.1  # alpha = 0.1
               atr = p.get('atr') if isinstance(p.get('atr'), (int, float)) else None
               pred_up = p['decision'] == "Long"


               # Обновляем умные веса
               update_weights_smart(ww, pred_up, p['prediction_price'], current_price, factor, atr)


               # === Бальная система (Score) ===
               price_change_pct = (current_price - p['prediction_price']) / p['prediction_price'] * 100 \
                   if p['prediction_price'] != 0 else 0


               actual_up = current_price > p['prediction_price']
               prediction_correct = (pred_up == actual_up)


               if prediction_correct:
                   score_change = price_change_pct
               else:
                   score_change = -abs(price_change_pct)


               old_score = ww.get('score', 0.0)
               ww['score'] = max(-500, min(500, old_score + score_change))


               updated_any = True


               logging.info(
                   f"[{sym}] {'✅' if prediction_correct else '❌'} | "
                   f"Pred: {p['decision']} | Change: {price_change_pct:+.2f}% | "
                   f"Score: {old_score:+.2f} → {ww['score']:+.2f}"
               )


           except Exception as e:
               logging.error(f"Ошибка обновления для {sym}: {e}")


       # Сохраняем обновлённые веса
       if updated_any:
           save_json(WEIGHTS_FILE, weights)


       # Обновляем таблицу в GUI
       self.root.after(0, lambda: self.display_results(predictions))
       self.root.after(0, lambda: self.update_training_status("Проверка завершена", "green"))


   def on_closing(self):
       settings["tremor_enabled"] = self.tremor_enabled.get()
       settings["tremor_interval"] = self.tremor_interval.get()
       settings["dev_filter_enabled"] = self.dev_filter_enabled.get()
       settings["dev_threshold"] = self.dev_threshold.get()


       # === НОВЫЕ СТРОКИ ===
       settings["auto_single_enabled"] = self.auto_single_enabled.get()
       settings["auto_single_interval"] = self.auto_single_interval.get()


       save_json(SETTINGS_FILE, settings)
       self.root.destroy()


   def setup_ui(self):
       top = tk.Frame(self.root)
       top.pack(pady=10, fill='x')


       tk.Label(top, text="Пара:").grid(row=0, column=0, padx=5)
       self.pair_var = tk.StringVar()
       self.pair_cb = ttk.Combobox(top, textvariable=self.pair_var, values=self.all_pairs, width=15)
       self.pair_cb.grid(row=0, column=1, padx=5)
       self.pair_cb.bind('<KeyRelease>', self.filter_pairs)


       tk.Label(top, text="Горизонт (сек):").grid(row=0, column=2, padx=5)
       self.horizon_seconds = settings["horizon_seconds"]
       self.horizon_label = tk.Label(top, text=f"{self.horizon_seconds} сек")
       self.horizon_label.grid(row=0, column=3, padx=5)
       self.horizon_entry = tk.Entry(top, width=8)
       self.horizon_entry.grid(row=0, column=4, padx=5)
       self.horizon_entry.insert(0, str(self.horizon_seconds))
       tk.Label(top, text="(1+)").grid(row=0, column=5, padx=5)
       tk.Button(top, text="OK", command=self.save_horizon).grid(row=0, column=6, padx=5)


       opts = tk.Frame(self.root)
       opts.pack(pady=5)
       self.use_lstm = tk.BooleanVar(value=True)
       self.manual_weights = tk.BooleanVar()
       self.disable_weights = tk.BooleanVar()
       self.exclude_perfect = tk.BooleanVar()
       self.auto_retrain = tk.BooleanVar(value=True)
       self.use_top_mode = tk.BooleanVar()
       self.use_n_candles = tk.BooleanVar(value=False)


       tk.Checkbutton(opts, text="LSTM", variable=self.use_lstm).pack(side='left', padx=10)
       tk.Checkbutton(opts, text="Ручные веса", variable=self.manual_weights).pack(side='left', padx=10)
       tk.Checkbutton(opts, text="Без весов", variable=self.disable_weights).pack(side='left', padx=10)
       tk.Checkbutton(opts, text="Искл. 100%", variable=self.exclude_perfect).pack(side='left', padx=10)
       tk.Checkbutton(opts, text="Авто-обучение", variable=self.auto_retrain).pack(side='left', padx=10)
       tk.Checkbutton(opts, text="Авто-топ", variable=self.use_top_mode, command=self.toggle_auto).pack(side='left',
                                                                                                        padx=10)
       # === НОВАЯ ГАЛОЧКА ===
       tk.Checkbutton(opts, text="Авто одна монета", variable=self.auto_single_enabled,
                      command=self.toggle_auto_single).pack(side='left', padx=10)


       # Настройка интервала
       single_frame = tk.Frame(opts)
       single_frame.pack(side='left', padx=8)
       tk.Label(single_frame, text="Каждые:").pack(side='left')
       tk.Entry(single_frame, width=5, textvariable=self.auto_single_interval).pack(side='left', padx=3)
       tk.Label(single_frame, text="сек").pack(side='left')


       tk.Checkbutton(opts, text="N свечей (тренд)", variable=self.use_n_candles).pack(side='left', padx=10)
       tk.Checkbutton(opts, text="Dev Forecast", variable=self.dev_forecast_enabled).pack(side='left', padx=10)


       # === Дрожание цены ===
       tremor_frame = tk.Frame(opts)
       tremor_frame.pack(side='left', padx=20)
       tk.Checkbutton(tremor_frame, text="Дрожание цены", variable=self.tremor_enabled,
                      command=self.on_tremor_toggle).pack(side='left')
       tk.Label(tremor_frame, text="Интервал (5-1000 сек):").pack(side='left')
       self.tremor_entry = tk.Entry(tremor_frame, width=6, textvariable=self.tremor_interval)
       self.tremor_entry.pack(side='left', padx=2)
       tk.Button(tremor_frame, text="OK", command=self.save_tremor_interval).pack(side='left')


       # === ПАНЕЛЬ НАСТРОЕК ===
       settings_frame = tk.Frame(self.root)
       settings_frame.pack(pady=5)


       # Интервал
       interval_frame = tk.Frame(settings_frame)
       interval_frame.pack(side='left', padx=20)
       self.auto_interval = settings["auto_interval"]
       self.current_interval_label = tk.Label(interval_frame, text=f"Интервал: {self.auto_interval} сек")
       self.current_interval_label.pack(side='left')
       tk.Label(interval_frame, text="Новый:").pack(side='left', padx=5)
       self.interval_entry = tk.Entry(interval_frame, width=6)
       self.interval_entry.pack(side='left', padx=5)
       tk.Button(interval_frame, text="OK", command=self.save_interval).pack(side='left')


       # Множитель
       multiplier_frame = tk.Frame(settings_frame)
       multiplier_frame.pack(side='left', padx=20)
       self.weight_multiplier = settings["weight_multiplier"]
       self.current_multiplier_label = tk.Label(multiplier_frame, text=f"Множитель: x{self.weight_multiplier}")
       self.current_multiplier_label.pack(side='left')
       tk.Label(multiplier_frame, text="Новый:").pack(side='left', padx=5)
       self.multiplier_entry = tk.Entry(multiplier_frame, width=8)
       self.multiplier_entry.pack(side='left', padx=5)
       self.multiplier_entry.insert(0, str(self.weight_multiplier))
       tk.Label(multiplier_frame, text="(x1+)").pack(side='left', padx=5)
       tk.Button(multiplier_frame, text="OK", command=self.save_multiplier).pack(side='left')
       self.extreme_mode_label = tk.Label(multiplier_frame, text="", fg="red")
       self.extreme_mode_label.pack(side='left', padx=10)


       # Свечи (N)
       history_frame = tk.Frame(settings_frame)
       history_frame.pack(side='left', padx=20)
       self.history_candles = settings["history_candles"]
       self.current_history_label = tk.Label(history_frame, text=f"Свечи (N): {self.history_candles}")
       self.current_history_label.pack(side='left')
       tk.Label(history_frame, text="Новое:").pack(side='left', padx=5)
       self.history_entry = tk.Entry(history_frame, width=6)
       self.history_entry.pack(side='left', padx=5)
       self.history_entry.insert(0, str(self.history_candles))
       tk.Label(history_frame, text="(2-1500)").pack(side='left', padx=5)
       tk.Button(history_frame, text="OK", command=self.save_history).pack(side='left')


       # Dev filter
       dev_frame = tk.Frame(settings_frame)
       dev_frame.pack(side='left', padx=20)
       tk.Checkbutton(dev_frame, text="Фильтр dev", variable=self.dev_filter_enabled).pack(side='left')
       self.current_dev_label = tk.Label(dev_frame, text=f"Порог: {self.dev_threshold.get()}")
       self.current_dev_label.pack(side='left', padx=5)
       tk.Label(dev_frame, text="Новый:").pack(side='left', padx=5)
       self.dev_entry = tk.Entry(dev_frame, width=6, textvariable=self.dev_threshold)
       self.dev_entry.pack(side='left', padx=5)
       tk.Button(dev_frame, text="OK", command=self.save_dev_threshold).pack(side='left')


       self.training_status = tk.Label(self.root, text="Обучение: не запущено", fg="gray")
       self.training_status.pack(pady=2)


       self.last_check_label = tk.Label(self.root, text="Проверка: - | Монета: -")
       self.last_check_label.pack(pady=5)


       self.progress = ttk.Progressbar(self.root, mode='determinate')
       self.progress.pack(fill='x', padx=20, pady=5)


       btns = tk.Frame(self.root)
       btns.pack(pady=5)
       tk.Button(btns, text="Прогноз", command=self.predict).pack(side='left', padx=5)
       tk.Button(btns, text="Топ", command=self.predict_top).pack(side='left', padx=5)
       tk.Button(btns, text="Обучение", command=self.start_training).pack(side='left', padx=5)
       tk.Button(btns, text="Продолжить обучение", command=self.resume_training).pack(side='left', padx=5)
       tk.Button(btns, text="Стоп", command=self.stop_training_func).pack(side='left', padx=5)
       tk.Button(btns, text="Веса", command=self.open_weights_editor).pack(side='left', padx=5)


       # Обновлённые колонки: убрали Long Pos/Short Pos, добавили время/цену
       # Обновлённые колонки с EV$
       cols = ('Pair', 'Up', 'Down', 'Diff', 'Change', 'Decision', 'Acc', 'Streak', 'EV$', 'Score',
               'N-Candles', 'Tremor', 'Long Vol', 'Short Vol', 'Pred Time', 'Pred Price',
               'Verif Time', 'Verif Price', 'Fund(bps)', 'Fund MA', 'Curr Fund', 'L/S Ratio', 'OI')


       widths = [70, 45, 45, 45, 55, 70, 45, 70, 55, 65, 60, 60, 70, 70, 100, 70, 100, 70, 60, 60, 70, 50, 70]


       self.tree = ttk.Treeview(self.root, columns=cols, show='headings', height=15)
       for c, w in zip(cols, widths):
           self.tree.heading(c, text=c)
           self.tree.column(c, width=w, anchor='center')
       self.tree.pack(fill='both', expand=True, padx=20, pady=10)


       scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.tree.yview)
       self.tree.configure(yscrollcommand=scrollbar.set)
       scrollbar.pack(side='right', fill='y')


   def save_dev_threshold(self):
       try:
           val = float(self.dev_entry.get())
           if val >= 0:
               self.dev_threshold.set(val)
               settings["dev_threshold"] = val
               self.current_dev_label.config(text=f"Порог: {val}")
       except:
           pass


   def on_tremor_toggle(self):
       settings["tremor_enabled"] = self.tremor_enabled.get()


   def save_horizon(self):
       try:
           val = int(self.horizon_entry.get())
           if val >= 1:
               self.horizon_seconds = val
               settings["horizon_seconds"] = val
               self.horizon_label.config(text=f"{val} сек")
       except:
           pass


   def save_interval(self):
       try:
           val = int(self.interval_entry.get())
           if val > 0:
               self.auto_interval = val
               settings["auto_interval"] = val
               self.current_interval_label.config(text=f"Интервал: {val} сек")
       except:
           pass


   def save_multiplier(self):
       try:
           val = float(self.multiplier_entry.get())
           if val >= 1:
               self.weight_multiplier = val
               settings["weight_multiplier"] = val
               self.current_multiplier_label.config(text=f"Множитель: x{val}")
               if val > 10:
                   self.extreme_mode_label.config(text="ЭКСТРИМ-РЕЖИМ", fg="red")
               else:
                   self.extreme_mode_label.config(text="")
       except:
           pass


   def save_history(self):
       try:
           val = int(self.history_entry.get())
           if 2 <= val <= 1500:
               self.history_candles = val
               settings["history_candles"] = val
               self.current_history_label.config(text=f"Свечи (N): {val}")
       except:
           pass


   def save_tremor_interval(self):
       try:
           val = int(self.tremor_interval.get())
           if 5 <= val <= 1000:
               settings["tremor_interval"] = val
               self.tremor_interval.set(val)
       except:
           pass


   def filter_pairs(self, event):
       typed = self.pair_var.get().upper()
       filtered = [p for p in self.all_pairs if typed in p]
       self.pair_cb['values'] = filtered


   def update_progress(self, val):
       self.progress['value'] = val
       self.root.update_idletasks()


   def update_training_status(self, text, color="black"):
       self.training_status.config(text=text, fg=color)
       self.root.update_idletasks()


   def update_last_check(self, text):
       self.last_check_label.config(text=f"Проверка: {datetime.now():%H:%M:%S} | {text}")


   def toggle_auto(self):
       if self.use_top_mode.get():
           if not self.auto_thread or not self.auto_thread.is_alive():
               self.auto_thread = threading.Thread(target=self.auto_loop, daemon=True)
               self.auto_thread.start()
       else:
           self.stop_training = True


   def auto_loop(self):
       while self.use_top_mode.get():
           self.predict_top()
           time.sleep(self.auto_interval)


   def combine_nc_tremor(self, base_result, tremor):
       """
       Комбинирует N-свечей и tremor:
       - Если направления совпадают: 100% в эту сторону.
       - Если разные: 55% в tremor, 45% в N-свечей.
       Возвращает обновлённые up_prob, down_prob, prob_diff, decision, predicted_change.
       predicted_change остаётся от N-свечей (в %).
       """
       if not tremor or 'tremor_up_prob' not in tremor:
           return base_result  # Без tremor — как было


       nc_dir = base_result['n_candles_dir']
       nc_up = base_result.get('n_candles_up_prob', base_result['up_prob'])  # 100 или 0 для N-свечей
       tr_dir = tremor['direction']
       tr_up = tremor['tremor_up_prob']  # 100 или 0


       same_dir = nc_dir == tr_dir


       if same_dir:
           final_up = 100 if nc_dir == "Long" else 0
       else:
           # 55% в tremor, 45% в N-свечей
           if tr_dir == "Long":
               final_up = 55 + 45 * (nc_up / 100)
           else:
               final_up = 45 * (nc_up / 100)


       final_up = round(final_up, 2)
       final_down = round(100 - final_up, 2)
       final_diff = round(abs(final_up - final_down), 2)
       final_decision = "Long" if final_up > final_down else "Short"


       base_result['up_prob'] = final_up
       base_result['down_prob'] = final_down
       base_result['prob_diff'] = final_diff
       base_result['decision'] = final_decision
       # predicted_change остаётся от N-свечей


       return base_result


   def predict(self):
       symbol = self.pair_var.get().strip()
       if not symbol:
           return
       horizon = self.horizon_seconds
       self.tree.delete(*self.tree.get_children())
       self.tree.insert('', 'end', values=('Загрузка...',) + ('',) * 20)  # 21 для новых колонок


       def run():
           w = None if self.disable_weights.get() else weights
           lstm = self.use_lstm.get() and not self.use_n_candles.get()
           use_n = self.use_n_candles.get()
           n_val = self.history_candles if use_n else 500
           use_dev = self.dev_forecast_enabled.get()


           base_result = predict_symbol(symbol, horizon, lstm, w, self.history_candles, use_n, n_val, use_dev)
           if not base_result:
               self.root.after(0, lambda: self.display_results([]))
               return


           tremor = None
           tremor_str = "-"
           if use_n and self.tremor_enabled.get():
               self.root.after(0, lambda: self.tree.delete(*self.tree.get_children()) or
                                          self.tree.insert('', 'end', values=('Мониторинг дрожания...',) + ('',) * 20))
               interval = settings["tremor_interval"]
               tremor = monitor_tremor(symbol, interval)
               if tremor:
                   tremor_str = f"{tremor['direction']} ({tremor['confidence']}%) [{tremor['method']}]"
                   logging.info(f"[TREMOR FINAL] {symbol}: {tremor['direction']}")
               else:
                   tremor_str = "Нет данных"


           # Комбинируем только в режиме N-свечей + tremor
           if use_n and tremor:
               base_result = self.combine_nc_tremor(base_result, tremor)


           base_result['tremor'] = tremor_str
           # Для predict: verification не доступна, оставляем пустыми
           base_result['verification_time'] = '-'
           base_result['verification_price'] = '-'


           self.root.after(0, lambda: self.display_results([base_result]))


       threading.Thread(target=run, daemon=True).start()


   def predict_top(self):
       symbols = get_top_symbols(self.dev_filter_enabled.get(), self.dev_threshold.get())
       if not symbols:
           self.tree.delete(*self.tree.get_children())
           self.tree.insert('', 'end', values=('Нету подходящего топа, ждём следующий прогноз',) + ('',) * 20)
           return


       horizon = self.horizon_seconds
       self.tree.delete(*self.tree.get_children())
       self.tree.insert('', 'end', values=('Проверка топ-1...',) + ('',) * 20)


       def run():
           w = None if self.disable_weights.get() else weights
           lstm = self.use_lstm.get() and not self.use_n_candles.get()
           use_n = self.use_n_candles.get()
           n_val = self.history_candles if use_n else 500
           use_dev = self.dev_forecast_enabled.get()
           result = None
           used_symbol = None


           for idx, symbol in enumerate(symbols):
               if self.stop_training: break
               self.root.after(0, lambda s=symbol, i=idx + 1: self.update_last_check(f"{s} (топ-{i})"))


               base = predict_symbol(symbol, horizon, lstm, w, self.history_candles, use_n, n_val, use_dev)
               if not base:
                   continue


               tremor = None
               tremor_str = "-"
               if use_n and self.tremor_enabled.get():
                   interval = settings["tremor_interval"]
                   tremor = monitor_tremor(symbol, interval)
                   if tremor:
                       tremor_str = f"{tremor['direction']} ({tremor['confidence']}%) [{tremor['method']}]"


               if use_n and tremor:
                   base = self.combine_nc_tremor(base, tremor)


               base['tremor'] = tremor_str
               # Для predict_top: verification не доступна
               base['verification_time'] = '-'
               base['verification_price'] = '-'
               result = base
               used_symbol = symbol
               break


           self.root.after(0, lambda: self.display_results([result] if result else []))
           if result:
               self.root.after(0, lambda: self.update_last_check(f"ГОТОВО: {used_symbol}"))


       threading.Thread(target=run, daemon=True).start()


   def display_results(self, results):
       self.tree.delete(*self.tree.get_children())


       # Сколько последних прогнозов учитывать
       window = settings.get("recent_accuracy_window", 10)


       for r in results:
           if not r:
               continue


           sym = r['symbol']
           w = weights.get(sym, {})


           # Расчёт процента из последних N прогнозов
           recent_acc = get_recent_accuracy(w, window)


           streak = r.get('streak', 0)
           score = w.get('score', 0.0)


           streak_display = f"{streak} ({recent_acc}%)"


           # EV$ считаем по недавней точности
           ev_dollars = calculate_ev(
               recent_acc,
               settings.get("rr_ratio", 2.0),
               settings.get("trade_size_usd", 20.0),
               settings.get("commission_rate", 0.00045)
           )


           n_dir = r.get('n_candles_dir', r['decision'])
           tremor_str = r.get('tremor', '-')
           long_vol = r.get('long_volume', 0)
           short_vol = r.get('short_volume', 0)
           pred_time = r.get('prediction_time', '-')
           pred_price = round(r.get('prediction_price', 0), 6)
           verif_time = r.get('verification_time', '-')
           verif_price = round(r.get('verification_price', 0), 6) if isinstance(r.get('verification_price'),
                                                                                (int, float)) else '-'


           funding_rate = round(r.get('funding_rate', 0), 2)
           funding_ma = round(r.get('funding_rate_ma', 0), 2)
           curr_funding = round(r.get('current_funding_rate', 0), 2)
           long_short_ratio = round(r.get('long_short_ratio', 1.0), 2)
           oi = round(r.get('oi', 0), 2)


           self.tree.insert('', 'end', values=(
               sym, r['up_prob'], r['down_prob'], r['prob_diff'],
               r['predicted_change'], r['decision'], f"{recent_acc:.1f}%",
               streak_display,
               ev_dollars,
               round(score, 2),
               n_dir, tremor_str, long_vol, short_vol,
               pred_time, pred_price, verif_time, verif_price,
               funding_rate, funding_ma, curr_funding, long_short_ratio, oi
           ))


           save_json(RECOMMENDED_TRADE_FILE, r)
           self.update_last_check(sym)


   def open_weights_editor(self):
       WeightsEditor(self.root, weights, self.all_pairs, save_json)


   def start_training(self):
       if self.auto_thread and self.auto_thread.is_alive():
           self.update_training_status("Обучение уже запущено", "red")
           return
       self.stop_training = False
       self.auto_thread = threading.Thread(target=self.training_loop, daemon=True)
       self.auto_thread.start()
       self.update_training_status("Обучение запущено...", "green")


   def stop_training_func(self):
       self.stop_training = True
       self.update_training_status("Обучение остановлено", "red")


   def training_loop(self):
       horizon = self.horizon_seconds
       wait = max(horizon, 60)
       alpha = 0.1
       multiplier = self.weight_multiplier


       selected_symbol = self.pair_var.get().strip()
       symbols_to_train = [selected_symbol] if selected_symbol else self.all_pairs


       self.root.after(0, lambda: self.update_training_status(f"Обучение: {len(symbols_to_train)} монет", "blue"))


       if not self.auto_retrain.get():
           self.single_training_cycle(symbols_to_train, horizon, wait, alpha, multiplier)
           self.root.after(0, lambda: self.update_training_status("Обучение завершено (один цикл)", "green"))
           return


       while not self.stop_training and self.auto_retrain.get():
           self.single_training_cycle(symbols_to_train, horizon, wait, alpha, multiplier)


       self.root.after(0, lambda: self.update_training_status("Обучение остановлено", "red"))


   def single_training_cycle(self, symbols, horizon, wait, alpha, multiplier):
       start = time.time()


       # === 1. Создаём сессию (снимок текущего цикла) ===
       session = {
           'symbols': symbols[:],  # список монет
           'horizon': horizon,
           'wait': wait,
           'multiplier': multiplier,
           'start_time': datetime.now().isoformat(),
           'predictions': [],  # сюда сохраним прогнозы
           'verified': False,
           'elapsed_wait': 0  # сколько уже отжали во время ожидания
       }


       w = None if self.disable_weights.get() else weights
       lstm = self.use_lstm.get() and not self.use_n_candles.get()
       use_n = self.use_n_candles.get()
       n_val = self.history_candles if use_n else 500
       use_dev = self.dev_forecast_enabled.get()


       predictions = []


       self.root.after(0, lambda: self.update_training_status("Прогноз...", "blue"))
       self.root.after(0, lambda: self.update_progress(0))


       # === 2. Делаем прогнозы ===
       with ThreadPoolExecutor(max_workers=5) as exec:
           futures = [exec.submit(predict_symbol, s, horizon, lstm, w,
                                  self.history_candles, use_n, n_val, use_dev)
                      for s in symbols]


           for i, f in enumerate(futures):
               if self.stop_training:
                   save_training_session(session)
                   return
               r = f.result()
               if r:
                   predictions.append(r)
                   self.root.after(0, lambda s=r['symbol']: self.update_last_check(s))
               self.update_progress((i + 1) / len(symbols) * 100)


       if not predictions:
           self.root.after(0, lambda: self.update_training_status("Нет прогноза", "red"))
           time.sleep(3)
           return


       # === 3. Дрожание цены (если нужно) ===
       for p in predictions:
           if use_n and self.tremor_enabled.get():
               tremor = monitor_tremor(p['symbol'], settings["tremor_interval"])
               if tremor:
                   p = self.combine_nc_tremor(p, tremor)
                   p['tremor'] = f"{tremor['direction']} ({tremor['confidence']}%) [{tremor['method']}]"


       # Сохраняем прогнозы в сессию
       session['predictions'] = predictions
       save_training_session(session)  # ← Важное сохранение!


       self.root.after(0, lambda: self.display_results(predictions))
       self.root.after(0, lambda: self.update_training_status(f"Ожидание {wait} сек...", "orange"))


       # === 4. УМНОЕ ОЖИДАНИЕ С АВТОСОХРАНЕНИЕМ ===
       elapsed = time.time() - start
       sleep_time = max(1, wait - elapsed)


       save_every = 600  # сохранять каждые 10 минут (600 секунд)
       last_save_time = time.time()


       for second in range(int(sleep_time)):
           if self.stop_training:
               save_training_session(session)
               return

           time.sleep(1)

           # Обновляем, сколько уже ждём
           session['elapsed_wait'] = int(time.time() - start)


           # Автосохранение каждые 10 минут
           if time.time() - last_save_time >= save_every:
               save_training_session(session)
               last_save_time = time.time()
               logging.info(f"Сессия обучения сохранена (прошло {session['elapsed_wait']} сек)")


           # Прогресс-бар
           remaining = wait - (time.time() - start)
           progress = int(100 * (wait - remaining) / wait)
           self.root.after(0, lambda p=progress: self.update_progress(p))


       # === 5. ПРОВЕРКА ПРОГНОЗОВ ===
       if self.stop_training:
           save_training_session(session)
           return


       self.perform_verification(predictions)   # ← Главный вызов


       # === 6. УДАЛЯЕМ СЕССИЮ — цикл успешно завершён ===
       if os.path.exists(TRAINING_SESSION_FILE):
           try:
               os.remove(TRAINING_SESSION_FILE)
           except:
               pass


       self.root.after(0, lambda: self.update_training_status("Цикл завершён", "green"))


       if not self.manual_weights.get() and not self.disable_weights.get():
           for p in predictions:
               sym = p['symbol']
               if self.exclude_perfect.get() and weights.get(sym, {}).get('accuracy', 0) >= 1.0:
                   continue
               try:
                   current_price_data = client.get_symbol_ticker(symbol=sym)
                   current_price = float(current_price_data['price'])


                   p['verification_time'] = verification_time
                   p['verification_price'] = current_price


                   if sym not in weights:
                       weights[sym] = {
                           'up_weight': 1.0, 'down_weight': 1.0,
                           'correct': 0, 'total': 0, 'streak': 0,
                           'score': 0.0, 'history': []
                       }


                   ww = weights[sym]
                   factor = multiplier * alpha
                   atr = p.get('atr') if isinstance(p.get('atr'), (int, float)) else None
                   pred_up = p['decision'] == "Long"


                   update_weights_smart(ww, pred_up, p['prediction_price'], current_price, factor, atr)


                   # Бальная система
                   price_change_pct = (current_price - p['prediction_price']) / p['prediction_price'] * 100 if p[
                                                                                                                   'prediction_price'] != 0 else 0
                   actual_up = current_price > p['prediction_price']
                   prediction_correct = (pred_up == actual_up)


                   if prediction_correct:
                       score_change = price_change_pct
                   else:
                       score_change = -abs(price_change_pct)


                   old_score = ww.get('score', 0.0)
                   ww['score'] = max(-500, min(500, old_score + score_change))


                   logging.info(
                       f"[{sym}] {'✅' if prediction_correct else '❌'} | Pred: {p['decision']} | Change: {price_change_pct:+.2f}% | Score: {old_score:+.2f} → {ww['score']:+.2f}")


               except Exception as e:
                   logging.error(f"Ошибка обновления для {sym}: {e}")


           save_json(WEIGHTS_FILE, weights)


       self.root.after(0, lambda: self.display_results(predictions))
       self.root.after(0, lambda: self.update_training_status("Цикл завершён", "green"))


       # === 6. УДАЛЯЕМ СЕССИЮ — цикл успешно завершён ===
       if os.path.exists(TRAINING_SESSION_FILE):
           try:
               os.remove(TRAINING_SESSION_FILE)
           except:
               pass


   def resume_training(self):
       """Кнопка 'Продолжить обучение'"""
       session = load_training_session()
       if not session:
           self.update_training_status("Нет сохранённой незавершённой сессии", "red")
           return


       self.current_session = session
       self.stop_training = False
       self.auto_thread = threading.Thread(target=self.resumed_training_loop, daemon=True)
       self.auto_thread.start()
       self.update_training_status("✅ Возобновляем обучение...", "green")


   def resumed_training_loop(self):
       """Цикл, который запускается при продолжении"""
       session = self.current_session
       if not session:
           return


       symbols = session.get('symbols', [])
       horizon = session.get('horizon', self.horizon_seconds)
       wait = session.get('wait', 60)
       multiplier = session.get('multiplier', self.weight_multiplier)
       alpha = 0.1


       # === Самое важное: если прогнозы были сделаны, но проверка ещё не прошла ===
       if session.get('predictions') and not session.get('verified', False):
           self.root.after(0, lambda: self.update_training_status("Проверяем старые прогнозы...", "blue"))


           # Делаем проверку тех прогнозов, которые были сохранены
           self.perform_verification(session['predictions'])


           session['verified'] = True
           save_training_session(session)


       # === Продолжаем обычные циклы обучения ===
       while not self.stop_training and self.auto_retrain.get():
           self.single_training_cycle(symbols, horizon, wait, alpha, multiplier)


           # После успешного полного цикла удаляем сессию
           if os.path.exists(TRAINING_SESSION_FILE):
               try:
                   os.remove(TRAINING_SESSION_FILE)
               except:
                   pass


   def extreme_retrain(self, symbol, horizon_seconds, was_correct, multiplier):
       try:
           interval_binance, horizon = get_binance_interval(horizon_seconds)
           df = get_historical_data(symbol, interval_binance, limit=self.history_candles)
           if df is None or len(df) < 50: return
           df = calculate_indicators(df)
           if df is None: return


           X, y_class, _ = prepare_data(df, horizon)
           if len(X) < 50: return


           latest = df.tail(1)[FEATURES]


           clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
           clf.fit(X, y_class)
           pred_up = clf.predict_proba(latest)[0][1] > 0.5


           repeat_count = max(1, int(multiplier / 10))
           if was_correct:
               correct_class = 1 if pred_up else 0
               mask = y_class == correct_class
               X_boost = pd.concat([X] + [X[mask]] * repeat_count, ignore_index=True)
               y_boost = pd.concat([y_class] + [y_class[mask]] * repeat_count, ignore_index=True)
           else:
               wrong_class = 0 if pred_up else 1
               mask = y_class == wrong_class
               X_boost = pd.concat([X] + [X[mask]] * repeat_count, ignore_index=True)
               y_boost = pd.concat([y_class] + [y_class[mask]] * repeat_count, ignore_index=True)


           clf.fit(X_boost, y_boost)


           ww = weights[symbol]
           if was_correct:
               ww['correct'] += 1
               ww['streak'] = ww.get('streak', 0) + 1
               key = 'up_weight' if pred_up else 'down_weight'
               ww[key] *= (1 + 0.5 * (multiplier / 100))
           else:
               ww['streak'] = 0
               key = 'up_weight' if pred_up else 'down_weight'
               ww[key] *= 0.5


       except Exception as e:
           logging.error(f"Extreme retrain error: {e}")

   # ==================== НОВАЯ АВТО-ФУНКЦИЯ ====================
   def toggle_auto_single(self):
       """Включает и выключает авто-прогноз одной монеты"""
       if self.auto_single_enabled.get():
           if not self.auto_single_thread or not self.auto_single_thread.is_alive():
               self.auto_single_thread = threading.Thread(target=self.auto_single_loop, daemon=True)
               self.auto_single_thread.start()
               print("✅ Авто-прогноз одной монеты ЗАПУЩЕН")
       else:
           print("⛔ Авто-прогноз одной монеты ОСТАНОВЛЕН")


   def auto_single_loop(self):
       """Главный цикл: каждые N секунд делает прогноз выбранной монеты"""
       while self.auto_single_enabled.get():
           symbol = self.pair_var.get().strip()


           if symbol:
               self.root.after(0, lambda s=symbol: self.update_last_check(f"Авто-одна → {s}"))
               self.run_single_prediction(symbol)
           else:
               self.root.after(0, lambda: self.update_last_check("Выбери монету!"))


           # Ждём нужное количество секунд
           wait_seconds = self.auto_single_interval.get()
           for _ in range(wait_seconds):
               if not self.auto_single_enabled.get():
                   return
               time.sleep(1)


   def run_single_prediction(self, symbol):
       """Делает один прогноз по выбранной монете"""


       def task():
           w = None if self.disable_weights.get() else weights
           lstm = self.use_lstm.get() and not self.use_n_candles.get()
           use_n = self.use_n_candles.get()
           n_val = self.history_candles if use_n else 500
           use_dev = self.dev_forecast_enabled.get()


           result = predict_symbol(symbol, self.horizon_seconds, lstm, w,
                                   self.history_candles, use_n, n_val, use_dev)


           if not result:
               return


           # Дрожание цены (если включено)
           tremor_str = "-"
           if use_n and self.tremor_enabled.get():
               tremor = monitor_tremor(symbol, settings.get("tremor_interval", 60))
               if tremor:
                   tremor_str = f"{tremor['direction']} ({tremor['confidence']}%)"
                   if use_n:
                       result = self.combine_nc_tremor(result, tremor)


           result['tremor'] = tremor_str
           result['verification_time'] = '-'
           result['verification_price'] = '-'


           self.root.after(0, lambda: self.display_results([result]))

       threading.Thread(target=task, daemon=True).start()



# === ЗАПУСК ===
if __name__ == "__main__":
   root = tk.Tk()
   app = CryptoApp(root)
   root.mainloop()

