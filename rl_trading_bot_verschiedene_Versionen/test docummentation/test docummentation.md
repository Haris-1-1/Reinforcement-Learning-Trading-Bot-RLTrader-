v1
# ════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.001  # Strafe damit er nicht zu oft tradet
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # 
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 100000,  # 
            'log_interval': 10000
        }
    }

======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2020-01-01 to 2024-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 1461 rows of data
Date range: 2020-01-01 00:00:00 to 2023-12-31 00:00:00
Price range: $4970.79 - $67566.83
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 1412
Train set: 1129 rows (80%)
Test set: 283 rows (20%)

 Original Train Prices: $4970.79 - $67566.83
 Original Test Prices: $25124.68 - $44166.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 1129 days
 Test data: 283 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $4970.79 - $67566.83
   Price change over period: 194.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 100,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/100,000 | Ep   8 | ε=0.923 | Avg Reward: -0.3163
Step  20,000/100,000 | Ep  17 | ε=0.843 | Avg Reward: -0.2252
  └─ Episode 20: Reward=+0.3645 | Portfolio=$16,909 | Trades=395 | ε=0.818
Step  30,000/100,000 | Ep  26 | ε=0.770 | Avg Reward: +0.1185
Step  40,000/100,000 | Ep  35 | ε=0.703 | Avg Reward: +0.8296
  └─ Episode 40: Reward=+1.6044 | Portfolio=$55,638 | Trades=356 | ε=0.669
Step  50,000/100,000 | Ep  44 | ε=0.643 | Avg Reward: +1.3173
Step  60,000/100,000 | Ep  53 | ε=0.587 | Avg Reward: +1.7878
  └─ Episode 60: Reward=+2.4424 | Portfolio=$112,370 | Trades=383 | ε=0.547
Step  70,000/100,000 | Ep  62 | ε=0.536 | Avg Reward: +1.7784
Step  80,000/100,000 | Ep  70 | ε=0.495 | Avg Reward: +2.1588
Step  90,000/100,000 | Ep  79 | ε=0.452 | Avg Reward: +2.6126
  └─ Episode 80: Reward=+2.8900 | Portfolio=$182,850 | Trades=327 | ε=0.448
Step 100,000/100,000 | Ep  88 | ε=0.413 | Avg Reward: +2.8094

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 89
Final Epsilon: 0.4088
Mean Reward: 1.3286
Best Episode Reward: 4.0061
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251208_142444.pkl
Config saved to results/config_v2_20251208_142444.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $3,440,047.62
Return:  +34300.48%
Trades:  299
Fees:    $556726.03

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $25124.68 - $44166.60
   Price change over period: 53.7%
   Using 21 features for state
Initial: $10,000.00
Final:   $10,032.67
Return:  +0.33%
Trades:  72
Fees:    $1413.68

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |   +34300.48% |       +0.33%
Buy & Hold           |     +194.12% |      +53.73%
--------------------------------------------------
Outperformance       |   +34106.35% |      -53.40%

 Final Values:
   Agent Train: $3,440,047.62  |  Buy&Hold: $29,412.27
   Agent Test:  $10,032.67  |  Buy&Hold: $15,372.91

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251208_142444.pkl
Final Epsilon: 0.4088
Total Episodes: 89

 Agent made profit, but didn't beat Buy & Hold
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-> 

v2
======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2020-01-01 to 2024-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 1461 rows of data
Date range: 2020-01-01 00:00:00 to 2023-12-31 00:00:00
Price range: $4970.79 - $67566.83
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 1412
Train set: 1129 rows (80%)
Test set: 283 rows (20%)

 Original Train Prices: $4970.79 - $67566.83
 Original Test Prices: $25124.68 - $44166.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 1129 days
 Test data: 283 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $4970.79 - $67566.83
   Price change over period: 194.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 100,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/100,000 | Ep   8 | ε=0.923 | Avg Reward: -2.3362
Step  20,000/100,000 | Ep  17 | ε=0.843 | Avg Reward: -2.1242
  └─ Episode 20: Reward=-1.8576 | Portfolio=$10,180 | Trades=361 | ε=0.818
Step  30,000/100,000 | Ep  26 | ε=0.770 | Avg Reward: -1.5166
Step  40,000/100,000 | Ep  35 | ε=0.703 | Avg Reward: -0.9318
  └─ Episode 40: Reward=-0.2281 | Portfolio=$49,649 | Trades=353 | ε=0.669
Step  50,000/100,000 | Ep  44 | ε=0.643 | Avg Reward: -0.8388
Step  60,000/100,000 | Ep  53 | ε=0.587 | Avg Reward: +0.0175
  └─ Episode 60: Reward=+0.0508 | Portfolio=$42,417 | Trades=289 | ε=0.547
Step  70,000/100,000 | Ep  62 | ε=0.536 | Avg Reward: +0.3114
Step  80,000/100,000 | Ep  70 | ε=0.495 | Avg Reward: +0.4362
Step  90,000/100,000 | Ep  79 | ε=0.452 | Avg Reward: +1.0087
  └─ Episode 80: Reward=+1.2158 | Portfolio=$143,383 | Trades=305 | ε=0.448
Step 100,000/100,000 | Ep  88 | ε=0.413 | Avg Reward: +1.3319

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 89
Final Epsilon: 0.4088
Mean Reward: -0.4235
Best Episode Reward: 2.6848
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251208_142606.pkl
Config saved to results/config_v2_20251208_142606.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $2,824,853.01
Return:  +28148.53%
Trades:  183
Fees:    $274969.72

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $25124.68 - $44166.60
   Price change over period: 53.7%
   Using 21 features for state
Initial: $10,000.00
Final:   $11,826.14
Return:  +18.26%
Trades:  48
Fees:    $952.74

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |   +28148.53% |      +18.26%
Buy & Hold           |     +194.12% |      +53.73%
--------------------------------------------------
Outperformance       |   +27954.41% |      -35.47%

 Final Values:
   Agent Train: $2,824,853.01  |  Buy&Hold: $29,412.27
   Agent Test:  $11,826.14  |  Buy&Hold: $15,372.91

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251208_142606.pkl
Final Epsilon: 0.4088
Total Episodes: 89

 Agent made profit, but didn't beat Buy & Hold
# ════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 100000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

v3
    # CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2020-01-01',
            'end_date': '2024-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 200000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2020-01-01 to 2024-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 1461 rows of data
Date range: 2020-01-01 00:00:00 to 2023-12-31 00:00:00
Price range: $4970.79 - $67566.83
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 1412
Train set: 1129 rows (80%)
Test set: 283 rows (20%)

 Original Train Prices: $4970.79 - $67566.83
 Original Test Prices: $25124.68 - $44166.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 1129 days
 Test data: 283 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $4970.79 - $67566.83
   Price change over period: 194.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 200,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/200,000 | Ep   8 | ε=0.923 | Avg Reward: -2.5727
Step  20,000/200,000 | Ep  17 | ε=0.843 | Avg Reward: -2.5756
  └─ Episode 20: Reward=-1.0622 | Portfolio=$25,050 | Trades=367 | ε=0.818
Step  30,000/200,000 | Ep  26 | ε=0.770 | Avg Reward: -1.1296
Step  40,000/200,000 | Ep  35 | ε=0.703 | Avg Reward: -0.8668
  └─ Episode 40: Reward=-0.2579 | Portfolio=$45,503 | Trades=338 | ε=0.669
Step  50,000/200,000 | Ep  44 | ε=0.643 | Avg Reward: -0.5567
Step  60,000/200,000 | Ep  53 | ε=0.587 | Avg Reward: -0.2616
  └─ Episode 60: Reward=+0.9930 | Portfolio=$136,154 | Trades=317 | ε=0.547
Step  70,000/200,000 | Ep  62 | ε=0.536 | Avg Reward: +0.3932
Step  80,000/200,000 | Ep  70 | ε=0.495 | Avg Reward: +0.6245
Step  90,000/200,000 | Ep  79 | ε=0.452 | Avg Reward: +1.2330
  └─ Episode 80: Reward=+0.6889 | Portfolio=$90,161 | Trades=305 | ε=0.448
Step 100,000/200,000 | Ep  88 | ε=0.413 | Avg Reward: +1.1900
Step 110,000/200,000 | Ep  97 | ε=0.377 | Avg Reward: +1.4717
  └─ Episode 100: Reward=+2.5152 | Portfolio=$415,283 | Trades=267 | ε=0.366
Step 120,000/200,000 | Ep 106 | ε=0.345 | Avg Reward: +1.9164
Step 130,000/200,000 | Ep 115 | ε=0.315 | Avg Reward: +2.0763
  └─ Episode 120: Reward=+1.8158 | Portfolio=$195,908 | Trades=265 | ε=0.299
Step 140,000/200,000 | Ep 124 | ε=0.288 | Avg Reward: +2.1597
Step 150,000/200,000 | Ep 132 | ε=0.265 | Avg Reward: +2.5632
  └─ Episode 140: Reward=+2.5928 | Portfolio=$494,640 | Trades=285 | ε=0.245
Step 160,000/200,000 | Ep 141 | ε=0.242 | Avg Reward: +2.8004
Step 170,000/200,000 | Ep 150 | ε=0.221 | Avg Reward: +3.0740
Step 180,000/200,000 | Ep 159 | ε=0.202 | Avg Reward: +3.2799
  └─ Episode 160: Reward=+4.5517 | Portfolio=$2,349,752 | Trades=235 | ε=0.200
Step 190,000/200,000 | Ep 168 | ε=0.185 | Avg Reward: +3.4065
Step 200,000/200,000 | Ep 177 | ε=0.169 | Avg Reward: +3.4850

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 178
Final Epsilon: 0.1671
Mean Reward: 1.1269
Best Episode Reward: 4.5517
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251208_143300.pkl
Config saved to results/config_v2_20251208_143300.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $3,249,803.98
Return:  +32398.04%
Trades:  179
Fees:    $335637.73

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $25124.68 - $44166.60
   Price change over period: 53.7%
   Using 21 features for state
Initial: $10,000.00
Final:   $12,653.44
Return:  +26.53%
Trades:  50
Fees:    $1069.40

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |   +32398.04% |      +26.53%
Buy & Hold           |     +194.12% |      +53.73%
--------------------------------------------------
Outperformance       |   +32203.92% |      -27.19%

 Final Values:
   Agent Train: $3,249,803.98  |  Buy&Hold: $29,412.27
   Agent Test:  $12,653.44  |  Buy&Hold: $15,372.91

======================================================================
 TRAINING COMPLETE!

v4
# ════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 200000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 200,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/200,000 | Ep  18 | ε=0.835 | Avg Reward: -0.9360
  └─ Episode 20: Reward=-0.5363 | Portfolio=$15,465 | Trades=165 | ε=0.818
Step  20,000/200,000 | Ep  36 | ε=0.696 | Avg Reward: -0.6086
  └─ Episode 40: Reward=-0.4716 | Portfolio=$17,141 | Trades=168 | ε=0.669
Step  30,000/200,000 | Ep  55 | ε=0.575 | Avg Reward: +0.0408
  └─ Episode 60: Reward=-0.0033 | Portfolio=$25,751 | Trades=165 | ε=0.547
Step  40,000/200,000 | Ep  73 | ε=0.480 | Avg Reward: +0.3837
  └─ Episode 80: Reward=+0.4545 | Portfolio=$34,989 | Trades=150 | ε=0.448
Step  50,000/200,000 | Ep  91 | ε=0.401 | Avg Reward: +0.4544
  └─ Episode 100: Reward=+0.7931 | Portfolio=$49,322 | Trades=145 | ε=0.366
Step  60,000/200,000 | Ep 110 | ε=0.331 | Avg Reward: +0.6918
  └─ Episode 120: Reward=+0.7383 | Portfolio=$52,084 | Trades=166 | ε=0.299
Step  70,000/200,000 | Ep 128 | ε=0.276 | Avg Reward: +1.0771
  └─ Episode 140: Reward=+1.2556 | Portfolio=$75,440 | Trades=146 | ε=0.245
Step  80,000/200,000 | Ep 147 | ε=0.228 | Avg Reward: +1.1008
  └─ Episode 160: Reward=+1.2956 | Portfolio=$71,543 | Trades=128 | ε=0.200
Step  90,000/200,000 | Ep 165 | ε=0.190 | Avg Reward: +1.2155
  └─ Episode 180: Reward=+1.3584 | Portfolio=$75,579 | Trades=128 | ε=0.164
Step 100,000/200,000 | Ep 183 | ε=0.159 | Avg Reward: +1.4532
  └─ Episode 200: Reward=+1.4454 | Portfolio=$78,372 | Trades=122 | ε=0.134
Step 110,000/200,000 | Ep 202 | ε=0.131 | Avg Reward: +1.4848
  └─ Episode 220: Reward=+1.6967 | Portfolio=$102,815 | Trades=128 | ε=0.110
Step 120,000/200,000 | Ep 220 | ε=0.110 | Avg Reward: +1.5327
Step 130,000/200,000 | Ep 238 | ε=0.091 | Avg Reward: +1.6931
  └─ Episode 240: Reward=+1.7205 | Portfolio=$102,091 | Trades=122 | ε=0.090
Step 140,000/200,000 | Ep 257 | ε=0.076 | Avg Reward: +1.8029
  └─ Episode 260: Reward=+1.6520 | Portfolio=$98,863 | Trades=134 | ε=0.073
Step 150,000/200,000 | Ep 275 | ε=0.063 | Avg Reward: +1.8498
  └─ Episode 280: Reward=+1.8647 | Portfolio=$116,355 | Trades=122 | ε=0.060
Step 160,000/200,000 | Ep 294 | ε=0.052 | Avg Reward: +1.8568
  └─ Episode 300: Reward=+1.7624 | Portfolio=$111,325 | Trades=132 | ε=0.049
Step 170,000/200,000 | Ep 312 | ε=0.043 | Avg Reward: +1.8332
  └─ Episode 320: Reward=+1.9530 | Portfolio=$124,264 | Trades=118 | ε=0.040
Step 180,000/200,000 | Ep 330 | ε=0.036 | Avg Reward: +1.9768
  └─ Episode 340: Reward=+1.9285 | Portfolio=$124,504 | Trades=122 | ε=0.033
Step 190,000/200,000 | Ep 349 | ε=0.030 | Avg Reward: +1.9587
  └─ Episode 360: Reward=+1.9192 | Portfolio=$120,202 | Trades=116 | ε=0.027
Step 200,000/200,000 | Ep 367 | ε=0.025 | Avg Reward: +1.9783

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 368
Final Epsilon: 0.0248
Mean Reward: 1.1101
Best Episode Reward: 2.1607
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_221146.pkl
Config saved to results/config_v2_20251209_221146.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $143,057.26
Return:  +1330.57%
Trades:  118
Fees:    $12984.86

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $13,263.77
Return:  +32.64%
Trades:  30
Fees:    $696.16

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |    +1330.57% |      +32.64%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |    +1188.49% |      -24.44%

 Final Values:
   Agent Train: $143,057.26  |  Buy&Hold: $24,208.72
   Agent Test:  $13,263.77  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_221146.pkl
Final Epsilon: 0.0248
Total Episodes: 368

 Agent made profit, but didn't beat Buy & Hold

v5
# V5 KONFIGURATION: Trendverfolgung erhöhen und Strafe reduzieren
config = {
    'data': {
        'symbol': 'BTC-USD',
        'start_date': '2023-01-01',
        'end_date': '2025-01-01', # Behält den kurzen, aktuellen Trainingszeitraum bei
        'interval': '1d',
        'test_split': 0.2
    },
    'environment': {
        'initial_cash': 10000.0,
        'trading_fee_maker': 0.001,
        'trading_fee_taker': 0.002,
        'slippage': 0.001,
        'trade_frequency_penalty': 0.002 # REDUZIERT, um mehr Trades zuzulassen
    },
    'q_learning': {
        'learning_rate': 0.1,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.99, 
        'n_bins': 20 # ERHÖHT, um Muster feiner zu erkennen
    },
    'training': {
        'total_timesteps': 200000, 
        'log_interval': 10000
    }
}
==================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50     
 Original Test Prices: $53948.75 - $106140.60     

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (20, 20, 20, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 200,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/200,000 | Ep  18 | ε=0.835 | Avg Reward: -0.3523
  └─ Episode 20: Reward=+0.1415 | Portfolio=$20,206 | Trades=163 | ε=0.818
Step  20,000/200,000 | Ep  36 | ε=0.696 | Avg Reward: -0.0674
  └─ Episode 40: Reward=-0.2380 | Portfolio=$14,544 | Trades=188 | ε=0.669
Step  30,000/200,000 | Ep  55 | ε=0.575 | Avg Reward: +0.4152
  └─ Episode 60: Reward=+0.1185 | Portfolio=$19,316 | Trades=164 | ε=0.547
Step  40,000/200,000 | Ep  73 | ε=0.480 | Avg Reward: +0.7149
  └─ Episode 80: Reward=+0.8428 | Portfolio=$39,673 | Trades=170 | ε=0.448
Step  50,000/200,000 | Ep  91 | ε=0.401 | Avg Reward: +1.0360
  └─ Episode 100: Reward=+1.3702 | Portfolio=$66,165 | Trades=163 | ε=0.366
Step  60,000/200,000 | Ep 110 | ε=0.331 | Avg Reward: +1.3635
  └─ Episode 120: Reward=+1.6535 | Portfolio=$89,998 | Trades=170 | ε=0.299
Step  70,000/200,000 | Ep 128 | ε=0.276 | Avg Reward: +1.4938
  └─ Episode 140: Reward=+1.0157 | Portfolio=$47,546 | Trades=175 | ε=0.245
Step  80,000/200,000 | Ep 147 | ε=0.228 | Avg Reward: +1.6769
  └─ Episode 160: Reward=+1.8577 | Portfolio=$100,326 | Trades=148 | ε=0.200
Step  90,000/200,000 | Ep 165 | ε=0.190 | Avg Reward: +1.8403
  └─ Episode 180: Reward=+2.0424 | Portfolio=$123,916 | Trades=154 | ε=0.164
Step 100,000/200,000 | Ep 183 | ε=0.159 | Avg Reward: +1.9295
  └─ Episode 200: Reward=+1.7552 | Portfolio=$99,215 | Trades=168 | ε=0.134
Step 110,000/200,000 | Ep 202 | ε=0.131 | Avg Reward: +1.9721
  └─ Episode 220: Reward=+1.8765 | Portfolio=$108,100 | Trades=170 | ε=0.110
Step 120,000/200,000 | Ep 220 | ε=0.110 | Avg Reward: +1.9250
Step 130,000/200,000 | Ep 238 | ε=0.091 | Avg Reward: +2.0506
  └─ Episode 240: Reward=+2.1647 | Portfolio=$138,691 | Trades=154 | ε=0.090
Step 140,000/200,000 | Ep 257 | ε=0.076 | Avg Reward: +2.1736
  └─ Episode 260: Reward=+1.9813 | Portfolio=$121,738 | Trades=170 | ε=0.073
Step 150,000/200,000 | Ep 275 | ε=0.063 | Avg Reward: +2.0715
  └─ Episode 280: Reward=+2.0860 | Portfolio=$131,045 | Trades=160 | ε=0.060
Step 160,000/200,000 | Ep 294 | ε=0.052 | Avg Reward: +2.2606
  └─ Episode 300: Reward=+2.0416 | Portfolio=$128,324 | Trades=170 | ε=0.049
Step 170,000/200,000 | Ep 312 | ε=0.043 | Avg Reward: +2.2570
  └─ Episode 320: Reward=+2.3442 | Portfolio=$167,924 | Trades=156 | ε=0.040
Step 180,000/200,000 | Ep 330 | ε=0.036 | Avg Reward: +2.2600
  └─ Episode 340: Reward=+2.3287 | Portfolio=$169,592 | Trades=164 | ε=0.033
Step 190,000/200,000 | Ep 349 | ε=0.030 | Avg Reward: +2.2979
  └─ Episode 360: Reward=+2.3923 | Portfolio=$179,561 | Trades=164 | ε=0.027
Step 200,000/200,000 | Ep 367 | ε=0.025 | Avg Reward: +2.2887

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 368
Final Epsilon: 0.0248
Mean Reward: 1.5419
Best Episode Reward: 2.4531
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_222830.pkl
Config saved to results/config_v2_20251209_222830.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $180,975.93
Return:  +1709.76%
Trades:  164
Fees:    $17801.48

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $10,356.82
Return:  +3.57%
Trades:  31
Fees:    $617.97

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test  
--------------------------------------------------  
Q-Learning Agent     |    +1709.76% |       +3.57%  
Buy & Hold           |     +142.09% |      +57.08%  
--------------------------------------------------  
Outperformance       |    +1567.67% |      -53.51%  

 Final Values:
   Agent Train: $180,975.93  |  Buy&Hold: $24,208.72
   Agent Test:  $10,356.82  |  Buy&Hold: $15,707.94 

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_222830.pkl    
Final Epsilon: 0.0248
Total Episodes: 368

v6
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50     
 Original Test Prices: $53948.75 - $106140.60     

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (20, 20, 20, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 500,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/500,000 | Ep  18 | ε=0.835 | Avg Reward: -0.8216
  └─ Episode 20: Reward=-1.2524 | Portfolio=$8,531 | Trades=179 | ε=0.818
Step  20,000/500,000 | Ep  36 | ε=0.696 | Avg Reward: -0.2927
  └─ Episode 40: Reward=-0.0466 | Portfolio=$22,628 | Trades=161 | ε=0.669
Step  30,000/500,000 | Ep  55 | ε=0.575 | Avg Reward: +0.0780
  └─ Episode 60: Reward=+0.3968 | Portfolio=$34,748 | Trades=147 | ε=0.547
Step  40,000/500,000 | Ep  73 | ε=0.480 | Avg Reward: +0.2003
  └─ Episode 80: Reward=+0.4057 | Portfolio=$35,587 | Trades=149 | ε=0.448
Step  50,000/500,000 | Ep  91 | ε=0.401 | Avg Reward: +0.7375
  └─ Episode 100: Reward=+0.8974 | Portfolio=$53,620 | Trades=148 | ε=0.366
Step  60,000/500,000 | Ep 110 | ε=0.331 | Avg Reward: +0.8886
  └─ Episode 120: Reward=+0.9100 | Portfolio=$50,993 | Trades=139 | ε=0.299
Step  70,000/500,000 | Ep 128 | ε=0.276 | Avg Reward: +1.1782
  └─ Episode 140: Reward=+1.4723 | Portfolio=$85,967 | Trades=134 | ε=0.245
Step  80,000/500,000 | Ep 147 | ε=0.228 | Avg Reward: +1.3669
  └─ Episode 160: Reward=+1.4977 | Portfolio=$93,651 | Trades=146 | ε=0.200
Step  90,000/500,000 | Ep 165 | ε=0.190 | Avg Reward: +1.5226
  └─ Episode 180: Reward=+1.7429 | Portfolio=$111,710 | Trades=132 | ε=0.164
Step 100,000/500,000 | Ep 183 | ε=0.159 | Avg Reward: +1.7537
  └─ Episode 200: Reward=+1.5734 | Portfolio=$93,487 | Trades=134 | ε=0.134
Step 110,000/500,000 | Ep 202 | ε=0.131 | Avg Reward: +1.6843
  └─ Episode 220: Reward=+1.6355 | Portfolio=$97,095 | Trades=128 | ε=0.110
Step 120,000/500,000 | Ep 220 | ε=0.110 | Avg Reward: +1.8764
Step 130,000/500,000 | Ep 238 | ε=0.091 | Avg Reward: +1.8745
  └─ Episode 240: Reward=+1.8449 | Portfolio=$123,774 | Trades=133 | ε=0.090
Step 140,000/500,000 | Ep 257 | ε=0.076 | Avg Reward: +1.9844
  └─ Episode 260: Reward=+2.0006 | Portfolio=$141,473 | Trades=132 | ε=0.073
Step 150,000/500,000 | Ep 275 | ε=0.063 | Avg Reward: +2.0030
  └─ Episode 280: Reward=+2.0215 | Portfolio=$136,182 | Trades=118 | ε=0.060
Step 160,000/500,000 | Ep 294 | ε=0.052 | Avg Reward: +2.0236
  └─ Episode 300: Reward=+2.0290 | Portfolio=$133,461 | Trades=116 | ε=0.049
Step 170,000/500,000 | Ep 312 | ε=0.043 | Avg Reward: +2.1041
  └─ Episode 320: Reward=+1.9995 | Portfolio=$128,290 | Trades=116 | ε=0.040
Step 180,000/500,000 | Ep 330 | ε=0.036 | Avg Reward: +2.1152
  └─ Episode 340: Reward=+2.2401 | Portfolio=$169,707 | Trades=122 | ε=0.033
Step 190,000/500,000 | Ep 349 | ε=0.030 | Avg Reward: +2.1534
  └─ Episode 360: Reward=+2.1989 | Portfolio=$161,267 | Trades=122 | ε=0.027
Step 200,000/500,000 | Ep 367 | ε=0.025 | Avg Reward: +2.1403
  └─ Episode 380: Reward=+2.2506 | Portfolio=$172,566 | Trades=122 | ε=0.022
Step 210,000/500,000 | Ep 386 | ε=0.021 | Avg Reward: +2.2081
  └─ Episode 400: Reward=+2.1727 | Portfolio=$157,363 | Trades=120 | ε=0.018
Step 220,000/500,000 | Ep 404 | ε=0.017 | Avg Reward: +2.1929
  └─ Episode 420: Reward=+2.2811 | Portfolio=$167,701 | Trades=114 | ε=0.015
Step 230,000/500,000 | Ep 422 | ε=0.014 | Avg Reward: +2.2668
  └─ Episode 440: Reward=+2.2791 | Portfolio=$173,031 | Trades=118 | ε=0.012
Step 240,000/500,000 | Ep 441 | ε=0.012 | Avg Reward: +2.1925
Step 250,000/500,000 | Ep 459 | ε=0.010 | Avg Reward: +2.2147
  └─ Episode 460: Reward=+2.2724 | Portfolio=$171,005 | Trades=118 | ε=0.010
Step 260,000/500,000 | Ep 477 | ε=0.010 | Avg Reward: +2.2035
  └─ Episode 480: Reward=+2.1257 | Portfolio=$150,887 | Trades=122 | ε=0.010
Step 270,000/500,000 | Ep 496 | ε=0.010 | Avg Reward: +2.2456
  └─ Episode 500: Reward=+2.2610 | Portfolio=$169,666 | Trades=120 | ε=0.010
Step 280,000/500,000 | Ep 514 | ε=0.010 | Avg Reward: +2.2292
  └─ Episode 520: Reward=+2.2302 | Portfolio=$163,842 | Trades=118 | ε=0.010
Step 290,000/500,000 | Ep 533 | ε=0.010 | Avg Reward: +2.2392
  └─ Episode 540: Reward=+2.2040 | Portfolio=$162,992 | Trades=122 | ε=0.010
Step 300,000/500,000 | Ep 551 | ε=0.010 | Avg Reward: +2.2105
  └─ Episode 560: Reward=+2.2724 | Portfolio=$171,006 | Trades=118 | ε=0.010
Step 310,000/500,000 | Ep 569 | ε=0.010 | Avg Reward: +2.2351
  └─ Episode 580: Reward=+2.2205 | Portfolio=$164,884 | Trades=120 | ε=0.010
Step 320,000/500,000 | Ep 588 | ε=0.010 | Avg Reward: +2.2255
  └─ Episode 600: Reward=+2.2713 | Portfolio=$170,877 | Trades=118 | ε=0.010
Step 330,000/500,000 | Ep 606 | ε=0.010 | Avg Reward: +2.2394
  └─ Episode 620: Reward=+2.2347 | Portfolio=$164,772 | Trades=118 | ε=0.010
Step 340,000/500,000 | Ep 624 | ε=0.010 | Avg Reward: +2.2416
  └─ Episode 640: Reward=+2.2482 | Portfolio=$167,645 | Trades=120 | ε=0.010
Step 350,000/500,000 | Ep 643 | ε=0.010 | Avg Reward: +2.2506
  └─ Episode 660: Reward=+2.2724 | Portfolio=$171,006 | Trades=118 | ε=0.010
Step 360,000/500,000 | Ep 661 | ε=0.010 | Avg Reward: +2.2356
  └─ Episode 680: Reward=+2.2583 | Portfolio=$169,593 | Trades=118 | ε=0.010
Step 370,000/500,000 | Ep 680 | ε=0.010 | Avg Reward: +2.2321
Step 380,000/500,000 | Ep 698 | ε=0.010 | Avg Reward: +2.2409
  └─ Episode 700: Reward=+2.2288 | Portfolio=$167,915 | Trades=120 | ε=0.010
Step 390,000/500,000 | Ep 716 | ε=0.010 | Avg Reward: +2.2345
  └─ Episode 720: Reward=+2.3933 | Portfolio=$189,648 | Trades=114 | ε=0.010
Step 400,000/500,000 | Ep 735 | ε=0.010 | Avg Reward: +2.2617
  └─ Episode 740: Reward=+2.2609 | Portfolio=$170,752 | Trades=120 | ε=0.010
Step 410,000/500,000 | Ep 753 | ε=0.010 | Avg Reward: +2.2509
  └─ Episode 760: Reward=+2.2063 | Portfolio=$161,709 | Trades=120 | ε=0.010
Step 420,000/500,000 | Ep 772 | ε=0.010 | Avg Reward: +2.2355
  └─ Episode 780: Reward=+2.2662 | Portfolio=$169,905 | Trades=118 | ε=0.010
Step 430,000/500,000 | Ep 790 | ε=0.010 | Avg Reward: +2.2489
  └─ Episode 800: Reward=+2.2057 | Portfolio=$167,395 | Trades=124 | ε=0.010
Step 440,000/500,000 | Ep 808 | ε=0.010 | Avg Reward: +2.2342
  └─ Episode 820: Reward=+2.2404 | Portfolio=$167,066 | Trades=118 | ε=0.010
Step 450,000/500,000 | Ep 827 | ε=0.010 | Avg Reward: +2.2479
  └─ Episode 840: Reward=+2.2530 | Portfolio=$167,746 | Trades=118 | ε=0.010
Step 460,000/500,000 | Ep 845 | ε=0.010 | Avg Reward: +2.2290
  └─ Episode 860: Reward=+2.2820 | Portfolio=$174,378 | Trades=120 | ε=0.010
Step 470,000/500,000 | Ep 863 | ε=0.010 | Avg Reward: +2.2358
  └─ Episode 880: Reward=+2.2632 | Portfolio=$171,175 | Trades=120 | ε=0.010
Step 480,000/500,000 | Ep 882 | ε=0.010 | Avg Reward: +2.2106
  └─ Episode 900: Reward=+2.1479 | Portfolio=$155,463 | Trades=122 | ε=0.010
Step 490,000/500,000 | Ep 900 | ε=0.010 | Avg Reward: +2.2187
Step 500,000/500,000 | Ep 919 | ε=0.010 | Avg Reward: +2.2419
  └─ Episode 920: Reward=+0.2943 | Portfolio=$14,063 | Trades=11 | ε=0.010

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 920
Final Epsilon: 0.0100
Mean Reward: 1.8551
Best Episode Reward: 2.3933
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_223523.pkl
Config saved to results/config_v2_20251209_223523.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $171,006.47
Return:  +1610.06%
Trades:  118
Fees:    $13902.63

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $13,471.74
Return:  +34.72%
Trades:  21
Fees:    $495.05

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test  
--------------------------------------------------  
Q-Learning Agent     |    +1610.06% |      +34.72%  
Buy & Hold           |     +142.09% |      +57.08%  
--------------------------------------------------  
Outperformance       |    +1467.98% |      -22.36%  

 Final Values:
   Agent Train: $171,006.47  |  Buy&Hold: $24,208.72
   Agent Test:  $13,471.74  |  Buy&Hold: $15,707.94 

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_223523.pkl    
Final Epsilon: 0.0100
Total Episodes: 920

 Agent made profit, but didn't beat Buy & Hold  

config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 20
        },
        'training': {
            'total_timesteps': 500000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

v7
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.004  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 500000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50     
 Original Test Prices: $53948.75 - $106140.60     

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 500,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/500,000 | Ep  18 | ε=0.835 | Avg Reward: -0.6090
  └─ Episode 20: Reward=-0.9993 | Portfolio=$9,051 | Trades=187 | ε=0.818
Step  20,000/500,000 | Ep  36 | ε=0.696 | Avg Reward: -0.2610
  └─ Episode 40: Reward=-0.2282 | Portfolio=$16,702 | Trades=164 | ε=0.669
Step  30,000/500,000 | Ep  55 | ε=0.575 | Avg Reward: +0.0867
  └─ Episode 60: Reward=+0.1917 | Portfolio=$23,846 | Trades=154 | ε=0.547
Step  40,000/500,000 | Ep  73 | ε=0.480 | Avg Reward: +0.5294
  └─ Episode 80: Reward=+1.0169 | Portfolio=$57,881 | Trades=172 | ε=0.448
Step  50,000/500,000 | Ep  91 | ε=0.401 | Avg Reward: +0.6334
  └─ Episode 100: Reward=+1.3291 | Portfolio=$63,933 | Trades=130 | ε=0.366
Step  60,000/500,000 | Ep 110 | ε=0.331 | Avg Reward: +0.7981
  └─ Episode 120: Reward=+1.1918 | Portfolio=$62,273 | Trades=154 | ε=0.299
Step  70,000/500,000 | Ep 128 | ε=0.276 | Avg Reward: +1.0858
  └─ Episode 140: Reward=+1.3495 | Portfolio=$63,805 | Trades=118 | ε=0.245
Step  80,000/500,000 | Ep 147 | ε=0.228 | Avg Reward: +1.2970
  └─ Episode 160: Reward=+1.3895 | Portfolio=$66,095 | Trades=120 | ε=0.200
Step  90,000/500,000 | Ep 165 | ε=0.190 | Avg Reward: +1.3652
  └─ Episode 180: Reward=+1.4091 | Portfolio=$72,292 | Trades=130 | ε=0.164
Step 100,000/500,000 | Ep 183 | ε=0.159 | Avg Reward: +1.4667
  └─ Episode 200: Reward=+1.4452 | Portfolio=$74,028 | Trades=124 | ε=0.134
Step 110,000/500,000 | Ep 202 | ε=0.131 | Avg Reward: +1.5445
  └─ Episode 220: Reward=+1.7160 | Portfolio=$90,669 | Trades=124 | ε=0.110
Step 120,000/500,000 | Ep 220 | ε=0.110 | Avg Reward: +1.6542
Step 130,000/500,000 | Ep 238 | ε=0.091 | Avg Reward: +1.6891
  └─ Episode 240: Reward=+1.6878 | Portfolio=$85,997 | Trades=120 | ε=0.090
Step 140,000/500,000 | Ep 257 | ε=0.076 | Avg Reward: +1.7323
  └─ Episode 260: Reward=+1.7923 | Portfolio=$95,067 | Trades=114 | ε=0.073
Step 150,000/500,000 | Ep 275 | ε=0.063 | Avg Reward: +1.8869
  └─ Episode 280: Reward=+1.7227 | Portfolio=$83,317 | Trades=106 | ε=0.060
Step 160,000/500,000 | Ep 294 | ε=0.052 | Avg Reward: +1.8902
  └─ Episode 300: Reward=+1.8531 | Portfolio=$99,028 | Trades=116 | ε=0.049
Step 170,000/500,000 | Ep 312 | ε=0.043 | Avg Reward: +1.8994
  └─ Episode 320: Reward=+1.9901 | Portfolio=$108,896 | Trades=106 | ε=0.040
Step 180,000/500,000 | Ep 330 | ε=0.036 | Avg Reward: +1.9313
  └─ Episode 340: Reward=+1.9132 | Portfolio=$100,299 | Trades=106 | ε=0.033
Step 190,000/500,000 | Ep 349 | ε=0.030 | Avg Reward: +1.9327
  └─ Episode 360: Reward=+1.7633 | Portfolio=$85,836 | Trades=104 | ε=0.027
Step 200,000/500,000 | Ep 367 | ε=0.025 | Avg Reward: +1.8929
  └─ Episode 380: Reward=+1.8598 | Portfolio=$92,524 | Trades=100 | ε=0.022
Step 210,000/500,000 | Ep 386 | ε=0.021 | Avg Reward: +2.0041
  └─ Episode 400: Reward=+2.0434 | Portfolio=$112,919 | Trades=104 | ε=0.018
Step 220,000/500,000 | Ep 404 | ε=0.017 | Avg Reward: +2.0353
  └─ Episode 420: Reward=+1.7778 | Portfolio=$88,351 | Trades=104 | ε=0.015
Step 230,000/500,000 | Ep 422 | ε=0.014 | Avg Reward: +2.0056
  └─ Episode 440: Reward=+1.9072 | Portfolio=$101,128 | Trades=108 | ε=0.012
Step 240,000/500,000 | Ep 441 | ε=0.012 | Avg Reward: +2.0125
Step 250,000/500,000 | Ep 459 | ε=0.010 | Avg Reward: +2.0431
  └─ Episode 460: Reward=+2.0646 | Portfolio=$114,483 | Trades=102 | ε=0.010
Step 260,000/500,000 | Ep 477 | ε=0.010 | Avg Reward: +2.0274
  └─ Episode 480: Reward=+2.0705 | Portfolio=$115,510 | Trades=102 | ε=0.010
Step 270,000/500,000 | Ep 496 | ε=0.010 | Avg Reward: +2.0458
  └─ Episode 500: Reward=+2.0253 | Portfolio=$109,886 | Trades=102 | ε=0.010
Step 280,000/500,000 | Ep 514 | ε=0.010 | Avg Reward: +2.0256
  └─ Episode 520: Reward=+1.9610 | Portfolio=$107,553 | Trades=110 | ε=0.010
Step 290,000/500,000 | Ep 533 | ε=0.010 | Avg Reward: +2.0464
  └─ Episode 540: Reward=+1.9325 | Portfolio=$103,108 | Trades=106 | ε=0.010
Step 300,000/500,000 | Ep 551 | ε=0.010 | Avg Reward: +1.9985
  └─ Episode 560: Reward=+2.0754 | Portfolio=$114,709 | Trades=100 | ε=0.010
Step 310,000/500,000 | Ep 569 | ε=0.010 | Avg Reward: +2.0453
  └─ Episode 580: Reward=+2.0924 | Portfolio=$118,002 | Trades=102 | ε=0.010
Step 320,000/500,000 | Ep 588 | ε=0.010 | Avg Reward: +2.0592
  └─ Episode 600: Reward=+2.0303 | Portfolio=$110,988 | Trades=102 | ε=0.010
Step 330,000/500,000 | Ep 606 | ε=0.010 | Avg Reward: +2.0390
  └─ Episode 620: Reward=+2.0333 | Portfolio=$111,283 | Trades=102 | ε=0.010
Step 340,000/500,000 | Ep 624 | ε=0.010 | Avg Reward: +2.0590
  └─ Episode 640: Reward=+2.0681 | Portfolio=$114,315 | Trades=100 | ε=0.010
Step 350,000/500,000 | Ep 643 | ε=0.010 | Avg Reward: +2.0131
  └─ Episode 660: Reward=+2.0368 | Portfolio=$112,574 | Trades=104 | ε=0.010
Step 360,000/500,000 | Ep 661 | ε=0.010 | Avg Reward: +2.0247
  └─ Episode 680: Reward=+1.9812 | Portfolio=$105,038 | Trades=100 | ε=0.010
Step 370,000/500,000 | Ep 680 | ε=0.010 | Avg Reward: +2.0449
Step 380,000/500,000 | Ep 698 | ε=0.010 | Avg Reward: +2.0386
  └─ Episode 700: Reward=+2.0718 | Portfolio=$114,291 | Trades=100 | ε=0.010
Step 390,000/500,000 | Ep 716 | ε=0.010 | Avg Reward: +2.0276
  └─ Episode 720: Reward=+2.0073 | Portfolio=$108,378 | Trades=102 | ε=0.010
Step 400,000/500,000 | Ep 735 | ε=0.010 | Avg Reward: +2.0367
  └─ Episode 740: Reward=+1.9722 | Portfolio=$104,954 | Trades=104 | ε=0.010
Step 410,000/500,000 | Ep 753 | ε=0.010 | Avg Reward: +2.0442
  └─ Episode 760: Reward=+2.0418 | Portfolio=$113,156 | Trades=104 | ε=0.010
Step 420,000/500,000 | Ep 772 | ε=0.010 | Avg Reward: +2.0404
  └─ Episode 780: Reward=+2.0993 | Portfolio=$119,134 | Trades=102 | ε=0.010
Step 430,000/500,000 | Ep 790 | ε=0.010 | Avg Reward: +2.0363
  └─ Episode 800: Reward=+2.0064 | Portfolio=$110,066 | Trades=106 | ε=0.010
Step 440,000/500,000 | Ep 808 | ε=0.010 | Avg Reward: +2.0498
  └─ Episode 820: Reward=+2.0754 | Portfolio=$114,709 | Trades=100 | ε=0.010
Step 450,000/500,000 | Ep 827 | ε=0.010 | Avg Reward: +2.0322
  └─ Episode 840: Reward=+1.9818 | Portfolio=$107,651 | Trades=106 | ε=0.010
Step 460,000/500,000 | Ep 845 | ε=0.010 | Avg Reward: +2.0184
  └─ Episode 860: Reward=+2.0754 | Portfolio=$114,709 | Trades=100 | ε=0.010
Step 470,000/500,000 | Ep 863 | ε=0.010 | Avg Reward: +2.0380
  └─ Episode 880: Reward=+2.0754 | Portfolio=$114,709 | Trades=100 | ε=0.010
Step 480,000/500,000 | Ep 882 | ε=0.010 | Avg Reward: +2.0583
  └─ Episode 900: Reward=+2.1033 | Portfolio=$116,988 | Trades=98 | ε=0.010
Step 490,000/500,000 | Ep 900 | ε=0.010 | Avg Reward: +2.0547
Step 500,000/500,000 | Ep 919 | ε=0.010 | Avg Reward: +2.0139
  └─ Episode 920: Reward=+0.2959 | Portfolio=$13,925 | Trades=12 | ε=0.010

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 920
Final Epsilon: 0.0100
Mean Reward: 1.6914
Best Episode Reward: 2.1316
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_224144.pkl
Config saved to results/config_v2_20251209_224144.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $114,708.60
Return:  +1047.09%
Trades:  100
Fees:    $9464.40

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $11,863.06
Return:  +18.63%
Trades:  28
Fees:    $554.23

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test  
--------------------------------------------------  
Q-Learning Agent     |    +1047.09% |      +18.63%  
Buy & Hold           |     +142.09% |      +57.08%  
--------------------------------------------------  
Outperformance       |     +905.00% |      -38.45%  

 Final Values:
   Agent Train: $114,708.60  |  Buy&Hold: $24,208.72
   Agent Test:  $11,863.06  |  Buy&Hold: $15,707.94 

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_224144.pkl    
Final Epsilon: 0.0100
Total Episodes: 920

 Agent made profit, but didn't beat Buy & Hold   

v8
======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50     
 Original Test Prices: $53948.75 - $106140.60     

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 300,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/300,000 | Ep  18 | ε=0.835 | Avg Reward: -2.1213
  └─ Episode 20: Reward=-1.4453 | Portfolio=$2,594 | Trades=148 | ε=0.818
Step  20,000/300,000 | Ep  36 | ε=0.696 | Avg Reward: -1.6009
  └─ Episode 40: Reward=-1.0044 | Portfolio=$3,913 | Trades=142 | ε=0.669
Step  30,000/300,000 | Ep  55 | ε=0.575 | Avg Reward: -1.0812
  └─ Episode 60: Reward=-1.4577 | Portfolio=$2,464 | Trades=130 | ε=0.547
Step  40,000/300,000 | Ep  73 | ε=0.480 | Avg Reward: -0.6515
  └─ Episode 80: Reward=-0.1250 | Portfolio=$9,305 | Trades=108 | ε=0.448
Step  50,000/300,000 | Ep  91 | ε=0.401 | Avg Reward: -0.4083
  └─ Episode 100: Reward=+0.2300 | Portfolio=$12,469 | Trades=85 | ε=0.366
Step  60,000/300,000 | Ep 110 | ε=0.331 | Avg Reward: -0.0465
  └─ Episode 120: Reward=+0.3643 | Portfolio=$14,225 | Trades=84 | ε=0.299
Step  70,000/300,000 | Ep 128 | ε=0.276 | Avg Reward: +0.1698
  └─ Episode 140: Reward=+0.5145 | Portfolio=$16,525 | Trades=82 | ε=0.245
Step  80,000/300,000 | Ep 147 | ε=0.228 | Avg Reward: +0.4605
  └─ Episode 160: Reward=+0.4520 | Portfolio=$15,546 | Trades=84 | ε=0.200
Step  90,000/300,000 | Ep 165 | ε=0.190 | Avg Reward: +0.5312
  └─ Episode 180: Reward=+1.1764 | Portfolio=$32,037 | Trades=66 | ε=0.164
Step 100,000/300,000 | Ep 183 | ε=0.159 | Avg Reward: +0.7361
  └─ Episode 200: Reward=+1.2171 | Portfolio=$33,247 | Trades=60 | ε=0.134
Step 110,000/300,000 | Ep 202 | ε=0.131 | Avg Reward: +1.0063
  └─ Episode 220: Reward=+1.4143 | Portfolio=$40,059 | Trades=57 | ε=0.110
Step 120,000/300,000 | Ep 220 | ε=0.110 | Avg Reward: +1.0799
Step 130,000/300,000 | Ep 238 | ε=0.091 | Avg Reward: +1.1158
  └─ Episode 240: Reward=+1.5754 | Portfolio=$47,083 | Trades=54 | ε=0.090
Step 140,000/300,000 | Ep 257 | ε=0.076 | Avg Reward: +1.3216
  └─ Episode 260: Reward=+1.0918 | Portfolio=$29,121 | Trades=62 | ε=0.073
Step 150,000/300,000 | Ep 275 | ε=0.063 | Avg Reward: +1.4006
  └─ Episode 280: Reward=+1.1491 | Portfolio=$30,587 | Trades=56 | ε=0.060
Step 160,000/300,000 | Ep 294 | ε=0.052 | Avg Reward: +1.4322
  └─ Episode 300: Reward=+1.5094 | Portfolio=$43,901 | Trades=54 | ε=0.049
Step 170,000/300,000 | Ep 312 | ε=0.043 | Avg Reward: +1.4943
  └─ Episode 320: Reward=+1.4817 | Portfolio=$43,048 | Trades=54 | ε=0.040
Step 180,000/300,000 | Ep 330 | ε=0.036 | Avg Reward: +1.4702
  └─ Episode 340: Reward=+1.4305 | Portfolio=$41,481 | Trades=62 | ε=0.033
Step 190,000/300,000 | Ep 349 | ε=0.030 | Avg Reward: +1.5499
  └─ Episode 360: Reward=+1.7116 | Portfolio=$53,826 | Trades=50 | ε=0.027
Step 200,000/300,000 | Ep 367 | ε=0.025 | Avg Reward: +1.6091
  └─ Episode 380: Reward=+1.7142 | Portfolio=$53,736 | Trades=48 | ε=0.022
Step 210,000/300,000 | Ep 386 | ε=0.021 | Avg Reward: +1.5777
  └─ Episode 400: Reward=+1.7220 | Portfolio=$53,877 | Trades=48 | ε=0.018
Step 220,000/300,000 | Ep 404 | ε=0.017 | Avg Reward: +1.6495
  └─ Episode 420: Reward=+1.7391 | Portfolio=$55,048 | Trades=46 | ε=0.015
Step 230,000/300,000 | Ep 422 | ε=0.014 | Avg Reward: +1.6779
  └─ Episode 440: Reward=+1.7391 | Portfolio=$55,048 | Trades=46 | ε=0.012
Step 240,000/300,000 | Ep 441 | ε=0.012 | Avg Reward: +1.6926
Step 250,000/300,000 | Ep 459 | ε=0.010 | Avg Reward: +1.6503
  └─ Episode 460: Reward=+1.7401 | Portfolio=$55,048 | Trades=46 | ε=0.010
Step 260,000/300,000 | Ep 477 | ε=0.010 | Avg Reward: +1.6728
  └─ Episode 480: Reward=+1.7085 | Portfolio=$53,530 | Trades=48 | ε=0.010
Step 270,000/300,000 | Ep 496 | ε=0.010 | Avg Reward: +1.7080
  └─ Episode 500: Reward=+1.7401 | Portfolio=$55,048 | Trades=46 | ε=0.010
Step 280,000/300,000 | Ep 514 | ε=0.010 | Avg Reward: +1.7006
  └─ Episode 520: Reward=+1.7401 | Portfolio=$55,048 | Trades=46 | ε=0.010
Step 290,000/300,000 | Ep 533 | ε=0.010 | Avg Reward: +1.6922
  └─ Episode 540: Reward=+1.6826 | Portfolio=$52,085 | Trades=48 | ε=0.010
Step 300,000/300,000 | Ep 551 | ε=0.010 | Avg Reward: +1.6764

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 552
Final Epsilon: 0.0100
Mean Reward: 0.8441
Best Episode Reward: 1.8375
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_225241.pkl
Config saved to results/config_v2_20251209_225241.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $55,048.15
Return:  +450.48%
Trades:  46
Fees:    $24291.28

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $11,867.91
Return:  +18.68%
Trades:  16
Fees:    $2425.03

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test  
--------------------------------------------------  
Q-Learning Agent     |     +450.48% |      +18.68%  
Buy & Hold           |     +142.09% |      +57.08%  
--------------------------------------------------  
Outperformance       |     +308.39% |      -38.40%  

 Final Values:
   Agent Train: $55,048.15  |  Buy&Hold: $24,208.72 
   Agent Test:  $11,867.91  |  Buy&Hold: $15,707.94 

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_225241.pkl    
Final Epsilon: 0.0100
Total Episodes: 552

config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.015,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.001  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

v9
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.01,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.003  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.90,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.9 (reaches 0.01 after ~43 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 300,000
Epsilon: 1.00  0.01
Decay rate: 0.9
============================================================

Step  10,000/300,000 | Ep  18 | ε=0.150 | Avg Reward: +0.2333
  └─ Episode 20: Reward=+0.9366 | Portfolio=$30,264 | Trades=76 | ε=0.122
Step  20,000/300,000 | Ep  36 | ε=0.023 | Avg Reward: +1.3871
  └─ Episode 40: Reward=+1.5216 | Portfolio=$52,972 | Trades=70 | ε=0.015
Step  30,000/300,000 | Ep  55 | ε=0.010 | Avg Reward: +1.5824
  └─ Episode 60: Reward=+1.5761 | Portfolio=$56,010 | Trades=70 | ε=0.010
Step  40,000/300,000 | Ep  73 | ε=0.010 | Avg Reward: +1.5573
  └─ Episode 80: Reward=+1.6251 | Portfolio=$57,874 | Trades=66 | ε=0.010
Step  50,000/300,000 | Ep  91 | ε=0.010 | Avg Reward: +1.5667
  └─ Episode 100: Reward=+1.6038 | Portfolio=$57,169 | Trades=68 | ε=0.010
Step  60,000/300,000 | Ep 110 | ε=0.010 | Avg Reward: +1.5855
  └─ Episode 120: Reward=+1.5845 | Portfolio=$56,385 | Trades=70 | ε=0.010
Step  70,000/300,000 | Ep 128 | ε=0.010 | Avg Reward: +1.5709
  └─ Episode 140: Reward=+1.5699 | Portfolio=$55,566 | Trades=70 | ε=0.010
Step  80,000/300,000 | Ep 147 | ε=0.010 | Avg Reward: +1.5427
  └─ Episode 160: Reward=+1.6206 | Portfolio=$57,612 | Trades=66 | ε=0.010
Step  90,000/300,000 | Ep 165 | ε=0.010 | Avg Reward: +1.5794
  └─ Episode 180: Reward=+1.5880 | Portfolio=$56,264 | Trades=68 | ε=0.010
Step 100,000/300,000 | Ep 183 | ε=0.010 | Avg Reward: +1.5922
  └─ Episode 200: Reward=+1.6239 | Portfolio=$57,969 | Trades=66 | ε=0.010
Step 110,000/300,000 | Ep 202 | ε=0.010 | Avg Reward: +1.5887
  └─ Episode 220: Reward=+1.6251 | Portfolio=$57,874 | Trades=66 | ε=0.010
Step 120,000/300,000 | Ep 220 | ε=0.010 | Avg Reward: +1.6093
Step 130,000/300,000 | Ep 238 | ε=0.010 | Avg Reward: +1.5849
  └─ Episode 240: Reward=+1.5884 | Portfolio=$56,436 | Trades=68 | ε=0.010
Step 140,000/300,000 | Ep 257 | ε=0.010 | Avg Reward: +1.5870
  └─ Episode 260: Reward=+1.6352 | Portfolio=$58,493 | Trades=66 | ε=0.010
Step 150,000/300,000 | Ep 275 | ε=0.010 | Avg Reward: +1.5953
  └─ Episode 280: Reward=+1.6185 | Portfolio=$57,568 | Trades=66 | ε=0.010
Step 160,000/300,000 | Ep 294 | ε=0.010 | Avg Reward: +1.5886
  └─ Episode 300: Reward=+1.6084 | Portfolio=$56,927 | Trades=66 | ε=0.010
Step 170,000/300,000 | Ep 312 | ε=0.010 | Avg Reward: +1.5649
  └─ Episode 320: Reward=+1.4566 | Portfolio=$49,736 | Trades=70 | ε=0.010
Step 180,000/300,000 | Ep 330 | ε=0.010 | Avg Reward: +1.5733
  └─ Episode 340: Reward=+1.4690 | Portfolio=$50,508 | Trades=70 | ε=0.010
Step 190,000/300,000 | Ep 349 | ε=0.010 | Avg Reward: +1.5779
  └─ Episode 360: Reward=+1.6251 | Portfolio=$57,874 | Trades=66 | ε=0.010
Step 200,000/300,000 | Ep 367 | ε=0.010 | Avg Reward: +1.5511
  └─ Episode 380: Reward=+1.6251 | Portfolio=$57,874 | Trades=66 | ε=0.010
Step 210,000/300,000 | Ep 386 | ε=0.010 | Avg Reward: +1.5358
  └─ Episode 400: Reward=+1.5961 | Portfolio=$56,407 | Trades=66 | ε=0.010
Step 220,000/300,000 | Ep 404 | ε=0.010 | Avg Reward: +1.5870
  └─ Episode 420: Reward=+1.6075 | Portfolio=$57,483 | Trades=68 | ε=0.010
Step 230,000/300,000 | Ep 422 | ε=0.010 | Avg Reward: +1.5577
  └─ Episode 440: Reward=+1.5471 | Portfolio=$54,075 | Trades=70 | ε=0.010
Step 240,000/300,000 | Ep 441 | ε=0.010 | Avg Reward: +1.5647
Step 250,000/300,000 | Ep 459 | ε=0.010 | Avg Reward: +1.5975
  └─ Episode 460: Reward=+1.5056 | Portfolio=$51,661 | Trades=68 | ε=0.010
Step 260,000/300,000 | Ep 477 | ε=0.010 | Avg Reward: +1.5655
  └─ Episode 480: Reward=+1.6105 | Portfolio=$56,466 | Trades=64 | ε=0.010
Step 270,000/300,000 | Ep 496 | ε=0.010 | Avg Reward: +1.5897
  └─ Episode 500: Reward=+1.4734 | Portfolio=$50,079 | Trades=68 | ε=0.010
Step 280,000/300,000 | Ep 514 | ε=0.010 | Avg Reward: +1.5625
  └─ Episode 520: Reward=+1.5314 | Portfolio=$53,888 | Trades=72 | ε=0.010
Step 290,000/300,000 | Ep 533 | ε=0.010 | Avg Reward: +1.5501
  └─ Episode 540: Reward=+1.6175 | Portfolio=$57,774 | Trades=68 | ε=0.010
Step 300,000/300,000 | Ep 551 | ε=0.010 | Avg Reward: +1.5813

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 552
Final Epsilon: 0.0100
Mean Reward: 1.4948
Best Episode Reward: 1.7320
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_231221.pkl
Config saved to results/config_v2_20251209_231221.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $57,874.39
Return:  +478.74%
Trades:  66
Fees:    $20707.44

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $10,716.23
Return:  +7.16%
Trades:  20
Fees:    $1897.78

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |     +478.74% |       +7.16%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +336.66% |      -49.92%

 Final Values:
   Agent Train: $57,874.39  |  Buy&Hold: $24,208.72
   Agent Test:  $10,716.23  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_231221.pkl
Final Epsilon: 0.0100
Total Episodes: 552

v10
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.90,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
=====================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.9 (reaches 0.01 after ~43 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 300,000
Epsilon: 1.00  0.01
Decay rate: 0.9
============================================================

Step  10,000/300,000 | Ep  18 | ε=0.150 | Avg Reward: +0.8814
  └─ Episode 20: Reward=+1.2302 | Portfolio=$69,889 | Trades=142 | ε=0.122
Step  20,000/300,000 | Ep  36 | ε=0.023 | Avg Reward: +1.6886
  └─ Episode 40: Reward=+1.6491 | Portfolio=$90,514 | Trades=112 | ε=0.015
Step  30,000/300,000 | Ep  55 | ε=0.010 | Avg Reward: +1.7963
  └─ Episode 60: Reward=+1.8337 | Portfolio=$106,086 | Trades=108 | ε=0.010
Step  40,000/300,000 | Ep  73 | ε=0.010 | Avg Reward: +1.7942
  └─ Episode 80: Reward=+1.8322 | Portfolio=$108,645 | Trades=110 | ε=0.010
Step  50,000/300,000 | Ep  91 | ε=0.010 | Avg Reward: +1.7801
  └─ Episode 100: Reward=+1.8738 | Portfolio=$114,213 | Trades=112 | ε=0.010
Step  60,000/300,000 | Ep 110 | ε=0.010 | Avg Reward: +1.7658
  └─ Episode 120: Reward=+1.8072 | Portfolio=$107,008 | Trades=112 | ε=0.010
Step  70,000/300,000 | Ep 128 | ε=0.010 | Avg Reward: +1.7941
  └─ Episode 140: Reward=+1.7834 | Portfolio=$101,031 | Trades=108 | ε=0.010
Step  80,000/300,000 | Ep 147 | ε=0.010 | Avg Reward: +1.7905
  └─ Episode 160: Reward=+1.7883 | Portfolio=$101,692 | Trades=106 | ε=0.010
Step  90,000/300,000 | Ep 165 | ε=0.010 | Avg Reward: +1.8210
  └─ Episode 180: Reward=+1.8406 | Portfolio=$108,340 | Trades=110 | ε=0.010
Step 100,000/300,000 | Ep 183 | ε=0.010 | Avg Reward: +1.8029
  └─ Episode 200: Reward=+1.8250 | Portfolio=$106,780 | Trades=108 | ε=0.010
Step 110,000/300,000 | Ep 202 | ε=0.010 | Avg Reward: +1.7722
  └─ Episode 220: Reward=+1.8226 | Portfolio=$107,586 | Trades=110 | ε=0.010
Step 120,000/300,000 | Ep 220 | ε=0.010 | Avg Reward: +1.8041
Step 130,000/300,000 | Ep 238 | ε=0.010 | Avg Reward: +1.8201
  └─ Episode 240: Reward=+1.8320 | Portfolio=$108,017 | Trades=110 | ε=0.010
Step 140,000/300,000 | Ep 257 | ε=0.010 | Avg Reward: +1.8082
  └─ Episode 260: Reward=+1.7746 | Portfolio=$101,721 | Trades=110 | ε=0.010
Step 150,000/300,000 | Ep 275 | ε=0.010 | Avg Reward: +1.7616
  └─ Episode 280: Reward=+1.8409 | Portfolio=$109,604 | Trades=110 | ε=0.010
Step 160,000/300,000 | Ep 294 | ε=0.010 | Avg Reward: +1.7829
  └─ Episode 300: Reward=+1.8282 | Portfolio=$107,653 | Trades=110 | ε=0.010
Step 170,000/300,000 | Ep 312 | ε=0.010 | Avg Reward: +1.7626
  └─ Episode 320: Reward=+1.8229 | Portfolio=$107,875 | Trades=112 | ε=0.010
Step 180,000/300,000 | Ep 330 | ε=0.010 | Avg Reward: +1.8113
  └─ Episode 340: Reward=+1.7982 | Portfolio=$105,964 | Trades=112 | ε=0.010
Step 190,000/300,000 | Ep 349 | ε=0.010 | Avg Reward: +1.8116
  └─ Episode 360: Reward=+1.6876 | Portfolio=$93,266 | Trades=110 | ε=0.010
Step 200,000/300,000 | Ep 367 | ε=0.010 | Avg Reward: +1.7624
  └─ Episode 380: Reward=+1.7969 | Portfolio=$106,230 | Trades=112 | ε=0.010
Step 210,000/300,000 | Ep 386 | ε=0.010 | Avg Reward: +1.8224
  └─ Episode 400: Reward=+1.7858 | Portfolio=$100,447 | Trades=106 | ε=0.010
Step 220,000/300,000 | Ep 404 | ε=0.010 | Avg Reward: +1.7827
  └─ Episode 420: Reward=+1.7721 | Portfolio=$99,436 | Trades=106 | ε=0.010
Step 230,000/300,000 | Ep 422 | ε=0.010 | Avg Reward: +1.7865
  └─ Episode 440: Reward=+1.7939 | Portfolio=$107,705 | Trades=116 | ε=0.010
Step 240,000/300,000 | Ep 441 | ε=0.010 | Avg Reward: +1.8043
Step 250,000/300,000 | Ep 459 | ε=0.010 | Avg Reward: +1.7821
  └─ Episode 460: Reward=+1.8272 | Portfolio=$108,090 | Trades=110 | ε=0.010
Step 260,000/300,000 | Ep 477 | ε=0.010 | Avg Reward: +1.7927
  └─ Episode 480: Reward=+1.8073 | Portfolio=$107,452 | Trades=112 | ε=0.010
Step 270,000/300,000 | Ep 496 | ε=0.010 | Avg Reward: +1.7936
  └─ Episode 500: Reward=+1.7773 | Portfolio=$101,084 | Trades=108 | ε=0.010
Step 280,000/300,000 | Ep 514 | ε=0.010 | Avg Reward: +1.8086
  └─ Episode 520: Reward=+1.8334 | Portfolio=$107,131 | Trades=108 | ε=0.010
Step 290,000/300,000 | Ep 533 | ε=0.010 | Avg Reward: +1.8174
  └─ Episode 540: Reward=+1.8282 | Portfolio=$107,653 | Trades=110 | ε=0.010
Step 300,000/300,000 | Ep 551 | ε=0.010 | Avg Reward: +1.7921

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 552
Final Epsilon: 0.0100
Mean Reward: 1.7358
Best Episode Reward: 1.9762
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_231838.pkl
Config saved to results/config_v2_20251209_231838.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $107,652.88
Return:  +976.53%
Trades:  110
Fees:    $10377.81

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $12,141.23
Return:  +21.41%
Trades:  12
Fees:    $261.82

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |     +976.53% |      +21.41%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +834.44% |      -35.67%

 Final Values:
   Agent Train: $107,652.88  |  Buy&Hold: $24,208.72
   Agent Test:  $12,141.23  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_231838.pkl
Final Epsilon: 0.0100
Total Episodes: 552

 Agent made profit, but didn't beat Buy & Hold
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader->

v11
 config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000
        }

 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.995 (reaches 0.01 after ~918 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 500,000
Epsilon: 1.00  0.01
Decay rate: 0.995
============================================================

Step  10,000/500,000 | Ep  18 | ε=0.914 | Avg Reward: -0.9674
  └─ Episode 20: Reward=-0.7355 | Portfolio=$14,280 | Trades=173 | ε=0.905
Step  20,000/500,000 | Ep  36 | ε=0.835 | Avg Reward: -0.8452
  └─ Episode 40: Reward=-0.1655 | Portfolio=$21,706 | Trades=158 | ε=0.818
Step  30,000/500,000 | Ep  55 | ε=0.759 | Avg Reward: -0.6689
  └─ Episode 60: Reward=-0.5449 | Portfolio=$15,083 | Trades=161 | ε=0.740
Step  40,000/500,000 | Ep  73 | ε=0.694 | Avg Reward: -0.4246
  └─ Episode 80: Reward=+0.0224 | Portfolio=$27,625 | Trades=175 | ε=0.670
Step  50,000/500,000 | Ep  91 | ε=0.634 | Avg Reward: -0.1411
  └─ Episode 100: Reward=+0.4301 | Portfolio=$38,306 | Trades=160 | ε=0.606
Step  60,000/500,000 | Ep 110 | ε=0.576 | Avg Reward: +0.0109
  └─ Episode 120: Reward=+0.0811 | Portfolio=$26,660 | Trades=157 | ε=0.548
Step  70,000/500,000 | Ep 128 | ε=0.526 | Avg Reward: +0.0882
  └─ Episode 140: Reward=+0.2544 | Portfolio=$31,490 | Trades=156 | ε=0.496
Step  80,000/500,000 | Ep 147 | ε=0.479 | Avg Reward: +0.3026
  └─ Episode 160: Reward=+0.4843 | Portfolio=$40,868 | Trades=163 | ε=0.448
Step  90,000/500,000 | Ep 165 | ε=0.437 | Avg Reward: +0.5532
  └─ Episode 180: Reward=+0.1874 | Portfolio=$27,118 | Trades=142 | ε=0.406
Step 100,000/500,000 | Ep 183 | ε=0.400 | Avg Reward: +0.4382
  └─ Episode 200: Reward=+0.8545 | Portfolio=$46,584 | Trades=128 | ε=0.367
Step 110,000/500,000 | Ep 202 | ε=0.363 | Avg Reward: +0.6052
  └─ Episode 220: Reward=+0.4227 | Portfolio=$35,198 | Trades=158 | ε=0.332
Step 120,000/500,000 | Ep 220 | ε=0.332 | Avg Reward: +0.5249
Step 130,000/500,000 | Ep 238 | ε=0.303 | Avg Reward: +0.7684
  └─ Episode 240: Reward=+0.9453 | Portfolio=$58,567 | Trades=149 | ε=0.300
Step 140,000/500,000 | Ep 257 | ε=0.276 | Avg Reward: +1.0360
  └─ Episode 260: Reward=+0.8303 | Portfolio=$54,068 | Trades=150 | ε=0.272
Step 150,000/500,000 | Ep 275 | ε=0.252 | Avg Reward: +0.8997
  └─ Episode 280: Reward=+0.8820 | Portfolio=$57,523 | Trades=152 | ε=0.246
Step 160,000/500,000 | Ep 294 | ε=0.229 | Avg Reward: +0.8409
  └─ Episode 300: Reward=+0.8820 | Portfolio=$45,750 | Trades=118 | ε=0.222
Step 170,000/500,000 | Ep 312 | ε=0.209 | Avg Reward: +1.0562
  └─ Episode 320: Reward=+0.9917 | Portfolio=$59,010 | Trades=134 | ε=0.201
Step 180,000/500,000 | Ep 330 | ε=0.191 | Avg Reward: +1.1292
  └─ Episode 340: Reward=+1.2943 | Portfolio=$67,504 | Trades=118 | ε=0.182
Step 190,000/500,000 | Ep 349 | ε=0.174 | Avg Reward: +1.1392
  └─ Episode 360: Reward=+1.2816 | Portfolio=$73,576 | Trades=126 | ε=0.165
Step 200,000/500,000 | Ep 367 | ε=0.159 | Avg Reward: +1.3011
  └─ Episode 380: Reward=+0.8520 | Portfolio=$45,623 | Trades=124 | ε=0.149
Step 210,000/500,000 | Ep 386 | ε=0.144 | Avg Reward: +1.2991
  └─ Episode 400: Reward=+1.6766 | Portfolio=$88,202 | Trades=102 | ε=0.135
Step 220,000/500,000 | Ep 404 | ε=0.132 | Avg Reward: +1.4727
  └─ Episode 420: Reward=+1.2275 | Portfolio=$66,342 | Trades=124 | ε=0.122
Step 230,000/500,000 | Ep 422 | ε=0.121 | Avg Reward: +1.3880
  └─ Episode 440: Reward=+1.6280 | Portfolio=$87,314 | Trades=104 | ε=0.110
Step 240,000/500,000 | Ep 441 | ε=0.110 | Avg Reward: +1.5498
Step 250,000/500,000 | Ep 459 | ε=0.100 | Avg Reward: +1.5066
  └─ Episode 460: Reward=+1.5537 | Portfolio=$86,967 | Trades=122 | ε=0.100
Step 260,000/500,000 | Ep 477 | ε=0.092 | Avg Reward: +1.5894
  └─ Episode 480: Reward=+1.5545 | Portfolio=$82,875 | Trades=110 | ε=0.090
Step 270,000/500,000 | Ep 496 | ε=0.083 | Avg Reward: +1.5595
  └─ Episode 500: Reward=+1.9188 | Portfolio=$118,078 | Trades=112 | ε=0.082
Step 280,000/500,000 | Ep 514 | ε=0.076 | Avg Reward: +1.6053
  └─ Episode 520: Reward=+1.6149 | Portfolio=$85,119 | Trades=102 | ε=0.074
Step 290,000/500,000 | Ep 533 | ε=0.069 | Avg Reward: +1.7132
  └─ Episode 540: Reward=+1.8188 | Portfolio=$100,369 | Trades=102 | ε=0.067
Step 300,000/500,000 | Ep 551 | ε=0.063 | Avg Reward: +1.6566
  └─ Episode 560: Reward=+1.7841 | Portfolio=$103,496 | Trades=110 | ε=0.060
Step 310,000/500,000 | Ep 569 | ε=0.058 | Avg Reward: +1.6862
  └─ Episode 580: Reward=+1.7867 | Portfolio=$102,452 | Trades=106 | ε=0.055
Step 320,000/500,000 | Ep 588 | ε=0.052 | Avg Reward: +1.7735
  └─ Episode 600: Reward=+1.6612 | Portfolio=$91,097 | Trades=108 | ε=0.049
Step 330,000/500,000 | Ep 606 | ε=0.048 | Avg Reward: +1.7434
  └─ Episode 620: Reward=+1.8215 | Portfolio=$107,167 | Trades=114 | ε=0.045
Step 340,000/500,000 | Ep 624 | ε=0.044 | Avg Reward: +1.7121
  └─ Episode 640: Reward=+1.6233 | Portfolio=$88,529 | Trades=112 | ε=0.040
Step 350,000/500,000 | Ep 643 | ε=0.040 | Avg Reward: +1.7837
  └─ Episode 660: Reward=+1.8124 | Portfolio=$102,288 | Trades=104 | ε=0.037
Step 360,000/500,000 | Ep 661 | ε=0.036 | Avg Reward: +1.8099
  └─ Episode 680: Reward=+1.8516 | Portfolio=$102,279 | Trades=98 | ε=0.033
Step 370,000/500,000 | Ep 680 | ε=0.033 | Avg Reward: +1.7623
Step 380,000/500,000 | Ep 698 | ε=0.030 | Avg Reward: +1.7682
  └─ Episode 700: Reward=+1.8143 | Portfolio=$104,159 | Trades=106 | ε=0.030
Step 390,000/500,000 | Ep 716 | ε=0.028 | Avg Reward: +1.8383
  └─ Episode 720: Reward=+1.6864 | Portfolio=$88,832 | Trades=100 | ε=0.027
Step 400,000/500,000 | Ep 735 | ε=0.025 | Avg Reward: +1.7756
  └─ Episode 740: Reward=+1.8661 | Portfolio=$106,650 | Trades=104 | ε=0.024
Step 410,000/500,000 | Ep 753 | ε=0.023 | Avg Reward: +1.8605
  └─ Episode 760: Reward=+1.8101 | Portfolio=$98,586 | Trades=98 | ε=0.022
Step 420,000/500,000 | Ep 772 | ε=0.021 | Avg Reward: +1.8742
  └─ Episode 780: Reward=+1.8480 | Portfolio=$105,524 | Trades=104 | ε=0.020
Step 430,000/500,000 | Ep 790 | ε=0.019 | Avg Reward: +1.8195
  └─ Episode 800: Reward=+1.9167 | Portfolio=$108,176 | Trades=96 | ε=0.018
Step 440,000/500,000 | Ep 808 | ε=0.017 | Avg Reward: +1.8442
  └─ Episode 820: Reward=+1.8476 | Portfolio=$102,961 | Trades=100 | ε=0.016
Step 450,000/500,000 | Ep 827 | ε=0.016 | Avg Reward: +1.8523
  └─ Episode 840: Reward=+1.9074 | Portfolio=$108,785 | Trades=100 | ε=0.015
Step 460,000/500,000 | Ep 845 | ε=0.014 | Avg Reward: +1.8766
  └─ Episode 860: Reward=+1.8597 | Portfolio=$103,636 | Trades=100 | ε=0.013
Step 470,000/500,000 | Ep 863 | ε=0.013 | Avg Reward: +1.8264
  └─ Episode 880: Reward=+1.8696 | Portfolio=$105,801 | Trades=100 | ε=0.012
Step 480,000/500,000 | Ep 882 | ε=0.012 | Avg Reward: +1.8780
  └─ Episode 900: Reward=+1.9324 | Portfolio=$111,524 | Trades=100 | ε=0.011
Step 490,000/500,000 | Ep 900 | ε=0.011 | Avg Reward: +1.8894
Step 500,000/500,000 | Ep 919 | ε=0.010 | Avg Reward: +1.8771
  └─ Episode 920: Reward=+0.3281 | Portfolio=$14,005 | Trades=6 | ε=0.010

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 920
Final Epsilon: 0.0100
Mean Reward: 1.1780
Best Episode Reward: 2.0151
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_232415.pkl
Config saved to results/config_v2_20251209_232415.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $108,593.07
Return:  +985.93%
Trades:  98
Fees:    $9462.55

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $11,987.20
Return:  +19.87%
Trades:  30
Fees:    $619.63

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |     +985.93% |      +19.87%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +843.84% |      -37.21%

 Final Values:
   Agent Train: $108,593.07  |  Buy&Hold: $24,208.72
   Agent Test:  $11,987.20  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_232415.pkl
Final Epsilon: 0.0100
Total Episodes: 920

v12
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.996,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.993 (reaches 0.01 after ~655 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 300,000
Epsilon: 1.00  0.01
Decay rate: 0.993
============================================================

Step  10,000/300,000 | Ep  18 | ε=0.881 | Avg Reward: -1.1035
  └─ Episode 20: Reward=-0.5707 | Portfolio=$15,401 | Trades=166 | ε=0.869
Step  20,000/300,000 | Ep  36 | ε=0.777 | Avg Reward: -0.7361
  └─ Episode 40: Reward=-0.7791 | Portfolio=$11,992 | Trades=163 | ε=0.755
Step  30,000/300,000 | Ep  55 | ε=0.680 | Avg Reward: -0.3426
  └─ Episode 60: Reward=-0.4265 | Portfolio=$16,716 | Trades=167 | ε=0.656
Step  40,000/300,000 | Ep  73 | ε=0.599 | Avg Reward: -0.0837
  └─ Episode 80: Reward=-0.2382 | Portfolio=$21,972 | Trades=166 | ε=0.570
Step  50,000/300,000 | Ep  91 | ε=0.528 | Avg Reward: +0.2456
  └─ Episode 100: Reward=+0.1100 | Portfolio=$32,514 | Trades=182 | ε=0.495
Step  60,000/300,000 | Ep 110 | ε=0.462 | Avg Reward: +0.1776
  └─ Episode 120: Reward=+0.4740 | Portfolio=$37,717 | Trades=148 | ε=0.430
Step  70,000/300,000 | Ep 128 | ε=0.407 | Avg Reward: +0.4255
  └─ Episode 140: Reward=+0.5546 | Portfolio=$44,647 | Trades=164 | ε=0.374
Step  80,000/300,000 | Ep 147 | ε=0.356 | Avg Reward: +0.7081
  └─ Episode 160: Reward=+1.1062 | Portfolio=$73,707 | Trades=150 | ε=0.325
Step  90,000/300,000 | Ep 165 | ε=0.314 | Avg Reward: +0.7168
  └─ Episode 180: Reward=+0.7454 | Portfolio=$49,779 | Trades=154 | ε=0.282
Step 100,000/300,000 | Ep 183 | ε=0.277 | Avg Reward: +0.9482
  └─ Episode 200: Reward=+1.1203 | Portfolio=$55,142 | Trades=118 | ε=0.245
Step 110,000/300,000 | Ep 202 | ε=0.242 | Avg Reward: +0.8494
  └─ Episode 220: Reward=+1.3307 | Portfolio=$87,144 | Trades=150 | ε=0.213
Step 120,000/300,000 | Ep 220 | ε=0.213 | Avg Reward: +1.1041
Step 130,000/300,000 | Ep 238 | ε=0.188 | Avg Reward: +1.2558
  └─ Episode 240: Reward=+1.3236 | Portfolio=$82,111 | Trades=146 | ε=0.185
Step 140,000/300,000 | Ep 257 | ε=0.164 | Avg Reward: +1.2363
  └─ Episode 260: Reward=+1.4810 | Portfolio=$100,309 | Trades=152 | ε=0.161
Step 150,000/300,000 | Ep 275 | ε=0.145 | Avg Reward: +1.2691
  └─ Episode 280: Reward=+1.3856 | Portfolio=$83,372 | Trades=138 | ε=0.140
Step 160,000/300,000 | Ep 294 | ε=0.127 | Avg Reward: +1.3099
  └─ Episode 300: Reward=+1.5068 | Portfolio=$94,459 | Trades=140 | ε=0.122
Step 170,000/300,000 | Ep 312 | ε=0.112 | Avg Reward: +1.4230
  └─ Episode 320: Reward=+1.4038 | Portfolio=$86,000 | Trades=144 | ε=0.106
Step 180,000/300,000 | Ep 330 | ε=0.098 | Avg Reward: +1.3809
  └─ Episode 340: Reward=+1.6092 | Portfolio=$110,199 | Trades=150 | ε=0.092
Step 190,000/300,000 | Ep 349 | ε=0.086 | Avg Reward: +1.5496
  └─ Episode 360: Reward=+1.3396 | Portfolio=$79,530 | Trades=142 | ε=0.080
Step 200,000/300,000 | Ep 367 | ε=0.076 | Avg Reward: +1.4633
  └─ Episode 380: Reward=+1.6749 | Portfolio=$109,957 | Trades=145 | ε=0.069
Step 210,000/300,000 | Ep 386 | ε=0.066 | Avg Reward: +1.5967
  └─ Episode 400: Reward=+1.5529 | Portfolio=$96,657 | Trades=140 | ε=0.060
Step 220,000/300,000 | Ep 404 | ε=0.059 | Avg Reward: +1.4873
  └─ Episode 420: Reward=+1.4443 | Portfolio=$90,136 | Trades=140 | ε=0.052
Step 230,000/300,000 | Ep 422 | ε=0.052 | Avg Reward: +1.4830
  └─ Episode 440: Reward=+1.7519 | Portfolio=$116,642 | Trades=136 | ε=0.045
Step 240,000/300,000 | Ep 441 | ε=0.045 | Avg Reward: +1.6196
Step 250,000/300,000 | Ep 459 | ε=0.040 | Avg Reward: +1.6113
  └─ Episode 460: Reward=+1.5718 | Portfolio=$96,289 | Trades=138 | ε=0.040
Step 260,000/300,000 | Ep 477 | ε=0.035 | Avg Reward: +1.6544
  └─ Episode 480: Reward=+1.7168 | Portfolio=$115,917 | Trades=144 | ε=0.034
Step 270,000/300,000 | Ep 496 | ε=0.031 | Avg Reward: +1.5865
  └─ Episode 500: Reward=+1.5229 | Portfolio=$98,831 | Trades=148 | ε=0.030
Step 280,000/300,000 | Ep 514 | ε=0.027 | Avg Reward: +1.6698
  └─ Episode 520: Reward=+1.5003 | Portfolio=$91,969 | Trades=144 | ε=0.026
Step 290,000/300,000 | Ep 533 | ε=0.024 | Avg Reward: +1.5951
  └─ Episode 540: Reward=+1.6957 | Portfolio=$114,409 | Trades=146 | ε=0.023
Step 300,000/300,000 | Ep 551 | ε=0.021 | Avg Reward: +1.6678

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 552
Final Epsilon: 0.0207
Mean Reward: 0.9811
Best Episode Reward: 1.8285
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_233057.pkl
Config saved to results/config_v2_20251209_233057.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $115,434.44
Return:  +1054.34%
Trades:  140
Fees:    $12556.51

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $13,797.27
Return:  +37.97%
Trades:  32
Fees:    $722.27

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |    +1054.34% |      +37.97%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +912.26% |      -19.11%

 Final Values:
   Agent Train: $115,434.44  |  Buy&Hold: $24,208.72
   Agent Test:  $13,797.27  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_233057.pkl
Final Epsilon: 0.0207
Total Episodes: 552

 Agent made profit, but didn't beat Buy & Hold
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader->

v13
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.998,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 500000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.998 (reaches 0.01 after ~2300 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 500,000
Epsilon: 1.00  0.01
Decay rate: 0.998
============================================================

Step  10,000/500,000 | Ep  18 | ε=0.965 | Avg Reward: -1.4597
  └─ Episode 20: Reward=-0.6117 | Portfolio=$14,414 | Trades=166 | ε=0.961
Step  20,000/500,000 | Ep  36 | ε=0.930 | Avg Reward: -0.9347
  └─ Episode 40: Reward=-0.9355 | Portfolio=$11,746 | Trades=186 | ε=0.923
Step  30,000/500,000 | Ep  55 | ε=0.896 | Avg Reward: -0.9577
  └─ Episode 60: Reward=-0.7906 | Portfolio=$14,512 | Trades=192 | ε=0.887
Step  40,000/500,000 | Ep  73 | ε=0.864 | Avg Reward: -0.7305
  └─ Episode 80: Reward=-0.5309 | Portfolio=$16,074 | Trades=165 | ε=0.852
Step  50,000/500,000 | Ep  91 | ε=0.833 | Avg Reward: -0.8635
  └─ Episode 100: Reward=-1.1541 | Portfolio=$8,841 | Trades=175 | ε=0.819
Step  60,000/500,000 | Ep 110 | ε=0.802 | Avg Reward: -0.6066
  └─ Episode 120: Reward=-0.7203 | Portfolio=$13,387 | Trades=172 | ε=0.786
Step  70,000/500,000 | Ep 128 | ε=0.774 | Avg Reward: -0.3655
  └─ Episode 140: Reward=-0.3831 | Portfolio=$17,228 | Trades=159 | ε=0.756
Step  80,000/500,000 | Ep 147 | ε=0.745 | Avg Reward: -0.5446
  └─ Episode 160: Reward=-0.4342 | Portfolio=$18,028 | Trades=182 | ε=0.726
Step  90,000/500,000 | Ep 165 | ε=0.719 | Avg Reward: -0.6761
  └─ Episode 180: Reward=-0.5704 | Portfolio=$15,484 | Trades=167 | ε=0.697
Step 100,000/500,000 | Ep 183 | ε=0.693 | Avg Reward: -0.3619
  └─ Episode 200: Reward=-0.7370 | Portfolio=$13,798 | Trades=177 | ε=0.670
Step 110,000/500,000 | Ep 202 | ε=0.667 | Avg Reward: -0.2066
  └─ Episode 220: Reward=-0.4616 | Portfolio=$17,964 | Trades=178 | ε=0.644
Step 120,000/500,000 | Ep 220 | ε=0.644 | Avg Reward: -0.1741
Step 130,000/500,000 | Ep 238 | ε=0.621 | Avg Reward: -0.1635
  └─ Episode 240: Reward=+0.0813 | Portfolio=$27,528 | Trades=164 | ε=0.618
Step 140,000/500,000 | Ep 257 | ε=0.598 | Avg Reward: -0.0555
  └─ Episode 260: Reward=-0.0037 | Portfolio=$25,371 | Trades=155 | ε=0.594
Step 150,000/500,000 | Ep 275 | ε=0.577 | Avg Reward: -0.0234
  └─ Episode 280: Reward=-0.2526 | Portfolio=$20,926 | Trades=176 | ε=0.571
Step 160,000/500,000 | Ep 294 | ε=0.555 | Avg Reward: -0.0084
  └─ Episode 300: Reward=+0.7174 | Portfolio=$44,617 | Trades=142 | ε=0.548
Step 170,000/500,000 | Ep 312 | ε=0.535 | Avg Reward: +0.0927
  └─ Episode 320: Reward=+0.5035 | Portfolio=$48,118 | Trades=177 | ε=0.527
Step 180,000/500,000 | Ep 330 | ε=0.517 | Avg Reward: +0.2097
  └─ Episode 340: Reward=+0.3900 | Portfolio=$36,125 | Trades=158 | ε=0.506
Step 190,000/500,000 | Ep 349 | ε=0.497 | Avg Reward: +0.2385
  └─ Episode 360: Reward=+0.2049 | Portfolio=$32,978 | Trades=174 | ε=0.486
Step 200,000/500,000 | Ep 367 | ε=0.480 | Avg Reward: +0.2222
  └─ Episode 380: Reward=+0.7444 | Portfolio=$48,334 | Trades=151 | ε=0.467
Step 210,000/500,000 | Ep 386 | ε=0.462 | Avg Reward: +0.3200
  └─ Episode 400: Reward=+0.3141 | Portfolio=$34,471 | Trades=164 | ε=0.449
Step 220,000/500,000 | Ep 404 | ε=0.445 | Avg Reward: +0.4643
  └─ Episode 420: Reward=+0.4116 | Portfolio=$39,663 | Trades=170 | ε=0.431
Step 230,000/500,000 | Ep 422 | ε=0.430 | Avg Reward: +0.3395
  └─ Episode 440: Reward=+0.0980 | Portfolio=$27,706 | Trades=161 | ε=0.414
Step 240,000/500,000 | Ep 441 | ε=0.414 | Avg Reward: +0.5420
Step 250,000/500,000 | Ep 459 | ε=0.399 | Avg Reward: +0.3806
  └─ Episode 460: Reward=+0.7117 | Portfolio=$42,385 | Trades=138 | ε=0.398
Step 260,000/500,000 | Ep 477 | ε=0.385 | Avg Reward: +0.6309
  └─ Episode 480: Reward=+0.9845 | Portfolio=$59,973 | Trades=152 | ε=0.383
Step 270,000/500,000 | Ep 496 | ε=0.370 | Avg Reward: +0.4550
  └─ Episode 500: Reward=+0.6514 | Portfolio=$41,027 | Trades=138 | ε=0.368
Step 280,000/500,000 | Ep 514 | ε=0.357 | Avg Reward: +0.4866
  └─ Episode 520: Reward=+0.4613 | Portfolio=$37,401 | Trades=153 | ε=0.353
Step 290,000/500,000 | Ep 533 | ε=0.344 | Avg Reward: +0.6652
  └─ Episode 540: Reward=+1.1157 | Portfolio=$62,500 | Trades=132 | ε=0.339
Step 300,000/500,000 | Ep 551 | ε=0.332 | Avg Reward: +0.4834
  └─ Episode 560: Reward=+0.9420 | Portfolio=$52,101 | Trades=132 | ε=0.326
Step 310,000/500,000 | Ep 569 | ε=0.320 | Avg Reward: +0.7247
  └─ Episode 580: Reward=+0.8233 | Portfolio=$51,979 | Trades=150 | ε=0.313
Step 320,000/500,000 | Ep 588 | ε=0.308 | Avg Reward: +0.8120
  └─ Episode 600: Reward=+0.7894 | Portfolio=$50,493 | Trades=146 | ε=0.301
Step 330,000/500,000 | Ep 606 | ε=0.297 | Avg Reward: +0.7481
  └─ Episode 620: Reward=+0.6460 | Portfolio=$44,078 | Trades=144 | ε=0.289
Step 340,000/500,000 | Ep 624 | ε=0.287 | Avg Reward: +0.7335
  └─ Episode 640: Reward=+0.5272 | Portfolio=$40,835 | Trades=150 | ε=0.278
Step 350,000/500,000 | Ep 643 | ε=0.276 | Avg Reward: +0.7660
  └─ Episode 660: Reward=+1.0459 | Portfolio=$64,551 | Trades=143 | ε=0.267
Step 360,000/500,000 | Ep 661 | ε=0.266 | Avg Reward: +0.8836
  └─ Episode 680: Reward=+1.0642 | Portfolio=$64,576 | Trades=145 | ε=0.256
Step 370,000/500,000 | Ep 680 | ε=0.256 | Avg Reward: +1.1488
Step 380,000/500,000 | Ep 698 | ε=0.247 | Avg Reward: +0.9030
  └─ Episode 700: Reward=+0.8346 | Portfolio=$53,986 | Trades=156 | ε=0.246
Step 390,000/500,000 | Ep 716 | ε=0.238 | Avg Reward: +0.9613
  └─ Episode 720: Reward=+1.2966 | Portfolio=$72,532 | Trades=128 | ε=0.237
Step 400,000/500,000 | Ep 735 | ε=0.230 | Avg Reward: +0.9536
  └─ Episode 740: Reward=+1.2550 | Portfolio=$72,768 | Trades=134 | ε=0.227
Step 410,000/500,000 | Ep 753 | ε=0.221 | Avg Reward: +1.0709
  └─ Episode 760: Reward=+0.9233 | Portfolio=$51,634 | Trades=132 | ε=0.218
Step 420,000/500,000 | Ep 772 | ε=0.213 | Avg Reward: +1.0453
  └─ Episode 780: Reward=+1.2305 | Portfolio=$72,817 | Trades=147 | ε=0.210
Step 430,000/500,000 | Ep 790 | ε=0.206 | Avg Reward: +1.1093
  └─ Episode 800: Reward=+1.3638 | Portfolio=$74,153 | Trades=124 | ε=0.202
Step 440,000/500,000 | Ep 808 | ε=0.198 | Avg Reward: +1.0915
  └─ Episode 820: Reward=+1.0656 | Portfolio=$56,162 | Trades=126 | ε=0.194
Step 450,000/500,000 | Ep 827 | ε=0.191 | Avg Reward: +1.1758
  └─ Episode 840: Reward=+1.3026 | Portfolio=$73,558 | Trades=133 | ε=0.186
Step 460,000/500,000 | Ep 845 | ε=0.184 | Avg Reward: +1.1998
  └─ Episode 860: Reward=+1.2521 | Portfolio=$76,478 | Trades=148 | ε=0.179
Step 470,000/500,000 | Ep 863 | ε=0.178 | Avg Reward: +1.2178
  └─ Episode 880: Reward=+1.3326 | Portfolio=$75,520 | Trades=134 | ε=0.172
Step 480,000/500,000 | Ep 882 | ε=0.171 | Avg Reward: +1.1989
  └─ Episode 900: Reward=+1.1343 | Portfolio=$64,471 | Trades=128 | ε=0.165
Step 490,000/500,000 | Ep 900 | ε=0.165 | Avg Reward: +1.3606
Step 500,000/500,000 | Ep 919 | ε=0.159 | Avg Reward: +1.2934
  └─ Episode 920: Reward=+0.1781 | Portfolio=$13,190 | Trades=20 | ε=0.159

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 920
Final Epsilon: 0.1585
Mean Reward: 0.3631
Best Episode Reward: 1.6397
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251209_233637.pkl
Config saved to results/config_v2_20251209_233637.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $110,580.22
Return:  +1005.80%
Trades:  108
Fees:    $10357.96

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $11,917.72
Return:  +19.18%
Trades:  30
Fees:    $615.79

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |    +1005.80% |      +19.18%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +863.72% |      -37.90%

 Final Values:
   Agent Train: $110,580.22  |  Buy&Hold: $24,208.72
   Agent Test:  $11,917.72  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251209_233637.pkl
Final Epsilon: 0.1585
Total Episodes: 920

 Agent made profit, but didn't beat Buy & Hold
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader->

v14
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 500000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True       
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.995 (reaches 0.01 after ~918 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 500,000
Epsilon: 1.00  0.01
Decay rate: 0.995
============================================================

Step  10,000/500,000 | Ep  18 | ε=0.914 | Avg Reward: -1.0429
  └─ Episode 20: Reward=-0.9250 | Portfolio=$10,985 | Trades=169 | ε=0.905
Step  20,000/500,000 | Ep  36 | ε=0.835 | Avg Reward: -0.9008
  └─ Episode 40: Reward=-0.6239 | Portfolio=$15,133 | Trades=176 | ε=0.818
Step  30,000/500,000 | Ep  55 | ε=0.759 | Avg Reward: -0.5633
  └─ Episode 60: Reward=+0.3823 | Portfolio=$38,131 | Trades=157 | ε=0.740
Step  40,000/500,000 | Ep  73 | ε=0.694 | Avg Reward: -0.4769
  └─ Episode 80: Reward=-0.6123 | Portfolio=$14,487 | Trades=173 | ε=0.670
Step  50,000/500,000 | Ep  91 | ε=0.634 | Avg Reward: -0.2502
  └─ Episode 100: Reward=-0.2572 | Portfolio=$21,703 | Trades=180 | ε=0.606
Step  60,000/500,000 | Ep 110 | ε=0.576 | Avg Reward: -0.0254
  └─ Episode 120: Reward=+0.3024 | Portfolio=$32,556 | Trades=164 | ε=0.548
Step  70,000/500,000 | Ep 128 | ε=0.526 | Avg Reward: -0.0315
  └─ Episode 140: Reward=-0.1807 | Portfolio=$20,678 | Trades=159 | ε=0.496
Step  80,000/500,000 | Ep 147 | ε=0.479 | Avg Reward: +0.1319
  └─ Episode 160: Reward=+0.6010 | Portfolio=$44,562 | Trades=162 | ε=0.448
Step  90,000/500,000 | Ep 165 | ε=0.437 | Avg Reward: +0.4941
  └─ Episode 180: Reward=+0.3082 | Portfolio=$32,283 | Trades=154 | ε=0.406
Step 100,000/500,000 | Ep 183 | ε=0.400 | Avg Reward: +0.4018
  └─ Episode 200: Reward=+0.4235 | Portfolio=$30,171 | Trades=125 | ε=0.367
Step 110,000/500,000 | Ep 202 | ε=0.363 | Avg Reward: +0.4922
  └─ Episode 220: Reward=+1.1096 | Portfolio=$62,481 | Trades=130 | ε=0.332
Step 120,000/500,000 | Ep 220 | ε=0.332 | Avg Reward: +0.6396
Step 130,000/500,000 | Ep 238 | ε=0.303 | Avg Reward: +0.7508
  └─ Episode 240: Reward=+1.0396 | Portfolio=$60,219 | Trades=138 | ε=0.300
Step 140,000/500,000 | Ep 257 | ε=0.276 | Avg Reward: +0.9888
  └─ Episode 260: Reward=+1.0020 | Portfolio=$52,396 | Trades=115 | ε=0.272
Step 150,000/500,000 | Ep 275 | ε=0.252 | Avg Reward: +0.9187
  └─ Episode 280: Reward=+0.9444 | Portfolio=$50,497 | Trades=130 | ε=0.246
Step 160,000/500,000 | Ep 294 | ε=0.229 | Avg Reward: +1.1438
  └─ Episode 300: Reward=+1.2329 | Portfolio=$63,959 | Trades=118 | ε=0.222
Step 170,000/500,000 | Ep 312 | ε=0.209 | Avg Reward: +1.2196
  └─ Episode 320: Reward=+1.0147 | Portfolio=$55,854 | Trades=136 | ε=0.201
Step 180,000/500,000 | Ep 330 | ε=0.191 | Avg Reward: +1.2581
  └─ Episode 340: Reward=+1.1116 | Portfolio=$57,447 | Trades=122 | ε=0.182
Step 190,000/500,000 | Ep 349 | ε=0.174 | Avg Reward: +1.1514
  └─ Episode 360: Reward=+0.9421 | Portfolio=$48,857 | Trades=120 | ε=0.165
Step 200,000/500,000 | Ep 367 | ε=0.159 | Avg Reward: +1.2739
  └─ Episode 380: Reward=+0.8324 | Portfolio=$43,588 | Trades=118 | ε=0.149
Step 210,000/500,000 | Ep 386 | ε=0.144 | Avg Reward: +1.2425
  └─ Episode 400: Reward=+1.2080 | Portfolio=$61,652 | Trades=122 | ε=0.135
Step 220,000/500,000 | Ep 404 | ε=0.132 | Avg Reward: +1.2639
  └─ Episode 420: Reward=+1.4406 | Portfolio=$77,260 | Trades=116 | ε=0.122
Step 230,000/500,000 | Ep 422 | ε=0.121 | Avg Reward: +1.4023
  └─ Episode 440: Reward=+1.3234 | Portfolio=$65,839 | Trades=108 | ε=0.110
Step 240,000/500,000 | Ep 441 | ε=0.110 | Avg Reward: +1.4839
Step 250,000/500,000 | Ep 459 | ε=0.100 | Avg Reward: +1.3972
  └─ Episode 460: Reward=+1.3832 | Portfolio=$70,387 | Trades=110 | ε=0.100
Step 260,000/500,000 | Ep 477 | ε=0.092 | Avg Reward: +1.5520
  └─ Episode 480: Reward=+1.4496 | Portfolio=$77,116 | Trades=114 | ε=0.090
Step 270,000/500,000 | Ep 496 | ε=0.083 | Avg Reward: +1.5775
  └─ Episode 500: Reward=+1.5737 | Portfolio=$82,077 | Trades=104 | ε=0.082
Step 280,000/500,000 | Ep 514 | ε=0.076 | Avg Reward: +1.5936
  └─ Episode 520: Reward=+1.6238 | Portfolio=$87,671 | Trades=108 | ε=0.074
Step 290,000/500,000 | Ep 533 | ε=0.069 | Avg Reward: +1.5485
  └─ Episode 540: Reward=+1.5287 | Portfolio=$78,199 | Trades=104 | ε=0.067
Step 300,000/500,000 | Ep 551 | ε=0.063 | Avg Reward: +1.6511
  └─ Episode 560: Reward=+1.5245 | Portfolio=$77,209 | Trades=104 | ε=0.060
Step 310,000/500,000 | Ep 569 | ε=0.058 | Avg Reward: +1.5883
  └─ Episode 580: Reward=+1.6773 | Portfolio=$90,913 | Trades=104 | ε=0.055
Step 320,000/500,000 | Ep 588 | ε=0.052 | Avg Reward: +1.6254
  └─ Episode 600: Reward=+1.3776 | Portfolio=$65,769 | Trades=102 | ε=0.049
Step 330,000/500,000 | Ep 606 | ε=0.048 | Avg Reward: +1.5603
  └─ Episode 620: Reward=+1.7188 | Portfolio=$94,229 | Trades=106 | ε=0.045
Step 340,000/500,000 | Ep 624 | ε=0.044 | Avg Reward: +1.6554
  └─ Episode 640: Reward=+1.7086 | Portfolio=$92,386 | Trades=102 | ε=0.040
Step 350,000/500,000 | Ep 643 | ε=0.040 | Avg Reward: +1.7084
  └─ Episode 660: Reward=+1.8498 | Portfolio=$104,825 | Trades=102 | ε=0.037
Step 360,000/500,000 | Ep 661 | ε=0.036 | Avg Reward: +1.7148
  └─ Episode 680: Reward=+1.7511 | Portfolio=$93,355 | Trades=100 | ε=0.033
Step 370,000/500,000 | Ep 680 | ε=0.033 | Avg Reward: +1.7331
Step 380,000/500,000 | Ep 698 | ε=0.030 | Avg Reward: +1.7560
  └─ Episode 700: Reward=+1.8358 | Portfolio=$103,002 | Trades=102 | ε=0.030
Step 390,000/500,000 | Ep 716 | ε=0.028 | Avg Reward: +1.7580
  └─ Episode 720: Reward=+1.7137 | Portfolio=$88,625 | Trades=96 | ε=0.027
Step 400,000/500,000 | Ep 735 | ε=0.025 | Avg Reward: +1.7733
  └─ Episode 740: Reward=+1.7872 | Portfolio=$96,812 | Trades=98 | ε=0.024
Step 410,000/500,000 | Ep 753 | ε=0.023 | Avg Reward: +1.7465
  └─ Episode 760: Reward=+1.7966 | Portfolio=$97,727 | Trades=98 | ε=0.022
Step 420,000/500,000 | Ep 772 | ε=0.021 | Avg Reward: +1.8006
  └─ Episode 780: Reward=+1.7029 | Portfolio=$90,367 | Trades=100 | ε=0.020
Step 430,000/500,000 | Ep 790 | ε=0.019 | Avg Reward: +1.7937
  └─ Episode 800: Reward=+1.6505 | Portfolio=$85,948 | Trades=102 | ε=0.018
Step 440,000/500,000 | Ep 808 | ε=0.017 | Avg Reward: +1.7739
  └─ Episode 820: Reward=+1.8131 | Portfolio=$100,366 | Trades=100 | ε=0.016
Step 450,000/500,000 | Ep 827 | ε=0.016 | Avg Reward: +1.8175
  └─ Episode 840: Reward=+1.6227 | Portfolio=$82,947 | Trades=99 | ε=0.015
Step 460,000/500,000 | Ep 845 | ε=0.014 | Avg Reward: +1.7917
  └─ Episode 860: Reward=+1.8253 | Portfolio=$100,098 | Trades=98 | ε=0.013
Step 470,000/500,000 | Ep 863 | ε=0.013 | Avg Reward: +1.8068
  └─ Episode 880: Reward=+1.7674 | Portfolio=$94,326 | Trades=96 | ε=0.012
Step 480,000/500,000 | Ep 882 | ε=0.012 | Avg Reward: +1.8147
  └─ Episode 900: Reward=+1.8539 | Portfolio=$103,466 | Trades=98 | ε=0.011
Step 490,000/500,000 | Ep 900 | ε=0.011 | Avg Reward: +1.8106
Step 500,000/500,000 | Ep 919 | ε=0.010 | Avg Reward: +1.8041
  └─ Episode 920: Reward=+0.2592 | Portfolio=$13,090 | Trades=6 | ε=0.010

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 920
Final Epsilon: 0.0100
Mean Reward: 1.1439
Best Episode Reward: 2.0157
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251212_231302.pkl
Config saved to results/config_v2_20251212_231302.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $101,425.46
Return:  +914.25%
Trades:  96
Fees:    $8249.34

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $12,385.20
Return:  +23.85%
Trades:  24
Fees:    $470.67

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |     +914.25% |      +23.85%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +772.17% |      -33.23%

 Final Values:
   Agent Train: $101,425.46  |  Buy&Hold: $24,208.72
   Agent Test:  $12,385.20  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251212_231302.pkl
Final Epsilon: 0.0100
Total Episodes: 920

 Agent made profit, but didn't beat Buy & Hold
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading 
bot\Reinforcement-Learning-Trading-Bot-RLTrader-> 

v15
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.15,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 500000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True       
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.995 (reaches 0.01 after ~918 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 500,000
Epsilon: 1.00  0.01
Decay rate: 0.995
============================================================

Step  10,000/500,000 | Ep  18 | ε=0.914 | Avg Reward: -1.0644
  └─ Episode 20: Reward=-0.9286 | Portfolio=$12,184 | Trades=188 | ε=0.905
Step  20,000/500,000 | Ep  36 | ε=0.835 | Avg Reward: -0.9138
  └─ Episode 40: Reward=-0.5217 | Portfolio=$17,403 | Trades=174 | ε=0.818
Step  30,000/500,000 | Ep  55 | ε=0.759 | Avg Reward: -0.5789
  └─ Episode 60: Reward=-0.6916 | Portfolio=$16,035 | Trades=191 | ε=0.740
Step  40,000/500,000 | Ep  73 | ε=0.694 | Avg Reward: -0.5452
  └─ Episode 80: Reward=-0.4486 | Portfolio=$17,208 | Trades=170 | ε=0.670
Step  50,000/500,000 | Ep  91 | ε=0.634 | Avg Reward: -0.3211
  └─ Episode 100: Reward=-0.0885 | Portfolio=$27,585 | Trades=178 | ε=0.606
Step  60,000/500,000 | Ep 110 | ε=0.576 | Avg Reward: -0.0898
  └─ Episode 120: Reward=+0.3588 | Portfolio=$32,465 | Trades=136 | ε=0.548
Step  70,000/500,000 | Ep 128 | ε=0.526 | Avg Reward: -0.0351
  └─ Episode 140: Reward=+0.1898 | Portfolio=$29,680 | Trades=158 | ε=0.496
Step  80,000/500,000 | Ep 147 | ε=0.479 | Avg Reward: +0.0138
  └─ Episode 160: Reward=+0.9549 | Portfolio=$56,898 | Trades=139 | ε=0.448
Step  90,000/500,000 | Ep 165 | ε=0.437 | Avg Reward: +0.3750
  └─ Episode 180: Reward=+0.7918 | Portfolio=$56,024 | Trades=157 | ε=0.406
Step 100,000/500,000 | Ep 183 | ε=0.400 | Avg Reward: +0.5088
  └─ Episode 200: Reward=+0.5211 | Portfolio=$39,993 | Trades=163 | ε=0.367
Step 110,000/500,000 | Ep 202 | ε=0.363 | Avg Reward: +0.3677
  └─ Episode 220: Reward=+0.1875 | Portfolio=$26,232 | Trades=144 | ε=0.332
Step 120,000/500,000 | Ep 220 | ε=0.332 | Avg Reward: +0.5454
Step 130,000/500,000 | Ep 238 | ε=0.303 | Avg Reward: +0.9272
  └─ Episode 240: Reward=+0.9927 | Portfolio=$55,310 | Trades=130 | ε=0.300
Step 140,000/500,000 | Ep 257 | ε=0.276 | Avg Reward: +0.8491
  └─ Episode 260: Reward=+0.7368 | Portfolio=$42,272 | Trades=126 | ε=0.272
Step 150,000/500,000 | Ep 275 | ε=0.252 | Avg Reward: +0.8836
  └─ Episode 280: Reward=+0.6884 | Portfolio=$41,535 | Trades=124 | ε=0.246
Step 160,000/500,000 | Ep 294 | ε=0.229 | Avg Reward: +0.9335
  └─ Episode 300: Reward=+1.0084 | Portfolio=$56,305 | Trades=130 | ε=0.222
Step 170,000/500,000 | Ep 312 | ε=0.209 | Avg Reward: +1.1277
  └─ Episode 320: Reward=+1.1580 | Portfolio=$60,132 | Trades=123 | ε=0.201
Step 180,000/500,000 | Ep 330 | ε=0.191 | Avg Reward: +1.0217
  └─ Episode 340: Reward=+1.1484 | Portfolio=$57,405 | Trades=114 | ε=0.182
Step 190,000/500,000 | Ep 349 | ε=0.174 | Avg Reward: +1.1501
  └─ Episode 360: Reward=+1.2051 | Portfolio=$63,863 | Trades=118 | ε=0.165
Step 200,000/500,000 | Ep 367 | ε=0.159 | Avg Reward: +1.1319
  └─ Episode 380: Reward=+1.0284 | Portfolio=$57,745 | Trades=124 | ε=0.149
Step 210,000/500,000 | Ep 386 | ε=0.144 | Avg Reward: +1.1804
  └─ Episode 400: Reward=+1.1794 | Portfolio=$66,283 | Trades=128 | ε=0.135
Step 220,000/500,000 | Ep 404 | ε=0.132 | Avg Reward: +1.3087
  └─ Episode 420: Reward=+1.3743 | Portfolio=$69,527 | Trades=108 | ε=0.122
Step 230,000/500,000 | Ep 422 | ε=0.121 | Avg Reward: +1.3278
  └─ Episode 440: Reward=+1.0875 | Portfolio=$55,895 | Trades=116 | ε=0.110
Step 240,000/500,000 | Ep 441 | ε=0.110 | Avg Reward: +1.3406
Step 250,000/500,000 | Ep 459 | ε=0.100 | Avg Reward: +1.4673
  └─ Episode 460: Reward=+1.2781 | Portfolio=$65,749 | Trades=118 | ε=0.100
Step 260,000/500,000 | Ep 477 | ε=0.092 | Avg Reward: +1.3872
  └─ Episode 480: Reward=+1.6086 | Portfolio=$81,947 | Trades=98 | ε=0.090
Step 270,000/500,000 | Ep 496 | ε=0.083 | Avg Reward: +1.4336
  └─ Episode 500: Reward=+1.5387 | Portfolio=$77,027 | Trades=96 | ε=0.082
Step 280,000/500,000 | Ep 514 | ε=0.076 | Avg Reward: +1.5112
  └─ Episode 520: Reward=+1.3847 | Portfolio=$68,149 | Trades=104 | ε=0.074
Step 290,000/500,000 | Ep 533 | ε=0.069 | Avg Reward: +1.5416
  └─ Episode 540: Reward=+1.1745 | Portfolio=$56,063 | Trades=110 | ε=0.067
Step 300,000/500,000 | Ep 551 | ε=0.063 | Avg Reward: +1.5750
  └─ Episode 560: Reward=+1.7103 | Portfolio=$90,329 | Trades=98 | ε=0.060
Step 310,000/500,000 | Ep 569 | ε=0.058 | Avg Reward: +1.5725
  └─ Episode 580: Reward=+1.5924 | Portfolio=$81,804 | Trades=102 | ε=0.055
Step 320,000/500,000 | Ep 588 | ε=0.052 | Avg Reward: +1.6365
  └─ Episode 600: Reward=+1.3716 | Portfolio=$69,224 | Trades=108 | ε=0.049
Step 330,000/500,000 | Ep 606 | ε=0.048 | Avg Reward: +1.5985
  └─ Episode 620: Reward=+1.6331 | Portfolio=$82,204 | Trades=94 | ε=0.045
Step 340,000/500,000 | Ep 624 | ε=0.044 | Avg Reward: +1.6787
  └─ Episode 640: Reward=+1.5738 | Portfolio=$79,268 | Trades=100 | ε=0.040
Step 350,000/500,000 | Ep 643 | ε=0.040 | Avg Reward: +1.6785
  └─ Episode 660: Reward=+1.6001 | Portfolio=$86,349 | Trades=112 | ε=0.037
Step 360,000/500,000 | Ep 661 | ε=0.036 | Avg Reward: +1.6400
  └─ Episode 680: Reward=+1.7792 | Portfolio=$93,732 | Trades=90 | ε=0.033
Step 370,000/500,000 | Ep 680 | ε=0.033 | Avg Reward: +1.6734
Step 380,000/500,000 | Ep 698 | ε=0.030 | Avg Reward: +1.7042
  └─ Episode 700: Reward=+1.6041 | Portfolio=$82,628 | Trades=100 | ε=0.030
Step 390,000/500,000 | Ep 716 | ε=0.028 | Avg Reward: +1.7058
  └─ Episode 720: Reward=+1.7063 | Portfolio=$89,327 | Trades=96 | ε=0.027
Step 400,000/500,000 | Ep 735 | ε=0.025 | Avg Reward: +1.7011
  └─ Episode 740: Reward=+1.8869 | Portfolio=$103,190 | Trades=90 | ε=0.024
Step 410,000/500,000 | Ep 753 | ε=0.023 | Avg Reward: +1.7211
  └─ Episode 760: Reward=+1.7107 | Portfolio=$88,318 | Trades=94 | ε=0.022
Step 420,000/500,000 | Ep 772 | ε=0.021 | Avg Reward: +1.7510
  └─ Episode 780: Reward=+1.5657 | Portfolio=$78,317 | Trades=96 | ε=0.020
Step 430,000/500,000 | Ep 790 | ε=0.019 | Avg Reward: +1.7138
  └─ Episode 800: Reward=+1.6314 | Portfolio=$82,689 | Trades=96 | ε=0.018
Step 440,000/500,000 | Ep 808 | ε=0.017 | Avg Reward: +1.7481
  └─ Episode 820: Reward=+1.7655 | Portfolio=$91,579 | Trades=92 | ε=0.016
Step 450,000/500,000 | Ep 827 | ε=0.016 | Avg Reward: +1.7530
  └─ Episode 840: Reward=+1.7846 | Portfolio=$91,871 | Trades=88 | ε=0.015
Step 460,000/500,000 | Ep 845 | ε=0.014 | Avg Reward: +1.7753
  └─ Episode 860: Reward=+1.7614 | Portfolio=$93,469 | Trades=94 | ε=0.013
Step 470,000/500,000 | Ep 863 | ε=0.013 | Avg Reward: +1.7668
  └─ Episode 880: Reward=+1.7846 | Portfolio=$94,252 | Trades=92 | ε=0.012
Step 480,000/500,000 | Ep 882 | ε=0.012 | Avg Reward: +1.7679
  └─ Episode 900: Reward=+1.8458 | Portfolio=$99,298 | Trades=90 | ε=0.011
Step 490,000/500,000 | Ep 900 | ε=0.011 | Avg Reward: +1.7629
Step 500,000/500,000 | Ep 919 | ε=0.010 | Avg Reward: +1.7854
  └─ Episode 920: Reward=+0.2330 | Portfolio=$13,080 | Trades=10 | ε=0.010

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 920
Final Epsilon: 0.0100
Mean Reward: 1.0821
Best Episode Reward: 1.8869
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251212_231604.pkl
Config saved to results/config_v2_20251212_231604.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $95,467.95
Return:  +854.68%
Trades:  90
Fees:    $7765.99

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $11,014.57
Return:  +10.15%
Trades:  30
Fees:    $578.55

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |     +854.68% |      +10.15%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +712.59% |      -46.93%

 Final Values:
   Agent Train: $95,467.95  |  Buy&Hold: $24,208.72
   Agent Test:  $11,014.57  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251212_231604.pkl
Final Epsilon: 0.0100
Total Episodes: 920

 Agent made profit, but didn't beat Buy & Hold
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-> 

v15
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 500000,  # MEHR! (war 50000)
            'log_interval': 10000
        }

======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True       
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.995 (reaches 0.01 after ~918 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 500,000
Epsilon: 1.00  0.01
Decay rate: 0.995
============================================================

Step  10,000/500,000 | Ep  18 | ε=0.914 | Avg Reward: -1.2329
  └─ Episode 20: Reward=-0.7236 | Portfolio=$15,678 | Trades=198 | ε=0.905
Step  20,000/500,000 | Ep  36 | ε=0.835 | Avg Reward: -0.7966
  └─ Episode 40: Reward=-0.6712 | Portfolio=$15,060 | Trades=172 | ε=0.818
Step  30,000/500,000 | Ep  55 | ε=0.759 | Avg Reward: -0.4271
  └─ Episode 60: Reward=-0.3373 | Portfolio=$19,050 | Trades=168 | ε=0.740
Step  40,000/500,000 | Ep  73 | ε=0.694 | Avg Reward: -0.4410
  └─ Episode 80: Reward=-0.0865 | Portfolio=$23,930 | Trades=157 | ε=0.670
Step  50,000/500,000 | Ep  91 | ε=0.634 | Avg Reward: -0.2022
  └─ Episode 100: Reward=+0.2022 | Portfolio=$33,430 | Trades=169 | ε=0.606
Step  60,000/500,000 | Ep 110 | ε=0.576 | Avg Reward: +0.0972
  └─ Episode 120: Reward=-0.4710 | Portfolio=$16,611 | Trades=166 | ε=0.548
Step  70,000/500,000 | Ep 128 | ε=0.526 | Avg Reward: -0.0713
  └─ Episode 140: Reward=+0.5651 | Portfolio=$41,894 | Trades=154 | ε=0.496
Step  80,000/500,000 | Ep 147 | ε=0.479 | Avg Reward: +0.4412
  └─ Episode 160: Reward=+0.6171 | Portfolio=$41,496 | Trades=152 | ε=0.448
Step  90,000/500,000 | Ep 165 | ε=0.437 | Avg Reward: +0.3853
  └─ Episode 180: Reward=+0.6893 | Portfolio=$44,851 | Trades=141 | ε=0.406
Step 100,000/500,000 | Ep 183 | ε=0.400 | Avg Reward: +0.5531
  └─ Episode 200: Reward=+1.0145 | Portfolio=$62,808 | Trades=144 | ε=0.367
Step 110,000/500,000 | Ep 202 | ε=0.363 | Avg Reward: +0.7688
  └─ Episode 220: Reward=+0.6634 | Portfolio=$44,042 | Trades=148 | ε=0.332
Step 120,000/500,000 | Ep 220 | ε=0.332 | Avg Reward: +0.8464
Step 130,000/500,000 | Ep 238 | ε=0.303 | Avg Reward: +0.8420
  └─ Episode 240: Reward=+0.7284 | Portfolio=$46,989 | Trades=146 | ε=0.300
Step 140,000/500,000 | Ep 257 | ε=0.276 | Avg Reward: +0.8884
  └─ Episode 260: Reward=+1.1063 | Portfolio=$67,376 | Trades=150 | ε=0.272
Step 150,000/500,000 | Ep 275 | ε=0.252 | Avg Reward: +0.8943
  └─ Episode 280: Reward=+0.8514 | Portfolio=$53,777 | Trades=142 | ε=0.246
Step 160,000/500,000 | Ep 294 | ε=0.229 | Avg Reward: +1.0859
  └─ Episode 300: Reward=+1.3774 | Portfolio=$80,093 | Trades=128 | ε=0.222
Step 170,000/500,000 | Ep 312 | ε=0.209 | Avg Reward: +1.1506
  └─ Episode 320: Reward=+1.4051 | Portfolio=$82,750 | Trades=124 | ε=0.201
Step 180,000/500,000 | Ep 330 | ε=0.191 | Avg Reward: +1.1818
  └─ Episode 340: Reward=+1.2701 | Portfolio=$70,642 | Trades=128 | ε=0.182
Step 190,000/500,000 | Ep 349 | ε=0.174 | Avg Reward: +1.3191
  └─ Episode 360: Reward=+1.1058 | Portfolio=$61,343 | Trades=128 | ε=0.165
Step 200,000/500,000 | Ep 367 | ε=0.159 | Avg Reward: +1.2654
  └─ Episode 380: Reward=+1.3430 | Portfolio=$72,629 | Trades=122 | ε=0.149
Step 210,000/500,000 | Ep 386 | ε=0.144 | Avg Reward: +1.3575
  └─ Episode 400: Reward=+1.0918 | Portfolio=$58,678 | Trades=128 | ε=0.135
Step 220,000/500,000 | Ep 404 | ε=0.132 | Avg Reward: +1.3009
  └─ Episode 420: Reward=+1.4313 | Portfolio=$73,305 | Trades=111 | ε=0.122
Step 230,000/500,000 | Ep 422 | ε=0.121 | Avg Reward: +1.4405
  └─ Episode 440: Reward=+1.3782 | Portfolio=$72,587 | Trades=118 | ε=0.110
Step 240,000/500,000 | Ep 441 | ε=0.110 | Avg Reward: +1.5282
Step 250,000/500,000 | Ep 459 | ε=0.100 | Avg Reward: +1.5591
  └─ Episode 460: Reward=+1.4528 | Portfolio=$81,705 | Trades=130 | ε=0.100
Step 260,000/500,000 | Ep 477 | ε=0.092 | Avg Reward: +1.5087
  └─ Episode 480: Reward=+1.7424 | Portfolio=$100,812 | Trades=114 | ε=0.090
Step 270,000/500,000 | Ep 496 | ε=0.083 | Avg Reward: +1.6435
  └─ Episode 500: Reward=+1.6173 | Portfolio=$86,266 | Trades=108 | ε=0.082
Step 280,000/500,000 | Ep 514 | ε=0.076 | Avg Reward: +1.6078
  └─ Episode 520: Reward=+1.5565 | Portfolio=$82,824 | Trades=114 | ε=0.074
Step 290,000/500,000 | Ep 533 | ε=0.069 | Avg Reward: +1.6662
  └─ Episode 540: Reward=+1.6013 | Portfolio=$89,398 | Trades=116 | ε=0.067
Step 300,000/500,000 | Ep 551 | ε=0.063 | Avg Reward: +1.5941
  └─ Episode 560: Reward=+1.6706 | Portfolio=$92,733 | Trades=114 | ε=0.060
Step 310,000/500,000 | Ep 569 | ε=0.058 | Avg Reward: +1.6388
  └─ Episode 580: Reward=+1.7413 | Portfolio=$93,046 | Trades=106 | ε=0.055
Step 320,000/500,000 | Ep 588 | ε=0.052 | Avg Reward: +1.7597
  └─ Episode 600: Reward=+1.5915 | Portfolio=$86,285 | Trades=114 | ε=0.049
Step 330,000/500,000 | Ep 606 | ε=0.048 | Avg Reward: +1.6827
  └─ Episode 620: Reward=+1.6959 | Portfolio=$91,755 | Trades=110 | ε=0.045
Step 340,000/500,000 | Ep 624 | ε=0.044 | Avg Reward: +1.7282
  └─ Episode 640: Reward=+1.7242 | Portfolio=$93,930 | Trades=106 | ε=0.040
Step 350,000/500,000 | Ep 643 | ε=0.040 | Avg Reward: +1.7152
  └─ Episode 660: Reward=+1.8420 | Portfolio=$101,214 | Trades=98 | ε=0.037
Step 360,000/500,000 | Ep 661 | ε=0.036 | Avg Reward: +1.7774
  └─ Episode 680: Reward=+1.8667 | Portfolio=$107,600 | Trades=106 | ε=0.033
Step 370,000/500,000 | Ep 680 | ε=0.033 | Avg Reward: +1.8114
Step 380,000/500,000 | Ep 698 | ε=0.030 | Avg Reward: +1.7738
  └─ Episode 700: Reward=+1.7492 | Portfolio=$94,127 | Trades=104 | ε=0.030
Step 390,000/500,000 | Ep 716 | ε=0.028 | Avg Reward: +1.7488
  └─ Episode 720: Reward=+1.7420 | Portfolio=$96,108 | Trades=110 | ε=0.027
Step 400,000/500,000 | Ep 735 | ε=0.025 | Avg Reward: +1.7968
  └─ Episode 740: Reward=+1.6672 | Portfolio=$88,805 | Trades=106 | ε=0.024
Step 410,000/500,000 | Ep 753 | ε=0.023 | Avg Reward: +1.7855
  └─ Episode 760: Reward=+1.9432 | Portfolio=$114,154 | Trades=102 | ε=0.022
Step 420,000/500,000 | Ep 772 | ε=0.021 | Avg Reward: +1.8208
  └─ Episode 780: Reward=+1.7353 | Portfolio=$94,894 | Trades=108 | ε=0.020
Step 430,000/500,000 | Ep 790 | ε=0.019 | Avg Reward: +1.7815
  └─ Episode 800: Reward=+1.8687 | Portfolio=$104,458 | Trades=102 | ε=0.018
Step 440,000/500,000 | Ep 808 | ε=0.017 | Avg Reward: +1.8389
  └─ Episode 820: Reward=+1.8783 | Portfolio=$105,992 | Trades=102 | ε=0.016
Step 450,000/500,000 | Ep 827 | ε=0.016 | Avg Reward: +1.8511
  └─ Episode 840: Reward=+1.8618 | Portfolio=$103,724 | Trades=102 | ε=0.015
Step 460,000/500,000 | Ep 845 | ε=0.014 | Avg Reward: +1.8529
  └─ Episode 860: Reward=+1.8400 | Portfolio=$104,070 | Trades=106 | ε=0.013
Step 470,000/500,000 | Ep 863 | ε=0.013 | Avg Reward: +1.8509
  └─ Episode 880: Reward=+1.8509 | Portfolio=$103,670 | Trades=104 | ε=0.012
Step 480,000/500,000 | Ep 882 | ε=0.012 | Avg Reward: +1.8562
  └─ Episode 900: Reward=+1.8586 | Portfolio=$101,651 | Trades=100 | ε=0.011
Step 490,000/500,000 | Ep 900 | ε=0.011 | Avg Reward: +1.8713
Step 500,000/500,000 | Ep 919 | ε=0.010 | Avg Reward: +1.8748
  └─ Episode 920: Reward=+0.2690 | Portfolio=$13,349 | Trades=6 | ε=0.010

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 920
Final Epsilon: 0.0100
Mean Reward: 1.1839
Best Episode Reward: 1.9761
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251212_231935.pkl
Config saved to results/config_v2_20251212_231935.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $106,610.40
Return:  +966.10%
Trades:  100
Fees:    $9398.13

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $11,432.10
Return:  +14.32%
Trades:  28
Fees:    $556.27

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |     +966.10% |      +14.32%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +824.02% |      -42.76%

 Final Values:
   Agent Train: $106,610.40  |  Buy&Hold: $24,208.72
   Agent Test:  $11,432.10  |  Buy&Hold: $15,707.94

v16
# ════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 200000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-01-01...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True       
  data = yf.download(
Loaded 731 rows of data
Date range: 2023-01-01 00:00:00 to 2024-12-31 00:00:00
Price range: $16625.08 - $106140.60
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 682
Train set: 545 rows (80%)
Test set: 137 rows (20%)

 Original Train Prices: $20187.24 - $73083.50
 Original Test Prices: $53948.75 - $106140.60

Normalizing features for neural network...
Features normalized ✓

 Train data: 545 days
 Test data: 137 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $73083.50
   Price change over period: 142.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 200,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/200,000 | Ep  18 | ε=0.835 | Avg Reward: -0.8332
  └─ Episode 20: Reward=-0.7032 | Portfolio=$13,560 | Trades=164 | ε=0.818
Step  20,000/200,000 | Ep  36 | ε=0.696 | Avg Reward: -0.4061
  └─ Episode 40: Reward=-0.5810 | Portfolio=$13,963 | Trades=157 | ε=0.669
Step  30,000/200,000 | Ep  55 | ε=0.575 | Avg Reward: -0.0607
  └─ Episode 60: Reward=+0.0140 | Portfolio=$27,079 | Trades=172 | ε=0.547
Step  40,000/200,000 | Ep  73 | ε=0.480 | Avg Reward: +0.3057
  └─ Episode 80: Reward=+0.5469 | Portfolio=$39,762 | Trades=149 | ε=0.448
Step  50,000/200,000 | Ep  91 | ε=0.401 | Avg Reward: +0.3988
  └─ Episode 100: Reward=+0.4553 | Portfolio=$37,377 | Trades=160 | ε=0.366
Step  60,000/200,000 | Ep 110 | ε=0.331 | Avg Reward: +0.5649
  └─ Episode 120: Reward=+1.0568 | Portfolio=$62,264 | Trades=136 | ε=0.299
Step  70,000/200,000 | Ep 128 | ε=0.276 | Avg Reward: +1.0419
  └─ Episode 140: Reward=+1.1048 | Portfolio=$57,816 | Trades=130 | ε=0.245
Step  80,000/200,000 | Ep 147 | ε=0.228 | Avg Reward: +1.0355
  └─ Episode 160: Reward=+1.4030 | Portfolio=$71,787 | Trades=112 | ε=0.200
Step  90,000/200,000 | Ep 165 | ε=0.190 | Avg Reward: +1.2090
  └─ Episode 180: Reward=+1.4377 | Portfolio=$78,636 | Trades=122 | ε=0.164
Step 100,000/200,000 | Ep 183 | ε=0.159 | Avg Reward: +1.3976
  └─ Episode 200: Reward=+1.7747 | Portfolio=$99,566 | Trades=106 | ε=0.134
Step 110,000/200,000 | Ep 202 | ε=0.131 | Avg Reward: +1.5361
  └─ Episode 220: Reward=+1.6564 | Portfolio=$86,073 | Trades=106 | ε=0.110
Step 120,000/200,000 | Ep 220 | ε=0.110 | Avg Reward: +1.5685
Step 130,000/200,000 | Ep 238 | ε=0.091 | Avg Reward: +1.6424
  └─ Episode 240: Reward=+1.8117 | Portfolio=$95,370 | Trades=94 | ε=0.090
Step 140,000/200,000 | Ep 257 | ε=0.076 | Avg Reward: +1.6547
  └─ Episode 260: Reward=+1.6547 | Portfolio=$87,290 | Trades=108 | ε=0.073
Step 150,000/200,000 | Ep 275 | ε=0.063 | Avg Reward: +1.7249
  └─ Episode 280: Reward=+1.6465 | Portfolio=$83,748 | Trades=100 | ε=0.060
Step 160,000/200,000 | Ep 294 | ε=0.052 | Avg Reward: +1.8381
  └─ Episode 300: Reward=+1.7137 | Portfolio=$89,934 | Trades=104 | ε=0.049
Step 170,000/200,000 | Ep 312 | ε=0.043 | Avg Reward: +1.8477
  └─ Episode 320: Reward=+1.9790 | Portfolio=$109,473 | Trades=92 | ε=0.040
Step 180,000/200,000 | Ep 330 | ε=0.036 | Avg Reward: +1.8134
  └─ Episode 340: Reward=+1.9750 | Portfolio=$112,233 | Trades=96 | ε=0.033
Step 190,000/200,000 | Ep 349 | ε=0.030 | Avg Reward: +1.9284
  └─ Episode 360: Reward=+1.9846 | Portfolio=$111,458 | Trades=90 | ε=0.027
Step 200,000/200,000 | Ep 367 | ε=0.025 | Avg Reward: +1.9244

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 368
Final Epsilon: 0.0248
Mean Reward: 1.0856
Best Episode Reward: 2.0557
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251212_232513.pkl
Config saved to results/config_v2_20251212_232513.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $113,063.57
Return:  +1030.64%
Trades:  90
Fees:    $8978.37

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $53948.75 - $106140.60
   Price change over period: 57.1%
   Using 21 features for state
Initial: $10,000.00
Final:   $11,996.80
Return:  +19.97%
Trades:  28
Fees:    $586.59

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |    +1030.64% |      +19.97%
Buy & Hold           |     +142.09% |      +57.08%
--------------------------------------------------
Outperformance       |     +888.55% |      -37.11%

 Final Values:
   Agent Train: $113,063.57  |  Buy&Hold: $24,208.72
   Agent Test:  $11,996.80  |  Buy&Hold: $15,707.94

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251212_232513.pkl
Final Epsilon: 0.0248
Total Episodes: 368

v18
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-01-01',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.996,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

v19
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-12-12',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.996,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }

======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-12-12...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 1076 rows of data
Date range: 2023-01-01 00:00:00 to 2025-12-11 00:00:00
Price range: $16625.08 - $124752.53
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 1027
Train set: 821 rows (80%)
Test set: 206 rows (20%)

 Original Train Prices: $20187.24 - $106446.01
 Original Test Prices: $84648.36 - $124752.53

Normalizing features for neural network...
Features normalized ✓

 Train data: 821 days
 Test data: 206 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $106446.01
   Price change over period: 334.1%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.996 (reaches 0.01 after ~1148 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 300,000
Epsilon: 1.00  0.01
Decay rate: 0.996
============================================================

Step  10,000/300,000 | Ep  12 | ε=0.953 | Avg Reward: -1.5521
  └─ Episode 20: Reward=-0.7169 | Portfolio=$21,863 | Trades=254 | ε=0.923
Step  20,000/300,000 | Ep  24 | ε=0.908 | Avg Reward: -1.2333
Step  30,000/300,000 | Ep  36 | ε=0.866 | Avg Reward: -1.3130
  └─ Episode 40: Reward=-1.5215 | Portfolio=$12,125 | Trades=272 | ε=0.852
Step  40,000/300,000 | Ep  48 | ε=0.825 | Avg Reward: -1.0280
  └─ Episode 60: Reward=-1.0674 | Portfolio=$16,046 | Trades=274 | ε=0.786
Step  50,000/300,000 | Ep  60 | ε=0.786 | Avg Reward: -1.0427
Step  60,000/300,000 | Ep  73 | ε=0.746 | Avg Reward: -0.7494
  └─ Episode 80: Reward=-0.8081 | Portfolio=$24,180 | Trades=284 | ε=0.726
Step  70,000/300,000 | Ep  85 | ε=0.711 | Avg Reward: -0.6163
Step  80,000/300,000 | Ep  97 | ε=0.678 | Avg Reward: -0.5488
  └─ Episode 100: Reward=+0.1200 | Portfolio=$47,636 | Trades=259 | ε=0.670
Step  90,000/300,000 | Ep 109 | ε=0.646 | Avg Reward: -0.1955
  └─ Episode 120: Reward=+0.2274 | Portfolio=$44,459 | Trades=218 | ε=0.618
Step 100,000/300,000 | Ep 121 | ε=0.616 | Avg Reward: -0.0529
Step 110,000/300,000 | Ep 134 | ε=0.584 | Avg Reward: +0.0516
  └─ Episode 140: Reward=+0.6563 | Portfolio=$69,831 | Trades=219 | ε=0.571
Step 120,000/300,000 | Ep 146 | ε=0.557 | Avg Reward: -0.0470
Step 130,000/300,000 | Ep 158 | ε=0.531 | Avg Reward: +0.2725
  └─ Episode 160: Reward=-0.4698 | Portfolio=$24,925 | Trades=245 | ε=0.527
Step 140,000/300,000 | Ep 170 | ε=0.506 | Avg Reward: +0.1100
  └─ Episode 180: Reward=+1.2082 | Portfolio=$124,634 | Trades=233 | ε=0.486
Step 150,000/300,000 | Ep 182 | ε=0.482 | Avg Reward: +0.5001
Step 160,000/300,000 | Ep 195 | ε=0.458 | Avg Reward: +0.5396
  └─ Episode 200: Reward=+0.5431 | Portfolio=$52,123 | Trades=200 | ε=0.449
Step 170,000/300,000 | Ep 207 | ε=0.436 | Avg Reward: +0.6553
Step 180,000/300,000 | Ep 219 | ε=0.416 | Avg Reward: +0.7354
  └─ Episode 220: Reward=+1.1830 | Portfolio=$119,181 | Trades=221 | ε=0.414
Step 190,000/300,000 | Ep 231 | ε=0.396 | Avg Reward: +0.7328
  └─ Episode 240: Reward=+0.9744 | Portfolio=$91,052 | Trades=219 | ε=0.382
Step 200,000/300,000 | Ep 243 | ε=0.378 | Avg Reward: +1.0254
Step 210,000/300,000 | Ep 256 | ε=0.358 | Avg Reward: +1.0045
  └─ Episode 260: Reward=+0.8179 | Portfolio=$77,244 | Trades=217 | ε=0.353
Step 220,000/300,000 | Ep 268 | ε=0.342 | Avg Reward: +1.1900
  └─ Episode 280: Reward=+0.7134 | Portfolio=$74,454 | Trades=226 | ε=0.326
Step 230,000/300,000 | Ep 280 | ε=0.326 | Avg Reward: +1.1423
Step 240,000/300,000 | Ep 292 | ε=0.310 | Avg Reward: +1.1998
  └─ Episode 300: Reward=+1.3013 | Portfolio=$139,613 | Trades=239 | ε=0.300
Step 250,000/300,000 | Ep 304 | ε=0.296 | Avg Reward: +1.2577
Step 260,000/300,000 | Ep 317 | ε=0.281 | Avg Reward: +1.2911
  └─ Episode 320: Reward=+1.0057 | Portfolio=$92,877 | Trades=219 | ε=0.277
Step 270,000/300,000 | Ep 329 | ε=0.267 | Avg Reward: +1.4571
  └─ Episode 340: Reward=+1.3646 | Portfolio=$132,685 | Trades=225 | ε=0.256
Step 280,000/300,000 | Ep 341 | ε=0.255 | Avg Reward: +1.4375
Step 290,000/300,000 | Ep 353 | ε=0.243 | Avg Reward: +1.6402
  └─ Episode 360: Reward=+1.9877 | Portfolio=$252,536 | Trades=220 | ε=0.236
Step 300,000/300,000 | Ep 365 | ε=0.232 | Avg Reward: +1.5858

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 366
Final Epsilon: 0.2306
Mean Reward: 0.3200
Best Episode Reward: 2.1691
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251212_235141.pkl
Config saved to results/config_v2_20251212_235141.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $501,984.07
Return:  +4919.84%
Trades:  195
Fees:    $49913.32

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $84648.36 - $124752.53
   Price change over period: -13.4%
   Using 21 features for state
Initial: $10,000.00
Final:   $6,283.67
Return:  -37.16%
Trades:  47
Fees:    $789.68

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |    +4919.84% |      -37.16%
Buy & Hold           |     +334.10% |      -13.37%
--------------------------------------------------
Outperformance       |    +4585.74% |      -23.79%

 Final Values:
   Agent Train: $501,984.07  |  Buy&Hold: $43,409.95
   Agent Test:  $6,283.67  |  Buy&Hold: $8,662.83

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251212_235141.pkl
Final Epsilon: 0.2306
Total Episodes: 366

v20
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2020-01-01',
            'end_date': '2025-11-11',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.996,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |    +2215.02% |      +10.69%
Buy & Hold           |     +286.16% |      +11.90%
--------------------------------------------------
Outperformance       |    +1928.86% |       -1.22%

 Final Values:
   Agent Train: $231,502.12  |  Buy&Hold: $38,616.07
   Agent Test:  $11,068.93  |  Buy&Hold: $11,190.46

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251212_235503.pkl
Final Epsilon: 0.2198
Total Episodes: 378

 Agent made profit, but didn't beat Buy & Hold
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-> 

v21
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2020-01-01',
            'end_date': '2025-11-11',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.996,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 300000,  # MEHR! (war 50000)
            'log_interval': 10000

v22
# CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2023-01-01',
            'end_date': '2025-11-11',
            'interval': '1d',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.005  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.999,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 500000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2023-01-01 to 2025-11-11...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 1045 rows of data
Date range: 2023-01-01 00:00:00 to 2025-11-10 00:00:00
Price range: $16625.08 - $124752.53
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 996
Train set: 796 rows (80%)
Test set: 200 rows (20%)

 Original Train Prices: $20187.24 - $106146.27
 Original Test Prices: $93754.84 - $124752.53

Normalizing features for neural network...
Features normalized ✓

 Train data: 796 days
 Test data: 200 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $20187.24 - $106146.27
   Price change over period: 286.2%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.995 (reaches 0.01 after ~918 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 500,000
Epsilon: 1.00  0.01
Decay rate: 0.995
============================================================

Step  10,000/500,000 | Ep  12 | ε=0.942 | Avg Reward: -1.5446
  └─ Episode 20: Reward=-1.4395 | Portfolio=$11,855 | Trades=264 | ε=0.905
Step  20,000/500,000 | Ep  25 | ε=0.882 | Avg Reward: -1.5957
Step  30,000/500,000 | Ep  37 | ε=0.831 | Avg Reward: -1.3343
  └─ Episode 40: Reward=-0.3607 | Portfolio=$27,549 | Trades=238 | ε=0.818
Step  40,000/500,000 | Ep  50 | ε=0.778 | Avg Reward: -1.0191
  └─ Episode 60: Reward=-1.0617 | Portfolio=$14,032 | Trades=225 | ε=0.740
Step  50,000/500,000 | Ep  62 | ε=0.733 | Avg Reward: -0.4166
Step  60,000/500,000 | Ep  75 | ε=0.687 | Avg Reward: -0.4715
  └─ Episode 80: Reward=-0.3378 | Portfolio=$26,388 | Trades=224 | ε=0.670
Step  70,000/500,000 | Ep  88 | ε=0.643 | Avg Reward: -0.3098
  └─ Episode 100: Reward=+0.1063 | Portfolio=$41,422 | Trades=217 | ε=0.606
Step  80,000/500,000 | Ep 100 | ε=0.606 | Avg Reward: -0.1444
Step  90,000/500,000 | Ep 113 | ε=0.568 | Avg Reward: +0.0583
  └─ Episode 120: Reward=+0.1524 | Portfolio=$39,063 | Trades=217 | ε=0.548
Step 100,000/500,000 | Ep 125 | ε=0.534 | Avg Reward: +0.2037
Step 110,000/500,000 | Ep 138 | ε=0.501 | Avg Reward: +0.3566
  └─ Episode 140: Reward=+0.7463 | Portfolio=$69,495 | Trades=215 | ε=0.496
Step 120,000/500,000 | Ep 150 | ε=0.471 | Avg Reward: +0.3638
  └─ Episode 160: Reward=-0.1627 | Portfolio=$24,040 | Trades=191 | ε=0.448
Step 130,000/500,000 | Ep 163 | ε=0.442 | Avg Reward: +0.4339
Step 140,000/500,000 | Ep 176 | ε=0.414 | Avg Reward: +0.6633
  └─ Episode 180: Reward=+1.0371 | Portfolio=$80,338 | Trades=196 | ε=0.406
Step 150,000/500,000 | Ep 188 | ε=0.390 | Avg Reward: +0.8621
  └─ Episode 200: Reward=+1.1761 | Portfolio=$104,226 | Trades=217 | ε=0.367
Step 160,000/500,000 | Ep 201 | ε=0.365 | Avg Reward: +0.9064
Step 170,000/500,000 | Ep 213 | ε=0.344 | Avg Reward: +0.8154
  └─ Episode 220: Reward=+1.2570 | Portfolio=$77,344 | Trades=157 | ε=0.332
Step 180,000/500,000 | Ep 226 | ε=0.322 | Avg Reward: +1.0462
Step 190,000/500,000 | Ep 238 | ε=0.303 | Avg Reward: +1.2653
  └─ Episode 240: Reward=+1.2745 | Portfolio=$119,946 | Trades=215 | ε=0.300
Step 200,000/500,000 | Ep 251 | ε=0.284 | Avg Reward: +1.3225
  └─ Episode 260: Reward=+1.5063 | Portfolio=$117,854 | Trades=187 | ε=0.272
Step 210,000/500,000 | Ep 264 | ε=0.266 | Avg Reward: +1.3779
Step 220,000/500,000 | Ep 276 | ε=0.251 | Avg Reward: +1.4262
  └─ Episode 280: Reward=+1.1009 | Portfolio=$104,044 | Trades=227 | ε=0.246
Step 230,000/500,000 | Ep 289 | ε=0.235 | Avg Reward: +1.3630
  └─ Episode 300: Reward=+1.9676 | Portfolio=$162,597 | Trades=173 | ε=0.222
Step 240,000/500,000 | Ep 301 | ε=0.221 | Avg Reward: +1.4993
Step 250,000/500,000 | Ep 314 | ε=0.207 | Avg Reward: +1.6175
  └─ Episode 320: Reward=+1.7200 | Portfolio=$136,886 | Trades=181 | ε=0.201
Step 260,000/500,000 | Ep 327 | ε=0.194 | Avg Reward: +1.6551
Step 270,000/500,000 | Ep 339 | ε=0.183 | Avg Reward: +1.6634
  └─ Episode 340: Reward=+2.0552 | Portfolio=$200,354 | Trades=187 | ε=0.182
Step 280,000/500,000 | Ep 352 | ε=0.171 | Avg Reward: +1.6822
  └─ Episode 360: Reward=+1.8374 | Portfolio=$169,472 | Trades=193 | ε=0.165
Step 290,000/500,000 | Ep 364 | ε=0.161 | Avg Reward: +1.8377
Step 300,000/500,000 | Ep 377 | ε=0.151 | Avg Reward: +1.8588
  └─ Episode 380: Reward=+1.6332 | Portfolio=$146,253 | Trades=205 | ε=0.149
Step 310,000/500,000 | Ep 389 | ε=0.142 | Avg Reward: +1.8428
  └─ Episode 400: Reward=+2.0686 | Portfolio=$199,755 | Trades=183 | ε=0.135
Step 320,000/500,000 | Ep 402 | ε=0.133 | Avg Reward: +1.8155
Step 330,000/500,000 | Ep 415 | ε=0.125 | Avg Reward: +2.1257
  └─ Episode 420: Reward=+2.1593 | Portfolio=$215,239 | Trades=183 | ε=0.122
Step 340,000/500,000 | Ep 427 | ε=0.118 | Avg Reward: +2.1697
  └─ Episode 440: Reward=+2.4566 | Portfolio=$282,935 | Trades=181 | ε=0.110
Step 350,000/500,000 | Ep 440 | ε=0.110 | Avg Reward: +2.1824
Step 360,000/500,000 | Ep 452 | ε=0.104 | Avg Reward: +2.1974
  └─ Episode 460: Reward=+2.3985 | Portfolio=$257,531 | Trades=179 | ε=0.100
Step 370,000/500,000 | Ep 465 | ε=0.097 | Avg Reward: +2.2540
Step 380,000/500,000 | Ep 477 | ε=0.092 | Avg Reward: +2.2508
  └─ Episode 480: Reward=+2.1627 | Portfolio=$200,942 | Trades=175 | ε=0.090
Step 390,000/500,000 | Ep 490 | ε=0.086 | Avg Reward: +2.2399
  └─ Episode 500: Reward=+2.1646 | Portfolio=$210,576 | Trades=186 | ε=0.082
Step 400,000/500,000 | Ep 503 | ε=0.080 | Avg Reward: +2.2644
Step 410,000/500,000 | Ep 515 | ε=0.076 | Avg Reward: +2.2025
  └─ Episode 520: Reward=+2.4560 | Portfolio=$252,193 | Trades=167 | ε=0.074
Step 420,000/500,000 | Ep 528 | ε=0.071 | Avg Reward: +2.4167
  └─ Episode 540: Reward=+2.4482 | Portfolio=$255,939 | Trades=169 | ε=0.067
Step 430,000/500,000 | Ep 540 | ε=0.067 | Avg Reward: +2.4408
Step 440,000/500,000 | Ep 553 | ε=0.063 | Avg Reward: +2.4748
  └─ Episode 560: Reward=+2.4015 | Portfolio=$250,936 | Trades=171 | ε=0.060
Step 450,000/500,000 | Ep 566 | ε=0.059 | Avg Reward: +2.4614
Step 460,000/500,000 | Ep 578 | ε=0.055 | Avg Reward: +2.5129
  └─ Episode 580: Reward=+2.4027 | Portfolio=$242,579 | Trades=167 | ε=0.055
Step 470,000/500,000 | Ep 591 | ε=0.052 | Avg Reward: +2.4193
  └─ Episode 600: Reward=+2.5770 | Portfolio=$298,014 | Trades=173 | ε=0.049
Step 480,000/500,000 | Ep 603 | ε=0.049 | Avg Reward: +2.4890
Step 490,000/500,000 | Ep 616 | ε=0.046 | Avg Reward: +2.4617
  └─ Episode 620: Reward=+2.5616 | Portfolio=$298,518 | Trades=179 | ε=0.045
Step 500,000/500,000 | Ep 628 | ε=0.043 | Avg Reward: +2.5430

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 629
Final Epsilon: 0.0427
Mean Reward: 1.2654
Best Episode Reward: 2.7913
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251213_000838.pkl
Config saved to results/config_v2_20251213_000838.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $331,988.63
Return:  +3219.89%
Trades:  161
Fees:    $30504.20

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $93754.84 - $124752.53
   Price change over period: 11.9%
   Using 21 features for state
Initial: $10,000.00
Final:   $12,259.87
Return:  +22.60%
Trades:  37
Fees:    $892.98

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |    +3219.89% |      +22.60%
Buy & Hold           |     +286.16% |      +11.90%
--------------------------------------------------
Outperformance       |    +2933.73% |      +10.69%

 Final Values:
   Agent Train: $331,988.63  |  Buy&Hold: $38,616.07
   Agent Test:  $12,259.87  |  Buy&Hold: $11,190.46

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251213_000838.pkl
Final Epsilon: 0.0427
Total Episodes: 629

 AGENT BEATS BUY & HOLD! 
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-> 

v24
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2025-10-20',
            'end_date': '2025-11-11',
            'interval': '5m',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.008  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.995,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.999,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 2000000,  # MEHR! (war 50000)
            'log_interval': 20000
        }
    }

 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2025-10-20 to 2025-11-11...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True 
  data = yf.download(
Loaded 6319 rows of data
Date range: 2025-10-20 00:00:00+00:00 to 2025-11-10 23:55:00+00:00
Price range: $99040.66 - $116070.29
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 6259
Train set: 5007 rows (80%)
Test set: 1252 rows (20%)

 Original Train Prices: $99040.66 - $116070.29
 Original Test Prices: $99343.85 - $106535.13

Normalizing features for neural network...
Features normalized ✓

 Train data: 5007 days
 Test data: 1252 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $99040.66 - $116070.29
   Price change over period: -6.6%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.999 (reaches 0.01 after ~4602 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 2,000,000
Epsilon: 1.00  0.01
Decay rate: 0.999
============================================================

Step  20,000/2,000,000 | Ep   3 | ε=0.997 | Avg Reward: -23.1034
Step  40,000/2,000,000 | Ep   7 | ε=0.993 | Avg Reward: -22.8878
Step  60,000/2,000,000 | Ep  11 | ε=0.989 | Avg Reward: -22.7111
Step  80,000/2,000,000 | Ep  15 | ε=0.985 | Avg Reward: -22.6638
Step 100,000/2,000,000 | Ep  19 | ε=0.981 | Avg Reward: -22.7129
  └─ Episode 20: Reward=-21.8134 | Portfolio=$80 | Trades=1591 | ε=0.980
Step 120,000/2,000,000 | Ep  23 | ε=0.977 | Avg Reward: -22.4588
Step 140,000/2,000,000 | Ep  27 | ε=0.973 | Avg Reward: -22.1565
Step 160,000/2,000,000 | Ep  31 | ε=0.969 | Avg Reward: -22.0105
Step 180,000/2,000,000 | Ep  35 | ε=0.966 | Avg Reward: -22.2401
Step 200,000/2,000,000 | Ep  39 | ε=0.962 | Avg Reward: -21.9990
  └─ Episode 40: Reward=-21.9014 | Portfolio=$79 | Trades=1595 | ε=0.961
Step 220,000/2,000,000 | Ep  43 | ε=0.958 | Avg Reward: -21.9163
Step 240,000/2,000,000 | Ep  47 | ε=0.954 | Avg Reward: -21.6935
Step 260,000/2,000,000 | Ep  51 | ε=0.950 | Avg Reward: -21.6366
Step 280,000/2,000,000 | Ep  55 | ε=0.946 | Avg Reward: -21.3047
Step 300,000/2,000,000 | Ep  59 | ε=0.943 | Avg Reward: -21.0917
  └─ Episode 60: Reward=-21.2975 | Portfolio=$80 | Trades=1579 | ε=0.942
Step 320,000/2,000,000 | Ep  63 | ε=0.939 | Avg Reward: -20.8511
Step 340,000/2,000,000 | Ep  67 | ε=0.935 | Avg Reward: -20.8141
Step 360,000/2,000,000 | Ep  71 | ε=0.931 | Avg Reward: -20.9604
Step 380,000/2,000,000 | Ep  75 | ε=0.928 | Avg Reward: -20.8896
Step 400,000/2,000,000 | Ep  79 | ε=0.924 | Avg Reward: -20.6747
  └─ Episode 80: Reward=-20.4474 | Portfolio=$99 | Trades=1515 | ε=0.923
Step 420,000/2,000,000 | Ep  83 | ε=0.920 | Avg Reward: -20.4976
Step 440,000/2,000,000 | Ep  87 | ε=0.917 | Avg Reward: -20.3574
Step 460,000/2,000,000 | Ep  91 | ε=0.913 | Avg Reward: -20.3627
Step 480,000/2,000,000 | Ep  95 | ε=0.909 | Avg Reward: -20.4371
Step 500,000/2,000,000 | Ep  99 | ε=0.906 | Avg Reward: -20.6275
  └─ Episode 100: Reward=-19.6029 | Portfolio=$117 | Trades=1444 | ε=0.905
Step 520,000/2,000,000 | Ep 103 | ε=0.902 | Avg Reward: -20.4311
Step 540,000/2,000,000 | Ep 107 | ε=0.898 | Avg Reward: -20.1390
Step 560,000/2,000,000 | Ep 111 | ε=0.895 | Avg Reward: -19.8909
Step 580,000/2,000,000 | Ep 115 | ε=0.891 | Avg Reward: -19.7233
Step 600,000/2,000,000 | Ep 119 | ε=0.888 | Avg Reward: -19.7639
  └─ Episode 120: Reward=-20.2460 | Portfolio=$101 | Trades=1517 | ε=0.887
Step 620,000/2,000,000 | Ep 123 | ε=0.884 | Avg Reward: -19.7713
Step 640,000/2,000,000 | Ep 127 | ε=0.881 | Avg Reward: -19.6827
Step 660,000/2,000,000 | Ep 131 | ε=0.877 | Avg Reward: -19.6934
Step 680,000/2,000,000 | Ep 135 | ε=0.874 | Avg Reward: -19.4526
Step 700,000/2,000,000 | Ep 139 | ε=0.870 | Avg Reward: -19.4599
  └─ Episode 140: Reward=-19.8723 | Portfolio=$109 | Trades=1494 | ε=0.869
Step 720,000/2,000,000 | Ep 143 | ε=0.867 | Avg Reward: -19.3317
Step 740,000/2,000,000 | Ep 147 | ε=0.863 | Avg Reward: -19.4900
Step 760,000/2,000,000 | Ep 151 | ε=0.860 | Avg Reward: -19.3556
Step 780,000/2,000,000 | Ep 155 | ε=0.856 | Avg Reward: -19.2518
Step 800,000/2,000,000 | Ep 159 | ε=0.853 | Avg Reward: -19.0340
  └─ Episode 160: Reward=-19.4286 | Portfolio=$110 | Trades=1464 | ε=0.852
Step 820,000/2,000,000 | Ep 163 | ε=0.850 | Avg Reward: -19.0798
Step 840,000/2,000,000 | Ep 167 | ε=0.846 | Avg Reward: -18.9288
Step 860,000/2,000,000 | Ep 171 | ε=0.843 | Avg Reward: -18.6479
Step 880,000/2,000,000 | Ep 175 | ε=0.839 | Avg Reward: -18.5557
Step 900,000/2,000,000 | Ep 179 | ε=0.836 | Avg Reward: -18.4320
  └─ Episode 180: Reward=-18.7705 | Portfolio=$132 | Trades=1418 | ε=0.835
Step 920,000/2,000,000 | Ep 183 | ε=0.833 | Avg Reward: -18.4920
Step 940,000/2,000,000 | Ep 187 | ε=0.829 | Avg Reward: -18.5007
Step 960,000/2,000,000 | Ep 191 | ε=0.826 | Avg Reward: -18.4656
Step 980,000/2,000,000 | Ep 195 | ε=0.823 | Avg Reward: -18.3380
Step 1,000,000/2,000,000 | Ep 199 | ε=0.819 | Avg Reward: -18.2875
  └─ Episode 200: Reward=-17.8439 | Portfolio=$180 | Trades=1360 | ε=0.819
Step 1,020,000/2,000,000 | Ep 203 | ε=0.816 | Avg Reward: -18.0298
Step 1,040,000/2,000,000 | Ep 207 | ε=0.813 | Avg Reward: -17.8072
Step 1,060,000/2,000,000 | Ep 211 | ε=0.810 | Avg Reward: -17.7833
Step 1,080,000/2,000,000 | Ep 215 | ε=0.806 | Avg Reward: -17.6170
Step 1,100,000/2,000,000 | Ep 219 | ε=0.803 | Avg Reward: -17.6810
  └─ Episode 220: Reward=-17.3039 | Portfolio=$185 | Trades=1308 | ε=0.802
Step 1,120,000/2,000,000 | Ep 223 | ε=0.800 | Avg Reward: -17.7079
Step 1,140,000/2,000,000 | Ep 227 | ε=0.797 | Avg Reward: -17.5361
Step 1,160,000/2,000,000 | Ep 231 | ε=0.794 | Avg Reward: -17.5504
Step 1,180,000/2,000,000 | Ep 235 | ε=0.790 | Avg Reward: -17.3892
Step 1,200,000/2,000,000 | Ep 239 | ε=0.787 | Avg Reward: -17.1825
  └─ Episode 240: Reward=-16.6751 | Portfolio=$225 | Trades=1260 | ε=0.787
Step 1,220,000/2,000,000 | Ep 243 | ε=0.784 | Avg Reward: -17.0964
Step 1,240,000/2,000,000 | Ep 247 | ε=0.781 | Avg Reward: -17.1357
Step 1,260,000/2,000,000 | Ep 251 | ε=0.778 | Avg Reward: -17.3582
Step 1,280,000/2,000,000 | Ep 255 | ε=0.775 | Avg Reward: -17.2225
Step 1,300,000/2,000,000 | Ep 259 | ε=0.772 | Avg Reward: -17.1485
  └─ Episode 260: Reward=-16.5485 | Portfolio=$219 | Trades=1266 | ε=0.771
Step 1,320,000/2,000,000 | Ep 263 | ε=0.769 | Avg Reward: -17.0293
Step 1,340,000/2,000,000 | Ep 267 | ε=0.766 | Avg Reward: -16.8813
Step 1,360,000/2,000,000 | Ep 271 | ε=0.763 | Avg Reward: -16.8066
Step 1,380,000/2,000,000 | Ep 275 | ε=0.759 | Avg Reward: -16.7587
Step 1,400,000/2,000,000 | Ep 279 | ε=0.756 | Avg Reward: -16.5754
  └─ Episode 280: Reward=-16.1499 | Portfolio=$237 | Trades=1242 | ε=0.756
Step 1,420,000/2,000,000 | Ep 283 | ε=0.753 | Avg Reward: -16.3708
Step 1,440,000/2,000,000 | Ep 287 | ε=0.750 | Avg Reward: -16.2709
Step 1,460,000/2,000,000 | Ep 291 | ε=0.747 | Avg Reward: -16.2920
Step 1,480,000/2,000,000 | Ep 295 | ε=0.744 | Avg Reward: -16.1735
Step 1,500,000/2,000,000 | Ep 299 | ε=0.741 | Avg Reward: -16.3289
  └─ Episode 300: Reward=-15.3491 | Portfolio=$276 | Trades=1193 | ε=0.741
Step 1,520,000/2,000,000 | Ep 303 | ε=0.738 | Avg Reward: -16.0996
Step 1,540,000/2,000,000 | Ep 307 | ε=0.736 | Avg Reward: -16.1675
Step 1,560,000/2,000,000 | Ep 311 | ε=0.733 | Avg Reward: -15.9173
Step 1,580,000/2,000,000 | Ep 315 | ε=0.730 | Avg Reward: -15.8017
Step 1,600,000/2,000,000 | Ep 319 | ε=0.727 | Avg Reward: -15.6886
  └─ Episode 320: Reward=-15.7933 | Portfolio=$258 | Trades=1210 | ε=0.726
Step 1,620,000/2,000,000 | Ep 323 | ε=0.724 | Avg Reward: -15.7337
Step 1,640,000/2,000,000 | Ep 327 | ε=0.721 | Avg Reward: -15.7565
Step 1,660,000/2,000,000 | Ep 331 | ε=0.718 | Avg Reward: -15.6804
Step 1,680,000/2,000,000 | Ep 335 | ε=0.715 | Avg Reward: -15.6536
Step 1,700,000/2,000,000 | Ep 339 | ε=0.712 | Avg Reward: -15.5269
  └─ Episode 340: Reward=-15.4682 | Portfolio=$268 | Trades=1193 | ε=0.712
Step 1,720,000/2,000,000 | Ep 343 | ε=0.710 | Avg Reward: -15.3585
Step 1,740,000/2,000,000 | Ep 347 | ε=0.707 | Avg Reward: -15.3654
Step 1,760,000/2,000,000 | Ep 351 | ε=0.704 | Avg Reward: -15.3635
Step 1,780,000/2,000,000 | Ep 355 | ε=0.701 | Avg Reward: -15.1176
Step 1,800,000/2,000,000 | Ep 359 | ε=0.698 | Avg Reward: -15.0951
  └─ Episode 360: Reward=-15.0057 | Portfolio=$283 | Trades=1183 | ε=0.698
Step 1,820,000/2,000,000 | Ep 363 | ε=0.695 | Avg Reward: -15.1980
Step 1,840,000/2,000,000 | Ep 367 | ε=0.693 | Avg Reward: -15.0438
Step 1,860,000/2,000,000 | Ep 371 | ε=0.690 | Avg Reward: -14.9345
Step 1,880,000/2,000,000 | Ep 375 | ε=0.687 | Avg Reward: -14.8137
Step 1,900,000/2,000,000 | Ep 379 | ε=0.684 | Avg Reward: -14.8377
  └─ Episode 380: Reward=-14.2686 | Portfolio=$338 | Trades=1111 | ε=0.684
Step 1,920,000/2,000,000 | Ep 383 | ε=0.682 | Avg Reward: -14.7485
Step 1,940,000/2,000,000 | Ep 387 | ε=0.679 | Avg Reward: -14.5643
Step 1,960,000/2,000,000 | Ep 391 | ε=0.676 | Avg Reward: -14.5561
Step 1,980,000/2,000,000 | Ep 395 | ε=0.674 | Avg Reward: -14.4011
Step 2,000,000/2,000,000 | Ep 399 | ε=0.671 | Avg Reward: -14.3640
  └─ Episode 400: Reward=-7.9658 | Portfolio=$1,512 | Trades=628 | ε=0.670

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 400
Final Epsilon: 0.6702
Mean Reward: -18.1805
Best Episode Reward: -7.9658
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251213_010806.pkl
Config saved to results/config_v2_20251213_010806.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $10,000.00
Return:  +0.00%
Trades:  0
Fees:    $0.00

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $99343.85 - $106535.13
   Price change over period: 2.8%
   Using 21 features for state
Initial: $10,000.00
Final:   $10,000.00
Return:  +0.00%
Trades:  0
Fees:    $0.00

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |       +0.00% |       +0.00%
Buy & Hold           |       -6.61% |       +2.84%
--------------------------------------------------
Outperformance       |       +6.61% |       -2.84%

 Final Values:
   Agent Train: $10,000.00  |  Buy&Hold: $9,339.32
   Agent Test:  $10,000.00  |  Buy&Hold: $10,283.92

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251213_010806.pkl
Final Epsilon: 0.6702
Total Episodes: 400

 Agent lost money - needs more training or tuning
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-> 

v24
======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2025-10-20 to 2025-11-11...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 6319 rows of data
Date range: 2025-10-20 00:00:00+00:00 to 2025-11-10 23:55:00+00:00
Price range: $99040.66 - $116070.29
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 6259
Train set: 5007 rows (80%)
Test set: 1252 rows (20%)

 Original Train Prices: $99040.66 - $116070.29
 Original Test Prices: $99343.85 - $106535.13

Normalizing features for neural network...
Features normalized ✓

 Train data: 5007 days
 Test data: 1252 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $99040.66 - $116070.29
   Price change over period: -6.6%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 2,000,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/2,000,000 | Ep   1 | ε=0.990 | Avg Reward: -4.8447
Step  20,000/2,000,000 | Ep   3 | ε=0.970 | Avg Reward: -4.9289
Step  30,000/2,000,000 | Ep   5 | ε=0.951 | Avg Reward: -4.9547
Step  40,000/2,000,000 | Ep   7 | ε=0.932 | Avg Reward: -4.9096
Step  50,000/2,000,000 | Ep   9 | ε=0.914 | Avg Reward: -4.8344
Step  60,000/2,000,000 | Ep  11 | ε=0.895 | Avg Reward: -4.7786
Step  70,000/2,000,000 | Ep  13 | ε=0.878 | Avg Reward: -4.6806
Step  80,000/2,000,000 | Ep  15 | ε=0.860 | Avg Reward: -4.5472
Step  90,000/2,000,000 | Ep  17 | ε=0.843 | Avg Reward: -4.4908
Step 100,000/2,000,000 | Ep  19 | ε=0.826 | Avg Reward: -4.4256
  └─ Episode 20: Reward=-4.1829 | Portfolio=$151 | Trades=1382 | ε=0.818
Step 110,000/2,000,000 | Ep  21 | ε=0.810 | Avg Reward: -4.3445
Step 120,000/2,000,000 | Ep  23 | ε=0.794 | Avg Reward: -4.2538
Step 130,000/2,000,000 | Ep  25 | ε=0.778 | Avg Reward: -4.1835
Step 140,000/2,000,000 | Ep  27 | ε=0.762 | Avg Reward: -4.0753
Step 150,000/2,000,000 | Ep  29 | ε=0.747 | Avg Reward: -4.0095
Step 160,000/2,000,000 | Ep  31 | ε=0.732 | Avg Reward: -3.9378
Step 170,000/2,000,000 | Ep  33 | ε=0.718 | Avg Reward: -3.8473
Step 180,000/2,000,000 | Ep  35 | ε=0.703 | Avg Reward: -3.7684
Step 190,000/2,000,000 | Ep  37 | ε=0.689 | Avg Reward: -3.6719
Step 200,000/2,000,000 | Ep  39 | ε=0.676 | Avg Reward: -3.5894
  └─ Episode 40: Reward=-3.4418 | Portfolio=$318 | Trades=1151 | ε=0.669
Step 210,000/2,000,000 | Ep  41 | ε=0.662 | Avg Reward: -3.5111
Step 220,000/2,000,000 | Ep  43 | ε=0.649 | Avg Reward: -3.4538
Step 230,000/2,000,000 | Ep  45 | ε=0.636 | Avg Reward: -3.3866
Step 240,000/2,000,000 | Ep  47 | ε=0.624 | Avg Reward: -3.3206
Step 250,000/2,000,000 | Ep  49 | ε=0.611 | Avg Reward: -3.2498
Step 260,000/2,000,000 | Ep  51 | ε=0.599 | Avg Reward: -3.1830
Step 270,000/2,000,000 | Ep  53 | ε=0.587 | Avg Reward: -3.1353
Step 280,000/2,000,000 | Ep  55 | ε=0.575 | Avg Reward: -3.0906
Step 290,000/2,000,000 | Ep  57 | ε=0.564 | Avg Reward: -3.0360
Step 300,000/2,000,000 | Ep  59 | ε=0.553 | Avg Reward: -2.9809
  └─ Episode 60: Reward=-2.7474 | Portfolio=$637 | Trades=911 | ε=0.547
Step 310,000/2,000,000 | Ep  61 | ε=0.542 | Avg Reward: -2.9126
Step 320,000/2,000,000 | Ep  63 | ε=0.531 | Avg Reward: -2.8289
Step 330,000/2,000,000 | Ep  65 | ε=0.520 | Avg Reward: -2.7448
Step 340,000/2,000,000 | Ep  67 | ε=0.510 | Avg Reward: -2.6888
Step 350,000/2,000,000 | Ep  69 | ε=0.500 | Avg Reward: -2.6461
Step 360,000/2,000,000 | Ep  71 | ε=0.490 | Avg Reward: -2.6313
Step 370,000/2,000,000 | Ep  73 | ε=0.480 | Avg Reward: -2.6099
Step 380,000/2,000,000 | Ep  75 | ε=0.471 | Avg Reward: -2.5882
Step 390,000/2,000,000 | Ep  77 | ε=0.461 | Avg Reward: -2.5227
Step 400,000/2,000,000 | Ep  79 | ε=0.452 | Avg Reward: -2.4683
  └─ Episode 80: Reward=-2.2477 | Portfolio=$1,050 | Trades=749 | ε=0.448
Step 410,000/2,000,000 | Ep  81 | ε=0.443 | Avg Reward: -2.3984
Step 420,000/2,000,000 | Ep  83 | ε=0.434 | Avg Reward: -2.3541
Step 430,000/2,000,000 | Ep  85 | ε=0.426 | Avg Reward: -2.2784
Step 440,000/2,000,000 | Ep  87 | ε=0.417 | Avg Reward: -2.2547
Step 450,000/2,000,000 | Ep  89 | ε=0.409 | Avg Reward: -2.2133
Step 460,000/2,000,000 | Ep  91 | ε=0.401 | Avg Reward: -2.1696
Step 470,000/2,000,000 | Ep  93 | ε=0.393 | Avg Reward: -2.1007
Step 480,000/2,000,000 | Ep  95 | ε=0.385 | Avg Reward: -2.0605
Step 490,000/2,000,000 | Ep  97 | ε=0.377 | Avg Reward: -2.0260
Step 500,000/2,000,000 | Ep  99 | ε=0.370 | Avg Reward: -1.9655
  └─ Episode 100: Reward=-1.8725 | Portfolio=$1,529 | Trades=622 | ε=0.366
Step 510,000/2,000,000 | Ep 101 | ε=0.362 | Avg Reward: -1.9283
Step 520,000/2,000,000 | Ep 103 | ε=0.355 | Avg Reward: -1.9157
Step 530,000/2,000,000 | Ep 105 | ε=0.348 | Avg Reward: -1.8834
Step 540,000/2,000,000 | Ep 107 | ε=0.341 | Avg Reward: -1.8234
Step 550,000/2,000,000 | Ep 109 | ε=0.334 | Avg Reward: -1.7671
Step 560,000/2,000,000 | Ep 111 | ε=0.328 | Avg Reward: -1.7080
Step 570,000/2,000,000 | Ep 113 | ε=0.321 | Avg Reward: -1.6732
Step 580,000/2,000,000 | Ep 115 | ε=0.315 | Avg Reward: -1.6565
Step 590,000/2,000,000 | Ep 117 | ε=0.309 | Avg Reward: -1.6326
Step 600,000/2,000,000 | Ep 119 | ε=0.302 | Avg Reward: -1.6325
  └─ Episode 120: Reward=-1.4584 | Portfolio=$2,316 | Trades=500 | ε=0.299
Step 610,000/2,000,000 | Ep 121 | ε=0.296 | Avg Reward: -1.6019
Step 620,000/2,000,000 | Ep 123 | ε=0.290 | Avg Reward: -1.5709
Step 630,000/2,000,000 | Ep 125 | ε=0.285 | Avg Reward: -1.5278
Step 640,000/2,000,000 | Ep 127 | ε=0.279 | Avg Reward: -1.5135
Step 650,000/2,000,000 | Ep 129 | ε=0.273 | Avg Reward: -1.4895
Step 660,000/2,000,000 | Ep 131 | ε=0.268 | Avg Reward: -1.4792
Step 670,000/2,000,000 | Ep 133 | ε=0.263 | Avg Reward: -1.4484
Step 680,000/2,000,000 | Ep 135 | ε=0.257 | Avg Reward: -1.4220
Step 690,000/2,000,000 | Ep 137 | ε=0.252 | Avg Reward: -1.3718
Step 700,000/2,000,000 | Ep 139 | ε=0.247 | Avg Reward: -1.3332
  └─ Episode 140: Reward=-1.3719 | Portfolio=$2,525 | Trades=437 | ε=0.245
Step 710,000/2,000,000 | Ep 141 | ε=0.242 | Avg Reward: -1.3181
Step 720,000/2,000,000 | Ep 143 | ε=0.238 | Avg Reward: -1.2578
Step 730,000/2,000,000 | Ep 145 | ε=0.233 | Avg Reward: -1.2418
Step 740,000/2,000,000 | Ep 147 | ε=0.228 | Avg Reward: -1.2362
Step 750,000/2,000,000 | Ep 149 | ε=0.224 | Avg Reward: -1.2176
Step 760,000/2,000,000 | Ep 151 | ε=0.219 | Avg Reward: -1.2001
Step 770,000/2,000,000 | Ep 153 | ε=0.215 | Avg Reward: -1.1948
Step 780,000/2,000,000 | Ep 155 | ε=0.211 | Avg Reward: -1.1654
Step 790,000/2,000,000 | Ep 157 | ε=0.206 | Avg Reward: -1.1379
Step 800,000/2,000,000 | Ep 159 | ε=0.202 | Avg Reward: -1.1030
  └─ Episode 160: Reward=-0.9737 | Portfolio=$3,761 | Trades=342 | ε=0.200
Step 810,000/2,000,000 | Ep 161 | ε=0.198 | Avg Reward: -1.0554
Step 820,000/2,000,000 | Ep 163 | ε=0.194 | Avg Reward: -1.0400
Step 830,000/2,000,000 | Ep 165 | ε=0.190 | Avg Reward: -0.9883
Step 840,000/2,000,000 | Ep 167 | ε=0.187 | Avg Reward: -0.9611
Step 850,000/2,000,000 | Ep 169 | ε=0.183 | Avg Reward: -0.9487
Step 860,000/2,000,000 | Ep 171 | ε=0.179 | Avg Reward: -0.9379
Step 870,000/2,000,000 | Ep 173 | ε=0.176 | Avg Reward: -0.9324
Step 880,000/2,000,000 | Ep 175 | ε=0.172 | Avg Reward: -0.9393
Step 890,000/2,000,000 | Ep 177 | ε=0.169 | Avg Reward: -0.9297
Step 900,000/2,000,000 | Ep 179 | ε=0.165 | Avg Reward: -0.9058
  └─ Episode 180: Reward=-0.6766 | Portfolio=$5,065 | Trades=264 | ε=0.164
Step 910,000/2,000,000 | Ep 181 | ε=0.162 | Avg Reward: -0.8663
Step 920,000/2,000,000 | Ep 183 | ε=0.159 | Avg Reward: -0.8303
Step 930,000/2,000,000 | Ep 185 | ε=0.156 | Avg Reward: -0.8178
Step 940,000/2,000,000 | Ep 187 | ε=0.153 | Avg Reward: -0.8029
Step 950,000/2,000,000 | Ep 189 | ε=0.150 | Avg Reward: -0.8030
Step 960,000/2,000,000 | Ep 191 | ε=0.147 | Avg Reward: -0.8286
Step 970,000/2,000,000 | Ep 193 | ε=0.144 | Avg Reward: -0.8169
Step 980,000/2,000,000 | Ep 195 | ε=0.141 | Avg Reward: -0.8078
Step 990,000/2,000,000 | Ep 197 | ε=0.138 | Avg Reward: -0.7887
Step 1,000,000/2,000,000 | Ep 199 | ε=0.135 | Avg Reward: -0.7641
  └─ Episode 200: Reward=-0.6957 | Portfolio=$4,970 | Trades=224 | ε=0.134
Step 1,010,000/2,000,000 | Ep 201 | ε=0.133 | Avg Reward: -0.7318
Step 1,020,000/2,000,000 | Ep 203 | ε=0.130 | Avg Reward: -0.7175
Step 1,030,000/2,000,000 | Ep 205 | ε=0.127 | Avg Reward: -0.6996
Step 1,040,000/2,000,000 | Ep 207 | ε=0.125 | Avg Reward: -0.6783
Step 1,050,000/2,000,000 | Ep 209 | ε=0.122 | Avg Reward: -0.6534
Step 1,060,000/2,000,000 | Ep 211 | ε=0.120 | Avg Reward: -0.6264
Step 1,070,000/2,000,000 | Ep 213 | ε=0.118 | Avg Reward: -0.6203
Step 1,080,000/2,000,000 | Ep 215 | ε=0.115 | Avg Reward: -0.6065
Step 1,090,000/2,000,000 | Ep 217 | ε=0.113 | Avg Reward: -0.6025
Step 1,100,000/2,000,000 | Ep 219 | ε=0.111 | Avg Reward: -0.6178
  └─ Episode 220: Reward=-0.6178 | Portfolio=$5,371 | Trades=209 | ε=0.110
Step 1,110,000/2,000,000 | Ep 221 | ε=0.108 | Avg Reward: -0.6206
Step 1,120,000/2,000,000 | Ep 223 | ε=0.106 | Avg Reward: -0.6167
Step 1,130,000/2,000,000 | Ep 225 | ε=0.104 | Avg Reward: -0.5876
Step 1,140,000/2,000,000 | Ep 227 | ε=0.102 | Avg Reward: -0.5538
Step 1,150,000/2,000,000 | Ep 229 | ε=0.100 | Avg Reward: -0.5226
Step 1,160,000/2,000,000 | Ep 231 | ε=0.098 | Avg Reward: -0.5125
Step 1,170,000/2,000,000 | Ep 233 | ε=0.096 | Avg Reward: -0.4850
Step 1,180,000/2,000,000 | Ep 235 | ε=0.094 | Avg Reward: -0.4980
Step 1,190,000/2,000,000 | Ep 237 | ε=0.092 | Avg Reward: -0.5086
Step 1,200,000/2,000,000 | Ep 239 | ε=0.091 | Avg Reward: -0.5180
  └─ Episode 240: Reward=-0.4374 | Portfolio=$6,437 | Trades=157 | ε=0.090
Step 1,210,000/2,000,000 | Ep 241 | ε=0.089 | Avg Reward: -0.5007
Step 1,220,000/2,000,000 | Ep 243 | ε=0.087 | Avg Reward: -0.5027
Step 1,230,000/2,000,000 | Ep 245 | ε=0.085 | Avg Reward: -0.4869
Step 1,240,000/2,000,000 | Ep 247 | ε=0.084 | Avg Reward: -0.4769
Step 1,250,000/2,000,000 | Ep 249 | ε=0.082 | Avg Reward: -0.4538
Step 1,260,000/2,000,000 | Ep 251 | ε=0.080 | Avg Reward: -0.4675
Step 1,270,000/2,000,000 | Ep 253 | ε=0.079 | Avg Reward: -0.4560
Step 1,280,000/2,000,000 | Ep 255 | ε=0.077 | Avg Reward: -0.4625
Step 1,290,000/2,000,000 | Ep 257 | ε=0.076 | Avg Reward: -0.4421
Step 1,300,000/2,000,000 | Ep 259 | ε=0.074 | Avg Reward: -0.4312
  └─ Episode 260: Reward=-0.4517 | Portfolio=$6,344 | Trades=136 | ε=0.073
Step 1,310,000/2,000,000 | Ep 261 | ε=0.073 | Avg Reward: -0.4126
Step 1,320,000/2,000,000 | Ep 263 | ε=0.071 | Avg Reward: -0.4051
Step 1,330,000/2,000,000 | Ep 265 | ε=0.070 | Avg Reward: -0.3863
Step 1,340,000/2,000,000 | Ep 267 | ε=0.068 | Avg Reward: -0.3845
Step 1,350,000/2,000,000 | Ep 269 | ε=0.067 | Avg Reward: -0.3813
Step 1,360,000/2,000,000 | Ep 271 | ε=0.066 | Avg Reward: -0.3717
Step 1,370,000/2,000,000 | Ep 273 | ε=0.064 | Avg Reward: -0.3728
Step 1,380,000/2,000,000 | Ep 275 | ε=0.063 | Avg Reward: -0.3609
Step 1,390,000/2,000,000 | Ep 277 | ε=0.062 | Avg Reward: -0.3695
Step 1,400,000/2,000,000 | Ep 279 | ε=0.061 | Avg Reward: -0.3540
  └─ Episode 280: Reward=-0.3916 | Portfolio=$6,740 | Trades=128 | ε=0.060
Step 1,410,000/2,000,000 | Ep 281 | ε=0.059 | Avg Reward: -0.3524
Step 1,420,000/2,000,000 | Ep 283 | ε=0.058 | Avg Reward: -0.3332
Step 1,430,000/2,000,000 | Ep 285 | ε=0.057 | Avg Reward: -0.3345
Step 1,440,000/2,000,000 | Ep 287 | ε=0.056 | Avg Reward: -0.3233
Step 1,450,000/2,000,000 | Ep 289 | ε=0.055 | Avg Reward: -0.3319
Step 1,460,000/2,000,000 | Ep 291 | ε=0.054 | Avg Reward: -0.3188
Step 1,470,000/2,000,000 | Ep 293 | ε=0.053 | Avg Reward: -0.3099
Step 1,480,000/2,000,000 | Ep 295 | ε=0.052 | Avg Reward: -0.3003
Step 1,490,000/2,000,000 | Ep 297 | ε=0.051 | Avg Reward: -0.2938
Step 1,500,000/2,000,000 | Ep 299 | ε=0.050 | Avg Reward: -0.2800
  └─ Episode 300: Reward=-0.2770 | Portfolio=$7,557 | Trades=85 | ε=0.049
Step 1,510,000/2,000,000 | Ep 301 | ε=0.049 | Avg Reward: -0.2786
Step 1,520,000/2,000,000 | Ep 303 | ε=0.048 | Avg Reward: -0.2769
Step 1,530,000/2,000,000 | Ep 305 | ε=0.047 | Avg Reward: -0.2795
Step 1,540,000/2,000,000 | Ep 307 | ε=0.046 | Avg Reward: -0.2741
Step 1,550,000/2,000,000 | Ep 309 | ε=0.045 | Avg Reward: -0.2671
Step 1,560,000/2,000,000 | Ep 311 | ε=0.044 | Avg Reward: -0.2457
Step 1,570,000/2,000,000 | Ep 313 | ε=0.043 | Avg Reward: -0.2446
Step 1,580,000/2,000,000 | Ep 315 | ε=0.042 | Avg Reward: -0.2364
Step 1,590,000/2,000,000 | Ep 317 | ε=0.041 | Avg Reward: -0.2202
Step 1,600,000/2,000,000 | Ep 319 | ε=0.041 | Avg Reward: -0.2147
  └─ Episode 320: Reward=-0.1733 | Portfolio=$8,384 | Trades=82 | ε=0.040
Step 1,610,000/2,000,000 | Ep 321 | ε=0.040 | Avg Reward: -0.2284
Step 1,620,000/2,000,000 | Ep 323 | ε=0.039 | Avg Reward: -0.2217
Step 1,630,000/2,000,000 | Ep 325 | ε=0.038 | Avg Reward: -0.2194
Step 1,640,000/2,000,000 | Ep 327 | ε=0.037 | Avg Reward: -0.2389
Step 1,650,000/2,000,000 | Ep 329 | ε=0.037 | Avg Reward: -0.2422
Step 1,660,000/2,000,000 | Ep 331 | ε=0.036 | Avg Reward: -0.2251
Step 1,670,000/2,000,000 | Ep 333 | ε=0.035 | Avg Reward: -0.2341
Step 1,680,000/2,000,000 | Ep 335 | ε=0.034 | Avg Reward: -0.2396
Step 1,690,000/2,000,000 | Ep 337 | ε=0.034 | Avg Reward: -0.2251
Step 1,700,000/2,000,000 | Ep 339 | ε=0.033 | Avg Reward: -0.2211
  └─ Episode 340: Reward=-0.1565 | Portfolio=$8,524 | Trades=47 | ε=0.033
Step 1,710,000/2,000,000 | Ep 341 | ε=0.032 | Avg Reward: -0.2198
Step 1,720,000/2,000,000 | Ep 343 | ε=0.032 | Avg Reward: -0.1976
Step 1,730,000/2,000,000 | Ep 345 | ε=0.031 | Avg Reward: -0.1752
Step 1,740,000/2,000,000 | Ep 347 | ε=0.031 | Avg Reward: -0.1711
Step 1,750,000/2,000,000 | Ep 349 | ε=0.030 | Avg Reward: -0.1734
Step 1,760,000/2,000,000 | Ep 351 | ε=0.029 | Avg Reward: -0.1730
Step 1,770,000/2,000,000 | Ep 353 | ε=0.029 | Avg Reward: -0.1889
Step 1,780,000/2,000,000 | Ep 355 | ε=0.028 | Avg Reward: -0.1964
Step 1,790,000/2,000,000 | Ep 357 | ε=0.028 | Avg Reward: -0.2064
Step 1,800,000/2,000,000 | Ep 359 | ε=0.027 | Avg Reward: -0.1942
  └─ Episode 360: Reward=-0.1415 | Portfolio=$8,656 | Trades=63 | ε=0.027
Step 1,810,000/2,000,000 | Ep 361 | ε=0.027 | Avg Reward: -0.1950
Step 1,820,000/2,000,000 | Ep 363 | ε=0.026 | Avg Reward: -0.1834
Step 1,830,000/2,000,000 | Ep 365 | ε=0.026 | Avg Reward: -0.1778
Step 1,840,000/2,000,000 | Ep 367 | ε=0.025 | Avg Reward: -0.1553
Step 1,850,000/2,000,000 | Ep 369 | ε=0.025 | Avg Reward: -0.1469
Step 1,860,000/2,000,000 | Ep 371 | ε=0.024 | Avg Reward: -0.1252
Step 1,870,000/2,000,000 | Ep 373 | ε=0.024 | Avg Reward: -0.1244
Step 1,880,000/2,000,000 | Ep 375 | ε=0.023 | Avg Reward: -0.1082
Step 1,890,000/2,000,000 | Ep 377 | ε=0.023 | Avg Reward: -0.1152
Step 1,900,000/2,000,000 | Ep 379 | ε=0.022 | Avg Reward: -0.1151
  └─ Episode 380: Reward=-0.1288 | Portfolio=$8,763 | Trades=43 | ε=0.022
Step 1,910,000/2,000,000 | Ep 381 | ε=0.022 | Avg Reward: -0.1264
Step 1,920,000/2,000,000 | Ep 383 | ε=0.021 | Avg Reward: -0.1164
Step 1,930,000/2,000,000 | Ep 385 | ε=0.021 | Avg Reward: -0.1145
Step 1,940,000/2,000,000 | Ep 387 | ε=0.020 | Avg Reward: -0.1186
Step 1,950,000/2,000,000 | Ep 389 | ε=0.020 | Avg Reward: -0.1182
Step 1,960,000/2,000,000 | Ep 391 | ε=0.020 | Avg Reward: -0.1322
Step 1,970,000/2,000,000 | Ep 393 | ε=0.019 | Avg Reward: -0.1427
Step 1,980,000/2,000,000 | Ep 395 | ε=0.019 | Avg Reward: -0.1443
Step 1,990,000/2,000,000 | Ep 397 | ε=0.019 | Avg Reward: -0.1362
Step 2,000,000/2,000,000 | Ep 399 | ε=0.018 | Avg Reward: -0.1340
  └─ Episode 400: Reward=-0.0357 | Portfolio=$9,638 | Trades=13 | ε=0.018

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 400
Final Epsilon: 0.0180
Mean Reward: -1.2560
Best Episode Reward: -0.0062
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251213_015102.pkl
Config saved to results/config_v2_20251213_015102.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $9,676.55
Return:  -3.23%
Trades:  9
Fees:    $179.57

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $99343.85 - $106535.13
   Price change over period: 2.8%
   Using 21 features for state
Initial: $10,000.00
Final:   $10,536.25
Return:  +5.36%
Trades:  1
Fees:    $19.96

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |       -3.23% |       +5.36%
Buy & Hold           |       -6.61% |       +2.84%
--------------------------------------------------
Outperformance       |       +3.37% |       +2.52%

 Final Values:
   Agent Train: $9,676.55  |  Buy&Hold: $9,339.32
   Agent Test:  $10,536.25  |  Buy&Hold: $10,283.92

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251213_015102.pkl
Final Epsilon: 0.0180
Total Episodes: 400
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2025-10-20',
            'end_date': '2025-11-11',
            'interval': '5m',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.00  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 2000000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
v26
======================================================================
 RL TRADING BOT - Q-LEARNING TRAINING v2
======================================================================
FIXES: Real prices, faster epsilon decay, more training
======================================================================

======================================================================
 STEP 1: LOADING DATA
======================================================================
Loading BTC-USD data from 2025-10-20 to 2025-11-11...
c:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader-\rl_trading_bot\utils\data_loader.py:59: FutureWarning: YF.download() has changed argument auto_adjust default to True
  data = yf.download(
Loaded 6319 rows of data
Date range: 2025-10-20 00:00:00+00:00 to 2025-11-10 23:55:00+00:00
Price range: $99040.66 - $116070.29
Calculating technical indicators...
Added technical indicators. Remaining rows after dropna: 6259
Train set: 5007 rows (80%)
Test set: 1252 rows (20%)

 Original Train Prices: $99040.66 - $116070.29
 Original Test Prices: $99343.85 - $106535.13

Normalizing features for neural network...
Features normalized ✓

 Train data: 5007 days
 Test data: 1252 days

======================================================================
 STEP 2: CREATING ENVIRONMENT
======================================================================
 Using REAL prices for trading: $99040.66 - $116070.29
   Price change over period: -6.6%
   Using 21 features for state

======================================================================
 STEP 3: CREATING Q-LEARNING AGENT
======================================================================
Q-table shape: (15, 15, 15, 2, 3)
Epsilon decay: 0.99 (reaches 0.01 after ~458 episodes)

======================================================================
 STEP 4: TRAINING
======================================================================

============================================================
Q-LEARNING TRAINING
============================================================
Total timesteps: 2,000,000
Epsilon: 1.00  0.01
Decay rate: 0.99
============================================================

Step  10,000/2,000,000 | Ep   1 | ε=0.990 | Avg Reward: +0.0616
Step  20,000/2,000,000 | Ep   3 | ε=0.970 | Avg Reward: +0.0368
Step  30,000/2,000,000 | Ep   5 | ε=0.951 | Avg Reward: +0.0299
Step  40,000/2,000,000 | Ep   7 | ε=0.932 | Avg Reward: +0.0259
Step  50,000/2,000,000 | Ep   9 | ε=0.914 | Avg Reward: +0.0171
Step  60,000/2,000,000 | Ep  11 | ε=0.895 | Avg Reward: +0.0150
Step  70,000/2,000,000 | Ep  13 | ε=0.878 | Avg Reward: +0.0087
Step  80,000/2,000,000 | Ep  15 | ε=0.860 | Avg Reward: +0.0105
Step  90,000/2,000,000 | Ep  17 | ε=0.843 | Avg Reward: +0.0087
Step 100,000/2,000,000 | Ep  19 | ε=0.826 | Avg Reward: +0.0223
  └─ Episode 20: Reward=-0.0154 | Portfolio=$9,825 | Trades=1751 | ε=0.818
Step 110,000/2,000,000 | Ep  21 | ε=0.810 | Avg Reward: +0.0251
Step 120,000/2,000,000 | Ep  23 | ε=0.794 | Avg Reward: +0.0227
Step 130,000/2,000,000 | Ep  25 | ε=0.778 | Avg Reward: +0.0299
Step 140,000/2,000,000 | Ep  27 | ε=0.762 | Avg Reward: +0.0446
Step 150,000/2,000,000 | Ep  29 | ε=0.747 | Avg Reward: +0.0347
Step 160,000/2,000,000 | Ep  31 | ε=0.732 | Avg Reward: +0.0430
Step 170,000/2,000,000 | Ep  33 | ε=0.718 | Avg Reward: +0.0584
Step 180,000/2,000,000 | Ep  35 | ε=0.703 | Avg Reward: +0.0659
Step 190,000/2,000,000 | Ep  37 | ε=0.689 | Avg Reward: +0.0761
Step 200,000/2,000,000 | Ep  39 | ε=0.676 | Avg Reward: +0.0911
  └─ Episode 40: Reward=+0.1026 | Portfolio=$11,056 | Trades=1742 | ε=0.669
Step 210,000/2,000,000 | Ep  41 | ε=0.662 | Avg Reward: +0.0989
Step 220,000/2,000,000 | Ep  43 | ε=0.649 | Avg Reward: +0.1047
Step 230,000/2,000,000 | Ep  45 | ε=0.636 | Avg Reward: +0.1203
Step 240,000/2,000,000 | Ep  47 | ε=0.624 | Avg Reward: +0.1278
Step 250,000/2,000,000 | Ep  49 | ε=0.611 | Avg Reward: +0.1368
Step 260,000/2,000,000 | Ep  51 | ε=0.599 | Avg Reward: +0.1357
Step 270,000/2,000,000 | Ep  53 | ε=0.587 | Avg Reward: +0.1486
Step 280,000/2,000,000 | Ep  55 | ε=0.575 | Avg Reward: +0.1541
Step 290,000/2,000,000 | Ep  57 | ε=0.564 | Avg Reward: +0.1455
Step 300,000/2,000,000 | Ep  59 | ε=0.553 | Avg Reward: +0.1632
  └─ Episode 60: Reward=+0.1912 | Portfolio=$12,080 | Trades=1812 | ε=0.547
Step 310,000/2,000,000 | Ep  61 | ε=0.542 | Avg Reward: +0.1812
Step 320,000/2,000,000 | Ep  63 | ε=0.531 | Avg Reward: +0.1949
Step 330,000/2,000,000 | Ep  65 | ε=0.520 | Avg Reward: +0.1924
Step 340,000/2,000,000 | Ep  67 | ε=0.510 | Avg Reward: +0.2085
Step 350,000/2,000,000 | Ep  69 | ε=0.500 | Avg Reward: +0.2095
Step 360,000/2,000,000 | Ep  71 | ε=0.490 | Avg Reward: +0.2179
Step 370,000/2,000,000 | Ep  73 | ε=0.480 | Avg Reward: +0.2235
Step 380,000/2,000,000 | Ep  75 | ε=0.471 | Avg Reward: +0.2370
Step 390,000/2,000,000 | Ep  77 | ε=0.461 | Avg Reward: +0.2395
Step 400,000/2,000,000 | Ep  79 | ε=0.452 | Avg Reward: +0.2470
  └─ Episode 80: Reward=+0.2095 | Portfolio=$12,311 | Trades=1816 | ε=0.448
Step 410,000/2,000,000 | Ep  81 | ε=0.443 | Avg Reward: +0.2421
Step 420,000/2,000,000 | Ep  83 | ε=0.434 | Avg Reward: +0.2370
Step 430,000/2,000,000 | Ep  85 | ε=0.426 | Avg Reward: +0.2389
Step 440,000/2,000,000 | Ep  87 | ε=0.417 | Avg Reward: +0.2425
Step 450,000/2,000,000 | Ep  89 | ε=0.409 | Avg Reward: +0.2451
Step 460,000/2,000,000 | Ep  91 | ε=0.401 | Avg Reward: +0.2557
Step 470,000/2,000,000 | Ep  93 | ε=0.393 | Avg Reward: +0.2625
Step 480,000/2,000,000 | Ep  95 | ε=0.385 | Avg Reward: +0.2622
Step 490,000/2,000,000 | Ep  97 | ε=0.377 | Avg Reward: +0.2673
Step 500,000/2,000,000 | Ep  99 | ε=0.370 | Avg Reward: +0.2670
  └─ Episode 100: Reward=+0.2644 | Portfolio=$13,007 | Trades=1830 | ε=0.366
Step 510,000/2,000,000 | Ep 101 | ε=0.362 | Avg Reward: +0.2667
Step 520,000/2,000,000 | Ep 103 | ε=0.355 | Avg Reward: +0.2845
Step 530,000/2,000,000 | Ep 105 | ε=0.348 | Avg Reward: +0.2919
Step 540,000/2,000,000 | Ep 107 | ε=0.341 | Avg Reward: +0.2997
Step 550,000/2,000,000 | Ep 109 | ε=0.334 | Avg Reward: +0.3044
Step 560,000/2,000,000 | Ep 111 | ε=0.328 | Avg Reward: +0.3092
Step 570,000/2,000,000 | Ep 113 | ε=0.321 | Avg Reward: +0.3051
Step 580,000/2,000,000 | Ep 115 | ε=0.315 | Avg Reward: +0.3073
Step 590,000/2,000,000 | Ep 117 | ε=0.309 | Avg Reward: +0.3099
Step 600,000/2,000,000 | Ep 119 | ε=0.302 | Avg Reward: +0.3215
  └─ Episode 120: Reward=+0.2948 | Portfolio=$13,408 | Trades=1686 | ε=0.299
Step 610,000/2,000,000 | Ep 121 | ε=0.296 | Avg Reward: +0.3257
Step 620,000/2,000,000 | Ep 123 | ε=0.290 | Avg Reward: +0.3276
Step 630,000/2,000,000 | Ep 125 | ε=0.285 | Avg Reward: +0.3321
Step 640,000/2,000,000 | Ep 127 | ε=0.279 | Avg Reward: +0.3330
Step 650,000/2,000,000 | Ep 129 | ε=0.273 | Avg Reward: +0.3301
Step 660,000/2,000,000 | Ep 131 | ε=0.268 | Avg Reward: +0.3317
Step 670,000/2,000,000 | Ep 133 | ε=0.263 | Avg Reward: +0.3368
Step 680,000/2,000,000 | Ep 135 | ε=0.257 | Avg Reward: +0.3413
Step 690,000/2,000,000 | Ep 137 | ε=0.252 | Avg Reward: +0.3556
Step 700,000/2,000,000 | Ep 139 | ε=0.247 | Avg Reward: +0.3551
  └─ Episode 140: Reward=+0.3644 | Portfolio=$14,375 | Trades=1668 | ε=0.245
Step 710,000/2,000,000 | Ep 141 | ε=0.242 | Avg Reward: +0.3616
Step 720,000/2,000,000 | Ep 143 | ε=0.238 | Avg Reward: +0.3616
Step 730,000/2,000,000 | Ep 145 | ε=0.233 | Avg Reward: +0.3565
Step 740,000/2,000,000 | Ep 147 | ε=0.228 | Avg Reward: +0.3558
Step 750,000/2,000,000 | Ep 149 | ε=0.224 | Avg Reward: +0.3650
Step 760,000/2,000,000 | Ep 151 | ε=0.219 | Avg Reward: +0.3653
Step 770,000/2,000,000 | Ep 153 | ε=0.215 | Avg Reward: +0.3720
Step 780,000/2,000,000 | Ep 155 | ε=0.211 | Avg Reward: +0.3811
Step 790,000/2,000,000 | Ep 157 | ε=0.206 | Avg Reward: +0.3778
Step 800,000/2,000,000 | Ep 159 | ε=0.202 | Avg Reward: +0.3780
  └─ Episode 160: Reward=+0.3558 | Portfolio=$14,255 | Trades=1572 | ε=0.200
Step 810,000/2,000,000 | Ep 161 | ε=0.198 | Avg Reward: +0.3791
Step 820,000/2,000,000 | Ep 163 | ε=0.194 | Avg Reward: +0.3780
Step 830,000/2,000,000 | Ep 165 | ε=0.190 | Avg Reward: +0.3813
Step 840,000/2,000,000 | Ep 167 | ε=0.187 | Avg Reward: +0.3871
Step 850,000/2,000,000 | Ep 169 | ε=0.183 | Avg Reward: +0.3888
Step 860,000/2,000,000 | Ep 171 | ε=0.179 | Avg Reward: +0.3921
Step 870,000/2,000,000 | Ep 173 | ε=0.176 | Avg Reward: +0.3861
Step 880,000/2,000,000 | Ep 175 | ε=0.172 | Avg Reward: +0.3876
Step 890,000/2,000,000 | Ep 177 | ε=0.169 | Avg Reward: +0.3942
Step 900,000/2,000,000 | Ep 179 | ε=0.165 | Avg Reward: +0.4047
  └─ Episode 180: Reward=+0.4026 | Portfolio=$14,937 | Trades=1558 | ε=0.164
Step 910,000/2,000,000 | Ep 181 | ε=0.162 | Avg Reward: +0.4069
Step 920,000/2,000,000 | Ep 183 | ε=0.159 | Avg Reward: +0.4249
Step 930,000/2,000,000 | Ep 185 | ε=0.156 | Avg Reward: +0.4289
Step 940,000/2,000,000 | Ep 187 | ε=0.153 | Avg Reward: +0.4293
Step 950,000/2,000,000 | Ep 189 | ε=0.150 | Avg Reward: +0.4293
Step 960,000/2,000,000 | Ep 191 | ε=0.147 | Avg Reward: +0.4495
Step 970,000/2,000,000 | Ep 193 | ε=0.144 | Avg Reward: +0.4526
Step 980,000/2,000,000 | Ep 195 | ε=0.141 | Avg Reward: +0.4603
Step 990,000/2,000,000 | Ep 197 | ε=0.138 | Avg Reward: +0.4659
Step 1,000,000/2,000,000 | Ep 199 | ε=0.135 | Avg Reward: +0.4741
  └─ Episode 200: Reward=+0.4489 | Portfolio=$15,641 | Trades=1566 | ε=0.134
Step 1,010,000/2,000,000 | Ep 201 | ε=0.133 | Avg Reward: +0.4664
Step 1,020,000/2,000,000 | Ep 203 | ε=0.130 | Avg Reward: +0.4625
Step 1,030,000/2,000,000 | Ep 205 | ε=0.127 | Avg Reward: +0.4612
Step 1,040,000/2,000,000 | Ep 207 | ε=0.125 | Avg Reward: +0.4524
Step 1,050,000/2,000,000 | Ep 209 | ε=0.122 | Avg Reward: +0.4424
Step 1,060,000/2,000,000 | Ep 211 | ε=0.120 | Avg Reward: +0.4384
Step 1,070,000/2,000,000 | Ep 213 | ε=0.118 | Avg Reward: +0.4272
Step 1,080,000/2,000,000 | Ep 215 | ε=0.115 | Avg Reward: +0.4255
Step 1,090,000/2,000,000 | Ep 217 | ε=0.113 | Avg Reward: +0.4395
Step 1,100,000/2,000,000 | Ep 219 | ε=0.111 | Avg Reward: +0.4443
  └─ Episode 220: Reward=+0.4400 | Portfolio=$15,505 | Trades=1454 | ε=0.110
Step 1,110,000/2,000,000 | Ep 221 | ε=0.108 | Avg Reward: +0.4508
Step 1,120,000/2,000,000 | Ep 223 | ε=0.106 | Avg Reward: +0.4629
Step 1,130,000/2,000,000 | Ep 225 | ε=0.104 | Avg Reward: +0.4709
Step 1,140,000/2,000,000 | Ep 227 | ε=0.102 | Avg Reward: +0.4656
Step 1,150,000/2,000,000 | Ep 229 | ε=0.100 | Avg Reward: +0.4632
Step 1,160,000/2,000,000 | Ep 231 | ε=0.098 | Avg Reward: +0.4640
Step 1,170,000/2,000,000 | Ep 233 | ε=0.096 | Avg Reward: +0.4661
Step 1,180,000/2,000,000 | Ep 235 | ε=0.094 | Avg Reward: +0.4655
Step 1,190,000/2,000,000 | Ep 237 | ε=0.092 | Avg Reward: +0.4695
Step 1,200,000/2,000,000 | Ep 239 | ε=0.091 | Avg Reward: +0.4777
  └─ Episode 240: Reward=+0.5066 | Portfolio=$16,573 | Trades=1452 | ε=0.090
Step 1,210,000/2,000,000 | Ep 241 | ε=0.089 | Avg Reward: +0.4801
Step 1,220,000/2,000,000 | Ep 243 | ε=0.087 | Avg Reward: +0.4830
Step 1,230,000/2,000,000 | Ep 245 | ε=0.085 | Avg Reward: +0.4844
Step 1,240,000/2,000,000 | Ep 247 | ε=0.084 | Avg Reward: +0.4869
Step 1,250,000/2,000,000 | Ep 249 | ε=0.082 | Avg Reward: +0.4888
Step 1,260,000/2,000,000 | Ep 251 | ε=0.080 | Avg Reward: +0.4885
Step 1,270,000/2,000,000 | Ep 253 | ε=0.079 | Avg Reward: +0.4834
Step 1,280,000/2,000,000 | Ep 255 | ε=0.077 | Avg Reward: +0.4784
Step 1,290,000/2,000,000 | Ep 257 | ε=0.076 | Avg Reward: +0.4773
Step 1,300,000/2,000,000 | Ep 259 | ε=0.074 | Avg Reward: +0.4751
  └─ Episode 260: Reward=+0.5152 | Portfolio=$16,715 | Trades=1496 | ε=0.073
Step 1,310,000/2,000,000 | Ep 261 | ε=0.073 | Avg Reward: +0.4829
Step 1,320,000/2,000,000 | Ep 263 | ε=0.071 | Avg Reward: +0.4868
Step 1,330,000/2,000,000 | Ep 265 | ε=0.070 | Avg Reward: +0.4916
Step 1,340,000/2,000,000 | Ep 267 | ε=0.068 | Avg Reward: +0.4936
Step 1,350,000/2,000,000 | Ep 269 | ε=0.067 | Avg Reward: +0.4898
Step 1,360,000/2,000,000 | Ep 271 | ε=0.066 | Avg Reward: +0.4847
Step 1,370,000/2,000,000 | Ep 273 | ε=0.064 | Avg Reward: +0.4931
Step 1,380,000/2,000,000 | Ep 275 | ε=0.063 | Avg Reward: +0.4927
Step 1,390,000/2,000,000 | Ep 277 | ε=0.062 | Avg Reward: +0.4880
Step 1,400,000/2,000,000 | Ep 279 | ε=0.061 | Avg Reward: +0.4929
  └─ Episode 280: Reward=+0.4828 | Portfolio=$16,184 | Trades=1416 | ε=0.060
Step 1,410,000/2,000,000 | Ep 281 | ε=0.059 | Avg Reward: +0.4909
Step 1,420,000/2,000,000 | Ep 283 | ε=0.058 | Avg Reward: +0.4857
Step 1,430,000/2,000,000 | Ep 285 | ε=0.057 | Avg Reward: +0.4912
Step 1,440,000/2,000,000 | Ep 287 | ε=0.056 | Avg Reward: +0.4992
Step 1,450,000/2,000,000 | Ep 289 | ε=0.055 | Avg Reward: +0.5093
Step 1,460,000/2,000,000 | Ep 291 | ε=0.054 | Avg Reward: +0.5161
Step 1,470,000/2,000,000 | Ep 293 | ε=0.053 | Avg Reward: +0.5241
Step 1,480,000/2,000,000 | Ep 295 | ε=0.052 | Avg Reward: +0.5231
Step 1,490,000/2,000,000 | Ep 297 | ε=0.051 | Avg Reward: +0.5234
Step 1,500,000/2,000,000 | Ep 299 | ε=0.050 | Avg Reward: +0.5169
  └─ Episode 300: Reward=+0.5260 | Portfolio=$16,900 | Trades=1388 | ε=0.049
Step 1,510,000/2,000,000 | Ep 301 | ε=0.049 | Avg Reward: +0.5131
Step 1,520,000/2,000,000 | Ep 303 | ε=0.048 | Avg Reward: +0.5060
Step 1,530,000/2,000,000 | Ep 305 | ε=0.047 | Avg Reward: +0.4990
Step 1,540,000/2,000,000 | Ep 307 | ε=0.046 | Avg Reward: +0.4931
Step 1,550,000/2,000,000 | Ep 309 | ε=0.045 | Avg Reward: +0.4882
Step 1,560,000/2,000,000 | Ep 311 | ε=0.044 | Avg Reward: +0.4895
Step 1,570,000/2,000,000 | Ep 313 | ε=0.043 | Avg Reward: +0.4990
Step 1,580,000/2,000,000 | Ep 315 | ε=0.042 | Avg Reward: +0.5069
Step 1,590,000/2,000,000 | Ep 317 | ε=0.041 | Avg Reward: +0.5087
Step 1,600,000/2,000,000 | Ep 319 | ε=0.041 | Avg Reward: +0.5112
  └─ Episode 320: Reward=+0.5394 | Portfolio=$17,126 | Trades=1352 | ε=0.040
Step 1,610,000/2,000,000 | Ep 321 | ε=0.040 | Avg Reward: +0.5164
Step 1,620,000/2,000,000 | Ep 323 | ε=0.039 | Avg Reward: +0.5172
Step 1,630,000/2,000,000 | Ep 325 | ε=0.038 | Avg Reward: +0.5205
Step 1,640,000/2,000,000 | Ep 327 | ε=0.037 | Avg Reward: +0.5275
Step 1,650,000/2,000,000 | Ep 329 | ε=0.037 | Avg Reward: +0.5347
Step 1,660,000/2,000,000 | Ep 331 | ε=0.036 | Avg Reward: +0.5315
Step 1,670,000/2,000,000 | Ep 333 | ε=0.035 | Avg Reward: +0.5291
Step 1,680,000/2,000,000 | Ep 335 | ε=0.034 | Avg Reward: +0.5261
Step 1,690,000/2,000,000 | Ep 337 | ε=0.034 | Avg Reward: +0.5248
Step 1,700,000/2,000,000 | Ep 339 | ε=0.033 | Avg Reward: +0.5221
  └─ Episode 340: Reward=+0.4950 | Portfolio=$16,385 | Trades=1228 | ε=0.033
Step 1,710,000/2,000,000 | Ep 341 | ε=0.032 | Avg Reward: +0.5181
Step 1,720,000/2,000,000 | Ep 343 | ε=0.032 | Avg Reward: +0.5137
Step 1,730,000/2,000,000 | Ep 345 | ε=0.031 | Avg Reward: +0.5115
Step 1,740,000/2,000,000 | Ep 347 | ε=0.031 | Avg Reward: +0.5083
Step 1,750,000/2,000,000 | Ep 349 | ε=0.030 | Avg Reward: +0.5077
Step 1,760,000/2,000,000 | Ep 351 | ε=0.029 | Avg Reward: +0.5116
Step 1,770,000/2,000,000 | Ep 353 | ε=0.029 | Avg Reward: +0.5178
Step 1,780,000/2,000,000 | Ep 355 | ε=0.028 | Avg Reward: +0.5207
Step 1,790,000/2,000,000 | Ep 357 | ε=0.028 | Avg Reward: +0.5264
Step 1,800,000/2,000,000 | Ep 359 | ε=0.027 | Avg Reward: +0.5316
  └─ Episode 360: Reward=+0.5231 | Portfolio=$16,852 | Trades=1322 | ε=0.027
Step 1,810,000/2,000,000 | Ep 361 | ε=0.027 | Avg Reward: +0.5346
Step 1,820,000/2,000,000 | Ep 363 | ε=0.026 | Avg Reward: +0.5347
Step 1,830,000/2,000,000 | Ep 365 | ε=0.026 | Avg Reward: +0.5378
Step 1,840,000/2,000,000 | Ep 367 | ε=0.025 | Avg Reward: +0.5349
Step 1,850,000/2,000,000 | Ep 369 | ε=0.025 | Avg Reward: +0.5362
Step 1,860,000/2,000,000 | Ep 371 | ε=0.024 | Avg Reward: +0.5378
Step 1,870,000/2,000,000 | Ep 373 | ε=0.024 | Avg Reward: +0.5376
Step 1,880,000/2,000,000 | Ep 375 | ε=0.023 | Avg Reward: +0.5385
Step 1,890,000/2,000,000 | Ep 377 | ε=0.023 | Avg Reward: +0.5404
Step 1,900,000/2,000,000 | Ep 379 | ε=0.022 | Avg Reward: +0.5395
  └─ Episode 380: Reward=+0.5341 | Portfolio=$17,036 | Trades=1336 | ε=0.022
Step 1,910,000/2,000,000 | Ep 381 | ε=0.022 | Avg Reward: +0.5369
Step 1,920,000/2,000,000 | Ep 383 | ε=0.021 | Avg Reward: +0.5374
Step 1,930,000/2,000,000 | Ep 385 | ε=0.021 | Avg Reward: +0.5365
Step 1,940,000/2,000,000 | Ep 387 | ε=0.020 | Avg Reward: +0.5395
Step 1,950,000/2,000,000 | Ep 389 | ε=0.020 | Avg Reward: +0.5388
Step 1,960,000/2,000,000 | Ep 391 | ε=0.020 | Avg Reward: +0.5415
Step 1,970,000/2,000,000 | Ep 393 | ε=0.019 | Avg Reward: +0.5391
Step 1,980,000/2,000,000 | Ep 395 | ε=0.019 | Avg Reward: +0.5408
Step 1,990,000/2,000,000 | Ep 397 | ε=0.019 | Avg Reward: +0.5407
Step 2,000,000/2,000,000 | Ep 399 | ε=0.018 | Avg Reward: +0.5441
  └─ Episode 400: Reward=+0.2612 | Portfolio=$12,976 | Trades=750 | ε=0.018

============================================================
TRAINING COMPLETE!
============================================================
Episodes: 400
Final Epsilon: 0.0180
Mean Reward: 0.3816
Best Episode Reward: 0.5665
============================================================

======================================================================
 STEP 5: SAVING MODEL
======================================================================
Model saved to results/q_learning_v2_20251213_021252.pkl
Config saved to results/config_v2_20251213_021252.json

======================================================================
 STEP 6: EVALUATION ON TRAINING DATA
======================================================================
Initial: $10,000.00
Final:   $17,539.52
Return:  +75.40%
Trades:  1420
Fees:    $0.00

======================================================================
 STEP 7: EVALUATION ON TEST DATA
======================================================================
 Using REAL prices for trading: $99343.85 - $106535.13
   Price change over period: 2.8%
   Using 21 features for state
Initial: $10,000.00
Final:   $10,056.71
Return:  +0.57%
Trades:  342
Fees:    $0.00

======================================================================
 STEP 8: BUY & HOLD COMPARISON
======================================================================

Strategy             |        Train |         Test
--------------------------------------------------
Q-Learning Agent     |      +75.40% |       +0.57%
Buy & Hold           |       -6.61% |       +2.84%
--------------------------------------------------
Outperformance       |      +82.00% |       -2.27%

 Final Values:
   Agent Train: $17,539.52  |  Buy&Hold: $9,339.32
   Agent Test:  $10,056.71  |  Buy&Hold: $10,283.92

======================================================================
 TRAINING COMPLETE!
======================================================================
Model: results/q_learning_v2_20251213_021252.pkl
Final Epsilon: 0.0180
Total Episodes: 400

 Agent made profit, but didn't beat Buy & Hold
PS C:\Users\haris\OneDrive\Anlagen\Desktop\RL trading bot\Reinforcement-Learning-Trading-Bot-RLTrader->
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2025-10-20',
            'end_date': '2025-11-11',
            'interval': '5m',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.00,
            'trading_fee_taker': 0.00,
            'slippage': 0.00,
            'trade_frequency_penalty': 0.00  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 2000000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }
  
v27
config = {
        'data': {
            'symbol': 'BTC-USD',
            'start_date': '2025-10-10',
            'end_date': '2025-11-11',
            'interval': '1h',
            'test_split': 0.2
        },
        'environment': {
            'initial_cash': 10000.0,
            'trading_fee_maker': 0.001,
            'trading_fee_taker': 0.002,
            'slippage': 0.001,
            'trade_frequency_penalty': 0.002  # REDUZIERT!
        },
        'q_learning': {
            'learning_rate': 0.1,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.99,  # SCHNELLER! (war 0.995)
            'n_bins': 15
        },
        'training': {
            'total_timesteps': 1500000,  # MEHR! (war 50000)
            'log_interval': 10000
        }
    }