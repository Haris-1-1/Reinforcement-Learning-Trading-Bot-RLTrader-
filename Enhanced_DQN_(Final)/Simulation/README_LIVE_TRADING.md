# LIVE PAPER TRADING SYSTEM

**Status:**  Ready for Testing

---

##  WHAT'S INCLUDED

This package contains everything you need for **live paper trading** with your trained DQN model:

### Files:
1. **live_paper_trading.py** - Main trading bot (runs in real-time)
2. **live_dashboard.py** - Real-time visualization dashboard
3. **start_live_trading.py** - Easy launcher (recommended!)
4. **LIVE_TRADING_GUIDE.txt** - Complete documentation

---

##  QUICK START (3 Steps)

### Step 1: Copy Files to Your Project
Place all 4 files in your `Enhanced_DQN_Fixed/` folder:

```
Enhanced_DQN_Fixed/
├── train_enhanced_dqn.py
├── models/
│   └── best_model.pth          ← Must exist!
├── live_paper_trading.py       ← NEW
├── live_dashboard.py           ← NEW
├── start_live_trading.py       ← NEW
└── LIVE_TRADING_GUIDE.txt      ← NEW
```

### Step 2: Ensure Model is Trained
```bash
# Check if model exists
ls models/best_model.pth

# If not found, train first:
python train_enhanced_dqn.py
```

### Step 3: Start Trading
```bash
# Option A: Use the launcher (easiest!)
python start_live_trading.py

# Option B: Run directly
python live_paper_trading.py
```

---

##  FEATURES

### Live Trading Bot (`live_paper_trading.py`)
-  Loads your trained DQN model
-  Fetches live market data every minute
-  Calculates all 42 technical indicators
-  Makes buy/sell decisions in real-time
-  Tracks portfolio value
-  Logs all trades to JSON
-  No real money at risk!

### Real-Time Dashboard (`live_dashboard.py`)
-  Live price chart with trade markers
-  Portfolio value evolution
-  P/L tracking
-  Current statistics panel
-  Auto-updates every 5 seconds

### Easy Launcher (`start_live_trading.py`)
-  One-click menu interface
-  Automatic model verification
-  Launch trading + dashboard together
-  View guide directly

---

##  USAGE OPTIONS

### Option 1: Quick Test (Recommended First)
```bash
python start_live_trading.py
# Choose: 1 (Start Live Trading)
# Then select: 1 (Quick Test - 5 minutes)
```

This runs a 5-minute test to verify everything works.

### Option 2: With Dashboard
```bash
# Terminal 1 (Trading)
python live_paper_trading.py

# Terminal 2 (Dashboard)
python live_dashboard.py
```

Or use the launcher and select option 3!

### Option 3: Long Session
For serious testing:
- Duration: 1-2 hours
- Interval: 1m or 5m
- Gives meaningful performance data

---

##  WHAT TO EXPECT

### During Trading:
```
[14:23:45] Iteration 1
──────────────────────────────────────────
 Fetching latest data...
 Current Price: $42,156.78
 BUY  @ $42,177.89 | Coins: 0.237231

 Portfolio: $10,000.00 | P/L: $0.00 (+0.00%)
   Position: 100.0% | Cash: $0.00 | Coins: 0.237231

⏳ Next update in 60s...
```

### Trade Actions:
-  **BUY** - Bot enters position
-  **SELL** - Bot exits position
-  **HOLD** - Bot does nothing

### Final Report:
```
╔═══════════════════════════════════════╗
║  LIVE TRADING SESSION COMPLETE        ║
╠═══════════════════════════════════════╣
║  Final Portfolio: $10,234.56          ║
║  Total Profit/Loss: +$234.56 (+2.35%) ║
║  Total Trades: 8                      ║
╚═══════════════════════════════════════╝
```

---

##  OUTPUT FILES

All trading data is saved to:
```
logs/live_trading_log.json
```

This contains:
- Every trade (timestamp, price, action)
- Portfolio value history
- Configuration used
- Final statistics

---

## ⚙️ CONFIGURATION

Edit `live_paper_trading.py` to customize:

```python
LIVE_CONFIG = {
    "symbol": "BTC-USD",        # Try ETH-USD, AAPL, etc.
    "interval": "1m",           # 1m, 5m, 15m, 1h
    "initial_cash": 10000.0,    # Starting capital
    "update_interval": 60,      # Seconds between checks
}
```

---

##  INTERPRETING RESULTS

### Good Signs 
- Positive P/L after fees
- Buys near local lows, sells near highs
- Reasonable trade frequency
- Smooth portfolio growth

### Warning Signs 
- Negative P/L (losing money)
- Too many trades (>100/hour)
- Buying high, selling low
- Wild portfolio swings

### Need More Data 
- P/L near 0% (±0.5%)
- Very few trades (<5/hour)
- Market might be ranging

---

## ️ TROUBLESHOOTING

### "Model not found"
 Train model first: `python train_enhanced_dqn.py`

### "Error fetching data"
 Check internet connection
 Try different symbol (ETH-USD)

### "Bot never trades"
 Model might be cautious (normal in stable markets)
 Try volatile market times

### Dashboard not updating
 Ensure trading bot is running
 Check `logs/live_trading_log.json` exists

---

##  FULL DOCUMENTATION

For complete details, read:
```
LIVE_TRADING_GUIDE.txt
```

Or view it via the launcher (option 4).

---

##  IMPORTANT DISCLAIMERS

1. **This is PAPER TRADING ONLY**
   - No real money is used
   - Simulates trading with fake cash
   - For testing and learning

2. **Not Financial Advice**
   - Educational purposes only
   - Trading is risky
   - Past performance ≠ future results

3. **Paper ≠ Real Trading**
   - Real trading has emotions, slippage, latency
   - Paper success doesn't guarantee real success
   - Always start small if going to real money

---

##  RECOMMENDED WORKFLOW

1. **Quick Test** (5 minutes)
   - Verify system works
   - Check trades execute

2. **Short Session** (30 minutes)
   - See basic patterns
   - Evaluate performance

3. **Normal Session** (1-2 hours)
   - Get meaningful data
   - Calculate metrics

4. **Multiple Days** (if results good)
   - Test different market conditions
   - Different times of day
   - Different symbols

5. **Analysis**
   - Review logs
   - Calculate win rate, Sharpe ratio
   - Compare with backtest

6. **Decision**
   - If consistently profitable  Consider longer paper trading
   - If unprofitable  Retrain model or adjust parameters

---

##  SUPPORT

If you have issues:
1. Check `LIVE_TRADING_GUIDE.txt` (troubleshooting section)
2. Verify all dependencies installed
3. Ensure model trained successfully
4. Check internet connection

---

##  NEXT STEPS

After successful paper trading:

1.  Run on different symbols (ETH, stocks, etc.)
2.  Test different time intervals
3.  Analyze trade logs
4.  Calculate performance metrics
5.  Document patterns
6.  Only then consider real money (start tiny!)

---

**READY TO START?**

```bash
python start_live_trading.py
```

Good luck! 

---

*Remember: This is for educational purposes. Always do your own research and never risk money you can't afford to lose.*
