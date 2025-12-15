"""
Debug script to diagnose yfinance column issues
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

print("="*70)
print("YFinance Column Debug Script")
print("="*70)

# Download data
ticker = 'BTC-USD'
start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

print(f"\nDownloading {ticker} from {start_date} to {end_date}...")

data = yf.download(
    ticker,
    start=start_date,
    end=end_date,
    interval='1d',
    progress=False,
    auto_adjust=True
)

print(f"\n1. RAW DATA AFTER DOWNLOAD:")
print(f"   Shape: {data.shape}")
print(f"   Index type: {type(data.index)}")
print(f"   Columns type: {type(data.columns)}")
print(f"   Is MultiIndex: {isinstance(data.columns, pd.MultiIndex)}")
print(f"   Columns: {list(data.columns)}")

# Reset index
print(f"\n2. AFTER reset_index():")
data = data.reset_index()
print(f"   Shape: {data.shape}")
print(f"   Columns type: {type(data.columns)}")
print(f"   Is MultiIndex: {isinstance(data.columns, pd.MultiIndex)}")
print(f"   Columns: {list(data.columns)}")
print(f"   First column type: {type(data.columns[0])}")
if len(data.columns) > 1:
    print(f"   Second column type: {type(data.columns[1])}")

# Check if any are tuples
has_tuples = any(isinstance(col, tuple) for col in data.columns)
print(f"   Has tuple columns: {has_tuples}")

# Try flattening
print(f"\n3. FLATTENING ATTEMPT:")
if isinstance(data.columns, pd.MultiIndex):
    print("   Using MultiIndex.get_level_values(0)")
    data.columns = data.columns.get_level_values(0)
elif has_tuples:
    print("   Using list comprehension on tuples")
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
else:
    print("   No flattening needed")

print(f"   Columns after flatten: {list(data.columns)}")
print(f"   First column type: {type(data.columns[0])}")
if len(data.columns) > 1:
    print(f"   Second column type: {type(data.columns[1])}")

# Try to access Close column
print(f"\n4. ACCESSING 'Close' COLUMN:")
try:
    close_data = data['Close']
    print(f"   ✓ Success! Shape: {close_data.shape}, Type: {type(close_data)}")
    print(f"   First value: {close_data.iloc[0]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Try to calculate BB
print(f"\n5. CALCULATING BOLLINGER BANDS:")
try:
    bb_middle = data['Close'].rolling(window=20).mean()
    print(f"   BB_middle type: {type(bb_middle)}")
    print(f"   BB_middle shape: {bb_middle.shape}")
    
    bb_std = data['Close'].rolling(window=20).std()
    print(f"   BB_std type: {type(bb_std)}")
    print(f"   BB_std shape: {bb_std.shape}")
    
    bb_upper = bb_middle + (bb_std * 2)
    print(f"   BB_upper type: {type(bb_upper)}")
    print(f"   BB_upper shape: {bb_upper.shape}")
    
    data['BB_upper'] = bb_upper
    print(f"   ✓ Successfully set BB_upper!")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    print(f"\n   DEBUG INFO:")
    print(f"   data.columns = {data.columns}")
    print(f"   data.shape = {data.shape}")
    if 'Close' in data.columns:
        print(f"   data['Close'].shape = {data['Close'].shape}")

print("\n" + "="*70)
print("Debug complete!")
print("="*70)
