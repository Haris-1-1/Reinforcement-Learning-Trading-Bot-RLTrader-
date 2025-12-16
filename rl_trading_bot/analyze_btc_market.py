"""
Bitcoin Market Analysis 2018-2025
Understand market regimes for optimal RL training
WITH BEAUTIFUL VISUALIZATIONS!
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

# Try plotly first (interactive), fallback to matplotlib
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available - install with: pip install plotly")
    print("   Continuing without interactive charts...")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  Matplotlib not available - install with: pip install matplotlib")
    print("   Continuing without static charts...")


def create_visualizations(data, events, yearly_stats, splits, output_dir='results'):
    """Create all visualizations"""
    
    if not PLOTLY_AVAILABLE:
        print("\n⚠️  Plotly not installed - skipping visualizations")
        print("   Install with: pip install plotly")
        return
    
    print("\n" + "="*80)
    print("🎨 CREATING VISUALIZATIONS")
    print("="*80)
    
    # 1. Main price chart with regimes
    print("\n📊 Creating main price chart...")
    fig1 = go.Figure()
    
    # Price line
    fig1.add_trace(go.Scatter(
        x=data['Datetime'],
        y=data['Close'],
        name='BTC Price',
        line=dict(color='#f7931a', width=2),
        hovertemplate='<b>%{x}</b><br>Price: $%{y:,.0f}<extra></extra>'
    ))
    
    # Add regime backgrounds
    regime_colors = {
        'BULL': 'rgba(0, 255, 0, 0.1)',
        'BEAR': 'rgba(255, 0, 0, 0.1)',
        'CRASH': 'rgba(255, 0, 0, 0.2)',
        'PEAK': 'rgba(255, 165, 0, 0.1)'
    }
    
    for event_name, event_data in events.items():
        fig1.add_vrect(
            x0=event_data['start'],
            x1=event_data['end'],
            fillcolor=regime_colors.get(event_data['type'], 'rgba(128, 128, 128, 0.1)'),
            layer="below",
            line_width=0,
            annotation_text=event_name,
            annotation_position="top left"
        )
    
    fig1.update_layout(
        title='📊 Bitcoin Price 2018-2025 with Market Regimes',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis_type='log',
        height=600,
        template='plotly_dark',
        hovermode='x unified'
    )
    
    html1 = f'{output_dir}/btc_price_chart.html'
    fig1.write_html(html1)
    print(f"✓ Saved: {html1}")
    
    # 2. Yearly comparison
    print("\n📊 Creating yearly comparison...")
    df = pd.DataFrame(yearly_stats)
    
    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Annual Returns %', 'Volatility %', 'Max Drawdown %', 'Price Range'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}]]
    )
    
    # Returns
    colors = ['green' if x > 0 else 'red' for x in df['return_pct']]
    fig2.add_trace(go.Bar(
        x=df['year'], y=df['return_pct'], marker_color=colors,
        text=[f"{x:+.0f}%" for x in df['return_pct']], textposition='outside',
        showlegend=False
    ), row=1, col=1)
    
    # Volatility
    fig2.add_trace(go.Bar(
        x=df['year'], y=df['volatility'], marker_color='orange',
        text=[f"{x:.0f}%" for x in df['volatility']], textposition='outside',
        showlegend=False
    ), row=1, col=2)
    
    # Max DD
    fig2.add_trace(go.Bar(
        x=df['year'], y=df['max_drawdown'], marker_color='red',
        text=[f"{x:.0f}%" for x in df['max_drawdown']], textposition='outside',
        showlegend=False
    ), row=2, col=1)
    
    # Price range
    fig2.add_trace(go.Scatter(
        x=df['year'], y=df['max_price'], name='Max',
        mode='lines+markers', line=dict(color='green', width=2)
    ), row=2, col=2)
    fig2.add_trace(go.Scatter(
        x=df['year'], y=df['min_price'], name='Min',
        mode='lines+markers', line=dict(color='red', width=2),
        fill='tonexty', fillcolor='rgba(255,0,0,0.1)'
    ), row=2, col=2)
    
    fig2.update_layout(
        title='📊 Bitcoin Market Metrics by Year',
        height=800,
        template='plotly_dark'
    )
    fig2.update_yaxes(title_text="Return %", row=1, col=1)
    fig2.update_yaxes(title_text="Volatility %", row=1, col=2)
    fig2.update_yaxes(title_text="Max DD %", row=2, col=1)
    fig2.update_yaxes(title_text="Price", type="log", row=2, col=2)
    
    html2 = f'{output_dir}/btc_yearly_metrics.html'
    fig2.write_html(html2)
    print(f"✓ Saved: {html2}")
    
    # 3. Train/Test splits
    print("\n📊 Creating train/test split visualization...")
    fig3 = make_subplots(
        rows=len(splits), cols=1,
        shared_xaxes=True,
        subplot_titles=[s['name'] for s in splits],
        vertical_spacing=0.05
    )
    
    for idx, split_config in enumerate(splits, 1):
        # Gray background (all data)
        fig3.add_trace(go.Scatter(
            x=data['Datetime'], y=data['Close'],
            name='All Data', line=dict(color='gray', width=1),
            showlegend=(idx==1)
        ), row=idx, col=1)
        
        # Train (blue)
        train_mask = (data['Datetime'] >= split_config['train'][0]) & (data['Datetime'] <= split_config['train'][1])
        fig3.add_trace(go.Scatter(
            x=data[train_mask]['Datetime'], y=data[train_mask]['Close'],
            name='Train', line=dict(color='blue', width=3),
            showlegend=(idx==1)
        ), row=idx, col=1)
        
        # Test (red)
        test_mask = (data['Datetime'] >= split_config['test'][0]) & (data['Datetime'] <= split_config['test'][1])
        fig3.add_trace(go.Scatter(
            x=data[test_mask]['Datetime'], y=data[test_mask]['Close'],
            name='Test', line=dict(color='red', width=3),
            showlegend=(idx==1)
        ), row=idx, col=1)
        
        fig3.update_yaxes(title_text="Price", type="log", row=idx, col=1)
    
    fig3.update_layout(
        title='🎯 Train/Test Split Comparison<br><sub>Blue = Training, Red = Testing</sub>',
        height=250 * len(splits),
        template='plotly_dark',
        hovermode='x unified'
    )
    fig3.update_xaxes(title_text="Date", row=len(splits), col=1)
    
    html3 = f'{output_dir}/btc_train_test_splits.html'
    fig3.write_html(html3)
    print(f"✓ Saved: {html3}")
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS CREATED!")
    print("="*80)
    print(f"\nOpen these files in your browser:")
    print(f"  1. {html1}")
    print(f"  2. {html2}")
    print(f"  3. {html3}")
    print("="*80)


def analyze_btc_market():
    """Comprehensive BTC market analysis"""
    
    print("="*80)
    print("📊 BITCOIN MARKET ANALYSIS 2018-2025")
    print("="*80)
    print("Goal: Find optimal train/test periods for RL trading bot")
    print("="*80 + "\n")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: LOAD FULL HISTORY
    # ═══════════════════════════════════════════════════════════════
    print("📥 STEP 1: Loading BTC data (2018-2025)...")
    
    data = yf.download(
        'BTC-USD',
        start='2018-01-01',
        end='2025-12-16',
        interval='1d',
        progress=False
    )
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data = data.reset_index()
    if 'Date' in data.columns:
        data = data.rename(columns={'Date': 'Datetime'})
    
    print(f"✓ Loaded {len(data)} days of data")
    print(f"  Date Range: {data['Datetime'].min()} to {data['Datetime'].max()}")
    print(f"  Price Range: ${data['Close'].min():,.0f} - ${data['Close'].max():,.0f}")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: IDENTIFY MAJOR EVENTS & REGIMES
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("📈 STEP 2: Identifying Market Events & Regimes")
    print("="*80)
    
    events = {
        '2018 Bear Market': {'start': '2018-01-01', 'end': '2018-12-31', 'type': 'BEAR'},
        '2019 Recovery': {'start': '2019-01-01', 'end': '2019-12-31', 'type': 'BULL'},
        '2020 COVID Crash': {'start': '2020-03-01', 'end': '2020-03-31', 'type': 'CRASH'},
        '2020 Bull Start': {'start': '2020-10-01', 'end': '2020-12-31', 'type': 'BULL'},
        '2021 Bull Run': {'start': '2021-01-01', 'end': '2021-11-30', 'type': 'BULL'},
        '2021 Peak & Crash': {'start': '2021-11-01', 'end': '2022-01-31', 'type': 'PEAK'},
        '2022 Bear Market': {'start': '2022-01-01', 'end': '2022-12-31', 'type': 'BEAR'},
        '2023 Recovery': {'start': '2023-01-01', 'end': '2023-12-31', 'type': 'BULL'},
        '2024 Halving Rally': {'start': '2024-01-01', 'end': '2024-12-31', 'type': 'BULL'},
        '2025 New ATH': {'start': '2025-01-01', 'end': '2025-12-15', 'type': 'BULL'}
    }
    
    print("\n🎯 MAJOR MARKET EVENTS:")
    print("-" * 80)
    
    for event_name, event_data in events.items():
        mask = (data['Datetime'] >= event_data['start']) & (data['Datetime'] <= event_data['end'])
        period_data = data[mask]
        
        if len(period_data) > 0:
            start_price = period_data.iloc[0]['Close']
            end_price = period_data.iloc[-1]['Close']
            return_pct = ((end_price - start_price) / start_price) * 100
            
            min_price = period_data['Close'].min()
            max_price = period_data['Close'].max()
            volatility = period_data['Close'].pct_change().std() * np.sqrt(252) * 100
            
            regime_emoji = {
                'BULL': '🟢',
                'BEAR': '🔴',
                'CRASH': '💥',
                'PEAK': '⚠️'
            }
            
            print(f"{regime_emoji.get(event_data['type'], '⚪')} {event_name:20s}")
            print(f"   Price: ${start_price:>8,.0f} → ${end_price:>8,.0f} ({return_pct:+6.1f}%)")
            print(f"   Range: ${min_price:>8,.0f} - ${max_price:>8,.0f}")
            print(f"   Volatility: {volatility:.1f}%")
            print()
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: YEARLY ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("="*80)
    print("📅 STEP 3: Year-by-Year Breakdown")
    print("="*80 + "\n")
    
    data['Year'] = pd.to_datetime(data['Datetime']).dt.year
    
    yearly_stats = []
    
    for year in sorted(data['Year'].unique()):
        year_data = data[data['Year'] == year]
        
        start_price = year_data.iloc[0]['Close']
        end_price = year_data.iloc[-1]['Close']
        min_price = year_data['Close'].min()
        max_price = year_data['Close'].max()
        
        return_pct = ((end_price - start_price) / start_price) * 100
        max_drawdown = ((year_data['Close'].cummax() - year_data['Close']) / year_data['Close'].cummax()).max() * 100
        volatility = year_data['Close'].pct_change().std() * np.sqrt(252) * 100
        
        regime = '🟢 BULL' if return_pct > 20 else ('🔴 BEAR' if return_pct < -20 else '⚪ SIDEWAYS')
        
        yearly_stats.append({
            'year': year,
            'start_price': start_price,
            'end_price': end_price,
            'min_price': min_price,
            'max_price': max_price,
            'return_pct': return_pct,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'regime': regime
        })
        
        print(f"{year} {regime}")
        print(f"  Price:   ${start_price:>8,.0f} → ${end_price:>8,.0f}")
        print(f"  Return:  {return_pct:+.1f}%")
        print(f"  Range:   ${min_price:>8,.0f} - ${max_price:>8,.0f}")
        print(f"  Max DD:  {max_drawdown:.1f}%")
        print(f"  Vol:     {volatility:.1f}%")
        print()
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: TRAIN/TEST SPLIT ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("="*80)
    print("🎯 STEP 4: Train/Test Split Analysis")
    print("="*80)
    print("\nProblem: Agent trained on one regime, tested on different regime!")
    print()
    
    # Common splits people try
    splits = [
        {'name': '2020-2024 train, 2025 test', 'train': ('2020-01-01', '2024-12-31'), 'test': ('2025-01-01', '2025-12-15')},
        {'name': '2022-2024 train, 2025 test', 'train': ('2022-01-01', '2024-12-31'), 'test': ('2025-01-01', '2025-12-15')},
        {'name': '2018-2023 train, 2024-2025 test', 'train': ('2018-01-01', '2023-12-31'), 'test': ('2024-01-01', '2025-12-15')},
        {'name': '80/20 chronological', 'train': ('2018-01-01', '2023-09-30'), 'test': ('2023-10-01', '2025-12-15')},
    ]
    
    print("⚠️  SPLIT MISMATCH ANALYSIS:")
    print("-" * 80)
    
    for split_config in splits:
        print(f"\n📌 {split_config['name']}")
        
        # Train stats
        train_mask = (data['Datetime'] >= split_config['train'][0]) & (data['Datetime'] <= split_config['train'][1])
        train_data = data[train_mask]
        
        train_min = train_data['Close'].min()
        train_max = train_data['Close'].max()
        train_return = ((train_data.iloc[-1]['Close'] - train_data.iloc[0]['Close']) / train_data.iloc[0]['Close']) * 100
        
        # Test stats
        test_mask = (data['Datetime'] >= split_config['test'][0]) & (data['Datetime'] <= split_config['test'][1])
        test_data = data[test_mask]
        
        test_min = test_data['Close'].min()
        test_max = test_data['Close'].max()
        test_return = ((test_data.iloc[-1]['Close'] - test_data.iloc[0]['Close']) / test_data.iloc[0]['Close']) * 100
        
        # Check overlap
        train_range = (train_min, train_max)
        test_range = (test_min, test_max)
        
        # How much of test is outside train range?
        out_of_dist = 0
        if test_min < train_min:
            out_of_dist += (train_min - test_min) / train_min * 100
        if test_max > train_max:
            out_of_dist += (test_max - train_max) / train_max * 100
        
        print(f"  TRAIN: {len(train_data)} days, ${train_min:,.0f}-${train_max:,.0f}, {train_return:+.1f}%")
        print(f"  TEST:  {len(test_data)} days, ${test_min:,.0f}-${test_max:,.0f}, {test_return:+.1f}%")
        
        if out_of_dist > 20:
            print(f"  ⚠️  HIGH RISK: Test prices {out_of_dist:.0f}% outside train range!")
            print(f"      → Agent will see UNKNOWN price levels!")
        elif out_of_dist > 10:
            print(f"  ⚠️  MODERATE RISK: Test prices {out_of_dist:.0f}% outside train range")
        else:
            print(f"  ✅ LOW RISK: Test prices mostly within train range")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 5: RECOMMENDATIONS
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*80)
    print("💡 STEP 5: Recommendations")
    print("="*80 + "\n")
    
    print("🎯 OPTIMAL TRAINING STRATEGIES:\n")
    
    print("1️⃣  INCLUDE MULTIPLE REGIMES (BEST)")
    print("   Train: 2018-2024 (includes Bear, Bull, Crash, Recovery)")
    print("   Test:  2025")
    print("   ✅ Agent sees all market conditions")
    print("   ✅ Better generalization")
    print("   ⚠️  Longer training time")
    print()
    
    print("2️⃣  SIMILAR REGIME MATCHING")
    print("   Train: 2023-2024 (Bull market)")
    print("   Test:  2025 (Also Bull)")
    print("   ✅ Test similar to train")
    print("   ❌ Agent only knows Bull markets (fails in Bear)")
    print()
    
    print("3️⃣  ROLLING WINDOW")
    print("   Train: Last 2 years (e.g., 2023-2024)")
    print("   Test:  Next 6 months")
    print("   Re-train every 6 months")
    print("   ✅ Always uses recent data")
    print("   ⚠️  Needs regular re-training")
    print()
    
    print("4️⃣  WALK-FORWARD")
    print("   Multiple train/test periods:")
    print("   - Train 2018-2020, Test 2021")
    print("   - Train 2019-2021, Test 2022")
    print("   - Train 2020-2022, Test 2023")
    print("   - Train 2021-2023, Test 2024")
    print("   - Train 2022-2024, Test 2025")
    print("   ✅ Most robust evaluation")
    print("   ✅ Tests generalization across regimes")
    print("   ⚠️  Time-consuming")
    print()
    
    print("="*80)
    print("🎖️  RECOMMENDED FOR YOUR BOT:")
    print("="*80)
    print("\n📌 Option A (Best Generalization):")
    print("   start_date: '2018-01-01'")
    print("   end_date:   '2025-12-15'")
    print("   test_split: 0.15  # Last 15% = ~1 year test")
    print("   ")
    print("   Why: Agent learns Bear (2018, 2022), Bull (2019-2021, 2023-2025), Crash (2020)")
    print()
    
    print("📌 Option B (Faster, Still Good):")
    print("   start_date: '2020-01-01'")
    print("   end_date:   '2025-12-15'")
    print("   test_split: 0.2")
    print("   ")
    print("   Why: Covers COVID, Bear market 2022, Recovery 2023-2025")
    print()
    
    print("📌 Option C (Conservative - Match Regimes):")
    print("   start_date: '2023-01-01'")
    print("   end_date:   '2025-12-15'")
    print("   test_split: 0.2")
    print("   ")
    print("   Why: Both train and test are Bull markets (similar distribution)")
    print("   ⚠️  But: Agent won't handle Bear markets well!")
    print()
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 6: SAVE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print("="*80)
    print("💾 STEP 6: Saving Analysis")
    print("="*80 + "\n")
    
    # Save yearly stats
    df_yearly = pd.DataFrame(yearly_stats)
    df_yearly.to_csv('results/btc_yearly_analysis.csv', index=False)
    print("✓ Saved yearly stats to results/btc_yearly_analysis.csv")
    
    # Save price data
    data.to_csv('results/btc_full_history_2018_2025.csv', index=False)
    print("✓ Saved full price history to results/btc_full_history_2018_2025.csv")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 7: CREATE VISUALIZATIONS
    # ═══════════════════════════════════════════════════════════════
    create_visualizations(data, events, yearly_stats, splits, output_dir='results')
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE!")
    print("="*80)
    print("\nKey Takeaway:")
    print("  Your agent struggles because 2025 prices ($90k-$100k+) are")
    print("  MUCH HIGHER than what it saw in training!")
    print()
    print("  Solution: Train on data that INCLUDES high prices (2018-2025)")
    print("  or at least 2020-2025 to see the full $7k-$100k range.")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    analyze_btc_market()