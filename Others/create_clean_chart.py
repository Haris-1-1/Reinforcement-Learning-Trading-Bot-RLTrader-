"""
Create Clean BTC Chart 2020-2025 with Market Regimes
Simple, clear, presentation-ready
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

def create_clean_chart():
    """Generate clean HTML chart for 2020-2025"""
    
    print("="*80)
    print("🎨 CREATING CLEAN BTC CHART 2020-2025")
    print("="*80)
    
    # Load data
    print("\n📊 Loading BTC data...")
    data = yf.download('BTC-USD', start='2020-01-01', end='2025-12-15', interval='1d', progress=False)
    
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data = data.reset_index()
    if 'Date' in data.columns:
        data = data.rename(columns={'Date': 'Datetime'})
    
    print(f"✓ Loaded {len(data)} days")
    print(f"  Range: ${data['Close'].min():,.0f} - ${data['Close'].max():,.0f}")
    
    # Define regimes with exact dates
    regimes = [
        {'name': '2020 COVID Crash', 'start': '2020-03-01', 'end': '2020-03-31', 'color': 'rgba(255, 0, 0, 0.15)', 'type': 'CRASH'},
        {'name': '2020 Bull Start', 'start': '2020-10-01', 'end': '2020-12-31', 'color': 'rgba(0, 255, 100, 0.10)', 'type': 'BULL'},
        {'name': '2021 Bull Run', 'start': '2021-01-01', 'end': '2021-11-10', 'color': 'rgba(0, 255, 100, 0.10)', 'type': 'BULL'},
        {'name': '2021 Peak & Crash', 'start': '2021-11-10', 'end': '2022-01-31', 'color': 'rgba(255, 100, 0, 0.15)', 'type': 'PEAK'},
        {'name': '2022 Bear Market', 'start': '2022-01-01', 'end': '2022-12-31', 'color': 'rgba(200, 0, 0, 0.15)', 'type': 'BEAR'},
        {'name': '2023 Recovery', 'start': '2023-01-01', 'end': '2023-12-31', 'color': 'rgba(0, 255, 100, 0.10)', 'type': 'BULL'},
        {'name': '2024 Halving Rally', 'start': '2024-01-01', 'end': '2024-12-31', 'color': 'rgba(0, 255, 100, 0.10)', 'type': 'BULL'},
        {'name': '2025 New ATH', 'start': '2025-01-01', 'end': '2025-12-15', 'color': 'rgba(0, 255, 100, 0.10)', 'type': 'BULL'}
    ]
    
    # Prepare data for Plotly
    dates = data['Datetime'].astype(str).tolist()
    prices = data['Close'].tolist()
    
    # Create HTML with embedded Plotly
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Bitcoin Price 2020-2025 with Market Regimes</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            margin: 0;
            padding: 20px;
            background: #0a0e27;
            font-family: Arial, sans-serif;
        }}
        #chart {{
            width: 100%;
            height: 90vh;
        }}
        .info {{
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }}
        .info h1 {{
            margin: 0 0 10px 0;
            color: #f7931a;
        }}
    </style>
</head>
<body>
    <div class="info">
        <h1>🪙 Bitcoin Price 2020-2025 with Market Regimes</h1>
        <p>Training Period: 2020-mid2024 | Test Period: mid2024-2025</p>
    </div>
    <div id="chart"></div>
    
    <script>
        // Data
        const dates = {dates};
        const prices = {prices};
        
        // Main price trace
        const trace = {{
            x: dates,
            y: prices,
            type: 'scatter',
            mode: 'lines',
            name: 'BTC-USD',
            line: {{
                color: '#f7931a',
                width: 2.5
            }},
            hovertemplate: '<b>%{{x}}</b><br>Price: $%{{y:,.0f}}<extra></extra>'
        }};
        
        // Layout with regime backgrounds
        const layout = {{
            title: {{
                text: '',
                font: {{ size: 24, color: 'white' }}
            }},
            plot_bgcolor: '#0a0e27',
            paper_bgcolor: '#0a0e27',
            font: {{ color: 'white' }},
            xaxis: {{
                title: 'Date',
                gridcolor: '#1a2332',
                showgrid: true
            }},
            yaxis: {{
                title: 'Price (USD)',
                type: 'log',
                gridcolor: '#1a2332',
                showgrid: true,
                tickformat: '$,.0f'
            }},
            hovermode: 'x unified',
            shapes: ["""
    
    # Add regime rectangles
    for regime in regimes:
        html += f"""
                {{
                    type: 'rect',
                    xref: 'x',
                    yref: 'paper',
                    x0: '{regime['start']}',
                    x1: '{regime['end']}',
                    y0: 0,
                    y1: 1,
                    fillcolor: '{regime['color']}',
                    line: {{ width: 0 }},
                    layer: 'below'
                }},"""
    
    html += """
            ],
            annotations: ["""
    
    # Add regime labels
    for regime in regimes:
        # Calculate middle date for label
        start_date = pd.to_datetime(regime['start'])
        end_date = pd.to_datetime(regime['end'])
        mid_date = start_date + (end_date - start_date) / 2
        
        html += f"""
                {{
                    x: '{mid_date.strftime('%Y-%m-%d')}',
                    y: 1.02,
                    xref: 'x',
                    yref: 'paper',
                    text: '{regime['name']}',
                    showarrow: false,
                    font: {{
                        size: 11,
                        color: 'white'
                    }},
                    bgcolor: 'rgba(0,0,0,0.5)',
                    borderpad: 4
                }},"""
    
    html += f"""
            ]
        }};
        
        // Config
        const config = {{
            responsive: true,
            displayModeBar: true,
            displaylogo: false,
            toImageButtonOptions: {{
                format: 'png',
                filename: 'btc_2020_2025',
                height: 1080,
                width: 1920,
                scale: 2
            }}
        }};
        
        // Plot
        Plotly.newPlot('chart', [trace], layout, config);
    </script>
</body>
</html>"""
    
    # Save
    output_file = 'results/btc_chart_2020_2025.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"\n✓ Chart saved to: {output_file}")
    print("\n" + "="*80)
    print("✅ CHART READY!")
    print("="*80)
    print(f"\nOpen '{output_file}' in your browser!")
    print("\nFeatures:")
    print("  • Clean design like your image")
    print("  • Color-coded regime backgrounds")
    print("  • Log scale for better visibility")
    print("  • Interactive (zoom, pan, hover)")
    print("  • Download button (save as PNG)")
    print("  • Full 2020-2025 coverage")
    print("\n🎨 Perfect for presentations!")
    print("="*80)


if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    create_clean_chart()
