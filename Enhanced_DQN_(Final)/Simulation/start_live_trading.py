#!/usr/bin/env python3
"""
QUICK START - Live Paper Trading
=================================
Simple launcher for live paper trading
"""

import os
import sys
import subprocess

def main():
    print("\n" + "="*70)
    print("  LIVE PAPER TRADING - QUICK START")
    print("="*70)
    
    # Check if model exists
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'best_model.pth')

    if not os.path.exists(model_path):
        print("\nERROR: Trained model not found!")
        print(f"Looking for model at: {model_path}")
        print("\nYou need to train the model first:")
        print("  -> python train_enhanced_dqn.py")
        print("\nAfter training completes, run this script again.")
        return

    print(f"\nModel found: {model_path}")
    print("\nWhat would you like to do?")
    print("="*70)
    print("\n1. Start Live Trading (Console)")
    print("2. Start Dashboard (Real-time Charts)")
    print("3. Both (Trading + Dashboard in separate windows)")
    print("4. View Trading Guide")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == '1':
        print("\nStarting Live Trading...")
        subprocess.run([sys.executable, 'live_paper_trading.py'])

    elif choice == '2':
        print("\nStarting Dashboard...")
        print("Note: Live trading must be running for data to appear!")
        subprocess.run([sys.executable, 'live_dashboard.py'])

    elif choice == '3':
        print("\nStarting both...")
        print("\nThis will open:")
        print("  1. Trading bot in this window")
        print("  2. Dashboard in a new window")
        print("\nPress Enter to continue...")
        input()
        
        # Start dashboard in background
        if os.name == 'nt':  # Windows
            subprocess.Popen(['start', 'cmd', '/c', f'{sys.executable} live_dashboard.py'], shell=True)
        else:  # Mac/Linux
            subprocess.Popen([sys.executable, 'live_dashboard.py'])
        
        # Start trading in foreground
        subprocess.run([sys.executable, 'live_paper_trading.py'])
    
    elif choice == '4':
        print("\nOpening Trading Guide...")
        if os.path.exists('LIVE_TRADING_GUIDE.txt'):
            with open('LIVE_TRADING_GUIDE.txt', 'r', encoding='utf-8') as f:
                print(f.read())
        else:
            print("Guide not found!")

    elif choice == '5':
        print("\nGoodbye!")
        return

    else:
        print("\nInvalid choice!")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
