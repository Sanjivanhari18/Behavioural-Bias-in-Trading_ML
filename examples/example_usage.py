"""Example usage of the Behavioral Trading Analysis System."""

import os
import sys

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from behavioral_trading import BehavioralAnalyzer

def main():
    print("="*80)
    print("BEHAVIORAL TRADING ANALYSIS SYSTEM - EXAMPLE")
    print("="*80)
    print()
    
    # Initialize analyzer
    print("Step 1: Initializing analyzer...")
    analyzer = BehavioralAnalyzer(n_clusters=3, baseline_window=30)
    
    # Load tradebook (CSV for prototype)
    print("\nStep 2: Loading tradebook...")
    try:
        analyzer.load_tradebook("examples/user_tradebook.csv", file_type="csv")
        print(f"[OK] Loaded {len(analyzer.trades)} trades")
    except Exception as e:
        print(f"[ERROR] Error loading tradebook: {e}")
        return
    
    # Enrich with market data (automatically handles multiple symbols)
    print("\nStep 3: Enriching with market data (this may take a moment)...")
    print("  Note: System will automatically fetch OHLCV for each unique symbol in your trades")
    try:
        # Pass None to automatically detect and process all symbols
        analyzer.enrich_with_market_data()  # No symbol needed - auto-detects from trades
        print("[OK] Market data enrichment complete")
    except Exception as e:
        print(f"[ERROR] Error enriching market data: {e}")
        print("  Note: This requires internet connection and valid stock symbols")
        return
    
    # Run analysis
    print("\nStep 4: Running behavioral analysis...")
    try:
        results = analyzer.analyze()
        print("[OK] Analysis complete")
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Generate report first (so dashboard can read clean files)
    print("\nStep 5: Generating report...")
    try:
        os.makedirs("output", exist_ok=True)
        report = analyzer.generate_report(results, output_file="output/behavioral_report.txt")
        print("[OK] Report saved to: output/behavioral_report.txt")
        print("\n" + "="*80)
        print("REPORT PREVIEW (first 800 characters):")
        print("="*80)
        print(report[:800] + "...\n")
    except Exception as e:
        print(f"[ERROR] Error generating report: {e}")
    
    # Generate visualizations (dashboard will read the newly generated reports)
    print("\nStep 6: Generating visualizations...")
    try:
        viz_files = analyzer.visualize(results, output_dir="output/")
        print(f"[OK] Visualizations saved:")
        for name, path in viz_files.items():
            print(f"  - {name}: {path}")
    except Exception as e:
        print(f"[ERROR] Error generating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # Explain a specific trade
    print("\nStep 7: Explaining a specific trade...")
    try:
        trade_idx = min(5, len(analyzer.features) - 1)
        explanation = analyzer.explain_trade(trade_idx)
        print(f"Trade #{trade_idx} explanation:")
        print(explanation)
    except Exception as e:
        print(f"[ERROR] Error explaining trade: {e}")
    
    # Get summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    try:
        summary = analyzer.get_summary()
        print(f"Total trades: {summary['total_trades']}")
        if 'pnl' in summary:
            print(f"Total P&L: ${summary['pnl']['total']:.2f}")
            print(f"Average P&L per trade: ${summary['pnl']['average']:.2f}")
            print(f"Win rate: {summary['pnl']['win_rate']:.1%}")
        if 'clusters' in summary:
            print(f"\nBehavioral clusters:")
            for cluster_id, count in summary['clusters'].items():
                print(f"  Cluster {cluster_id}: {count} trades")
    except Exception as e:
        print(f"Error getting summary: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nCheck the 'output/' directory for:")
    print("  - HTML visualizations (open in browser)")
    print("  - Text report with detailed findings")
    print("\nTo use your own data:")
    print("  1. Create a CSV with your trades")
    print("  2. Update the file path in this script")
    print("  3. Update the symbol in enrich_with_market_data()")
    print("  4. Run again!")

if __name__ == "__main__":
    main()

