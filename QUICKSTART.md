# Quick Start Guide

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -c "import behavioral_trading; print('Installation successful!')"
   ```

## Running the Example

### Option 1: Run the Example Script

```bash
python examples/example_usage.py
```

This will:
- Load the sample tradebook (`examples/sample_tradebook.csv`)
- Enrich with AAPL market data
- Run full behavioral analysis
- Generate visualizations in `output/` directory
- Generate a text report

### Option 2: Use Python Interactively

```python
from behavioral_trading import BehavioralAnalyzer

# Initialize analyzer
analyzer = BehavioralAnalyzer(n_clusters=3, baseline_window=30)

# Load your tradebook
analyzer.load_tradebook("examples/sample_tradebook.csv", file_type="csv")

# Enrich with market data (use the symbol you traded)
analyzer.enrich_with_market_data("AAPL")

# Run analysis
results = analyzer.analyze()

# Generate visualizations (saved as HTML files)
viz_files = analyzer.visualize(results, output_dir="output/")
print(f"Visualizations saved: {list(viz_files.keys())}")

# Generate text report
report = analyzer.generate_report(results, output_file="output/behavioral_report.txt")
print(report[:500])  # Print first 500 chars

# Explain a specific trade
print("\n" + analyzer.explain_trade(5))

# Get summary statistics
summary = analyzer.get_summary()
print(f"\nTotal trades: {summary['total_trades']}")
print(f"Win rate: {summary['pnl']['win_rate']:.1%}")
```

## Using Your Own Data

### CSV Format

Your CSV should have these columns:
- `date`: Trade date (YYYY-MM-DD format)
- `symbol`: Stock symbol (e.g., AAPL)
- `side`: 'buy' or 'sell'
- `quantity`: Number of shares
- `price`: Trade price
- `order_type`: (optional) Market, Limit, etc.
- `realized_pnl`: (optional) Realized profit/loss

Example:
```csv
date,symbol,side,quantity,price,order_type,realized_pnl
2024-01-02,AAPL,buy,10,150.25,Market,0
2024-01-05,AAPL,sell,10,152.50,Market,22.50
```

### PDF Format

For PDF broker statements:
```python
analyzer.load_tradebook("your_tradebook.pdf", file_type="pdf")
```

The system will attempt to extract tables automatically. If your broker's PDF format is different, you may need to adjust the column mapping in `behavioral_trading/stage1_data/pdf_loader.py`.

## Output Files

After running analysis, you'll find:

### Visualizations (HTML files in `output/` directory)
- `regime_timeline.html`: Behavioral clusters over time with change points
- `deviation_plots.html`: Baseline deviation analysis
- `post_event.html`: Post-loss behavior patterns
- `performance_matrix.html`: Performance by market regime × behavioral cluster

### Reports
- `behavioral_report.txt`: Comprehensive text report with all findings

## Troubleshooting

### Issue: "No data found for symbol"
- Make sure you're using the correct stock symbol
- Check your internet connection (Yahoo Finance requires internet)
- Try a different symbol to test

### Issue: "Missing market data columns"
- Ensure you called `enrich_with_market_data()` before `analyze()`
- Check that the symbol matches your trades

### Issue: PDF parsing fails
- PDF format varies by broker
- Try exporting to CSV from your broker instead
- Or adjust the column mapping in `pdf_loader.py`

### Issue: Import errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (requires 3.8+)

## Advanced Usage

### Customize Number of Clusters
```python
analyzer = BehavioralAnalyzer(n_clusters=5)  # Find 5 behavioral modes
```

### Customize Baseline Window
```python
analyzer = BehavioralAnalyzer(baseline_window=60)  # 60-trade rolling window
```

### Analyze Multiple Symbols
If you trade multiple symbols, you'll need to enrich each separately or modify the code to handle multiple symbols.

### Export Results to CSV
```python
results = analyzer.analyze()
results['features'].to_csv('output/enriched_features.csv', index=False)
```

## Next Steps

1. Run the example to see how it works
2. Replace with your own tradebook data
3. Review the visualizations to understand your trading patterns
4. Use the explanations to identify behavioral changes
5. Adjust your trading based on insights!

