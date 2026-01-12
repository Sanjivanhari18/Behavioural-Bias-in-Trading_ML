"""Main application entry point for Behavioral Trading Analysis System."""

import pandas as pd
import logging
from typing import Optional, Dict
import os

from .stage1_data import (
    CSVTradebookLoader,
    PDFTradebookLoader,
    TradebookValidator,
    TradebookCleaner
)
from .utils.market_data import MarketDataFetcher
from .stage2_analysis import (
    BehavioralFeatureEngineer,
    BaselineConstructor,
    PatternDiscoverer,
    BehavioralStabilityAnalyzer
)
from .stage3_viz import (
    BehavioralVisualizer,
    BehavioralExplainer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehavioralAnalyzer:
    """Main class for behavioral trading analysis."""
    
    def __init__(self, n_clusters: int = 3, baseline_window: int = 30):
        """
        Initialize the behavioral analyzer.
        
        Args:
            n_clusters: Number of behavioral clusters to discover
            baseline_window: Rolling window size for baseline calculation
        """
        self.n_clusters = n_clusters
        self.baseline_window = baseline_window
        
        # Stage 1 components
        self.csv_loader = CSVTradebookLoader()
        self.pdf_loader = PDFTradebookLoader()
        self.validator = TradebookValidator()
        self.cleaner = TradebookCleaner()
        
        # Stage 2 components
        self.market_fetcher = MarketDataFetcher()
        self.feature_engineer = BehavioralFeatureEngineer()
        self.baseline_constructor = BaselineConstructor(window_size=baseline_window)
        self.pattern_discoverer = PatternDiscoverer(n_clusters=n_clusters)
        self.stability_analyzer = BehavioralStabilityAnalyzer(window_size=baseline_window)
        
        # Stage 3 components
        self.visualizer = BehavioralVisualizer()
        self.explainer = BehavioralExplainer()
        
        # Data storage
        self.trades: Optional[pd.DataFrame] = None
        self.enriched_trades: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.baselines: Optional[Dict] = None
        self.pattern_results: Optional[Dict] = None
        self.stability_results: Optional[Dict] = None
    
    def load_tradebook(self, filepath: str, file_type: Optional[str] = None) -> pd.DataFrame:
        """
        Load tradebook from CSV or PDF file.
        
        Args:
            filepath: Path to tradebook file
            file_type: 'csv' or 'pdf'. If None, inferred from extension.
        
        Returns:
            Loaded trades DataFrame
        """
        if file_type is None:
            file_type = os.path.splitext(filepath)[1].lower().lstrip('.')
        
        if file_type == 'csv':
            logger.info(f"Loading CSV tradebook from {filepath}")
            self.trades = self.csv_loader.load(filepath)
        elif file_type == 'pdf':
            logger.info(f"Loading PDF tradebook from {filepath}")
            self.trades = self.pdf_loader.load(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Use 'csv' or 'pdf'.")
        
        # Validate
        validation = self.validator.validate(self.trades)
        if not self.validator.is_valid():
            logger.error(f"Validation errors: {validation['errors']}")
            raise ValueError(f"Tradebook validation failed: {validation['errors']}")
        
        if validation['warnings']:
            logger.warning(f"Validation warnings: {validation['warnings']}")
        
        # Clean
        logger.info("Cleaning tradebook data...")
        self.trades = self.cleaner.clean(self.trades)
        
        logger.info(f"Successfully loaded {len(self.trades)} trades")
        return self.trades
    
    def enrich_with_market_data(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Enrich trades with market context data.
        
        Now supports multiple symbols automatically:
        - If symbol is provided, uses that symbol for all trades (backward compatible)
        - If symbol is None, automatically detects all unique symbols in trades
          and fetches OHLCV for each symbol separately
        
        Args:
            symbol: Stock symbol (optional - if None, uses symbol column from trades)
        
        Returns:
            Enriched trades DataFrame
        """
        if self.trades is None:
            raise ValueError("No trades loaded. Call load_tradebook() first.")
        
        if symbol is None:
            logger.info("Enriching trades with market data for all symbols in tradebook...")
        else:
            logger.info(f"Enriching trades with market data for {symbol}...")
        
        self.enriched_trades = self.market_fetcher.enrich_trades_with_market_data(
            self.trades, symbol=symbol
        )
        
        logger.info("Market data enrichment complete")
        return self.enriched_trades
    
    def analyze(self) -> Dict:
        """
        Run full behavioral analysis pipeline.
        
        Returns:
            Dictionary containing all analysis results
        """
        if self.enriched_trades is None:
            raise ValueError("No enriched trades available. Call enrich_with_market_data() first.")
        
        logger.info("Starting behavioral analysis...")
        
        # Stage 2.1: Feature Engineering
        logger.info("Engineering behavioral features...")
        self.features = self.feature_engineer.engineer_features(self.enriched_trades)
        
        # Stage 2.2: Baseline Construction
        logger.info("Constructing behavioral baselines...")
        self.baselines = self.baseline_constructor.construct_baselines(self.features)
        self.features = self.baseline_constructor.calculate_deviations(self.features)
        
        # Stage 2.3: Pattern Discovery
        logger.info("Discovering behavioral patterns...")
        self.pattern_results = self.pattern_discoverer.discover_patterns(self.features)
        
        # Add cluster labels to features
        if 'clusters' in self.pattern_results:
            self.features['behavioral_cluster'] = self.pattern_results['clusters']['labels']
        
        # Stage 2.4: Behavioral Stability Analysis (NEW)
        logger.info("Calculating behavioral stability score...")
        self.stability_results = self.stability_analyzer.calculate_stability_score(self.features)
        
        logger.info("Behavioral analysis complete")
        
        return {
            'features': self.features,
            'baselines': self.baselines,
            'patterns': self.pattern_results,
            'stability': self.stability_results
        }
    
    def visualize(self, results: Optional[Dict] = None, output_dir: str = "output/") -> Dict[str, str]:
        """
        Generate all visualizations.
        
        Args:
            results: Analysis results (if None, uses internal results)
            output_dir: Directory to save visualizations
        
        Returns:
            Dictionary mapping visualization names to file paths
        """
        if results is None:
            if self.features is None or self.pattern_results is None:
                raise ValueError("No analysis results available. Call analyze() first.")
            results = {
                'features': self.features,
                'patterns': self.pattern_results,
                'stability': self.stability_results
            }
        
        features = results['features']
        patterns = results['patterns']
        
        logger.info("Generating visualizations...")
        
        # Regime timeline
        if 'clusters' in patterns and 'labels' in patterns['clusters']:
            clusters = patterns['clusters']['labels']
            change_points = patterns.get('change_points', {}).get('indices', [])
            try:
                self.visualizer.create_regime_timeline(features, clusters, change_points)
            except Exception as e:
                logger.warning(f"Could not create regime timeline: {e}")
        
        # Deviation plots
        try:
            self.visualizer.create_deviation_plots(features)
        except Exception as e:
            logger.warning(f"Could not create deviation plots: {e}")
        
        # Post-event charts
        try:
            self.visualizer.create_post_event_charts(features)
        except Exception as e:
            logger.warning(f"Could not create post-event charts: {e}")
        
        # Performance matrix
        if 'clusters' in patterns and 'labels' in patterns['clusters']:
            clusters = patterns['clusters']['labels']
            try:
                self.visualizer.create_performance_matrix(features, clusters, self.n_clusters)
            except Exception as e:
                logger.warning(f"Could not create performance matrix: {e}")
        
        # Cluster timeline visualization
        if 'clusters' in patterns and 'labels' in patterns['clusters']:
            clusters = patterns['clusters']['labels']
            try:
                self.visualizer.create_cluster_timeline(features, clusters)
            except Exception as e:
                logger.warning(f"Could not create cluster timeline: {e}")
        
        # Trade Journey Timeline (user-friendly for retail investors)
        try:
            self.visualizer.create_trade_journey_timeline(features)
        except Exception as e:
            logger.warning(f"Could not create trade journey timeline: {e}")
        
        # Signal Following Scorecard (user-friendly for retail investors)
        try:
            self.visualizer.create_signal_following_scorecard(features)
        except Exception as e:
            logger.warning(f"Could not create signal scorecard: {e}")
        
        # Behavioral Stability Scorecard (NEW)
        if 'stability' in results and results['stability']:
            try:
                self.visualizer.create_stability_scorecard(results['stability'])
            except Exception as e:
                logger.warning(f"Could not create stability scorecard: {e}")
        
        # Save all figures
        os.makedirs(output_dir, exist_ok=True)
        self.visualizer.save_all_figures(output_dir)
        
        # Create unified dashboard
        try:
            dashboard_path = self.visualizer.create_unified_dashboard(
                output_dir=output_dir,
                xai_file="xai_explanation.txt",
                report_file="behavioral_report.txt"
            )
            logger.info(f"Unified dashboard created: {dashboard_path}")
        except Exception as e:
            logger.warning(f"Could not create unified dashboard: {e}")
        
        logger.info(f"Visualizations saved to {output_dir}")
        
        return {name: os.path.join(output_dir, f"{name}.html") 
                for name in self.visualizer.figures.keys()}
    
    def generate_report(self, results: Optional[Dict] = None, 
                       output_file: str = "behavioral_report.txt") -> str:
        """
        Generate comprehensive behavioral report.
        
        Args:
            results: Analysis results (if None, uses internal results)
            output_file: Path to save report
        
        Returns:
            Report text
        """
        if results is None:
            if self.features is None or self.baselines is None or self.pattern_results is None:
                raise ValueError("No analysis results available. Call analyze() first.")
            results = {
                'features': self.features,
                'baselines': self.baselines,
                'patterns': self.pattern_results,
                'stability': self.stability_results
            }
        
        logger.info("Generating behavioral report...")
        
        report = self.explainer.generate_report(
            results['features'],
            results['baselines'],
            results['patterns']
        )
        
        # Add stability score to report if available
        if results.get('stability') and results['stability'].get('stability_score') is not None:
            stability_section = "\n\n" + "=" * 80 + "\n"
            stability_section += "BEHAVIORAL STABILITY / CONSISTENCY SCORE\n"
            stability_section += "=" * 80 + "\n\n"
            stability_section += results['stability']['interpretation'] + "\n\n"
            stability_section += results['stability']['note'] + "\n"
            report += stability_section
        
        # Generate XAI summary paragraph
        xai_summary = self.explainer.generate_xai_summary(
            results['features'],
            results['baselines'],
            results['patterns'],
            results.get('stability')
        )
        
        # Save main report
        with open(output_file, 'w') as f:
            f.write(report)
        
        # Save XAI summary as separate file
        output_dir = os.path.dirname(output_file) if os.path.dirname(output_file) else "output"
        xai_output_file = os.path.join(output_dir, "xai_explanation.txt")
        with open(xai_output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("EXPLAINABLE AI (XAI) SUMMARY - BEHAVIORAL TRADING ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            f.write("This summary uses rule-based Natural Language Generation (NLG) and feature\n")
            f.write("contribution ranking to explain your trading behavior in simple, quantifiable terms.\n\n")
            f.write("-" * 80 + "\n")
            f.write("KEY FINDINGS:\n")
            f.write("-" * 80 + "\n\n")
            f.write(xai_summary)
            f.write("\n\n")
            f.write("=" * 80 + "\n")
            f.write("Generated using:\n")
            f.write("- Rule-based NLG for natural language explanations\n")
            f.write("- Feature contribution ranking to identify key behavioral drivers\n")
            f.write("- Statistical analysis of trading patterns and deviations\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Report saved to {output_file}")
        logger.info(f"XAI explanation saved to {xai_output_file}")
        return report
    
    def explain_trade(self, trade_idx: int) -> str:
        """
        Generate explanation for a specific trade.
        
        Args:
            trade_idx: Index of trade to explain
        
        Returns:
            Natural language explanation
        """
        if self.features is None:
            raise ValueError("No features available. Call analyze() first.")
        
        clusters = None
        if self.pattern_results and 'clusters' in self.pattern_results:
            clusters = self.pattern_results['clusters']['labels']
        
        return self.explainer.explain_trade(
            trade_idx,
            self.features,
            self.baselines,
            clusters
        )
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        if self.features is None:
            return {}
        
        summary = {
            'total_trades': len(self.features),
            'date_range': {
                'start': str(self.features['date'].min()),
                'end': str(self.features['date'].max())
            }
        }
        
        if 'realized_pnl' in self.features.columns:
            summary['pnl'] = {
                'total': float(self.features['realized_pnl'].sum()),
                'average': float(self.features['realized_pnl'].mean()),
                'win_rate': float((self.features['realized_pnl'] > 0).mean())
            }
        
        if 'behavioral_cluster' in self.features.columns:
            summary['clusters'] = {
                int(cluster): int((self.features['behavioral_cluster'] == cluster).sum())
                for cluster in self.features['behavioral_cluster'].unique()
            }
        
        return summary

