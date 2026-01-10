"""Explainable AI components for behavioral analysis."""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class BehavioralExplainer:
    """Provides explainable insights for behavioral analysis."""
    
    def __init__(self):
        self.explanations: Dict = {}
    
    def explain_trade(self, trade_idx: int, features: pd.DataFrame, 
                     baselines: Dict, clusters: Optional[np.ndarray] = None) -> str:
        """
        Generate natural language explanation for a specific trade.
        
        Args:
            trade_idx: Index of trade to explain
            features: DataFrame with trade features
            baselines: Baseline statistics dictionary
            clusters: Optional cluster labels
            
        Returns:
            Natural language explanation
        """
        if trade_idx >= len(features):
            return "Trade index out of range."
        
        trade = features.iloc[trade_idx]
        explanations = []
        
        # Market context
        if 'rsi_14' in trade:
            rsi = trade['rsi_14']
            if not pd.isna(rsi):
                if rsi < 30:
                    explanations.append(f"Entered during oversold conditions (RSI: {rsi:.1f})")
                elif rsi > 70:
                    explanations.append(f"Entered during overbought conditions (RSI: {rsi:.1f})")
                else:
                    explanations.append(f"Entered during neutral RSI conditions (RSI: {rsi:.1f})")
        
        # Market regime context
        if 'trend_regime' in trade:
            explanations.append(f"Market trend regime: {trade['trend_regime']}")
        if 'volatility_regime' in trade:
            explanations.append(f"Market volatility regime: {trade['volatility_regime']}")
        
        # Position size deviation
        if 'position_size_normalized_by_volatility' in trade and 'position_size_normalized_by_volatility' in baselines:
            baseline = baselines['position_size_normalized_by_volatility']
            size_norm = trade['position_size_normalized_by_volatility']
            if not pd.isna(size_norm):
                zscore = (size_norm - baseline['mean']) / (baseline['std'] + 1e-6)
                
                if abs(zscore) > 2:
                    explanations.append(
                        f"Risk-adjusted position size was {'much larger' if zscore > 0 else 'much smaller'} "
                        f"than typical ({zscore:.2f} standard deviations from baseline)"
                    )
                elif abs(zscore) > 1:
                    explanations.append(
                        f"Risk-adjusted position size was {'larger' if zscore > 0 else 'smaller'} "
                        f"than typical ({zscore:.2f} standard deviations from baseline)"
                    )
        
        # Holding duration
        if 'holding_duration_days' in trade and 'holding_duration_days' in baselines:
            baseline = baselines['holding_duration_days']
            duration = trade['holding_duration_days']
            if not pd.isna(duration):
                if duration > baseline['q75']:
                    explanations.append(f"Held for longer than usual ({duration:.1f} days vs median {baseline['median']:.1f} days)")
                elif duration < baseline['q25']:
                    explanations.append(f"Held for shorter than usual ({duration:.1f} days vs median {baseline['median']:.1f} days)")
        
        # Sequence context
        if 'trades_after_loss' in trade and trade['trades_after_loss'] > 0:
            explanations.append(
                f"This trade occurred after a loss, with {trade['trades_after_loss']} trades "
                f"executed within 48 hours (potential revenge trading pattern)"
            )
        
        # Entry/exit relative to EMA
        if 'entry_price_distance_from_ema20' in trade:
            ema_dist = trade['entry_price_distance_from_ema20']
            if not pd.isna(ema_dist):
                if ema_dist > 0.02:  # 2% above EMA
                    explanations.append(f"Entered {ema_dist*100:.1f}% above EMA(20) - chasing trend")
                elif ema_dist < -0.02:  # 2% below EMA
                    explanations.append(f"Entered {abs(ema_dist)*100:.1f}% below EMA(20) - contrarian entry")
        
        # Cluster assignment
        if clusters is not None and trade_idx < len(clusters):
            cluster_id = clusters[trade_idx]
            explanations.append(f"Belongs to behavioral cluster {cluster_id}")
        
        # Combine explanations
        if explanations:
            explanation = f"Trade on {trade['date']} ({trade.get('symbol', 'N/A')}):\n"
            explanation += "\n".join(f"  • {exp}" for exp in explanations)
        else:
            explanation = f"Trade on {trade['date']}: No significant deviations detected."
        
        return explanation
    
    def explain_cluster(self, cluster_id: int, cluster_analysis: Dict) -> str:
        """Explain what a behavioral cluster represents."""
        cluster_key = f'cluster_{cluster_id}'
        if cluster_key not in cluster_analysis:
            return f"Cluster {cluster_id} not found."
        
        info = cluster_analysis[cluster_key]
        
        explanation = f"Behavioral Cluster {cluster_id}:\n"
        explanation += f"  • Size: {info['size']} trades\n"
        explanation += f"  • Average trades per day: {info['avg_trades_per_day']:.2f}\n"
        explanation += f"  • Average position size ratio: {info['avg_position_size']:.4f}\n"
        explanation += f"  • Average holding duration: {info['avg_holding_duration']:.1f} days\n"
        explanation += f"  • Average P&L: ${info['avg_pnl']:.2f}\n"
        explanation += f"  • Win rate: {info['win_rate']:.1%}\n"
        
        # Behavioral interpretation
        if info['avg_trades_per_day'] > 3:
            explanation += "\nInterpretation: High-frequency trading mode"
        elif info['avg_position_size'] > 0.1:
            explanation += "\nInterpretation: Aggressive position sizing"
        elif info['avg_holding_duration'] < 2:
            explanation += "\nInterpretation: Short-term trading focus"
        else:
            explanation += "\nInterpretation: Moderate, disciplined trading"
        
        return explanation
    
    def explain_change_point(self, change_point_idx: int, segments: List[Dict]) -> str:
        """Explain a detected change point."""
        # Find segment containing this change point
        for i, segment in enumerate(segments):
            if segment['end_idx'] == change_point_idx:
                prev_segment = segments[i-1] if i > 0 else None
                next_segment = segments[i] if i < len(segments) else None
                
                if prev_segment and next_segment:
                    explanation = f"Behavioral Change Point Detected:\n"
                    explanation += f"  • Date: {segment['start_date']}\n"
                    explanation += f"  • Previous period avg trades/day: {prev_segment['avg_trades_per_day']:.2f}\n"
                    explanation += f"  • New period avg trades/day: {next_segment['avg_trades_per_day']:.2f}\n"
                    explanation += f"  • Previous period avg P&L: ${prev_segment['avg_pnl']:.2f}\n"
                    explanation += f"  • New period avg P&L: ${next_segment['avg_pnl']:.2f}\n"
                    
                    # Interpretation
                    if abs(next_segment['avg_trades_per_day'] - prev_segment['avg_trades_per_day']) > 1:
                        explanation += "\nInterpretation: Significant change in trading frequency detected."
                    if abs(next_segment['avg_pnl'] - prev_segment['avg_pnl']) > 100:
                        explanation += "\nInterpretation: Significant change in performance detected."
                    
                    return explanation
        
        return f"Change point at index {change_point_idx}: No detailed analysis available."
    
    def generate_feature_attribution(self, trade_idx: int, features: pd.DataFrame, 
                                    cluster_centers: np.ndarray, cluster_id: int) -> str:
        """Explain why a trade belongs to a specific cluster."""
        if trade_idx >= len(features):
            return "Trade index out of range."
        
        trade = features.iloc[trade_idx]
        feature_cols = [col for col in features.columns 
                       if col not in ['date', 'symbol', 'side', 'price', 'quantity'] 
                       and features[col].dtype in [np.float64, np.int64]]
        
        # Calculate distances to cluster centers
        trade_features = trade[feature_cols].fillna(0).values
        center = cluster_centers[cluster_id]
        
        # Feature contributions (difference from cluster center)
        contributions = {}
        for i, col in enumerate(feature_cols):
            if i < len(trade_features) and i < len(center):
                contributions[col] = abs(trade_features[i] - center[i])
        
        # Sort by contribution
        top_contributors = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        explanation = f"Trade belongs to Cluster {cluster_id} primarily due to:\n"
        for feature, contribution in top_contributors:
            trade_val = trade[feature] if feature in trade else 0
            center_val = center[feature_cols.index(feature)] if feature_cols.index(feature) < len(center) else 0
            explanation += f"  • {feature}: {trade_val:.2f} (cluster center: {center_val:.2f})\n"
        
        return explanation
    
    def generate_counterfactual(self, trade_idx: int, features: pd.DataFrame, 
                               baselines: Dict) -> str:
        """Generate counterfactual explanation."""
        if trade_idx >= len(features):
            return "Trade index out of range."
        
        trade = features.iloc[trade_idx]
        counterfactuals = []
        
        # Position size counterfactual
        if 'position_size_ratio' in trade and 'position_size_ratio' in baselines:
            baseline = baselines['position_size_ratio']
            current_size = trade['position_size_ratio']
            baseline_size = baseline['mean']
            
            if abs(current_size - baseline_size) > baseline['std']:
                counterfactuals.append(
                    f"If position size were within normal range ({baseline_size:.4f} instead of {current_size:.4f}), "
                    f"this trade would align with baseline behavior."
                )
        
        # Timing counterfactual
        if 'holding_duration' in trade and 'holding_duration' in baselines:
            baseline = baselines['holding_duration']
            current_duration = trade['holding_duration']
            if not pd.isna(current_duration):
                baseline_duration = baseline['median']
                if abs(current_duration - baseline_duration) > baseline['std']:
                    counterfactuals.append(
                        f"If holding duration were typical ({baseline_duration:.1f} days instead of {current_duration:.1f} days), "
                        f"this trade would match baseline patterns."
                    )
        
        if counterfactuals:
            explanation = "Counterfactual Analysis:\n"
            explanation += "\n".join(f"  • {cf}" for cf in counterfactuals)
        else:
            explanation = "This trade aligns with baseline behavior patterns."
        
        return explanation
    
    def generate_report(self, features: pd.DataFrame, baselines: Dict, 
                       pattern_results: Dict) -> str:
        """Generate comprehensive behavioral report."""
        report = []
        report.append("=" * 80)
        report.append("BEHAVIORAL TRADING ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        report.append("SUMMARY STATISTICS")
        report.append("-" * 80)
        report.append(f"Total trades analyzed: {len(features)}")
        report.append(f"Date range: {features['date'].min()} to {features['date'].max()}")
        if 'realized_pnl' in features.columns:
            report.append(f"Total P&L: ${features['realized_pnl'].sum():.2f}")
            report.append(f"Average P&L per trade: ${features['realized_pnl'].mean():.2f}")
            report.append(f"Win rate: {(features['realized_pnl'] > 0).mean():.1%}")
        report.append("")
        
        # Baseline summary
        report.append("BEHAVIORAL BASELINES")
        report.append("-" * 80)
        for feature, stats in baselines.items():
            if isinstance(stats, dict) and 'mean' in stats:
                report.append(f"{feature}:")
                report.append(f"  Mean: {stats['mean']:.4f}")
                report.append(f"  Median: {stats['median']:.4f}")
                report.append(f"  Std: {stats['std']:.4f}")
        report.append("")
        
        # Cluster analysis
        if 'clusters' in pattern_results and 'analysis' in pattern_results['clusters']:
            report.append("BEHAVIORAL CLUSTERS")
            report.append("-" * 80)
            for cluster_id, info in pattern_results['clusters']['analysis'].items():
                report.append(self.explain_cluster(int(cluster_id.split('_')[1]), 
                                                  pattern_results['clusters']['analysis']))
                report.append("")
        
        # Change points
        if 'change_points' in pattern_results and pattern_results['change_points']['indices']:
            report.append("BEHAVIORAL CHANGE POINTS")
            report.append("-" * 80)
            for cp_idx in pattern_results['change_points']['indices']:
                report.append(self.explain_change_point(cp_idx, 
                                                       pattern_results['change_points']['segments']))
                report.append("")
        
        # Anomalies
        if 'anomalies' in pattern_results and pattern_results['anomalies']['indices']:
            report.append("ANOMALOUS TRADES")
            report.append("-" * 80)
            report.append(f"Found {len(pattern_results['anomalies']['indices'])} anomalous trades")
            report.append("These trades deviate significantly from baseline behavior patterns.")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def generate_xai_summary(self, features: pd.DataFrame, baselines: Dict, 
                            pattern_results: Dict) -> str:
        """
        Generate an explainable AI summary paragraph (~100 words) using rule-based NLG.
        
        This uses:
        - Rule-based Natural Language Generation (NLG)
        - Feature contribution ranking to explain trade differences
        - Quantifiable statistics
        - Simple English explanations
        
        Args:
            features: DataFrame with trade features
            baselines: Baseline statistics dictionary
            pattern_results: Pattern discovery results
            
        Returns:
            Natural language summary paragraph (~100 words)
        """
        summary_parts = []
        
        # 1. Overall Performance Summary
        total_trades = len(features)
        if 'realized_pnl' in features.columns:
            total_pnl = features['realized_pnl'].sum()
            avg_pnl = features['realized_pnl'].mean()
            win_rate = (features['realized_pnl'] > 0).mean() * 100
            winning_trades = (features['realized_pnl'] > 0).sum()
            losing_trades = (features['realized_pnl'] < 0).sum()
            
            # Performance assessment
            if total_pnl < 0:
                perf_desc = f"lost ${abs(total_pnl):,.0f} overall"
            else:
                perf_desc = f"gained ${total_pnl:,.0f} overall"
            
            summary_parts.append(
                f"Analyzed {total_trades} trades: {perf_desc} with {win_rate:.0f}% win rate "
                f"({winning_trades} wins, {losing_trades} losses), averaging ${avg_pnl:.0f} per trade."
            )
        
        # 2. Behavioral Pattern Summary (from clusters)
        if 'clusters' in pattern_results and 'labels' in pattern_results['clusters']:
            clusters = pattern_results['clusters']['labels']
            unique_clusters = np.unique(clusters)
            
            # Find best and worst performing clusters
            cluster_performance = {}
            for cluster_id in unique_clusters:
                cluster_mask = clusters == cluster_id
                if 'realized_pnl' in features.columns:
                    cluster_pnl = features.loc[cluster_mask, 'realized_pnl'].mean()
                    cluster_size = cluster_mask.sum()
                    cluster_performance[cluster_id] = {
                        'pnl': cluster_pnl,
                        'size': cluster_size,
                        'win_rate': (features.loc[cluster_mask, 'realized_pnl'] > 0).mean() * 100
                    }
            
            if cluster_performance:
                best_cluster = max(cluster_performance.items(), key=lambda x: x[1]['pnl'])
                worst_cluster = min(cluster_performance.items(), key=lambda x: x[1]['pnl'])
                
                best_id, best_stats = best_cluster
                worst_id, worst_stats = worst_cluster
                
                summary_parts.append(
                    f"Identified {len(unique_clusters)} behavioral patterns: Cluster {best_id} performed best "
                    f"(${best_stats['pnl']:.0f} avg, {best_stats['win_rate']:.0f}% win rate, {best_stats['size']} trades) "
                    f"while Cluster {worst_id} underperformed (${worst_stats['pnl']:.0f} avg, {worst_stats['win_rate']:.0f}% win rate)."
                )
        
        # 3. Key Behavioral Deviations (Feature Contribution Ranking)
        key_features = [
            'position_size_normalized_by_volatility',
            'holding_duration_days',
            'trades_per_day',
            'entry_price_distance_from_ema20',
            'trades_after_loss'
        ]
        
        deviations = []
        for feature in key_features:
            if feature in features.columns and feature in baselines:
                baseline = baselines[feature]
                if 'mean' in baseline and 'std' in baseline:
                    # Calculate average absolute deviation
                    avg_deviation = abs(features[feature] - baseline['mean']).mean()
                    relative_dev = avg_deviation / (baseline['std'] + 1e-6)
                    
                    if relative_dev > 1.0:  # Significant deviation
                        feature_name = feature.replace('_', ' ').title()
                        deviations.append((feature_name, relative_dev))
        
        # Sort by deviation magnitude
        deviations.sort(key=lambda x: x[1], reverse=True)
        
        if deviations:
            top_deviation = deviations[0]
            summary_parts.append(
                f"Primary behavioral deviation: {top_deviation[0]} was {top_deviation[1]:.1f}x "
                f"more variable than normal, indicating inconsistent trading discipline."
            )
        
        # 4. Market Signal Following
        if 'rsi_14' in features.columns:
            overbought = (features['rsi_14'] > 70).sum()
            oversold = (features['rsi_14'] < 30).sum()
            
            if 'side' in features.columns:
                overbought_sells = ((features['rsi_14'] > 70) & (features['side'] == 'sell')).sum()
                oversold_buys = ((features['rsi_14'] < 30) & (features['side'] == 'buy')).sum()
                
                rsi_follow_rate = ((overbought_sells + oversold_buys) / (overbought + oversold) * 100) if (overbought + oversold) > 0 else 0
                
                summary_parts.append(
                    f"Signal adherence: Followed {rsi_follow_rate:.0f}% of RSI signals "
                    f"({overbought_sells + oversold_buys}/{overbought + oversold} signals acted upon)."
                )
        
        # 5. Change Points and Anomalies
        if 'change_points' in pattern_results and pattern_results['change_points']['indices']:
            num_change_points = len(pattern_results['change_points']['indices'])
            summary_parts.append(
                f"Detected {num_change_points} behavioral shift{'s' if num_change_points > 1 else ''}, "
                f"suggesting trading strategy evolved or emotional responses changed."
            )
        
        if 'anomalies' in pattern_results and pattern_results['anomalies']['indices']:
            num_anomalies = len(pattern_results['anomalies']['indices'])
            summary_parts.append(
                f"{num_anomalies} anomalous trade{'s' if num_anomalies > 1 else ''} deviated significantly "
                f"from baseline patterns, likely driven by emotional decisions or external factors."
            )
        
        # 6. Risk Management Assessment
        if 'position_size_normalized_by_volatility' in features.columns and 'volatility_rolling_std' in features.columns:
            high_vol_trades = (features['volatility_rolling_std'] > features['volatility_rolling_std'].quantile(0.75)).sum()
            if high_vol_trades > 0:
                avg_size_high_vol = features.loc[features['volatility_rolling_std'] > features['volatility_rolling_std'].quantile(0.75), 
                                                  'position_size_normalized_by_volatility'].mean()
                avg_size_normal = features.loc[features['volatility_rolling_std'] <= features['volatility_rolling_std'].quantile(0.75),
                                               'position_size_normalized_by_volatility'].mean()
                
                if avg_size_high_vol > avg_size_normal * 1.2:
                    summary_parts.append(
                        f"Risk concern: Position sizes were {avg_size_high_vol/avg_size_normal:.1f}x larger "
                        f"during high volatility periods, increasing exposure when markets were most unstable."
                    )
        
        # Combine all parts into a coherent paragraph
        full_summary = " ".join(summary_parts)
        
        # Ensure it's approximately 100 words (target: 90-110 words)
        words = full_summary.split()
        word_count = len(words)
        
        if word_count > 110:
            # Trim to ~100 words, keeping most important parts
            full_summary = " ".join(words[:100]) + "..."
        elif word_count < 90:
            # Add more details to reach ~100 words
            additional_info = []
            
            # Add feature contribution ranking details
            if deviations and len(deviations) > 1:
                second_dev = deviations[1] if len(deviations) > 1 else None
                if second_dev:
                    additional_info.append(
                        f"Secondary deviation: {second_dev[0]} showed {second_dev[1]:.1f}x normal variation."
                    )
            
            # Add holding duration insight if available
            if 'holding_duration_days' in features.columns and 'holding_duration_days' in baselines:
                baseline_duration = baselines['holding_duration_days']['median']
                avg_duration = features['holding_duration_days'].mean()
                if abs(avg_duration - baseline_duration) > baseline_duration * 0.3:
                    additional_info.append(
                        f"Average holding period was {avg_duration:.1f} days vs typical {baseline_duration:.1f} days."
                    )
            
            # Add trade frequency insight
            if 'trades_per_day' in features.columns:
                avg_freq = features['trades_per_day'].mean()
                additional_info.append(f"Trading frequency averaged {avg_freq:.1f} trades per day.")
            
            # Add concluding recommendation
            if 'realized_pnl' in features.columns and features['realized_pnl'].sum() < 0:
                additional_info.append("Recommendation: Improve signal following and maintain consistent risk management.")
            else:
                additional_info.append("Continue monitoring patterns to sustain performance.")
            
            # Add enough to reach ~100 words
            while len(full_summary.split()) < 95 and additional_info:
                full_summary += " " + additional_info.pop(0)
        
        # Final word count check and trim if needed
        final_words = full_summary.split()
        if len(final_words) > 110:
            full_summary = " ".join(final_words[:105])
        
        return full_summary

