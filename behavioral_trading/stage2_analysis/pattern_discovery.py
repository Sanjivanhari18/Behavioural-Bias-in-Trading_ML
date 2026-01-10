"""Pattern discovery using ML (clustering, change point detection, anomaly detection)."""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import ruptures as rpt
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class PatternDiscoverer:
    """Discovers behavioral patterns using ML."""
    
    def __init__(self, n_clusters: int = 3):
        """
        Args:
            n_clusters: Number of behavioral clusters to discover
        """
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.clusterer: Optional[KMeans] = None
        self.anomaly_detector: Optional[IsolationForest] = None
        self.change_points: Optional[List[int]] = None
        self.results: Optional[Dict] = None
    
    def discover_patterns(self, features: pd.DataFrame) -> Dict:
        """
        Discover behavioral patterns:
        - Clustering (behavioral regimes)
        - Change point detection
        - Anomaly detection
        """
        results = {}
        
        # Prepare feature matrix
        feature_cols = self._get_feature_columns(features)
        X = features[feature_cols].fillna(0).values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # A. Clustering
        results['clusters'] = self._cluster_behaviors(X_scaled, features)
        
        # B. Change Point Detection
        results['change_points'] = self._detect_change_points(X_scaled, features)
        
        # C. Anomaly Detection
        results['anomalies'] = self._detect_anomalies(X_scaled, features)
        
        self.results = results
        return results
    
    def _get_feature_columns(self, features: pd.DataFrame) -> List[str]:
        """Get columns to use for pattern discovery."""
        # Select behavioral features (exclude market data and identifiers)
        exclude = ['date', 'symbol', 'side', 'price', 'quantity', 'rsi_14', 'ema_20', 
                  'ema_50', 'Close', 'Date', 'date_only', 'trend_regime', 'volatility_regime']
        
        feature_cols = [col for col in features.columns 
                       if col not in exclude and features[col].dtype in [np.float64, np.int64]]
        
        # Prioritize key behavioral features
        priority_features = [
            'trades_per_day', 'trades_per_rolling_7days', 'trades_per_rolling_30days',
            'position_size_dollar_value', 'position_size_normalized_by_volatility',
            'holding_duration_days', 'holding_duration_vs_volatility',
            'time_gap_hours_since_last_trade', 'time_gap_days_since_last_trade',
            'entry_price_distance_from_ema20', 'entry_price_distance_from_ema50',
            'exit_price_distance_from_ema20', 'exit_price_distance_from_ema50'
        ]
        
        # Order: priority features first, then others
        ordered = [f for f in priority_features if f in feature_cols]
        ordered.extend([f for f in feature_cols if f not in ordered])
        
        return ordered[:20]  # Limit to top 20 features
    
    def _cluster_behaviors(self, X: np.ndarray, features: pd.DataFrame) -> Dict:
        """Cluster trades into behavioral regimes."""
        # Use GMM for soft clustering (probabilistic assignment)
        gmm = GaussianMixture(n_components=self.n_clusters, random_state=42, n_init=5)
        cluster_labels = gmm.fit_predict(X)
        
        # Also use K-Means for hard clusters (deterministic assignment)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X)
        
        # Store K-Means as primary clusterer (used for consistency)
        self.clusterer = kmeans
        
        # Use GMM labels for analysis (more flexible, handles overlapping behaviors)
        # Analyze clusters using GMM labels (more accurate for behavioral patterns)
        cluster_analysis = {}
        for i in range(self.n_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = features.loc[cluster_mask]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': cluster_mask.sum(),
                'avg_trades_per_day': cluster_data['trades_per_day'].mean() if 'trades_per_day' in cluster_data.columns else 0,
                'avg_position_size': cluster_data['position_size_ratio'].mean() if 'position_size_ratio' in cluster_data.columns else 0,
                'avg_holding_duration': cluster_data['holding_duration'].mean() if 'holding_duration' in cluster_data.columns else 0,
                'avg_pnl': cluster_data['realized_pnl'].mean() if 'realized_pnl' in cluster_data.columns else 0,
                'win_rate': (cluster_data['realized_pnl'] > 0).mean() if 'realized_pnl' in cluster_data.columns else 0
            }
        
        return {
            'labels': cluster_labels,
            'kmeans_labels': kmeans_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'analysis': cluster_analysis
        }
    
    def _detect_change_points(self, X: np.ndarray, features: pd.DataFrame) -> Dict:
        """Detect structural changes in behavior."""
        # Use ruptures for change point detection
        # X is already in (n_samples, n_features) format - ruptures expects this
        # For multivariate change point detection, use appropriate cost function
        try:
            # Use Pelt algorithm with RBF cost for multivariate time series
            # Pelt expects (n_samples, n_features) format
            algo = rpt.Pelt(model="rbf").fit(X)
            # Adaptive penalty based on data size (log scale)
            pen_value = 10 * np.log(len(X)) if len(X) > 1 else 10
            change_points = algo.predict(pen=pen_value)
            
            # Convert to indices (ruptures returns end indices, excluding first 0 and last n)
            change_point_indices = [cp - 1 for cp in change_points if cp > 0 and cp < len(X)]
        except Exception as e:
            logger.warning(f"Change point detection failed: {e}. Using empty result.")
            change_point_indices = []
        
        self.change_points = change_point_indices
        
        # Analyze segments
        segments = []
        prev_idx = 0
        for cp_idx in change_point_indices + [len(features)]:
            segment_data = features.iloc[prev_idx:cp_idx]
            segments.append({
                'start_idx': prev_idx,
                'end_idx': cp_idx,
                'start_date': segment_data['date'].iloc[0] if len(segment_data) > 0 else None,
                'end_date': segment_data['date'].iloc[-1] if len(segment_data) > 0 else None,
                'avg_trades_per_day': segment_data['trades_per_day'].mean() if 'trades_per_day' in segment_data.columns else 0,
                'avg_pnl': segment_data['realized_pnl'].mean() if 'realized_pnl' in segment_data.columns else 0
            })
            prev_idx = cp_idx
        
        # Get dates for change points (handle potential index issues)
        change_point_dates = []
        if len(change_point_indices) > 0 and 'date' in features.columns:
            try:
                change_point_dates = features.iloc[change_point_indices]['date'].tolist()
            except (IndexError, KeyError) as e:
                logger.warning(f"Could not extract dates for change points: {e}")
                change_point_dates = []
        
        return {
            'indices': change_point_indices,
            'dates': change_point_dates,
            'segments': segments
        }
    
    def _detect_anomalies(self, X: np.ndarray, features: pd.DataFrame) -> Dict:
        """Detect anomalous trades."""
        # Adaptive contamination based on data size
        # Use 5-10% contamination, but ensure minimum samples
        n_samples = len(X)
        contamination_rate = min(0.1, max(0.05, 5.0 / n_samples)) if n_samples > 0 else 0.1
        
        # Use Isolation Forest (better for high-dimensional data)
        iso_forest = IsolationForest(
            contamination=contamination_rate, 
            random_state=42,
            n_estimators=100  # More trees for better precision
        )
        iso_labels = iso_forest.fit_predict(X)
        iso_scores = iso_forest.score_samples(X)
        
        # Also use LOF (better for local anomalies)
        lof = LocalOutlierFactor(
            contamination=contamination_rate,
            n_neighbors=min(20, max(5, n_samples // 10))  # Adaptive neighbors
        )
        lof_labels = lof.fit_predict(X)
        lof_scores = lof.negative_outlier_factor_
        
        self.anomaly_detector = iso_forest
        
        # Combine results (anomaly if either method flags it)
        # This provides higher recall but may have lower precision
        combined_anomalies = (iso_labels == -1) | (lof_labels == -1)
        
        anomaly_indices = np.where(combined_anomalies)[0]
        
        return {
            'indices': anomaly_indices.tolist(),
            'iso_labels': iso_labels,
            'iso_scores': iso_scores,
            'lof_labels': lof_labels,
            'lof_scores': lof_scores,
            'combined_anomalies': combined_anomalies
        }
    
    def get_results(self) -> Dict:
        """Get pattern discovery results."""
        if self.results is None:
            raise ValueError("No results available. Call discover_patterns() first.")
        return self.results

