"""Visualization components for behavioral analysis."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Optional
import logging
import html

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class BehavioralVisualizer:
    """Creates visualizations for behavioral analysis."""
    
    def __init__(self):
        self.figures: Dict[str, go.Figure] = {}
    
    def create_regime_timeline(self, features: pd.DataFrame, clusters: np.ndarray, 
                              change_points: list) -> go.Figure:
        """Create behavioral regime timeline."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Behavioral Regimes Over Time', 'Market Regime Overlay'),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )
        
        # Behavioral clusters timeline
        fig.add_trace(
            go.Scatter(
                x=features['date'],
                y=clusters,
                mode='markers',
                marker=dict(
                    size=8,
                    color=clusters,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Behavioral Cluster")
                ),
                name='Behavioral Cluster',
                hovertemplate='Date: %{x}<br>Cluster: %{y}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Mark change points
        if change_points and len(change_points) > 0:
            for cp_idx in change_points:
                if cp_idx < len(features):
                    try:
                        change_date = features.iloc[cp_idx]['date']
                        # Add vertical line with annotation
                        fig.add_vline(
                            x=change_date,
                            line_dash="dash",
                            line_color="red",
                            line_width=2,
                            annotation_text=f"Change Point {cp_idx}",
                            annotation_position="top",
                            row=1, col=1
                        )
                        # Also add to bottom panel for visibility
                        fig.add_vline(
                            x=change_date,
                            line_dash="dash",
                            line_color="red",
                            line_width=2,
                            row=2, col=1
                        )
                    except (KeyError, IndexError) as e:
                        logger.warning(f"Could not add change point at index {cp_idx}: {e}")
                        continue
        
        # Market regime overlay (use volatility as proxy for market conditions)
        if 'volatility_rolling_std' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['date'],
                    y=features['volatility_rolling_std'],
                    mode='lines',
                    name='Volatility',
                    line=dict(color='blue', width=2),
                    hovertemplate='Date: %{x}<br>Volatility: %{y:.4f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Cluster", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        fig.update_layout(height=600, title_text="Behavioral Regime Timeline")
        
        self.figures['regime_timeline'] = fig
        return fig
    
    def create_deviation_plots(self, features: pd.DataFrame) -> go.Figure:
        """Create baseline vs deviation plots."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Position Size vs Volatility',
                'Trade Frequency vs RSI Regime',
                'Holding Duration Deviation',
                'Overall Deviation Score'
            )
        )
        
        # Position size vs volatility
        if 'position_size_normalized_by_volatility' in features.columns and 'volatility_rolling_std' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['volatility_rolling_std'],
                    y=features['position_size_normalized_by_volatility'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=features.get('realized_pnl', 0),
                        colorscale='RdYlGn',
                        showscale=True,
                        colorbar=dict(title="P&L", x=1.15)
                    ),
                    name='Position Size vs Volatility',
                    hovertemplate='Volatility: %{x:.4f}<br>Position Size Ratio: %{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Trade frequency vs RSI
        if 'trades_per_day' in features.columns and 'rsi_14' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['rsi_14'],
                    y=features['trades_per_day'],
                    mode='markers',
                    marker=dict(size=6, opacity=0.6),
                    name='Trade Frequency vs RSI',
                    hovertemplate='RSI: %{x:.2f}<br>Trades/Day: %{y:.2f}<extra></extra>'
                ),
                row=1, col=2
            )
        
        # Holding duration vs volatility
        if 'holding_duration_vs_volatility' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['date'],
                    y=features['holding_duration_vs_volatility'],
                    mode='lines+markers',
                    name='Holding Duration vs Volatility',
                    line=dict(color='purple', width=2),
                    hovertemplate='Date: %{x}<br>Holding Duration/Volatility: %{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            # Add baseline line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
        
        # Overall deviation score
        if 'overall_deviation_score' in features.columns:
            fig.add_trace(
                go.Scatter(
                    x=features['date'],
                    y=features['overall_deviation_score'],
                    mode='lines+markers',
                    name='Overall Deviation',
                    line=dict(color='red', width=2),
                    hovertemplate='Date: %{x}<br>Deviation Score: %{y:.2f}<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Volatility", row=1, col=1)
        fig.update_yaxes(title_text="Position Size Ratio", row=1, col=1)
        fig.update_xaxes(title_text="RSI (14)", row=1, col=2)
        fig.update_yaxes(title_text="Trades per Day", row=1, col=2)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Holding Duration/Volatility", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Deviation Score", row=2, col=2)
        
        fig.update_layout(height=800, title_text="Baseline Deviation Analysis")
        
        self.figures['deviation_plots'] = fig
        return fig
    
    def create_post_event_charts(self, features: pd.DataFrame) -> go.Figure:
        """Create post-event behavior charts."""
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Trades After Loss', 'Risk Escalation After Loss'),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )
        
        # Check if we have loss data - use realized_pnl < 0 if is_loss doesn't exist
        if 'is_loss' in features.columns:
            loss_mask = features['is_loss'] == 1
        elif 'realized_pnl' in features.columns:
            loss_mask = features['realized_pnl'] < 0
        else:
            loss_mask = pd.Series([False] * len(features), index=features.index)
        
        # Trades after loss
        if loss_mask.any() and 'trades_after_loss' in features.columns:
            loss_data = features[loss_mask].copy()
            loss_data = loss_data[loss_data['trades_after_loss'].notna()]
            
            if len(loss_data) > 0:
                fig.add_trace(
                    go.Bar(
                        x=loss_data['date'],
                        y=loss_data['trades_after_loss'],
                        name='Trades in Next 48h',
                        marker_color='orange',
                        hovertemplate='Date: %{x}<br>Trades After: %{y}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # Risk escalation (position size after loss)
        # Use position_size_normalized_by_volatility for trades after losses
        if loss_mask.any() and 'position_size_normalized_by_volatility' in features.columns:
            loss_data = features[loss_mask].copy()
            loss_data = loss_data[loss_data['position_size_normalized_by_volatility'].notna()]
            
            if len(loss_data) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=loss_data['date'],
                        y=loss_data['position_size_normalized_by_volatility'],
                        mode='markers+lines',
                        name='Position Size After Loss',
                        marker=dict(size=10, color='red', opacity=0.7),
                        line=dict(color='red', width=2),
                        hovertemplate='Date: %{x}<br>Position Size: %{y:.2f}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Update axes with better date formatting
        fig.update_xaxes(
            title_text="Date",
            tickangle=-45,
            type='date',
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Date",
            tickangle=-45,
            type='date',
            row=2, col=1
        )
        fig.update_yaxes(title_text="Trades in Next 48h", row=1, col=1)
        fig.update_yaxes(title_text="Position Size (Normalized)", row=2, col=1)
        
        fig.update_layout(
            height=700,
            title_text="Post-Loss Behavior Analysis",
            xaxis=dict(type='date'),
            xaxis2=dict(type='date')
        )
        
        self.figures['post_event'] = fig
        return fig
    
    def create_performance_matrix(self, features: pd.DataFrame, clusters: np.ndarray, 
                                 n_clusters: int = 3) -> go.Figure:
        """Create regime performance matrix."""
        # Create matrix: market regimes x behavioral clusters
        volatility_regimes = ['low', 'medium', 'high']
        trend_regimes = ['uptrend', 'downtrend', 'sideways']
        
        # Build matrix data: rows = regimes, cols = clusters
        matrix_pnl = []
        matrix_winrate = []
        regime_labels = []
        
        # Check which regime columns exist
        has_vol_regime = 'volatility_regime' in features.columns
        has_trend_regime = 'trend_regime' in features.columns
        has_pnl = 'realized_pnl' in features.columns
        
        if not (has_vol_regime or has_trend_regime) or not has_pnl:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Performance matrix requires volatility_regime/trend_regime and realized_pnl columns",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(title="Regime Performance Matrix - Data Not Available", height=400)
            self.figures['performance_matrix'] = fig
            return fig
        
        # Use volatility regime as primary (or trend if vol not available)
        if has_vol_regime:
            regimes_to_use = volatility_regimes
            regime_col = 'volatility_regime'
        else:
            regimes_to_use = trend_regimes
            regime_col = 'trend_regime'
        
        # Build matrix
        for regime in regimes_to_use:
            regime_mask = features[regime_col] == regime
            if not regime_mask.any():
                continue
            
            row_pnl = []
            row_winrate = []
            
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                combined_mask = regime_mask & cluster_mask
                
                if combined_mask.any():
                    avg_pnl = features.loc[combined_mask, 'realized_pnl'].mean()
                    win_rate = (features.loc[combined_mask, 'realized_pnl'] > 0).mean()
                    row_pnl.append(avg_pnl)
                    row_winrate.append(win_rate * 100)  # Convert to percentage
                else:
                    row_pnl.append(np.nan)
                    row_winrate.append(np.nan)
            
            if any(not np.isnan(p) for p in row_pnl):  # Only add if we have data
                matrix_pnl.append(row_pnl)
                matrix_winrate.append(row_winrate)
                regime_labels.append(regime.title())
        
        # Create heatmap with P&L
        if matrix_pnl and len(matrix_pnl) > 0:
            cluster_labels = [f'Cluster {i}' for i in range(n_clusters)]
            
            # Create text annotations with both P&L and win rate
            text_matrix = []
            for i, row_pnl in enumerate(matrix_pnl):
                text_row = []
                for j, pnl in enumerate(row_pnl):
                    if not np.isnan(pnl):
                        wr = matrix_winrate[i][j] if i < len(matrix_winrate) and j < len(matrix_winrate[i]) else 0
                        text_row.append(f"P&L: ${pnl:.2f}<br>Win: {wr:.1f}%")
                    else:
                        text_row.append("")
                text_matrix.append(text_row)
            
            fig = go.Figure(data=go.Heatmap(
                z=matrix_pnl,
                x=cluster_labels,
                y=regime_labels,
                colorscale='RdYlGn',
                text=text_matrix,
                texttemplate="%{text}",
                textfont={"size": 11},
                colorbar=dict(title="Avg P&L ($)"),
                hovertemplate='Regime: %{y}<br>Cluster: %{x}<br>Avg P&L: $%{z:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Regime Performance Matrix (Average P&L by Market Regime and Behavioral Cluster)",
                xaxis_title="Behavioral Cluster",
                yaxis_title="Market Regime",
                height=max(400, len(regime_labels) * 80)
            )
            
            self.figures['performance_matrix'] = fig
            return fig
        
        # Fallback: empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available for performance matrix",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(title="Regime Performance Matrix", height=400)
        self.figures['performance_matrix'] = fig
        return fig
    
    def create_cluster_timeline(self, features: pd.DataFrame, clusters: np.ndarray) -> go.Figure:
        """
        Create a line graph showing trades over time with clusters highlighted/circled.
        
        Args:
            features: DataFrame with trade features
            clusters: Cluster labels for each trade
            
        Returns:
            Plotly figure with cluster visualization
        """
        fig = go.Figure()
        
        # Get unique clusters
        unique_clusters = np.unique(clusters)
        
        # Color palette for clusters
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Plot each cluster separately
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_data = features.loc[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Sort by date
            cluster_data = cluster_data.sort_values('date')
            
            # Use price or cumulative P&L for y-axis
            if 'price' in cluster_data.columns:
                y_values = cluster_data['price']
                y_label = 'Trade Price'
            elif 'realized_pnl' in cluster_data.columns:
                y_values = cluster_data['realized_pnl'].cumsum()
                y_label = 'Cumulative P&L'
            else:
                y_values = range(len(cluster_data))
                y_label = 'Trade Index'
            
            # Plot line for this cluster
            fig.add_trace(go.Scatter(
                x=cluster_data['date'],
                y=y_values,
                mode='lines+markers',
                name=f'Cluster {cluster_id}',
                line=dict(
                    color=colors[cluster_id % len(colors)],
                    width=3
                ),
                marker=dict(
                    size=12,
                    color=colors[cluster_id % len(colors)],
                    line=dict(width=2, color='white'),
                    symbol='circle'
                ),
                hovertemplate=f'<b>Cluster {cluster_id}</b><br>' +
                            'Date: %{x}<br>' +
                            f'{y_label}: %{{y:.2f}}<br>' +
                            '<extra></extra>'
            ))
            
            # Add circles/highlights around cluster points
            fig.add_trace(go.Scatter(
                x=cluster_data['date'],
                y=y_values,
                mode='markers',
                name=f'Cluster {cluster_id} Highlight',
                marker=dict(
                    size=20,
                    color=colors[cluster_id % len(colors)],
                    opacity=0.2,
                    line=dict(width=3, color=colors[cluster_id % len(colors)])
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add annotations for cluster labels at first occurrence
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_data = features.loc[cluster_mask]
            
            if len(cluster_data) > 0:
                cluster_data = cluster_data.sort_values('date')
                first_trade = cluster_data.iloc[0]
                
                if 'price' in first_trade:
                    y_val = first_trade['price']
                elif 'realized_pnl' in features.columns:
                    y_val = features.loc[:first_trade.name, 'realized_pnl'].sum() if first_trade.name > 0 else first_trade.get('realized_pnl', 0)
                else:
                    y_val = 0
                
                fig.add_annotation(
                    x=first_trade['date'],
                    y=y_val,
                    text=f'<b>Cluster {cluster_id}</b>',
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=colors[cluster_id % len(colors)],
                    bgcolor='white',
                    bordercolor=colors[cluster_id % len(colors)],
                    borderwidth=2,
                    font=dict(size=12, color=colors[cluster_id % len(colors)])
                )
        
        fig.update_layout(
            title='Behavioral Clusters Over Time',
            xaxis_title='Date',
            yaxis_title=y_label,
            hovermode='closest',
            height=600,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        self.figures['cluster_timeline'] = fig
        return fig
    
    def create_trade_journey_timeline(self, features: pd.DataFrame) -> go.Figure:
        """
        Create a user-friendly trade journey timeline showing each trade's entry to exit journey.
        
        This visualization helps amateur investors easily see:
        - When they entered and exited each trade
        - Profit/loss for each trade
        - Which stocks they traded
        - Visual journey from entry to exit
        """
        fig = go.Figure()
        
        # Sort by date
        df = features.sort_values('date').copy()
        
        # Group trades by symbol for better visualization
        if 'symbol' in df.columns:
            symbols = df['symbol'].unique()
            colors = px.colors.qualitative.Set3[:len(symbols)]
            symbol_colors = dict(zip(symbols, colors))
        else:
            symbol_colors = {}
        
        # Create entry and exit points
        entry_dates = []
        exit_dates = []
        entry_prices = []
        exit_prices = []
        pnl_values = []
        symbols_list = []
        trade_labels = []
        
        # Process buy-sell pairs - match by reconstructing positions
        if 'side' in df.columns and 'symbol' in df.columns:
            buy_trades = df[df['side'] == 'buy'].copy()
            sell_trades = df[df['side'] == 'sell'].copy()
            
            # Match buys with sells by symbol and date order
            for idx, buy in buy_trades.iterrows():
                symbol = buy.get('symbol', 'UNKNOWN')
                buy_date = buy['date']
                buy_price = buy.get('price', 0)
                buy_qty = buy.get('quantity', 0)
                
                # Find corresponding sell (same symbol, after buy date)
                matching_sells = sell_trades[
                    (sell_trades['symbol'] == symbol) & 
                    (sell_trades['date'] > buy_date)
                ].sort_values('date')
                
                if len(matching_sells) > 0:
                    # Take first matching sell
                    sell = matching_sells.iloc[0]
                    sell_date = sell['date']
                    sell_price = sell.get('price', 0)
                    sell_qty = sell.get('quantity', 0)
                    
                    # Calculate P&L
                    if 'realized_pnl' in sell:
                        pnl = sell['realized_pnl']
                    else:
                        # Estimate P&L
                        qty = min(buy_qty, sell_qty)
                        pnl = (sell_price - buy_price) * qty
                    
                    entry_dates.append(buy_date)
                    exit_dates.append(sell_date)
                    entry_prices.append(buy_price)
                    exit_prices.append(sell_price)
                    pnl_values.append(pnl)
                    symbols_list.append(symbol)
                    trade_labels.append(f"{symbol}<br>Entry: ${buy_price:.2f}<br>Exit: ${sell_price:.2f}<br>P&L: ${pnl:.2f}")
        else:
            # Fallback: use all trades as individual points
            for idx, row in df.iterrows():
                entry_dates.append(row['date'])
                exit_dates.append(row['date'])
                entry_prices.append(row.get('price', 0))
                exit_prices.append(row.get('price', 0))
                pnl_values.append(row.get('realized_pnl', 0))
                symbols_list.append(row.get('symbol', 'UNKNOWN'))
        
        # Add entry points (green)
        if entry_dates:
            fig.add_trace(go.Scatter(
                x=entry_dates,
                y=entry_prices,
                mode='markers',
                name='Entry Points',
                marker=dict(
                    size=12,
                    color='green',
                    symbol='triangle-up',
                    line=dict(width=2, color='darkgreen')
                ),
                text=[f"Entry: {s}<br>Price: ${p:.2f}" for s, p in zip(symbols_list, entry_prices)],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add exit points (color by P&L)
        if exit_dates:
            exit_colors = ['red' if pnl < 0 else 'green' for pnl in pnl_values]
            fig.add_trace(go.Scatter(
                x=exit_dates,
                y=exit_prices,
                mode='markers',
                name='Exit Points',
                marker=dict(
                    size=12,
                    color=exit_colors,
                    symbol='triangle-down',
                    line=dict(width=2, color='darkred')
                ),
                text=[f"Exit: {s}<br>Price: ${p:.2f}<br>P&L: ${pnl:.2f}" 
                      for s, p, pnl in zip(symbols_list, exit_prices, pnl_values)],
                hovertemplate='%{text}<extra></extra>'
            ))
        
        # Add journey lines (connecting entry to exit)
        for i in range(len(entry_dates)):
            line_color = 'red' if pnl_values[i] < 0 else 'green'
            line_width = 2 if abs(pnl_values[i]) > 100 else 1
            fig.add_trace(go.Scatter(
                x=[entry_dates[i], exit_dates[i]],
                y=[entry_prices[i], exit_prices[i]],
                mode='lines',
                name=f'Trade {i+1}',
                line=dict(color=line_color, width=line_width, dash='dash'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add P&L annotations
        for i in range(len(exit_dates)):
            annotation_text = f"${pnl_values[i]:.2f}"
            fig.add_annotation(
                x=exit_dates[i],
                y=exit_prices[i],
                text=annotation_text,
                showarrow=True,
                arrowhead=2,
                arrowcolor='red' if pnl_values[i] < 0 else 'green',
                bgcolor='white',
                bordercolor='red' if pnl_values[i] < 0 else 'green',
                borderwidth=1,
                font=dict(size=10, color='red' if pnl_values[i] < 0 else 'green')
            )
        
        fig.update_layout(
            title='Trade Journey Timeline - Your Trading Path',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=700,
            hovermode='closest',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        self.figures['trade_journey'] = fig
        return fig
    
    def create_signal_following_scorecard(self, features: pd.DataFrame) -> go.Figure:
        """
        Create a user-friendly scorecard showing how well the investor followed technical signals.
        
        This helps amateur investors understand:
        - Did they follow RSI signals?
        - Did they follow MACD signals?
        - Did they follow EMA trend signals?
        - Visual indicators of missed opportunities
        """
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'RSI Signal Following', 'MACD Signal Following',
                'EMA Trend Following', 'Overall Signal Score',
                'Missed Opportunities', 'Signal Adherence Summary'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "table"}]],
            vertical_spacing=0.25,  # Increased spacing to prevent text overlap
            row_heights=[0.35, 0.35, 0.30],  # Adjusted row heights - more space for row 1
            horizontal_spacing=0.15
        )
        
        # RSI Signal Analysis
        if 'rsi_14' in features.columns:
            # Overbought signals (RSI > 70) - should sell
            overbought_mask = features['rsi_14'] > 70
            overbought_trades = features[overbought_mask]
            
            # Oversold signals (RSI < 30) - should buy
            oversold_mask = features['rsi_14'] < 30
            oversold_trades = features[oversold_mask]
            
            # Count actual trades at these signals
            overbought_sells = len(overbought_trades[overbought_trades.get('side', '') == 'sell']) if 'side' in features.columns else 0
            oversold_buys = len(oversold_trades[oversold_trades.get('side', '') == 'buy']) if 'side' in features.columns else 0
            
            fig.add_trace(go.Bar(
                x=['Overbought Signals<br>(Should Sell)', 'Oversold Signals<br>(Should Buy)'],
                y=[len(overbought_trades), len(oversold_trades)],
                name='Total Signals',
                marker_color='lightblue',
                text=[len(overbought_trades), len(oversold_trades)],
                textposition='outside'
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=['Overbought Signals<br>(Should Sell)', 'Oversold Signals<br>(Should Buy)'],
                y=[overbought_sells, oversold_buys],
                name='Trades Taken',
                marker_color='green',
                text=[overbought_sells, oversold_buys],
                textposition='outside'
            ), row=1, col=1)
        
        # MACD Signal Analysis
        if 'macd_line' in features.columns and 'macd_signal' in features.columns:
            # MACD bullish: macd_line > macd_signal (should buy)
            # MACD bearish: macd_line < macd_signal (should sell)
            bullish_mask = features['macd_line'] > features['macd_signal']
            bearish_mask = features['macd_line'] < features['macd_signal']
            
            bullish_trades = features[bullish_mask]
            bearish_trades = features[bearish_mask]
            
            bullish_buys = len(bullish_trades[bullish_trades.get('side', '') == 'buy']) if 'side' in features.columns else 0
            bearish_sells = len(bearish_trades[bearish_trades.get('side', '') == 'sell']) if 'side' in features.columns else 0
            
            fig.add_trace(go.Bar(
                x=['Bullish Signals<br>(Should Buy)', 'Bearish Signals<br>(Should Sell)'],
                y=[len(bullish_trades), len(bearish_trades)],
                name='Total Signals',
                marker_color='lightblue',
                showlegend=False,
                text=[len(bullish_trades), len(bearish_trades)],
                textposition='outside'
            ), row=1, col=2)
            
            fig.add_trace(go.Bar(
                x=['Bullish Signals<br>(Should Buy)', 'Bearish Signals<br>(Should Sell)'],
                y=[bullish_buys, bearish_sells],
                name='Trades Taken',
                marker_color='green',
                showlegend=False,
                text=[bullish_buys, bearish_sells],
                textposition='outside'
            ), row=1, col=2)
        
        # EMA Trend Following
        if 'trend_regime' in features.columns and 'side' in features.columns:
            uptrend_trades = features[features['trend_regime'] == 'uptrend']
            downtrend_trades = features[features['trend_regime'] == 'downtrend']
            
            uptrend_buys = len(uptrend_trades[uptrend_trades['side'] == 'buy'])
            downtrend_sells = len(downtrend_trades[downtrend_trades['side'] == 'sell'])
            
            fig.add_trace(go.Bar(
                x=['Uptrend<br>(Should Buy)', 'Downtrend<br>(Should Sell)'],
                y=[len(uptrend_trades), len(downtrend_trades)],
                name='Total Signals',
                marker_color='lightblue',
                showlegend=False,
                text=[len(uptrend_trades), len(downtrend_trades)],
                textposition='outside'
            ), row=2, col=1)
            
            fig.add_trace(go.Bar(
                x=['Uptrend<br>(Should Buy)', 'Downtrend<br>(Should Sell)'],
                y=[uptrend_buys, downtrend_sells],
                name='Trades Taken',
                marker_color='green',
                showlegend=False,
                text=[uptrend_buys, downtrend_sells],
                textposition='outside'
            ), row=2, col=1)
        
        # Overall Signal Score (Gauge)
        total_signals = 0
        signals_followed = 0
        
        if 'rsi_14' in features.columns:
            total_signals += len(overbought_trades) + len(oversold_trades)
            signals_followed += overbought_sells + oversold_buys
        
        if 'macd_line' in features.columns:
            total_signals += len(bullish_trades) + len(bearish_trades)
            signals_followed += bullish_buys + bearish_sells
        
        if 'trend_regime' in features.columns:
            total_signals += len(uptrend_trades) + len(downtrend_trades)
            signals_followed += uptrend_buys + downtrend_sells
        
        signal_score = (signals_followed / total_signals * 100) if total_signals > 0 else 0
        
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=signal_score,
            domain={'x': [0.1, 0.9], 'y': [0.15, 0.85]},  # Reduced domain to add padding and prevent overlap
            title={'text': "Signal Following<br>Score (%)", 'font': {'size': 14}},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen" if signal_score >= 70 else "orange" if signal_score >= 50 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 70], 'color': "gray"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ), row=2, col=2)
        
        # Missed Opportunities
        missed_opportunities = []
        if 'rsi_14' in features.columns:
            missed_overbought = len(overbought_trades) - overbought_sells
            missed_oversold = len(oversold_trades) - oversold_buys
            if missed_overbought > 0:
                missed_opportunities.append(['RSI Overbought', missed_overbought, 'Missed Sell Signal'])
            if missed_oversold > 0:
                missed_opportunities.append(['RSI Oversold', missed_oversold, 'Missed Buy Signal'])
        
        if 'macd_line' in features.columns:
            missed_bullish = len(bullish_trades) - bullish_buys
            missed_bearish = len(bearish_trades) - bearish_sells
            if missed_bullish > 0:
                missed_opportunities.append(['MACD Bullish', missed_bullish, 'Missed Buy Signal'])
            if missed_bearish > 0:
                missed_opportunities.append(['MACD Bearish', missed_bearish, 'Missed Sell Signal'])
        
        if missed_opportunities:
            opp_df = pd.DataFrame(missed_opportunities, columns=['Signal Type', 'Count', 'Action'])
            fig.add_trace(go.Bar(
                x=opp_df['Signal Type'],
                y=opp_df['Count'],
                marker_color='orange',
                text=opp_df['Count'],
                textposition='outside',
                name='Missed Opportunities'
            ), row=3, col=1)
        
        # Summary Table
        summary_data = []
        if 'rsi_14' in features.columns:
            rsi_score = ((overbought_sells + oversold_buys) / (len(overbought_trades) + len(oversold_trades)) * 100) if (len(overbought_trades) + len(oversold_trades)) > 0 else 0
            summary_data.append(['RSI Signals', f"{signals_followed}/{total_signals}", f"{rsi_score:.1f}%"])
        
        if 'macd_line' in features.columns:
            macd_score = ((bullish_buys + bearish_sells) / (len(bullish_trades) + len(bearish_trades)) * 100) if (len(bullish_trades) + len(bearish_trades)) > 0 else 0
            summary_data.append(['MACD Signals', f"{bullish_buys + bearish_sells}/{len(bullish_trades) + len(bearish_trades)}", f"{macd_score:.1f}%"])
        
        if 'trend_regime' in features.columns:
            trend_score = ((uptrend_buys + downtrend_sells) / (len(uptrend_trades) + len(downtrend_trades)) * 100) if (len(uptrend_trades) + len(downtrend_trades)) > 0 else 0
            summary_data.append(['EMA Trend', f"{uptrend_buys + downtrend_sells}/{len(uptrend_trades) + len(downtrend_trades)}", f"{trend_score:.1f}%"])
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data, columns=['Signal Type', 'Followed/Total', 'Score'])
            fig.add_trace(go.Table(
                header=dict(values=list(summary_df.columns),
                           fill_color='paleturquoise',
                           align='left'),
                cells=dict(values=[summary_df[col] for col in summary_df.columns],
                          fill_color='lavender',
                          align='left')
            ), row=3, col=2)
        
        fig.update_layout(
            height=1300,  # Increased height to accommodate spacing
            title_text="Signal Following Scorecard - How Well Did You Follow Technical Indicators?",
            showlegend=True,
            margin=dict(t=120, b=80, l=60, r=60)  # Increased bottom margin to prevent overlap
        )
        
        # Add extra bottom margin to MACD chart to prevent overlap with gauge
        fig.update_layout(
            xaxis2=dict(
                automargin=True,
                tickangle=-45
            )
        )
        
        # Add explanation annotation for gauge calculation (positioned below gauge)
        if total_signals > 0:
            fig.add_annotation(
                text=f"<b>Gauge Calculation:</b><br>" +
                     f"Score = (Trades Following Signals / Total Signals) × 100<br>" +
                     f"Your Score: {signals_followed}/{total_signals} = {signal_score:.1f}%<br>" +
                     f"<b>Interpretation:</b><br>" +
                     f"• 70-100%: Excellent signal following<br>" +
                     f"• 50-70%: Good signal following<br>" +
                     f"• 0-50%: Needs improvement",
                xref="paper", yref="paper",
                x=0.75, y=0.10,  # Positioned lower to avoid overlap
                xanchor="left",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor="black",
                borderwidth=1,
                font=dict(size=9, color="black"),
                row=2, col=2
            )
        
        # Update axes with proper spacing
        fig.update_xaxes(title_text="Signal Type", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(
            title_text="Signal Type", 
            row=1, col=2,
            tickangle=-45,  # Angle labels to prevent overlap
            automargin=True  # Automatically adjust margins
        )
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_xaxes(title_text="Trend Type", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_xaxes(title_text="Signal Type", row=3, col=1)
        fig.update_yaxes(title_text="Missed Count", row=3, col=1)
        
        self.figures['signal_scorecard'] = fig
        return fig
    
    def create_stability_scorecard(self, stability_results: Dict) -> go.Figure:
        """
        Create a visualization for the Behavioral Stability / Consistency Score.
        
        This score measures how consistent a trader's behavior is over time.
        It does NOT measure skill or profitability - only consistency.
        
        Args:
            stability_results: Dictionary from BehavioralStabilityAnalyzer.calculate_stability_score()
            
        Returns:
            Plotly figure with stability score visualization
        """
        from plotly.subplots import make_subplots
        
        if stability_results.get('stability_score') is None:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data to calculate behavioral stability score.",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=14)
            )
            fig.update_layout(title="Behavioral Stability Score - Data Not Available", height=400)
            self.figures['stability_scorecard'] = fig
            return fig
        
        score = stability_results['stability_score']
        components = stability_results.get('components', {})
        
        # Create subplots: gauge at top, component breakdown below
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Behavioral Stability Score', 'Component Breakdown'),
            vertical_spacing=0.25,
            row_heights=[0.5, 0.5],
            specs=[[{"type": "indicator"}], [{"type": "bar"}]]
        )
        
        # Main gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Behavioral Stability<br>Consistency Index"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkgreen" if score >= 70 else "orange" if score >= 50 else "red"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 50], 'color': "gray"},
                    {'range': [50, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ), row=1, col=1)
        
        # Component breakdown bar chart
        if components:
            comp_names = []
            comp_scores = []
            for comp_name, comp_data in components.items():
                if comp_data.get('stability_score') is not None:
                    comp_names.append(comp_data.get('feature_name', comp_name))
                    comp_scores.append(comp_data['stability_score'])
            
            if comp_names:
                fig.add_trace(go.Bar(
                    x=comp_names,
                    y=comp_scores,
                    marker_color=['green' if s >= 70 else 'orange' if s >= 50 else 'red' for s in comp_scores],
                    text=[f"{s:.1f}" for s in comp_scores],
                    textposition='outside',
                    name='Component Stability'
                ), row=2, col=1)
        
        # Add interpretation text
        interpretation = stability_results.get('interpretation', '')
        note = stability_results.get('note', '')
        
        fig.add_annotation(
            text=f"<b>Interpretation:</b><br>{interpretation.split('Overall Assessment:')[1].split('Component Analysis:')[0] if 'Overall Assessment:' in interpretation else ''}<br><br><b>Note:</b> {note}",
            xref="paper", yref="paper",
            x=0.5, y=-0.15,
            xanchor="center",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10)
        )
        
        fig.update_xaxes(title_text="Behavioral Component", row=2, col=1)
        fig.update_yaxes(title_text="Stability Score (0-100)", row=2, col=1, range=[0, 100])
        
        fig.update_layout(
            height=800,
            title_text="Behavioral Stability / Consistency Score<br><sub>This score measures consistency of behavior, not skill or profitability</sub>",
            showlegend=False
        )
        
        self.figures['stability_scorecard'] = fig
        return fig
    
    def save_all_figures(self, output_dir: str = "output/"):
        """Save all figures to HTML files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for name, fig in self.figures.items():
            filepath = os.path.join(output_dir, f"{name}.html")
            fig.write_html(filepath)
            logger.info(f"Saved {name} to {filepath}")
    
    def create_unified_dashboard(self, output_dir: str = "output/", 
                                xai_file: str = "xai_explanation.txt",
                                report_file: str = "behavioral_report.txt") -> str:
        """
        Create a unified dark-themed dashboard HTML page with all visualizations and reports.
        
        Args:
            output_dir: Directory containing visualization files and reports
            xai_file: Filename of XAI explanation text file
            report_file: Filename of behavioral report text file
            
        Returns:
            Path to the generated dashboard HTML file
        """
        import os
        
        # Visualization order and descriptions
        viz_order = [
            ('regime_timeline', 'Behavioral Regime Timeline', 'Shows how your trading behavior clusters changed over time, with market regime overlays'),
            ('cluster_timeline', 'Cluster Distribution Timeline', 'Distribution of behavioral clusters across your trading period'),
            ('trade_journey', 'Trade Journey Timeline', 'Individual trade journeys from entry to exit with P&L'),
            ('signal_scorecard', 'Signal Following Scorecard', 'How well you followed technical indicators (RSI, MACD, EMA)'),
            ('stability_scorecard', 'Behavioral Stability Score', 'Consistency of your trading behavior over time (does not measure skill or profitability)'),
            ('post_event', 'Post-Loss Behavior Analysis', 'Trading behavior after losses (revenge trading patterns)'),
            ('performance_matrix', 'Performance Matrix', 'Performance across different market regimes and behavioral clusters'),
            ('deviation_plots', 'Behavioral Deviation Plots', 'How your trading deviates from baseline patterns')
        ]
        
        # Read text files
        xai_path = os.path.join(output_dir, xai_file)
        report_path = os.path.join(output_dir, report_file)
        
        xai_content = ""
        if os.path.exists(xai_path):
            try:
                with open(xai_path, 'r', encoding='utf-8') as f:
                    xai_content = f.read()
            except UnicodeDecodeError:
                with open(xai_path, 'r', encoding='utf-8', errors='replace') as f:
                    xai_content = f.read()
        
        report_content = ""
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    report_content = f.read()
            except UnicodeDecodeError:
                with open(report_path, 'r', encoding='utf-8', errors='replace') as f:
                    report_content = f.read()
        
        # Generate HTML
        html_parts = []
        
        # HTML Header with dark theme CSS
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Behavioral Trading Analysis Dashboard</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            background: #000000;
            color: #ffffff;
            line-height: 1.7;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            padding: 40px 20px;
            border-bottom: 2px solid #333333;
            margin-bottom: 40px;
            background: #1a1a1a;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }
        
        .header h1 {
            font-size: 2.2em;
            color: #ffffff;
            margin-bottom: 12px;
            font-weight: 600;
        }
        
        .header p {
            color: #e9ecef;
            font-size: 1.05em;
        }
        
        .section {
            background: #1a1a1a;
            border: 1px solid #333333;
            border-radius: 8px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
        }
        
        .section-title {
            font-size: 1.6em;
            color: #ffffff;
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 2px solid #333333;
            font-weight: 600;
        }
        
        .section-description {
            color: #e9ecef;
            font-size: 0.95em;
            margin-bottom: 24px;
            line-height: 1.6;
        }
        
        .viz-container {
            background: #0a0a0a;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            border: 1px solid #333333;
        }
        
        .viz-container iframe {
            width: 100%;
            height: 700px;
            border: none;
            border-radius: 4px;
            background: #ffffff;
        }
        
        .text-content {
            background: #0a0a0a;
            border-radius: 6px;
            padding: 24px;
            margin: 20px 0;
            border: 1px solid #333333;
            white-space: pre-wrap;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', monospace;
            font-size: 0.9em;
            line-height: 1.8;
            color: #ffffff;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .text-content::-webkit-scrollbar {
            width: 8px;
        }
        
        .text-content::-webkit-scrollbar-track {
            background: #1a1a1a;
            border-radius: 4px;
        }
        
        .text-content::-webkit-scrollbar-thumb {
            background: #6c757d;
            border-radius: 4px;
        }
        
        .text-content::-webkit-scrollbar-thumb:hover {
            background: #868e96;
        }
        
        .download-btn {
            display: inline-block;
            background: #6c757d;
            color: #ffffff;
            padding: 10px 20px;
            text-decoration: none;
            border: none;
            border-radius: 5px;
            font-weight: 500;
            margin: 10px 10px 10px 0;
            transition: background-color 0.2s ease;
            cursor: pointer;
            font-size: 0.95em;
            font-family: inherit;
        }
        
        .download-btn:hover {
            background: #868e96;
        }
        
        .download-btn:active {
            background: #5a6268;
        }
        
        .scale-info {
            background: #0a0a0a;
            border-left: 3px solid #6c757d;
            padding: 14px 16px;
            margin: 16px 0;
            border-radius: 4px;
            font-size: 0.9em;
            color: #e9ecef;
        }
        
        .scale-info strong {
            color: #ffffff;
        }
        
        .nav-menu {
            position: fixed;
            top: 20px;
            right: 20px;
            background: #1a1a1a;
            border: 1px solid #333333;
            border-radius: 6px;
            padding: 16px;
            z-index: 1000;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
        }
        
        .nav-menu h3 {
            color: #ffffff;
            font-size: 0.95em;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .nav-menu a {
            display: block;
            color: #e9ecef;
            text-decoration: none;
            padding: 6px 0;
            font-size: 0.9em;
            transition: color 0.2s;
        }
        
        .nav-menu a:hover {
            color: #ffffff;
        }
        
        @media (max-width: 768px) {
            body {
                padding: 12px;
            }
            
            .nav-menu {
                position: relative;
                top: 0;
                right: 0;
                margin-bottom: 20px;
            }
            
            .header {
                padding: 30px 15px;
            }
            
            .header h1 {
                font-size: 1.6em;
            }
            
            .section {
                padding: 20px;
            }
            
            .viz-container iframe {
                height: 500px;
            }
        }
    </style>
    <script>
        function downloadFile(filename) {
            // Create a temporary anchor element to trigger download
            const link = document.createElement('a');
            link.href = filename;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    </script>
</head>
<body>
    <div class="nav-menu">
        <h3>Navigation</h3>
        <a href="#visualizations">Visualizations</a>
        <a href="#xai-explanation">XAI Summary</a>
        <a href="#behavioral-report">Full Report</a>
    </div>
    
    <div class="container">
        <div class="header">
            <h1>Behavioral Trading Analysis Dashboard</h1>
            <p>Comprehensive analysis of your trading behavior, patterns, and performance</p>
        </div>
        
        <div id="visualizations">
            <div class="section">
                <h2 class="section-title">Interactive Visualizations</h2>
                <p class="section-description">Explore your trading patterns through interactive charts. Hover over data points for detailed information.</p>
""")
        
        # Add visualizations
        for viz_name, viz_title, viz_desc in viz_order:
            viz_file = os.path.join(output_dir, f"{viz_name}.html")
            if os.path.exists(viz_file):
                html_parts.append(f"""
                <div class="section">
                    <h3 class="section-title">{viz_title}</h3>
                    <p class="section-description">{viz_desc}</p>
                    <div class="scale-info">
                        <strong>Note:</strong> Interactive charts - zoom, hover for details, toggle series via legend.
                    </div>
                    <div class="viz-container">
                        <iframe src="{viz_name}.html" title="{viz_title}"></iframe>
                    </div>
                </div>
""")
        
        html_parts.append("""
        </div>
        
        <div id="xai-explanation">
            <div class="section">
                <h2 class="section-title">Explainable AI Summary</h2>
                <p class="section-description">AI-generated explanation of your trading behavior using rule-based Natural Language Generation and feature contribution ranking.</p>
                <button onclick="downloadFile('xai_explanation.txt')" class="download-btn">Download XAI Explanation</button>
                <div class="text-content">""")
        
        # Add XAI content (escape HTML)
        xai_escaped = html.escape(xai_content) if xai_content else "XAI explanation file not found."
        html_parts.append(xai_escaped)
        
        html_parts.append("""
                </div>
            </div>
        </div>
        
        <div id="behavioral-report">
            <div class="section">
                <h2 class="section-title">Detailed Behavioral Report</h2>
                <p class="section-description">Comprehensive analysis including statistics, baselines, clusters, change points, and anomalies.</p>
                <button onclick="downloadFile('behavioral_report.txt')" class="download-btn">Download Full Report</button>
                <div class="text-content">""")
        
        # Add report content (escape HTML)
        report_escaped = html.escape(report_content) if report_content else "Behavioral report file not found."
        html_parts.append(report_escaped)
        
        html_parts.append("""
                </div>
            </div>
        </div>
        
        <div class="section" style="text-align: center; padding: 30px; background: #0a0a0a;">
            <p style="color: #e9ecef; font-size: 0.9em;">
                Generated by Behavioral Trading Analysis System<br>
                Using Explainable AI (XAI) and Advanced Pattern Recognition
            </p>
        </div>
    </div>
</body>
</html>""")
        
        # Write HTML file
        dashboard_path = os.path.join(output_dir, "dashboard.html")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(''.join(html_parts))
        
        logger.info(f"Unified dashboard saved to {dashboard_path}")
        return dashboard_path

