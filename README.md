# Behavioral Trading Analysis System

A clean, 3-stage system for analyzing trading behavior through market conditions, focusing on self-awareness rather than market prediction.

## Philosophy

> **Behavioral analysis ≠ prediction ≠ optimization**
> 
> The system does not try to beat the market. It tries to help the investor understand *themselves under different market conditions*.

## System Architecture

### Stage 1: Data Ingestion & Normalization
- CSV tradebook support (prototype)
- PDF tradebook parsing (production)
- Data validation and cleaning
- Position reconstruction

### Stage 2: Behavioral & Contextual Analysis
- Market context enrichment (Yahoo Finance)
- Behavioral feature engineering
- Baseline construction (statistical)
- Pattern discovery (clustering, change point detection, anomaly detection)

### Stage 3: Visualization & Explainability
- Interactive dashboards
- Behavioral regime timelines
- Baseline deviation analysis
- Explainable AI explanations


## Key Features

- **No Market Prediction**: Focuses on understanding behavior, not predicting prices
- **Explainable**: All insights are explainable and traceable
- **Personalized Baselines**: Statistical baselines per user, not global assumptions
- **Regime-Aware**: Behavior analysis conditioned on market regimes
- **Production-Ready**: Handles real-world PDF tradebooks from brokers

## Model Choices 

| Model | Why Used | Where |
|-------|----------|-------|
| K-Means/GMM | Discover behavioral modes | Pattern discovery |
| Change Point Detection | Detect behavior shifts | Pattern discovery |
| Isolation Forest | Find unusual trades | Anomaly detection |
| Statistical Baselines | Explainable, stable | Baseline construction |
| Rule-based NLG | Trustworthy explanations | XAI |

## License

MIT

