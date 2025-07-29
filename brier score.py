def calculate_survival_brier_scores(model, X, y, event, 
                                  distribution='logistic', 
                                  time_points=None):
    """
    Calculate time-dependent Brier scores for XGBoost AFT models
    
    Args:
        model: Trained XGBoost AFT model
        X: Feature matrix (any population - train/val/oot)
        y: Survival times
        event: Event indicators (1=event, 0=censored)
        distribution: 'normal', 'logistic', or 'extreme'
        time_points: List of time points for evaluation
    
    Returns:
        Dictionary with Brier scores and IBS
    """
    if time_points is None:
        time_points = np.arange(30, 366, 30)
    
    # Get AFT predictions (log-scale survival times)
    dmatrix = xgb.DMatrix(X)
    log_pred = np.log(model.predict(dmatrix))  # XGBoost AFT predicts median survival time
    
    # Estimate scale parameter from residuals (observed events only)
    observed_mask = event == 1
    if observed_mask.sum() > 0:
        residuals = np.log(y[observed_mask]) - log_pred[observed_mask]
        
        if distribution == 'normal':
            scale_param = np.std(residuals)
        elif distribution == 'logistic':
            # For logistic distribution, scale = std * sqrt(3) / pi
            scale_param = np.std(residuals) * np.sqrt(3) / np.pi
        elif distribution == 'extreme':
            # For Gumbel distribution, scale = std * sqrt(6) / pi  
            scale_param = np.std(residuals) * np.sqrt(6) / np.pi
        else:
            raise ValueError("Distribution must be 'normal', 'logistic', or 'extreme'")
    else:
        scale_param = 1.0  # Default fallback
    
    print(f"Population size: {len(X):,}")
    print(f"Events observed: {observed_mask.sum():,} ({observed_mask.mean():.1%})")
    print(f"Estimated scale parameter: {scale_param:.3f}")
    
    def survival_probability(log_pred_time, eval_time, dist, scale):
        """Calculate survival probability S(t) for different AFT distributions"""
        # Standardized residual: (log(t) - log(predicted_median)) / scale
        z = (np.log(eval_time) - log_pred_time) / scale
        
        if dist == 'normal':
            # Normal AFT: S(t) = 1 - Φ(z)
            return 1 - stats.norm.cdf(z)
        elif dist == 'logistic':
            # Logistic AFT: S(t) = 1 / (1 + exp(z))
            return 1 / (1 + np.exp(np.clip(z, -500, 500)))  # Clip to prevent overflow
        elif dist == 'extreme':
            # Extreme AFT (Gumbel): S(t) = exp(-exp(z))
            return np.exp(-np.exp(np.clip(z, -500, 50)))  # Clip to prevent overflow
    
    brier_scores = []
    
    for t in time_points:
        # Calculate predicted survival probability at time t
        surv_prob = survival_probability(log_pred, t, distribution, scale_param)
        
        # Actual survival status at time t
        survived_t = (y > t) | ((y <= t) & (event == 0))
        survived_t = survived_t.astype(float)
        
        # Only include subjects at risk at time t
        at_risk_mask = y >= t
        
        if at_risk_mask.sum() > 50:  # Need sufficient sample size
            # Calculate Brier score
            bs = brier_score_loss(survived_t[at_risk_mask], 
                                surv_prob[at_risk_mask])
            
            brier_scores.append({
                'time': t,
                'brier_score': bs,
                'n_at_risk': at_risk_mask.sum(),
                'mean_pred_prob': surv_prob[at_risk_mask].mean(),
                'actual_survival_rate': survived_t[at_risk_mask].mean()
            })
    
    # Calculate Integrated Brier Score (IBS)
    if len(brier_scores) > 1:
        times = [bs['time'] for bs in brier_scores]
        scores = [bs['brier_score'] for bs in brier_scores]
        ibs = np.trapz(scores, times) / (times[-1] - times[0])
    else:
        ibs = np.nan
    
    return {
        'time_dependent_brier_scores': brier_scores,
        'integrated_brier_score': ibs,
        'scale_parameter_used': scale_param,
        'distribution': distribution,
        'population_type': 'custom'  # Since we don't know if it's train/val/oot
    }

# Usage examples for different populations:

# 1. On training data
brier_train = calculate_survival_brier_scores(
    model=analyzer.model,
    X=analyzer.model_data['X_train'], 
    y=analyzer.model_data['y_train'], 
    event=analyzer.model_data['event_train'],
    distribution='logistic'
)

# 2. On validation data
brier_val = calculate_survival_brier_scores(
    model=analyzer.model,
    X=analyzer.model_data['X_val'], 
    y=analyzer.model_data['y_val'], 
    event=analyzer.model_data['event_val'],
    distribution='logistic'
)

# 3. On OOT data (if available)
if 'X_oot' in analyzer.model_data:
    brier_oot = calculate_survival_brier_scores(
        model=analyzer.model,
        X=analyzer.model_data['X_oot'], 
        y=analyzer.model_data['y_oot'], 
        event=analyzer.model_data['event_oot'],
        distribution='logistic'
    )

# 4. On specific subsets (e.g., new hires in 2023)
subset_mask = (data['hire_date'] >= '2023-01-01') & (data['hire_date'] < '2024-01-01')
if subset_mask.sum() > 0:
    brier_subset = calculate_survival_brier_scores(
        model=analyzer.model,
        X=X_processed[subset_mask], 
        y=y_processed[subset_mask], 
        event=event_processed[subset_mask],
        distribution='logistic'
    )

print(f"Train IBS: {brier_train['integrated_brier_score']:.4f}")
print(f"Val IBS: {brier_val['integrated_brier_score']:.4f}")

####################------------###################
# Compare performance across populations
populations = {
    'train': (X_train, y_train, event_train),
    'val': (X_val, y_val, event_val),
    'oot': (X_oot, y_oot, event_oot)
}

results = {}
for pop_name, (X, y, event) in populations.items():
    results[pop_name] = calculate_survival_brier_scores(
        model, X, y, event, distribution='logistic'
    )
    print(f"{pop_name.upper()} IBS: {results[pop_name]['integrated_brier_score']:.4f}")
