def calculate_survival_brier_scores(model, X, y, event, 
                                  distribution='normal', 
                                  time_points=None):
    """
    Calculate Brier scores for XGBoost AFT survival models
    """
    if time_points is None:
        time_points = np.arange(30, 366, 30)
    
    # XGBoost AFT predictions are in original scale (days)
    dmatrix = xgb.DMatrix(X)
    pred_survival_time = model.predict(dmatrix)
    
    # Get base_margin (log-scale predictions) for scale estimation
    # This is more robust than using residuals
    try:
        # XGBoost internally works on log scale
        log_pred = model.predict(dmatrix, output_margin=True)
        
        # Estimate scale from observed events
        observed_mask = event == 1
        if observed_mask.sum() > 10:  # Need sufficient events
            log_y_observed = np.log(y[observed_mask])
            log_pred_observed = log_pred[observed_mask]
            residuals = log_y_observed - log_pred_observed
            
            if distribution == 'normal':
                scale_param = np.std(residuals, ddof=1)
            elif distribution == 'logistic':
                scale_param = np.std(residuals, ddof=1) * np.sqrt(3) / np.pi
            elif distribution == 'extreme':
                scale_param = np.std(residuals, ddof=1) * np.sqrt(6) / np.pi
        else:
            # Fallback: use a reasonable default
            scale_param = 0.5
    except:
        # Fallback if output_margin doesn't work
        scale_param = 0.5
    
    def survival_probability(pred_time, eval_time, dist, scale):
        """Calculate S(t) = P(T > t) using AFT formulation"""
        # Ensure positive values
        pred_time = np.maximum(pred_time, 1e-6)
        eval_time = np.maximum(eval_time, 1e-6)
        
        # AFT: z = (log(t) - log(T_pred)) / scale
        z = (np.log(eval_time) - np.log(pred_time)) / scale
        z = np.clip(z, -50, 50)  # Prevent overflow/underflow
        
        if dist == 'normal':
            return 1 - stats.norm.cdf(z)
        elif dist == 'logistic':
            exp_z = np.exp(z)
            return 1 / (1 + exp_z)
        elif dist == 'extreme':
            return np.exp(-np.exp(z))
        else:
            raise ValueError(f"Unknown distribution: {dist}")
    
    brier_scores = []
    
    for t in time_points:
        # Calculate predicted survival probability at time t
        surv_prob = survival_probability(pred_survival_time, t, distribution, scale_param)
        
        # True survival status at time t
        # survived_t = 1 if subject survived beyond time t, 0 otherwise
        survived_t = ((y > t) | (event == 0)).astype(float)
        
        # Calculate Brier score (no at-risk filtering needed)
        bs = brier_score_loss(survived_t, surv_prob)
        
        # For reporting purposes, calculate number who could be evaluated
        n_evaluable = len(y)  # All subjects contribute
        n_events_by_t = ((y <= t) & (event == 1)).sum()
        
        brier_scores.append({
            'time': t,
            'brier_score': bs,
            'n_evaluable': n_evaluable,
            'n_events_by_t': n_events_by_t,
            'mean_surv_prob': np.mean(surv_prob)
        })
    
    # Calculate Integrated Brier Score (IBS)
    if len(brier_scores) > 1:
        times = np.array([bs['time'] for bs in brier_scores])
        scores = np.array([bs['brier_score'] for bs in brier_scores])
        
        # Use trapezoidal rule for integration
        ibs = np.trapz(scores, times) / (times[-1] - times[0])
    else:
        ibs = np.nan
    
    return {
        'integrated_brier_score': ibs,
        'time_dependent_brier_scores': brier_scores,
        'scale_parameter_used': scale_param,
        'distribution_used': distribution
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
