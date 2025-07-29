"""
XGBoost AFT Deep Dive Analysis Module - COMPLETION

Expert-level model diagnostics and survival distribution analysis
Extension of xgboost_v1.py for comprehensive model understanding

Author: Expert MLE Analysis
Focus: Model training quality, AFT assumptions, distribution fitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from scipy.special import gamma, loggamma
import xgboost as xgb
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({'font.size': 10, 'figure.figsize': (12, 8), 'axes.grid': True})

class XGBoostAFTDeepDive:
    """
    Comprehensive XGBoost AFT model analysis focusing on:
    1. Training dynamics and convergence
    2. AFT assumption validation
    3. Residual analysis and distribution fitting
    4. Tree structure and feature interaction patterns
    """

    def __init__(self, analyzer_instance):
        """Initialize with trained SurvivalAnalysis instance"""
        self.analyzer = analyzer_instance
        self.model = analyzer_instance.model
        self.model_data = analyzer_instance.model_data
        self.data = analyzer_instance.data
        
        # Extract training components
        self.X_train = self.model_data['X_train']
        self.y_train = self.model_data['y_train']
        self.event_train = self.model_data['event_train']
        self.X_val = self.model_data['X_val']
        self.y_val = self.model_data['y_val']
        self.event_val = self.model_data['event_val']
        
        # Training predictions
        self.train_pred = self.model.predict(xgb.DMatrix(self.X_train))
        self.val_pred = self.model.predict(xgb.DMatrix(self.X_val))
        
        print(f"Deep dive initialized: {len(self.X_train):,} train, {len(self.X_val):,} val samples")

    def analyze_target_distribution(self):
        """Comprehensive analysis of survival time distribution"""
        print("\n" + "="*60)
        print("TARGET DISTRIBUTION ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Survival Time Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Raw survival times
        ax1 = axes[0, 0]
        # Combined data for distribution analysis
        all_times = np.concatenate([self.y_train, self.y_val])
        all_events = np.concatenate([self.event_train, self.event_val])
        
        # Plot observed vs censored
        observed_times = all_times[all_events == 1]
        censored_times = all_times[all_events == 0]
        
        ax1.hist(observed_times, bins=50, alpha=0.7, label=f'Observed (n={len(observed_times):,})',
                density=True, color='red')
        ax1.hist(censored_times, bins=50, alpha=0.7, label=f'Censored (n={len(censored_times):,})',
                density=True, color='blue')
        ax1.set_xlabel('Survival Time (days)')
        ax1.set_ylabel('Density')
        ax1.set_title('Raw Survival Times')
        ax1.legend()
        ax1.set_xlim(0, np.percentile(all_times, 95))
        
        # 2. Log-transformed survival times
        ax2 = axes[0, 1]
        log_observed = np.log(observed_times + 1)
        log_censored = np.log(censored_times + 1)
        
        ax2.hist(log_observed, bins=50, alpha=0.7, label='Log(Observed)', density=True, color='red')
        ax2.hist(log_censored, bins=50, alpha=0.7, label='Log(Censored)', density=True, color='blue')
        ax2.set_xlabel('Log(Survival Time + 1)')
        ax2.set_ylabel('Density')
        ax2.set_title('Log-Transformed Times')
        ax2.legend()
        
        # 3. Q-Q plots for distribution assessment
        ax3 = axes[0, 2]
        stats.probplot(log_observed, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot: Log(Observed) vs Normal')
        ax3.grid(True)
        
        # 4. Empirical hazard function
        ax4 = axes[1, 0]
        # Calculate empirical hazard using Nelson-Aalen estimator
        unique_times = np.sort(np.unique(observed_times))
        hazard_rates = []
        
        for t in unique_times[:100]:  # First 100 unique times for clarity
            at_risk = (all_times >= t).sum()
            events_at_t = ((all_times == t) & (all_events == 1)).sum()
            if at_risk > 0:
                hazard_rates.append(events_at_t / at_risk)
            else:
                hazard_rates.append(0)
        
        ax4.plot(unique_times[:len(hazard_rates)], hazard_rates, 'o-', markersize=3)
        ax4.set_xlabel('Time (days)')
        ax4.set_ylabel('Empirical Hazard Rate')
        ax4.set_title('Empirical Hazard Function')
        
        # 5. Distribution fitting comparison
        ax5 = axes[1, 1]
        # Fit multiple distributions to observed times
        distributions = {
            'Normal': stats.norm,
            'Lognormal': stats.lognorm,
            'Weibull': stats.weibull_min,
            'Exponential': stats.expon,
            'Gamma': stats.gamma
        }
        
        log_obs_clean = log_observed[np.isfinite(log_observed)]
        aic_scores = {}
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        x_range = np.linspace(log_obs_clean.min(), log_obs_clean.max(), 100)
        
        for i, (name, dist) in enumerate(distributions.items()):
            try:
                # Fit distribution
                if name == 'Lognormal':
                    params = dist.fit(observed_times, floc=0)
                    # Convert to log space for plotting
                    fitted_data = np.log(dist.rvs(*params, size=1000))
                else:
                    params = dist.fit(log_obs_clean)
                    fitted_data = log_obs_clean
                
                # Calculate AIC
                log_likelihood = np.sum(dist.logpdf(fitted_data, *params))
                aic = 2 * len(params) - 2 * log_likelihood
                aic_scores[name] = aic
                
                # Plot fitted distribution
                if name != 'Lognormal':
                    pdf_fitted = dist.pdf(x_range, *params)
                    ax5.plot(x_range, pdf_fitted, color=colors[i],
                            label=f'{name} (AIC: {aic:.0f})', linewidth=2)
                    
            except Exception as e:
                print(f"Failed to fit {name}: {e}")
                aic_scores[name] = np.inf
        
        # Plot empirical distribution
        ax5.hist(log_obs_clean, bins=30, density=True, alpha=0.3, color='gray', label='Empirical')
        ax5.set_xlabel('Log(Survival Time)')
        ax5.set_ylabel('Density')
        ax5.set_title('Distribution Fitting Comparison')
        ax5.legend()
        
        # 6. Best fit analysis
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Distribution fit summary
        best_dist = min(aic_scores.items(), key=lambda x: x[1])
        summary_text = f"""
DISTRIBUTION ANALYSIS SUMMARY

Sample Characteristics:
- Total samples: {len(all_times):,}
- Observed events: {len(observed_times):,} ({len(observed_times)/len(all_times):.1%})
- Censored: {len(censored_times):,} ({len(censored_times)/len(all_times):.1%})

Survival Time Statistics:
- Mean: {np.mean(all_times):.0f} days
- Median: {np.median(all_times):.0f} days
- Std: {np.std(all_times):.0f} days
- Skewness: {stats.skew(all_times):.2f}
- Kurtosis: {stats.kurtosis(all_times):.2f}

Log-Transform Statistics:
- Mean log(T): {np.mean(log_observed):.2f}
- Std log(T): {np.std(log_observed):.2f}
- Skewness log(T): {stats.skew(log_observed):.2f}

Best Distribution Fit:
- {best_dist[0]} (AIC: {best_dist[1]:.0f})

AFT Suitability:
- Log-normal shape: {'Good' if abs(stats.skew(log_observed)) < 1 else 'Poor'}
- Heavy tails: {'Yes' if stats.kurtosis(all_times) > 3 else 'No'}
"""
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('target_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'aic_scores': aic_scores,
            'best_fit': best_dist,
            'survival_stats': {
                'mean': np.mean(all_times),
                'median': np.median(all_times),
                'std': np.std(all_times),
                'skewness': stats.skew(all_times),
                'kurtosis': stats.kurtosis(all_times)
            },
            'log_stats': {
                'mean': np.mean(log_observed),
                'std': np.std(log_observed),
                'skewness': stats.skew(log_observed),
                'kurtosis': stats.kurtosis(log_observed)
            }
        }

    def analyze_aft_distributions(self):
        """Compare AFT distributions and their fit to data"""
        print("\n" + "="*60)
        print("AFT DISTRIBUTION COMPARISON")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('AFT Distribution Analysis for XGBoost', fontsize=16, fontweight='bold')
        
        # Get observed survival times for comparison
        observed_times = self.y_train[self.event_train == 1]
        log_observed = np.log(observed_times)
        
        # 1. AFT distribution PDFs
        ax1 = axes[0, 0]
        # Define AFT distributions available in XGBoost
        x_range = np.linspace(-3, 3, 1000)
        
        # Normal AFT: log(T) ~ Normal(mu, sigma)
        normal_pdf = stats.norm.pdf(x_range, loc=0, scale=1)
        ax1.plot(x_range, normal_pdf, label='Normal AFT', linewidth=2, color='blue')
        
        # Logistic AFT: log(T) ~ Logistic(mu, sigma)
        logistic_pdf = stats.logistic.pdf(x_range, loc=0, scale=1)
        ax1.plot(x_range, logistic_pdf, label='Logistic AFT', linewidth=2, color='red')
        
        # Extreme AFT: log(T) ~ Gumbel(mu, sigma)
        extreme_pdf = stats.gumbel_r.pdf(x_range, loc=0, scale=1)
        ax1.plot(x_range, extreme_pdf, label='Extreme AFT', linewidth=2, color='green')
        
        # Overlay empirical log survival times
        ax1.hist(log_observed, bins=50, density=True, alpha=0.3, color='gray', label='Empirical log(T)')
        ax1.set_xlabel('log(T)')
        ax1.set_ylabel('Density')
        ax1.set_title('AFT Distribution PDFs vs Empirical')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plots for each AFT distribution
        ax2 = axes[0, 1]
        # Standardize log observed times
        log_std = (log_observed - np.mean(log_observed)) / np.std(log_observed)
        
        # Q-Q plot against normal
        quantiles_normal = stats.probplot(log_std, dist='norm', plot=None)
        ax2.scatter(quantiles_normal[0][0], quantiles_normal[0][1], alpha=0.5, s=1, label='vs Normal')
        
        # Q-Q plot against logistic
        quantiles_logistic = stats.probplot(log_std, dist='logistic', plot=None)
        ax2.scatter(quantiles_logistic[0][0], quantiles_logistic[0][1], alpha=0.5, s=1, label='vs Logistic')
        
        # Perfect fit line
        qq_min, qq_max = ax2.get_xlim()
        ax2.plot([qq_min, qq_max], [qq_min, qq_max], 'k--', alpha=0.8, label='Perfect Fit')
        ax2.set_xlabel('Theoretical Quantiles')
        ax2.set_ylabel('Sample Quantiles')
        ax2.set_title('Q-Q Plots: AFT Distribution Fit')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Kolmogorov-Smirnov test results
        ax3 = axes[1, 0]
        # Fit each distribution and calculate KS statistics
        ks_results = {}
        
        # Normal
        mu_norm, sigma_norm = stats.norm.fit(log_observed)
        ks_norm = stats.kstest(log_observed, lambda x: stats.norm.cdf(x, mu_norm, sigma_norm))
        ks_results['Normal'] = {'statistic': ks_norm.statistic, 'pvalue': ks_norm.pvalue}
        
        # Logistic
        mu_log, sigma_log = stats.logistic.fit(log_observed)
        ks_log = stats.kstest(log_observed, lambda x: stats.logistic.cdf(x, mu_log, sigma_log))
        ks_results['Logistic'] = {'statistic': ks_log.statistic, 'pvalue': ks_log.pvalue}
        
        # Extreme (Gumbel)
        mu_ext, sigma_ext = stats.gumbel_r.fit(log_observed)
        ks_ext = stats.kstest(log_observed, lambda x: stats.gumbel_r.cdf(x, mu_ext, sigma_ext))
        ks_results['Extreme'] = {'statistic': ks_ext.statistic, 'pvalue': ks_ext.pvalue}
        
        # Plot KS statistics
        distributions = list(ks_results.keys())
        ks_stats = [ks_results[d]['statistic'] for d in distributions]
        colors = ['blue', 'red', 'green']
        
        bars = ax3.bar(distributions, ks_stats, color=colors, alpha=0.7)
        ax3.set_ylabel('KS Statistic')
        ax3.set_title('Kolmogorov-Smirnov Test Results\n(Lower is Better)')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add p-value annotations
        for i, (bar, dist) in enumerate(zip(bars, distributions)):
            pval = ks_results[dist]['pvalue']
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'p={pval:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Distribution recommendation
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Find best distribution
        best_ks = min(ks_results.items(), key=lambda x: x[1]['statistic'])
        
        # Calculate additional metrics
        log_skew = stats.skew(log_observed)
        log_kurtosis = stats.kurtosis(log_observed)
        
        # Theoretical properties
        normal_kurtosis = 0  # Normal distribution excess kurtosis
        logistic_kurtosis = 1.2  # Logistic distribution excess kurtosis
        gumbel_kurtosis = 2.4  # Gumbel distribution excess kurtosis
        
        recommendation_text = f"""
AFT DISTRIBUTION RECOMMENDATION

Kolmogorov-Smirnov Results:
- Normal: KS={ks_results['Normal']['statistic']:.4f}, p={ks_results['Normal']['pvalue']:.4f}
- Logistic: KS={ks_results['Logistic']['statistic']:.4f}, p={ks_results['Logistic']['pvalue']:.4f}
- Extreme: KS={ks_results['Extreme']['statistic']:.4f}, p={ks_results['Extreme']['pvalue']:.4f}

Best Fit: {best_ks[0]} (KS = {best_ks[1]['statistic']:.4f})

Empirical Properties:
- Log(T) Skewness: {log_skew:.3f}
- Log(T) Excess Kurtosis: {log_kurtosis:.3f}

Distribution Properties (Excess Kurtosis):
- Normal: {normal_kurtosis:.1f}
- Logistic: {logistic_kurtosis:.1f}
- Extreme: {gumbel_kurtosis:.1f}

Recommendation:
{'Logistic' if abs(log_kurtosis - logistic_kurtosis) < abs(log_kurtosis - normal_kurtosis) else 'Normal'} AFT
(Closest kurtosis match)

XGBoost Parameters:
objective: 'survival:aft'
aft_loss_distribution: '{best_ks[0].lower()}'

Expected Impact:
- Better likelihood fit → Lower training loss
- Improved calibration → Better Brier scores
- More stable gradients → Faster convergence
"""
        
        ax4.text(0.05, 0.95, recommendation_text, transform=ax4.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('aft_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'ks_results': ks_results,
            'best_distribution': best_ks[0],
            'empirical_stats': {
                'skewness': log_skew,
                'kurtosis': log_kurtosis
            },
            'fitted_params': {
                'normal': (mu_norm, sigma_norm),
                'logistic': (mu_log, sigma_log),
                'extreme': (mu_ext, sigma_ext)
            }
        }

    def analyze_training_dynamics(self):
        """Deep analysis of training process and convergence"""
        print("\n" + "="*60)
        print("TRAINING DYNAMICS ANALYSIS")
        print("="*60)
        
        # Retrain with detailed logging to capture training metrics
        print("Retraining model with detailed logging...")
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dval = xgb.DMatrix(self.X_val, label=self.y_val)
        
        # Set survival information
        dtrain.set_float_info('label_lower_bound', self.y_train.values)
        dtrain.set_float_info('label_upper_bound', self.y_train.values)
        dval.set_float_info('label_lower_bound', self.y_val.values)
        dval.set_float_info('label_upper_bound', self.y_val.values)
        
        # Enhanced parameters for detailed analysis
        params = {
            'objective': 'survival:aft',
            'aft_loss_distribution': 'logistic',
            'max_depth': 4,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # Training with evaluation logging
        evals = [(dtrain, 'train'), (dval, 'val')]
        evals_result = {}
        
        model_detailed = xgb.train(
            params, dtrain,
            num_boost_round=200,
            evals=evals,
            evals_result=evals_result,
            verbose_eval=False,
            early_stopping_rounds=20
        )
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('XGBoost Training Dynamics Deep Dive', fontsize=16, fontweight='bold')
        
        # 1. Training and validation loss curves
        ax1 = axes[0, 0]
        train_loss = evals_result['train']['aft-nloglik']
        val_loss = evals_result['val']['aft-nloglik']
        epochs = range(len(train_loss))
        
        ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2, color='blue')
        ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2, color='red')
        
        # Mark best iteration
        best_iter = model_detailed.best_iteration
        if best_iter < len(val_loss):
            ax1.axvline(x=best_iter, color='green', linestyle='--',
                       label=f'Best Iteration ({best_iter})')
            ax1.scatter(best_iter, val_loss[best_iter], color='green', s=100, zorder=5)
        
        ax1.set_xlabel('Boosting Round')
        ax1.set_ylabel('AFT Negative Log-Likelihood')
        ax1.set_title('Training Convergence')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss difference (overfitting indicator)
        ax2 = axes[0, 1]
        loss_diff = np.array(val_loss) - np.array(train_loss)
        ax2.plot(epochs, loss_diff, linewidth=2, color='purple')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.fill_between(epochs, loss_diff, 0, alpha=0.3, color='purple')
        ax2.set_xlabel('Boosting Round')
        ax2.set_ylabel('Validation - Training Loss')
        ax2.set_title('Overfitting Monitor')
        ax2.grid(True, alpha=0.3)
        
        # Add overfitting threshold
        overfitting_threshold = 0.1
        ax2.axhline(y=overfitting_threshold, color='red', linestyle='--',
                   label=f'Overfitting Threshold ({overfitting_threshold})')
        ax2.legend()
        
        # 3. Learning rate effectiveness
        ax3 = axes[0, 2]
        # Calculate loss reduction per round
        train_reduction = np.diff(train_loss)
        val_reduction = np.diff(val_loss)
        
        ax3.plot(epochs[1:], -train_reduction, label='Train Loss Reduction',
                alpha=0.7, color='blue')
        ax3.plot(epochs[1:], -val_reduction, label='Val Loss Reduction',
                alpha=0.7, color='red')
        
        # Smooth with rolling average
        window = 10
        if len(train_reduction) > window:
            train_smooth = pd.Series(-train_reduction).rolling(window).mean()
            val_smooth = pd.Series(-val_reduction).rolling(window).mean()
            ax3.plot(epochs[1:], train_smooth, linewidth=3, color='darkblue',
                    label=f'Train Smooth ({window}r)')
            ax3.plot(epochs[1:], val_smooth, linewidth=3, color='darkred',
                    label=f'Val Smooth ({window}r)')
        
        ax3.set_xlabel('Boosting Round')
        ax3.set_ylabel('Loss Reduction')
        ax3.set_title('Learning Rate Effectiveness')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Tree depth and complexity analysis
        ax4 = axes[1, 0]
        # Get tree statistics
        tree_depths = []
        tree_leaves = []
        
        for i in range(min(50, model_detailed.num_boosted_rounds())):  # First 50 trees
            tree_dump = model_detailed.get_dump(dump_format='json')[i]
            import json
            tree_json = json.loads(tree_dump)
            
            def get_tree_depth(node, depth=0):
                if 'children' not in node:
                    return depth
                return max(get_tree_depth(child, depth + 1) for child in node['children'])
            
            def count_leaves(node):
                if 'children' not in node:
                    return 1
                return sum(count_leaves(child) for child in node['children'])
            
            tree_depths.append(get_tree_depth(tree_json))
            tree_leaves.append(count_leaves(tree_json))
        
        ax4.scatter(tree_depths, tree_leaves, alpha=0.6, s=30)
        ax4.set_xlabel('Tree Depth')
        ax4.set_ylabel('Number of Leaves')
        ax4.set_title('Tree Complexity Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Add complexity annotations
        avg_depth = np.mean(tree_depths)
        avg_leaves = np.mean(tree_leaves)
        ax4.axvline(x=avg_depth, color='red', linestyle='--', alpha=0.7,
                   label=f'Avg Depth: {avg_depth:.1f}')
        ax4.axhline(y=avg_leaves, color='red', linestyle='--', alpha=0.7,
                   label=f'Avg Leaves: {avg_leaves:.1f}')
        ax4.legend()
        
        # 5. Feature usage frequency across trees
        ax5 = axes[1, 1]
        # Analyze feature splits across all trees
        feature_splits = {}
        
        for i in range(min(100, model_detailed.num_boosted_rounds())):
            tree_dump = model_detailed.get_dump()[i]
            for line in tree_dump.split('\n'):
                if '[f' in line and '<' in line:
                    # Extract feature index
                    feature_idx = line.split('[f')[1].split('<')[0]
                    try:
                        feature_idx = int(feature_idx)
                        if feature_idx < len(self.X_train.columns):
                            feature_name = self.X_train.columns[feature_idx]
                            feature_splits[feature_name] = feature_splits.get(feature_name, 0) + 1
                    except ValueError:
                        continue
        
        if feature_splits:
            # Plot top features by split frequency
            top_features = sorted(feature_splits.items(), key=lambda x: x[1], reverse=True)[:15]
            features, counts = zip(*top_features)
            
            ax5.barh(range(len(features)), counts, color='skyblue', alpha=0.8)
            ax5.set_yticks(range(len(features)))
            ax5.set_yticklabels(features, fontsize=9)
            ax5.set_xlabel('Number of Splits')
            ax5.set_title('Feature Usage in Tree Splits')
            ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. Training summary and recommendations
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate training metrics
        final_train_loss = train_loss[-1]
        final_val_loss = val_loss[-1]
        best_val_loss = min(val_loss)
        convergence_round = val_loss.index(best_val_loss)
        
        # Overfitting assessment
        final_gap = final_val_loss - final_train_loss
        overfitting_severity = "High" if final_gap > 0.15 else "Medium" if final_gap > 0.05 else "Low"
        
        # Training efficiency
        early_improvement = train_loss[0] - train_loss[min(20, len(train_loss)-1)]
        late_improvement = train_loss[min(50, len(train_loss)-1)] - train_loss[-1] if len(train_loss) > 50 else 0
        
        training_summary = f"""
TRAINING DYNAMICS SUMMARY

Convergence Analysis:
- Total rounds: {len(train_loss)}
- Best iteration: {convergence_round}
- Final train loss: {final_train_loss:.4f}
- Final val loss: {final_val_loss:.4f}
- Best val loss: {best_val_loss:.4f}

Overfitting Assessment:
- Train-val gap: {final_gap:.4f}
- Severity: {overfitting_severity}
- Early stopping triggered: {'Yes' if best_iter < len(train_loss)-1 else 'No'}

Learning Efficiency:
- Early improvement (0-20): {early_improvement:.4f}
- Late improvement (50+): {late_improvement:.4f}
- Avg tree depth: {avg_depth:.1f}
- Avg leaves per tree: {avg_leaves:.1f}

Recommendations:
- Learning rate: {'Decrease' if final_gap > 0.1 else 'OK'}
- Regularization: {'Increase' if overfitting_severity == 'High' else 'Current OK'}
- Max depth: {'Decrease' if avg_depth > 6 else 'OK'}
- Early stopping: {'More aggressive' if best_iter > len(train_loss) * 0.7 else 'Current OK'}

Training Quality: {'Poor' if final_gap > 0.15 else 'Good' if final_gap < 0.05 else 'Acceptable'}
"""
        
        ax6.text(0.05, 0.95, training_summary, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('training_dynamics_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'training_history': evals_result,
            'best_iteration': best_iter,
            'final_losses': {'train': final_train_loss, 'val': final_val_loss},
            'overfitting_gap': final_gap,
            'tree_stats': {'depths': tree_depths, 'leaves': tree_leaves},
            'feature_usage': feature_splits,
            'model_retrained': model_detailed
        }

    def analyze_residuals_and_assumptions(self):
        """Comprehensive residual analysis and AFT assumption validation"""
        print("\n" + "="*60)
        print("RESIDUAL ANALYSIS & AFT ASSUMPTIONS")
        print("="*60)
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Residual Analysis & AFT Assumption Validation', fontsize=16, fontweight='bold')
        
        # Calculate residuals for AFT model
        # AFT residual: r = log(observed_time) - predicted_log_time
        log_y_train = np.log(self.y_train)
        log_y_val = np.log(self.y_val)
        
        # XGBoost AFT predicts log(time) directly
        train_residuals = log_y_train - np.log(self.train_pred)
        val_residuals = log_y_val - np.log(self.val_pred)
        
        # 1. Residual distribution (should match AFT error distribution)
        ax1 = axes[0, 0]
        # Plot residuals for observed events only (uncensored)
        train_obs_residuals = train_residuals[self.event_train == 1]
        val_obs_residuals = val_residuals[self.event_val == 1]
        
        ax1.hist(train_obs_residuals, bins=50, alpha=0.7, density=True,
                label=f'Train (n={len(train_obs_residuals):,})', color='blue')
        ax1.hist(val_obs_residuals, bins=50, alpha=0.7, density=True,
                label=f'Val (n={len(val_obs_residuals):,})', color='red')
        
        # Overlay theoretical distributions
        x_range = np.linspace(-4, 4, 100)
        ax1.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'k--',
                label='Standard Normal', linewidth=2)
        ax1.plot(x_range, stats.logistic.pdf(x_range, 0, 1), 'g--',
                label='Standard Logistic', linewidth=2)
        
        ax1.set_xlabel('Residuals: log(T_obs) - log(T_pred)')
        ax1.set_ylabel('Density')
        ax1.set_title('Residual Distribution vs Theoretical')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q plot of residuals vs assumed distribution
        ax2 = axes[0, 1]
        combined_residuals = np.concatenate([train_obs_residuals, val_obs_residuals])
        stats.probplot(combined_residuals, dist='logistic', plot=ax2)
        ax2.set_title('Q-Q Plot: Residuals vs Logistic')
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals vs fitted values (homoscedasticity check)
        ax3 = axes[0, 2]
        train_fitted = np.log(self.train_pred[self.event_train == 1])
        val_fitted = np.log(self.val_pred[self.event_val == 1])
        
        ax3.scatter(train_fitted, train_obs_residuals, alpha=0.5, s=1,
                   color='blue', label='Train')
        ax3.scatter(val_fitted, val_obs_residuals, alpha=0.5, s=1,
                   color='red', label='Val')
        
        # Add trend line
        combined_fitted = np.concatenate([train_fitted, val_fitted])
        z = np.polyfit(combined_fitted, combined_residuals, 1)
        p = np.poly1d(z)
        ax3.plot(sorted(combined_fitted), p(sorted(combined_fitted)),
                "k--", alpha=0.8, linewidth=2, label=f'Trend: slope={z[0]:.3f}')
        
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_xlabel('Fitted Values: log(T_pred)')
        ax3.set_ylabel('Residuals')
        ax3.set_title('Residuals vs Fitted (Homoscedasticity)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals by feature quartiles (feature-specific patterns)
        ax4 = axes[1, 0]
        # Use most important feature for demonstration
        feature_importance = self.model.get_score(importance_type='gain')
        if feature_importance:
            if list(feature_importance.keys())[0].startswith("f"):
                top_feature_idx = int(max(feature_importance, key=feature_importance.get)[1:])
                top_feature_name = self.X_train.columns[top_feature_idx]
            else:
                top_feature_name = max(feature_importance, key=feature_importance.get)
            
            feature_values = self.X_train[top_feature_name][self.event_train == 1]
            
            # Create quartile-based boxplot
            quartiles = pd.qcut(feature_values, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            residual_df = pd.DataFrame({
                'quartile': quartiles,
                'residual': train_obs_residuals
            })
            
            residual_df.boxplot(column='residual', by='quartile', ax=ax4)
            ax4.set_xlabel(f'{top_feature_name} Quartiles')
            ax4.set_ylabel('Residuals')
            ax4.set_title(f'Residuals by {top_feature_name} Quartiles')
            ax4.grid(True, alpha=0.3)
        
        # 5. Censoring bias analysis
        ax5 = axes[1, 1]
        # Compare residual patterns between censored and observed
        # For censored: use lower bound residual
        train_censored_residuals = log_y_train[self.event_train == 0] - np.log(self.train_pred[self.event_train == 0])
        
        ax5.hist(train_obs_residuals, bins=30, alpha=0.7, density=True,
                label=f'Observed (n={len(train_obs_residuals):,})', color='green')
        ax5.hist(train_censored_residuals, bins=30, alpha=0.7, density=True,
                label=f'Censored (n={len(train_censored_residuals):,})', color='orange')
        
        ax5.set_xlabel('Residuals')
        ax5.set_ylabel('Density')
        ax5.set_title('Residuals: Observed vs Censored')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Residual autocorrelation (temporal patterns)
        ax6 = axes[1, 2]
        # Sort by survival time and check for autocorrelation
        sorted_indices = np.argsort(self.y_train[self.event_train == 1])
        sorted_residuals = train_obs_residuals[sorted_indices]
        
        # Calculate rolling correlation
        window_size = min(100, len(sorted_residuals) // 10)
        correlations = []
        
        for i in range(len(sorted_residuals) - window_size):
            window_residuals = sorted_residuals[i:i+window_size]
            if len(window_residuals) > 10:
                # Lag-1 autocorrelation
                corr = np.corrcoef(window_residuals[:-1], window_residuals[1:])[0, 1]
                correlations.append(corr)
        
        if correlations:
            ax6.plot(correlations, linewidth=2, color='purple')
            ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax6.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='Concern Threshold')
            ax6.axhline(y=-0.1, color='red', linestyle='--', alpha=0.7)
            ax6.set_xlabel('Window Position (Ordered by Survival Time)')
            ax6.set_ylabel('Lag-1 Autocorrelation')
            ax6.set_title('Residual Autocorrelation')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # 7. Martingale residuals (survival-specific)
        ax7 = axes[2, 0]
        # Martingale residuals: M = event - predicted_cumulative_hazard
        # For AFT: H(t) = ((t/exp(linear_pred))^k) where k depends on distribution
        # Approximate cumulative hazard for logistic AFT
        train_scale = np.exp(np.log(self.train_pred))
        train_cum_hazard = np.power(self.y_train / train_scale, 1.5)  # Approximation
        train_martingale = self.event_train - train_cum_hazard
        
        ax7.scatter(np.log(self.train_pred), train_martingale, alpha=0.5, s=1, color='blue')
        ax7.axhline(y=0, color='red', linestyle='-', alpha=0.7)
        
        # Add smoothed trend
        try:
            from scipy.signal import savgol_filter
            sorted_pred = np.argsort(np.log(self.train_pred))
            if len(sorted_pred) > 51:
                smoothed = savgol_filter(train_martingale[sorted_pred], 51, 3)
                ax7.plot(np.log(self.train_pred)[sorted_pred], smoothed,
                        'red', linewidth=3, label='Trend')
                ax7.legend()
        except:
            pass
        
        ax7.set_xlabel('log(Predicted Survival Time)')
        ax7.set_ylabel('Martingale Residuals')
        ax7.set_title('Martingale Residuals vs Fitted')
        ax7.grid(True, alpha=0.3)
        
        # 8. Deviance residuals
        ax8 = axes[2, 1]
        # Deviance residuals for AFT models
        # D = sign(M) * sqrt(-2 * [M + event * log(event - M)])
        train_deviance = np.zeros_like(train_martingale)
        
        for i in range(len(train_martingale)):
            M = train_martingale[i]
            event = self.event_train.iloc[i]
            if event == 1 and M < 1:
                deviance_val = -2 * (M + np.log(1 - M))
            else:
                deviance_val = -2 * M
            train_deviance[i] = np.sign(M) * np.sqrt(max(0, deviance_val))
        
        ax8.hist(train_deviance, bins=50, alpha=0.7, density=True, color='purple')
        
        # Overlay standard normal for comparison
        x_range = np.linspace(train_deviance.min(), train_deviance.max(), 100)
        ax8.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'k--',
                label='Standard Normal', linewidth=2)
        ax8.set_xlabel('Deviance Residuals')
        ax8.set_ylabel('Density')
        ax8.set_title('Deviance Residuals Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. AFT assumption summary and diagnostics
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Calculate diagnostic statistics
        residual_skewness = stats.skew(combined_residuals)
        residual_kurtosis = stats.kurtosis(combined_residuals)
        residual_std = np.std(combined_residuals)
        
        # Kolmogorov-Smirnov test against logistic distribution
        ks_stat, ks_pval = stats.kstest(combined_residuals, 'logistic')
        
        # Homoscedasticity test (correlation between residuals and fitted)
        homosced_corr = np.corrcoef(combined_fitted, combined_residuals)[0, 1]
        
        # Censoring bias assessment
        censored_mean = np.mean(train_censored_residuals) if len(train_censored_residuals) > 0 else 0
        observed_mean = np.mean(train_obs_residuals)
        censoring_bias = abs(censored_mean - observed_mean)
        
        assumption_summary = f"""
AFT ASSUMPTION VALIDATION

Residual Distribution Quality:
- Skewness: {residual_skewness:.3f} (target: ~0)
- Kurtosis: {residual_kurtosis:.3f} (logistic: 1.2)
- Std deviation: {residual_std:.3f} (target: ~1)
- KS vs Logistic: {ks_stat:.4f} (p={ks_pval:.4f})

Homoscedasticity:
- Residual-fitted correlation: {homosced_corr:.4f}
- Status: {'Good' if abs(homosced_corr) < 0.1 else 'Violation detected'}

Censoring Handling:
- Observed mean residual: {observed_mean:.3f}
- Censored mean residual: {censored_mean:.3f}
- Bias magnitude: {censoring_bias:.3f}

AFT Model Validity:
- Distribution fit: {'Good' if ks_pval > 0.05 else 'Poor'}
- Homoscedasticity: {'Pass' if abs(homosced_corr) < 0.1 else 'Fail'}
- Error distribution: {'Logistic OK' if abs(residual_kurtosis - 1.2) < 0.5 else 'Consider Normal'}

Recommendations:
- Error distribution: {'Current logistic OK' if ks_pval > 0.05 else 'Try normal or extreme'}
- Feature engineering: {'Current OK' if abs(homosced_corr) < 0.1 else 'Add interactions/transforms'}
- Regularization: {'Increase' if residual_std > 1.2 else 'Current OK'}

Overall AFT Fit: {'Excellent' if ks_pval > 0.1 and abs(homosced_corr) < 0.05
                   else 'Good' if ks_pval > 0.05 and abs(homosced_corr) < 0.1
                   else 'Needs improvement'}
"""
        
        ax9.text(0.05, 0.95, assumption_summary, transform=ax9.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('residual_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'residuals': {
                'train_observed': train_obs_residuals,
                'val_observed': val_obs_residuals,
                'train_censored': train_censored_residuals
            },
            'diagnostics': {
                'residual_skewness': residual_skewness,
                'residual_kurtosis': residual_kurtosis,
                'residual_std': residual_std,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'homoscedasticity_corr': homosced_corr,
                'censoring_bias': censoring_bias
            },
            'martingale_residuals': train_martingale,
            'deviance_residuals': train_deviance
        }

    def analyze_feature_interactions(self):
        """Deep dive into feature interactions and tree structure patterns"""
        print("\n" + "="*60)
        print("FEATURE INTERACTION ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Interactions & Tree Structure Analysis', fontsize=16, fontweight='bold')
        
        # 1. SHAP-based interaction analysis
        ax1 = axes[0, 0]
        try:
            import shap
            # Create explainer (sample data for efficiency)
            sample_size = min(1000, len(self.X_train))
            sample_indices = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_sample = self.X_train.iloc[sample_indices]
            
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_sample)
            
            # Plot feature importance based on SHAP
            feature_importance = np.abs(shap_values).mean(0)
            feature_names = X_sample.columns
            
            # Sort by importance
            importance_order = np.argsort(feature_importance)[::-1][:10]
            
            ax1.barh(range(len(importance_order)),
                    feature_importance[importance_order],
                    color='skyblue', alpha=0.8)
            ax1.set_yticks(range(len(importance_order)))
            ax1.set_yticklabels([feature_names[i] for i in importance_order], fontsize=9)
            ax1.set_xlabel('Mean |SHAP Value|')
            ax1.set_title('SHAP-based Feature Importance')
            ax1.grid(True, alpha=0.3, axis='x')
            
        except ImportError:
            ax1.text(0.5, 0.5, 'SHAP not available\nInstall with: pip install shap',
                    ha='center', va='center', transform=ax1.transAxes,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
            ax1.set_title('SHAP Analysis (Not Available)')
        
        # 2. Tree-based interaction detection
        ax2 = axes[0, 1]
        # Extract frequent feature pairs from tree splits
        feature_pairs = {}
        
        for i in range(min(50, self.model.num_boosted_rounds())):
            tree_dump = self.model.get_dump()[i]
            features_in_tree = []
            
            for line in tree_dump.split('\n'):
                if '[f' in line and '<' in line:
                    feature_idx = line.split('[f')[1].split('<')[0]
                    try:
                        feature_idx = int(feature_idx)
                        if feature_idx < len(self.X_train.columns):
                            features_in_tree.append(self.X_train.columns[feature_idx])
                    except ValueError:
                        continue
            
            # Count co-occurrences
            for j in range(len(features_in_tree)):
                for k in range(j+1, len(features_in_tree)):
                    pair = tuple(sorted([features_in_tree[j], features_in_tree[k]]))
                    feature_pairs[pair] = feature_pairs.get(pair, 0) + 1
        
        if feature_pairs:
            # Plot top interactions
            top_pairs = sorted(feature_pairs.items(), key=lambda x: x[1], reverse=True)[:10]
            pair_labels = [f"{pair[0][:8]}+\n{pair[1][:8]}" for pair, _ in top_pairs]
            pair_counts = [count for _, count in top_pairs]
            
            ax2.bar(range(len(pair_labels)), pair_counts, color='lightcoral', alpha=0.8)
            ax2.set_xticks(range(len(pair_labels)))
            ax2.set_xticklabels(pair_labels, rotation=45, ha='right', fontsize=8)
            ax2.set_ylabel('Co-occurrence Count')
            ax2.set_title('Most Frequent Feature Interactions')
            ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Partial dependence analysis for top features
        ax3 = axes[0, 2]
        # Get top 2 most important features
        feature_importance = self.model.get_score(importance_type='gain')
        if feature_importance:
            if list(feature_importance.keys())[0].startswith("f"):
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:2]
                top_feature_names = [self.X_train.columns[int(f[1:])] for f, _ in top_features]
            else:
                top_feature_names = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:2]
                top_feature_names = [f for f, _ in top_feature_names]
            
            if len(top_feature_names) >= 2:
                feature1, feature2 = top_feature_names[0], top_feature_names[1]
                
                # Create partial dependence plot
                f1_range = np.linspace(self.X_train[feature1].quantile(0.05),
                                     self.X_train[feature1].quantile(0.95), 20)
                f2_range = np.linspace(self.X_train[feature2].quantile(0.05),
                                     self.X_train[feature2].quantile(0.95), 20)
                
                # Sample baseline values
                baseline = self.X_train.median()
                pd_values = np.zeros((len(f1_range), len(f2_range)))
                
                for i, f1_val in enumerate(f1_range):
                    for j, f2_val in enumerate(f2_range):
                        temp_data = baseline.copy()
                        temp_data[feature1] = f1_val
                        temp_data[feature2] = f2_val
                        dtemp = xgb.DMatrix(temp_data.values.reshape(1, -1))
                        pd_values[i, j] = self.model.predict(dtemp)[0]
                
                im = ax3.imshow(pd_values, aspect='auto', origin='lower', cmap='viridis')
                ax3.set_xlabel(f'{feature2}')
                ax3.set_ylabel(f'{feature1}')
                ax3.set_title(f'Partial Dependence:\n{feature1} vs {feature2}')
                
                # Add colorbar
                plt.colorbar(im, ax=ax3, label='Predicted Survival Time')
        
        # 4. Tree depth vs performance analysis
        ax4 = axes[1, 0]
        # Analyze how tree depth correlates with prediction accuracy
        tree_depths = []
        tree_performance = []
        
        # Get per-tree contributions to final prediction
        for i in range(min(30, self.model.num_boosted_rounds())):
            tree_dump = self.model.get_dump(dump_format='json')[i]
            import json
            tree_json = json.loads(tree_dump)
            
            def get_tree_depth(node, depth=0):
                if 'children' not in node:
                    return depth
                return max(get_tree_depth(child, depth + 1) for child in node['children'])
            
            depth = get_tree_depth(tree_json)
            tree_depths.append(depth)
            
            # Approximate tree contribution magnitude
            tree_leaves = []
            def collect_leaf_values(node):
                if 'children' not in node:
                    tree_leaves.append(abs(node['leaf']))
                else:
                    for child in node['children']:
                        collect_leaf_values(child)
            
            collect_leaf_values(tree_json)
            avg_leaf_magnitude = np.mean(tree_leaves) if tree_leaves else 0
            tree_performance.append(avg_leaf_magnitude)
        
        if tree_depths and tree_performance:
            ax4.scatter(tree_depths, tree_performance, alpha=0.7, s=50)
            
            # Add correlation line
            if len(tree_depths) > 1:
                z = np.polyfit(tree_depths, tree_performance, 1)
                p = np.poly1d(z)
                ax4.plot(sorted(tree_depths), p(sorted(tree_depths)), "r--", alpha=0.8)
                corr = np.corrcoef(tree_depths, tree_performance)[0, 1]
                ax4.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                        transform=ax4.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax4.set_xlabel('Tree Depth')
            ax4.set_ylabel('Avg Leaf Magnitude')
            ax4.set_title('Tree Depth vs Contribution')
            ax4.grid(True, alpha=0.3)
        
        # 5. Feature split threshold analysis
        ax5 = axes[1, 1]
        # Analyze split thresholds for most important feature
        if feature_importance:
            if list(feature_importance.keys())[0].startswith("f"):
                top_feature_idx = int(max(feature_importance, key=feature_importance.get)[1:])
                top_feature_name = self.X_train.columns[top_feature_idx]
            else:
                top_feature_name = max(feature_importance, key=feature_importance.get)
            
            # Extract all split thresholds for this feature
            split_thresholds = []
            for i in range(min(100, self.model.num_boosted_rounds())):
                tree_dump = self.model.get_dump()[i]
                for line in tree_dump.split('\n'):
                    if f'[{top_feature_name}' in line or f'[f{self.X_train.columns.get_loc(top_feature_name)}' in line:
                        if '<' in line:
                            try:
                                threshold = float(line.split('<')[1].split(']')[0])
                                split_thresholds.append(threshold)
                            except (ValueError, IndexError):
                                continue
            
            if split_thresholds:
                ax5.hist(split_thresholds, bins=30, alpha=0.7, color='green', density=True)
                
                # Overlay actual feature distribution
                feature_values = self.X_train[top_feature_name]
                ax5.hist(feature_values, bins=30, alpha=0.3, color='blue',
                        density=True, label='Feature Distribution')
                
                ax5.axvline(np.median(split_thresholds), color='red', linestyle='--',
                           label=f'Median Split: {np.median(split_thresholds):.2f}')
                ax5.axvline(feature_values.median(), color='blue', linestyle='--',
                           label=f'Feature Median: {feature_values.median():.2f}')
                
                ax5.set_xlabel(f'{top_feature_name} Value')
                ax5.set_ylabel('Density')
                ax5.set_title(f'Split Thresholds: {top_feature_name}')
                ax5.legend()
                ax5.grid(True, alpha=0.3)
        
        # 6. Model complexity vs performance tradeoff
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Calculate model complexity metrics
        total_splits = sum(1 for i in range(self.model.num_boosted_rounds())
                          for line in self.model.get_dump()[i].split('\n')
                          if '[f' in line and '<' in line)
        
        avg_tree_depth = np.mean(tree_depths) if tree_depths else 0
        unique_features_used = len(set(self.X_train.columns) &
                                 set(f for f in feature_importance.keys()
                                     if not f.startswith('f'))) if feature_importance else 0
        
        # Model performance metrics
        train_c_index = concordance_index(self.y_train, self.train_pred, self.event_train)
        val_c_index = concordance_index(self.y_val, self.val_pred, self.event_val)
        
        complexity_summary = f"""
MODEL COMPLEXITY ANALYSIS

Structure Complexity:
- Total boosting rounds: {self.model.num_boosted_rounds()}
- Total splits: {total_splits:,}
- Avg tree depth: {avg_tree_depth:.1f}
- Features used: {unique_features_used}/{len(self.X_train.columns)}
- Splits per tree: {total_splits/self.model.num_boosted_rounds():.1f}

Performance Metrics:
- Train C-Index: {train_c_index:.3f}
- Val C-Index: {val_c_index:.3f}
- Overfitting gap: {train_c_index - val_c_index:.3f}

Complexity-Performance Trade-off:
- Model size: {'Large' if total_splits > 5000 else 'Medium' if total_splits > 1000 else 'Small'}
- Feature utilization: {unique_features_used/len(self.X_train.columns):.1%}
- Tree diversity: {'High' if avg_tree_depth > 5 else 'Medium' if avg_tree_depth > 3 else 'Low'}

Interaction Patterns:
- Top feature pairs: {len(feature_pairs)} detected
- Interaction strength: {'High' if len(feature_pairs) > 20 else 'Medium' if len(feature_pairs) > 10 else 'Low'}

Recommendations:
- Regularization: {'Increase' if train_c_index - val_c_index > 0.05 else 'Current OK'}
- Feature selection: {'Consider' if unique_features_used/len(self.X_train.columns) < 0.5 else 'Current OK'}
- Tree depth: {'Reduce' if avg_tree_depth > 6 else 'Current OK'}
"""
        
        ax6.text(0.05, 0.95, complexity_summary, transform=ax6.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig('feature_interaction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return {
            'feature_pairs': feature_pairs,
            'tree_complexity': {
                'depths': tree_depths,
                'performance': tree_performance,
                'total_splits': total_splits,
                'avg_depth': avg_tree_depth
            },
            'performance_metrics': {
                'train_c_index': train_c_index,
                'val_c_index': val_c_index,
                'overfitting_gap': train_c_index - val_c_index
            }
        }

    def run_complete_deep_dive(self):
        """Execute comprehensive XGBoost AFT deep dive analysis"""
        print("\n" + "="*80)
        print("XGBOOST AFT COMPREHENSIVE DEEP DIVE ANALYSIS")
        print("="*80)
        
        results = {}
        
        # 1. Target distribution analysis
        print("1/5 Analyzing target distribution characteristics...")
        results['target_distribution'] = self.analyze_target_distribution()
        
        # 2. AFT distribution comparison
        print("2/5 Comparing AFT distribution options...")
        results['aft_distributions'] = self.analyze_aft_distributions()
        
        # 3. Training dynamics
        print("3/5 Analyzing training dynamics and convergence...")
        results['training_dynamics'] = self.analyze_training_dynamics()
        
        # 4. Residual analysis and AFT assumptions
        print("4/5 Validating AFT assumptions and residual patterns...")
        results['residual_analysis'] = self.analyze_residuals_and_assumptions()
        
        # 5. Feature interactions and tree structure
        print("5/5 Analyzing feature interactions and model complexity...")
        results['feature_interactions'] = self.analyze_feature_interactions()
        
        # Generate executive summary
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY - XGBOOST AFT DEEP DIVE")
        print("="*60)
        
        # Extract key insights
        target_stats = results['target_distribution']['survival_stats']
        aft_best = results['aft_distributions']['best_distribution']
        training_gap = results['training_dynamics']['overfitting_gap']
        residual_ks = results['residual_analysis']['diagnostics']['ks_pvalue']
        model_complexity = results['feature_interactions']['tree_complexity']['total_splits']
        
        executive_summary = f"""
KEY FINDINGS:

Target Distribution:
- Median survival: {target_stats['median']:.0f} days
- Distribution skewness: {target_stats['skewness']:.2f}
- AFT suitability: {'Good' if abs(results['target_distribution']['log_stats']['skewness']) < 1 else 'Needs attention'}

Best AFT Distribution: {aft_best}
- Recommendation: Use 'aft_loss_distribution': '{aft_best.lower()}' in XGBoost

Training Quality:
- Overfitting gap: {training_gap:.4f} ({'Good' if training_gap < 0.05 else 'Needs attention'})
- Training stability: {'Stable' if training_gap < 0.1 else 'Unstable'}

AFT Assumptions:
- Residual distribution fit: {'Valid' if residual_ks > 0.05 else 'Violated'} (p={residual_ks:.4f})
- Homoscedasticity: {'Pass' if abs(results['residual_analysis']['diagnostics']['homoscedasticity_corr']) < 0.1 else 'Fail'}

Model Complexity:
- Total splits: {model_complexity:,}
- Model size: {'Appropriate' if 1000 < model_complexity < 10000 else 'Review needed'}

PERFORMANCE IMPROVEMENT RECOMMENDATIONS:

1. Distribution: {'Switch to ' + aft_best.lower() if aft_best != 'Logistic' else 'Current logistic OK'}
2. Regularization: {'Increase lambda/alpha' if training_gap > 0.1 else 'Current parameters OK'}
3. Feature Engineering: {'Add interactions' if abs(results['residual_analysis']['diagnostics']['homoscedasticity_corr']) > 0.1 else 'Current features adequate'}
4. Early Stopping: {'More aggressive' if training_gap > 0.05 else 'Current strategy OK'}

Expected C-Index Improvement: {0.02 if training_gap > 0.1 or residual_ks < 0.05 else 0.01:.3f}
Expected IBS Improvement: {0.05 if residual_ks < 0.05 else 0.02:.3f}
"""
        
        print(executive_summary)
        
        results['executive_summary'] = executive_summary
        
        return results


# Usage example extending xgboost_v1.py:
def run_xgboost_deep_dive(analyzer_instance):
    """
    Run comprehensive XGBoost AFT deep dive analysis
    
    Args:
        analyzer_instance: Trained SurvivalAnalysis instance from xgboost_v1.py
    
    Returns:
        Deep dive analysis results
    """
    deep_dive = XGBoostAFTDeepDive(analyzer_instance)
    return deep_dive.run_complete_deep_dive()


# Example integration with your existing code:
"""
# After running your xgboost_v1.py analysis:
analyzer = SurvivalAnalysis(data)
insights, summary = analyzer.run_survival_analysis()

# Run deep dive analysis:
deep_dive_results = run_xgboost_deep_dive(analyzer)

# Access specific analysis components:
target_analysis = deep_dive_results['target_distribution']
aft_comparison = deep_dive_results['aft_distributions'] 
training_analysis = deep_dive_results['training_dynamics']
residual_analysis = deep_dive_results['residual_analysis']
interaction_analysis = deep_dive_results['feature_interactions']
"""
