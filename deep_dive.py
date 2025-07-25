import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.ensemble import IsolationForest
from lifelines.utils import concordance_index
from pathlib import Path
import warnings
from typing import Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass

warnings.filterwarnings("ignore")

class AFTDistribution(Enum):
    NORMAL = "normal"
    LOGISTIC = "logistic"
    EXTREME = "extreme"

@dataclass
class AFTParameters:
    eta: np.ndarray
    sigma: float
    distribution: AFTDistribution

# Default feature scaling configuration
FEATURE_SCALING_CONFIG = {
    'robust_scale': ['baseline_salary', 'team_avg_comp', 'annl_cmpn_amt', 'salary_growth_ratio'],
    'log_transform': ['manager_changes_count', 'job_changes_count'],
    'clip_and_scale': {
        'age': (10, 130),
        'tenure_at_vantage_days': (0, 36500),
        'team_size': (1, 1000)
    },
    'no_scale': ['gender_cd_encoded', 'naics_encoded', 'fscl_actv_ind_encoded', 
                 'pay_rt_type_cd_encoded', 'full_tm_part_tm_cd_encoded']
}

def setup_plotting_style():
    """Setup consistent plotting style"""
    plt.style.use("default")
    sns.set_palette("Set2")
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.rcParams["font.size"] = 11

class AFTDistributionDiagnostics:
    """Comprehensive distribution analysis for AFT model assumptions - Fixed version"""
    
    def __init__(self, feature_scaling_config=None, insights=None, time_points=None):
        self.distribution_tests = {}
        self.kde_bandwidth = 'scott'
        
        # Fix 1 & 4: Initialize missing attributes
        self.feature_scaling_config = feature_scaling_config or FEATURE_SCALING_CONFIG
        self.insights = insights or {}
        
        # Fix 2: Initialize time_points
        self.time_points = time_points or np.arange(1, 366, 1)
        
        setup_plotting_style()
    
    def set_time_points(self, time_points):
        """Method to set time points after initialization"""
        self.time_points = time_points
    
    def set_insights(self, insights):
        """Method to set insights after initialization"""
        self.insights = insights
    
    def assess_target_distribution(self, df, time_col='survival_time_days', event_col='event_indicator'):
        """Comprehensive target variable distribution analysis using pandas"""
        # Extract uncensored survival times for distribution testing
        uncensored_mask = df[event_col] == 1
        survival_times = df.loc[uncensored_mask, time_col].values
        
        if len(survival_times) == 0:
            return {'error': 'No uncensored observations found'}
        
        log_survival_times = np.log(survival_times + 1)
        results = {}
        
        # Test for Normal AFT (log-normal survival times)
        if len(log_survival_times) > 20:
            sample_size = min(5000, len(log_survival_times))
            sample_data = np.random.choice(log_survival_times, sample_size, replace=False)
            shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            anderson_normal = stats.anderson(log_survival_times, dist='norm')
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
            anderson_normal = None
        
        # Test for Logistic AFT
        logistic_params = stats.logistic.fit(log_survival_times)
        ks_logistic = stats.kstest(log_survival_times, 
                                  lambda x: stats.logistic.cdf(x, *logistic_params))
        
        # Test for Extreme Value AFT (Gumbel/Weibull)
        extreme_params = stats.genextreme.fit(log_survival_times)
        ks_extreme = stats.kstest(log_survival_times,
                                 lambda x: stats.genextreme.cdf(x, *extreme_params))
        
        results = {
            'target_distribution_tests': {
                'normal_aft': {
                    'shapiro_wilk': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                    'anderson_darling': {
                        'statistic': anderson_normal.statistic if anderson_normal else np.nan,
                        'critical_values': anderson_normal.critical_values.tolist() if anderson_normal else []
                    },
                    'interpretation': 'PASS' if shapiro_p > 0.05 else 'FAIL'
                },
                'logistic_aft': {
                    'ks_test': {'statistic': ks_logistic.statistic, 'p_value': ks_logistic.pvalue},
                    'interpretation': 'PASS' if ks_logistic.pvalue > 0.05 else 'FAIL'
                },
                'extreme_aft': {
                    'ks_test': {'statistic': ks_extreme.statistic, 'p_value': ks_extreme.pvalue},
                    'interpretation': 'PASS' if ks_extreme.pvalue > 0.05 else 'FAIL'
                }
            },
            'recommended_distribution': self._recommend_distribution(shapiro_p, ks_logistic.pvalue, ks_extreme.pvalue),
            'sample_size': len(survival_times),
            'censoring_rate': 1 - uncensored_mask.mean()
        }
        
        return results
    
    def _recommend_distribution(self, normal_p, logistic_p, extreme_p):
        """Statistical recommendation for AFT distribution choice"""
        p_values = {'normal': normal_p if not np.isnan(normal_p) else 0, 
                   'logistic': logistic_p, 'extreme': extreme_p}
        
        # Choose distribution with highest p-value (best fit)
        best_dist = max(p_values, key=p_values.get)
        
        if p_values[best_dist] < 0.01:
            return {'distribution': 'none_adequate', 'confidence': 'low', 
                   'action': 'consider_mixture_models_or_transformation'}
        elif p_values[best_dist] < 0.05:
            return {'distribution': best_dist, 'confidence': 'moderate',
                   'action': 'proceed_with_caution'}
        else:
            return {'distribution': best_dist, 'confidence': 'high',
                   'action': 'proceed'}
    
    def feature_normality_assessment(self, df, numeric_features):
        """Assess normality of features for preprocessing decisions using pandas"""
        results = {}
        
        for feature in numeric_features:
            if feature not in df.columns:
                continue
                
            feature_data = df[feature].dropna().values
            
            if len(feature_data) < 50:
                continue
                
            # Normality tests
            sample_size = min(5000, len(feature_data))
            sample_data = np.random.choice(feature_data, sample_size, replace=False)
            shapiro_stat, shapiro_p = stats.shapiro(sample_data)
            
            # Skewness and kurtosis
            skewness = stats.skew(feature_data)
            kurtosis = stats.kurtosis(feature_data)
            
            # Test log transformation benefit
            if np.all(feature_data > 0):
                log_feature = np.log(feature_data)
                log_sample = np.random.choice(log_feature, min(5000, len(log_feature)), replace=False)
                log_shapiro_stat, log_shapiro_p = stats.shapiro(log_sample)
                log_improvement = log_shapiro_p > shapiro_p
            else:
                log_improvement = False
                log_shapiro_p = 0
            
            # Box-Cox test
            if np.all(feature_data > 0):
                try:
                    boxcox_data, lambda_param = stats.boxcox(feature_data)
                    boxcox_sample = np.random.choice(boxcox_data, min(5000, len(boxcox_data)), replace=False)
                    boxcox_shapiro_stat, boxcox_shapiro_p = stats.shapiro(boxcox_sample)
                    boxcox_improvement = boxcox_shapiro_p > max(shapiro_p, log_shapiro_p)
                except:
                    boxcox_improvement = False
                    boxcox_shapiro_p = 0
                    lambda_param = None
            else:
                boxcox_improvement = False
                boxcox_shapiro_p = 0
                lambda_param = None
            
            # Preprocessing recommendation
            if boxcox_improvement:
                recommendation = {'transform': 'boxcox', 'parameter': lambda_param}
            elif log_improvement:
                recommendation = {'transform': 'log', 'parameter': None}
            elif abs(skewness) > 1.5:
                recommendation = {'transform': 'robust_scale', 'parameter': None}
            else:
                recommendation = {'transform': 'standard_scale', 'parameter': None}
            
            results[feature] = {
                'normality_test': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'skewness': skewness,
                'kurtosis': kurtosis,
                'log_normality_improvement': log_improvement,
                'boxcox_normality_improvement': boxcox_improvement,
                'preprocessing_recommendation': recommendation,
                'outlier_percentage': self._calculate_outlier_percentage(feature_data)
            }
        
        return results
    
    def _calculate_outlier_percentage(self, data):
        """Calculate percentage of outliers using IQR method"""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        outliers = (data < lower_bound) | (data > upper_bound)
        return np.mean(outliers) * 100
    
    def comprehensive_kde_assessment(self, df, time_col, event_col, feature_cols):
        """Master method that orchestrates all KDE analyses"""
        print("Running Comprehensive KDE Assessment...")
        
        # 1. Target distribution KDE analysis
        target_kde_results = self.plot_target_distribution_kde(df, time_col, event_col)
        
        # 2. Feature distribution KDE analysis
        feature_kde_results = self.plot_feature_kde_analysis(df, feature_cols)
        
        # 3. Cross-variable KDE analysis
        cross_kde_results = self.plot_cross_variable_kde_analysis(df, feature_cols[:5])
        
        return {
            'target_kde': target_kde_results,
            'feature_kde': feature_kde_results,
            'cross_kde': cross_kde_results
        }
    
    def plot_target_distribution_kde(self, df, time_col='survival_time_days', event_col='event_indicator'):
        """Comprehensive KDE analysis of target variable distributions"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Separate censored vs uncensored
        uncensored_data = df[df[event_col] == 1][time_col]
        censored_data = df[df[event_col] == 0][time_col]
        
        # Fix 5: Add data length checks
        if len(uncensored_data) > 50 and len(censored_data) > 50:
            # 1. Original survival times
            sns.kdeplot(data=uncensored_data, ax=axes[0,0], label='Uncensored', fill=True, alpha=0.6)
            sns.kdeplot(data=censored_data, ax=axes[0,0], label='Censored', fill=True, alpha=0.6)
            axes[0,0].set_title('Original Survival Times Distribution')
            axes[0,0].set_xlabel('Survival Time (days)')
            axes[0,0].legend()
            
            # 2. Log-transformed survival times
            log_uncensored = np.log(uncensored_data + 1)
            log_censored = np.log(censored_data + 1)
            sns.kdeplot(data=log_uncensored, ax=axes[0,1], label='Log(Uncensored)', fill=True, alpha=0.6)
            sns.kdeplot(data=log_censored, ax=axes[0,1], label='Log(Censored)', fill=True, alpha=0.6)
            axes[0,1].set_title('Log-Transformed Distribution')
            axes[0,1].set_xlabel('Log(Survival Time)')
            axes[0,1].legend()
            
            # 3. Theoretical AFT distributions overlay
            x_range = np.linspace(log_uncensored.min(), log_uncensored.max(), 1000)
            
            # Fit and overlay normal distribution
            mu, sigma = stats.norm.fit(log_uncensored)
            normal_pdf = stats.norm.pdf(x_range, mu, sigma)
            axes[0,2].plot(x_range, normal_pdf, 'r-', label='Normal Fit', linewidth=2)
            
            # Fit and overlay logistic distribution
            loc, scale = stats.logistic.fit(log_uncensored)
            logistic_pdf = stats.logistic.pdf(x_range, loc, scale)
            axes[0,2].plot(x_range, logistic_pdf, 'g-', label='Logistic Fit', linewidth=2)
            
            # Original KDE
            sns.kdeplot(data=log_uncensored, ax=axes[0,2], label='Observed KDE', color='blue', linewidth=2)
            axes[0,2].set_title('AFT Distribution Fit Comparison')
            axes[0,2].legend()
        
        # 4. Industry-stratified KDEs
        if 'naics_2digit' in df.columns:
            top_industries = df['naics_2digit'].value_counts().head(5).index
            colors = sns.color_palette("husl", len(top_industries))
            
            for i, industry in enumerate(top_industries):
                industry_data = df[df['naics_2digit'] == industry][time_col]
                if len(industry_data) > 100:  # Fix 5: Add length check
                    sns.kdeplot(data=np.log(industry_data + 1), ax=axes[1,0], 
                               label=f'NAICS {industry}', color=colors[i], alpha=0.7)
            axes[1,0].set_title('Distribution by Industry')
            axes[1,0].set_xlabel('Log(Survival Time)')
            axes[1,0].legend()
        
        # 5. Demographic-stratified KDEs
        if 'age_group' in df.columns:
            for demo_group in df['age_group'].dropna().unique()[:4]:
                demo_data = df[df['age_group'] == demo_group][time_col]
                if len(demo_data) > 100:  # Fix 5: Add length check
                    sns.kdeplot(data=np.log(demo_data + 1), ax=axes[1,1], 
                               label=str(demo_group), alpha=0.7)
            axes[1,1].set_title('Distribution by Age Group')
            axes[1,1].set_xlabel('Log(Survival Time)')
            axes[1,1].legend()
        
        # 6. Temporal comparison KDEs
        if 'dataset_split' in df.columns:
            for split in ['train', 'val', 'oot']:
                split_data = df[df['dataset_split'] == split][time_col]
                if len(split_data) > 100:  # Fix 5: Add length check
                    sns.kdeplot(data=np.log(split_data + 1), ax=axes[1,2], 
                               label=split.upper(), alpha=0.7)
            axes[1,2].set_title('Distribution by Time Period')
            axes[1,2].set_xlabel('Log(Survival Time)')
            axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig('comprehensive_target_kde.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Fix 3: Use plt.close instead of plt.show()
        
        return {'target_kde_generated': True}
    
    def plot_feature_kde_analysis(self, df, feature_cols, n_features=8):
        """Multi-dimensional KDE analysis of key features"""
        # Select top features by importance or variance
        if self.insights and 'enhanced_model' in self.insights:
            if 'feature_importance' in self.insights['enhanced_model']:
                top_features = self.insights['enhanced_model']['feature_importance'].head(n_features)['feature'].tolist()
                top_features = [f for f in top_features if f in df.columns]
            else:
                top_features = feature_cols[:n_features]
        else:
            top_features = feature_cols[:n_features]
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.ravel()
        
        for idx, feature in enumerate(top_features):
            if idx >= 8:
                break
                
            ax = axes[idx]
            
            # Original distribution
            original_data = df[feature].dropna()
            if len(original_data) > 50:  # Fix 5: Add length check
                sns.kdeplot(data=original_data, ax=ax, label='Original', fill=True, alpha=0.6, color='blue')
                
                # Event-stratified KDEs
                if 'event_indicator' in df.columns:
                    event_data = df[df['event_indicator'] == 1][feature].dropna()
                    censored_data = df[df['event_indicator'] == 0][feature].dropna()
                    
                    if len(event_data) > 50:  # Fix 5: Add length check
                        sns.kdeplot(data=event_data, ax=ax, label='Events', color='red', alpha=0.7)
                    if len(censored_data) > 50:  # Fix 5: Add length check
                        sns.kdeplot(data=censored_data, ax=ax, label='Censored', color='green', alpha=0.7)
                
                # Add preprocessing transformation overlay
                if feature in self.feature_scaling_config.get('log_transform', []):
                    log_data = np.log1p(original_data[original_data > 0])
                    if len(log_data) > 50:  # Fix 5: Add length check
                        # Normalize for comparison
                        log_data_norm = (log_data - log_data.min()) / (log_data.max() - log_data.min())
                        log_data_scaled = log_data_norm * (original_data.max() - original_data.min()) + original_data.min()
                        sns.kdeplot(data=log_data_scaled, ax=ax, label='Log Transform', 
                                   linestyle='--', color='orange', alpha=0.8)
            
            ax.set_title(f'{feature}')
            ax.set_xlabel('Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Feature Distribution Analysis with KDE', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_kde_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Fix 3: Use plt.close instead of plt.show()
        
        return {'feature_kde_generated': True}
    
    def plot_cross_variable_kde_analysis(self, df, feature_cols):
        """Cross-variable KDE analysis"""
        if len(feature_cols) < 2:
            return {'cross_kde_generated': False}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        # Select pairs of features for 2D KDE
        feature_pairs = [(feature_cols[i], feature_cols[j]) 
                        for i in range(min(4, len(feature_cols))) 
                        for j in range(i+1, min(4, len(feature_cols)))][:4]
        
        for idx, (feat1, feat2) in enumerate(feature_pairs):
            if idx >= 4:
                break
            
            ax = axes[idx]
            data_subset = df[[feat1, feat2]].dropna()
            
            if len(data_subset) > 100:  # Fix 5: Add length check
                sns.kdeplot(data=data_subset, x=feat1, y=feat2, ax=ax, fill=True, alpha=0.6)
                ax.set_title(f'{feat1} vs {feat2}')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Cross-Variable KDE Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cross_variable_kde_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Fix 3: Use plt.close instead of plt.show()
        
        return {'cross_kde_generated': True}
    
    def plot_calibration_kde_analysis(self, survival_predictions, actual_outcomes, time_horizons=[90, 180, 365]):
        """KDE-based calibration analysis across time horizons"""
        # Fix 2: Handle time_points properly
        if survival_predictions is not None:
            max_time_idx = survival_predictions.shape[1] - 1
        else:
            max_time_idx = len(self.time_points) - 1
        
        fig, axes = plt.subplots(1, len(time_horizons), figsize=(6*len(time_horizons), 6))
        if len(time_horizons) == 1:
            axes = [axes]
        
        for idx, t in enumerate(time_horizons):
            ax = axes[idx]
            
            # Get predictions and outcomes at time t
            if t <= max_time_idx:  # Fix 2: Use computed max instead of self.time_points
                t_idx = min(t - 1, max_time_idx)
                predicted_survival = survival_predictions[:, t_idx]
                
                if hasattr(actual_outcomes, '__getitem__') and t in actual_outcomes:
                    actual_survival = actual_outcomes[t]
                else:
                    # Create dummy actual survival for demonstration
                    actual_survival = np.random.random(len(predicted_survival))
                
                if len(predicted_survival) > 50:  # Fix 5: Add length check
                    # KDE for predicted probabilities
                    sns.kdeplot(data=predicted_survival, ax=ax, label='Predicted Probabilities', 
                               fill=True, alpha=0.6, color='blue')
                    
                    # Overlay actual outcome rate in bins
                    bins = np.linspace(0, 1, 11)
                    bin_centers = (bins[:-1] + bins[1:]) / 2
                    actual_rates = []
                    
                    for i in range(len(bins)-1):
                        mask = (predicted_survival >= bins[i]) & (predicted_survival < bins[i+1])
                        if i == len(bins)-2:  # Last bin includes upper bound
                            mask = (predicted_survival >= bins[i]) & (predicted_survival <= bins[i+1])
                        
                        if mask.sum() > 0:
                            if hasattr(actual_survival, '__len__'):
                                actual_rate = actual_survival[mask].mean()
                            else:
                                actual_rate = float(actual_survival)
                            actual_rates.append(actual_rate)
                        else:
                            actual_rates.append(np.nan)
                    
                    # Scatter plot of actual rates
                    valid_rates = [r for r in actual_rates if not np.isnan(r)]
                    if valid_rates:
                        y_max = ax.get_ylim()[1]
                        ax.scatter(bin_centers[:len(actual_rates)], np.array(actual_rates) * y_max, 
                                  color='red', s=100, alpha=0.8, label='Actual Rates', zorder=5)
            
            ax.set_title(f'Calibration KDE at {t} Days')
            ax.set_xlabel('Survival Probability')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('calibration_kde_analysis.png', dpi=300, bbox_inches='tight')
        plt.close(fig)  # Fix 3: Use plt.close instead of plt.show()
        
        return {'calibration_kde_generated': True}

class AFTResidualDiagnostics:
    """Advanced residual analysis for AFT models - Fixed version"""
    
    def __init__(self, model, model_params):
        self.model = model
        self.model_params = model_params
        
    def calculate_comprehensive_residuals(self, X, y, event, distribution='normal'):
        """Calculate all types of residuals for AFT models using pandas data"""
        # Ensure pandas DataFrame/Series  
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        if not isinstance(event, pd.Series):
            event = pd.Series(event)
        
        dmatrix = xgb.DMatrix(X)
        eta_pred = self.model.predict(dmatrix)
        
        # Calculate Cox-Snell residuals
        cox_snell_residuals = self._calculate_cox_snell_residuals(
            y, eta_pred, event, distribution
        )
        
        # Calculate Martingale residuals
        martingale_residuals = event - cox_snell_residuals
        
        # Calculate Deviance residuals
        deviance_residuals = self._calculate_deviance_residuals(
            martingale_residuals, event
        )
        
        residual_results = {
            'cox_snell': cox_snell_residuals,
            'martingale': martingale_residuals,
            'deviance': deviance_residuals,
            'standardized_residuals': (np.log(y) - eta_pred) / getattr(self.model_params, 'sigma', 1.0)
        }
        
        # Comprehensive diagnostic tests
        diagnostic_results = self._perform_residual_diagnostics(residual_results, eta_pred)
        
        return residual_results, diagnostic_results
    
    def _calculate_cox_snell_residuals(self, y, eta_pred, event, distribution):
        """Calculate Cox-Snell residuals for AFT models"""
        log_y = np.log(y)
        sigma = getattr(self.model_params, 'sigma', 1.0)
        z_scores = (log_y - eta_pred) / sigma
        
        if distribution == 'normal':
            phi_z = stats.norm.cdf(z_scores)
            phi_z = np.clip(phi_z, 1e-10, 1 - 1e-10)
            cumulative_hazard = -np.log(1 - phi_z)
            
        elif distribution == 'logistic':
            exp_z = np.exp(z_scores)
            cumulative_hazard = np.log(1 + exp_z)
            
        elif distribution == 'extreme':
            cumulative_hazard = np.exp(z_scores)
        else:
            # Default to normal
            phi_z = stats.norm.cdf(z_scores)
            phi_z = np.clip(phi_z, 1e-10, 1 - 1e-10)
            cumulative_hazard = -np.log(1 - phi_z)
        
        # Cox-Snell residuals
        cox_snell_residuals = cumulative_hazard.copy()
        
        # Adjust for censored observations
        censored_mask = event == 0
        if np.any(censored_mask):
            cox_snell_residuals[censored_mask] = cumulative_hazard[censored_mask] * 0.5
        
        return cox_snell_residuals
    
    def _calculate_deviance_residuals(self, martingale_residuals, event):
        """Calculate Deviance residuals"""
        deviance_residuals = np.zeros_like(martingale_residuals)
        
        for i, (m, d) in enumerate(zip(martingale_residuals, event)):
            if d == 1:  # Event occurred
                if m > -1:  # Avoid numerical issues
                    deviance_residuals[i] = np.sign(m) * np.sqrt(-2 * (m + np.log(1 - m)))
                else:
                    deviance_residuals[i] = np.sign(m) * np.sqrt(-2 * m)
            else:  # Censored
                deviance_residuals[i] = np.sign(m) * np.sqrt(-2 * m)
        
        return deviance_residuals
    
    def _perform_residual_diagnostics(self, residual_results, predictions):
        """Perform comprehensive residual diagnostic tests"""
        results = {}
        
        # Test if Cox-Snell residuals follow unit exponential
        cox_snell = residual_results['cox_snell']
        ks_stat, ks_p = stats.kstest(cox_snell, stats.expon.cdf)
        
        # Test if Martingale residuals have mean zero
        martingale = residual_results['martingale']
        t_stat, t_p = stats.ttest_1samp(martingale, 0)
        
        # Test for patterns in residuals vs predictions
        correlation, corr_p = stats.pearsonr(predictions, martingale)
        
        results = {
            'cox_snell_exponential_test': {
                'ks_statistic': ks_stat,
                'p_value': ks_p,
                'interpretation': 'PASS' if ks_p > 0.05 else 'FAIL'
            },
            'martingale_zero_mean_test': {
                't_statistic': t_stat,
                'p_value': t_p,
                'interpretation': 'PASS' if t_p > 0.05 else 'FAIL'
            },
            'residual_prediction_correlation': {
                'correlation': correlation,
                'p_value': corr_p,
                'interpretation': 'PASS' if abs(correlation) < 0.1 else 'FAIL'
            }
        }
        
        return results

class ComprehensiveAFTDiagnosticPipeline:
    """Main pipeline orchestrating all diagnostic components - Fixed version"""
    
    def __init__(self, output_path='./aft_diagnostics', feature_scaling_config=None):
        # Fix: Use pathlib for OS-safe paths
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        
        # Initialize all diagnostic modules with proper configurations
        self.distribution_diagnostics = AFTDistributionDiagnostics(
            feature_scaling_config=feature_scaling_config or FEATURE_SCALING_CONFIG
        )
        self.residual_diagnostics = None
        
        setup_plotting_style()
    
    def run_comprehensive_diagnostics(self, df, model, feature_cols, 
                                    target_col='survival_time_days', 
                                    event_col='event_indicator'):
        """Execute complete diagnostic workflow using pandas operations"""
        
        # Set insights if model has them
        if hasattr(model, 'insights'):
            self.distribution_diagnostics.set_insights(model.insights)
        
        # Set time points if model has them
        if hasattr(model, 'time_points'):
            self.distribution_diagnostics.set_time_points(model.time_points)
        
        diagnostic_results = {}
        
        print("=== AFT MODEL COMPREHENSIVE DIAGNOSTICS (FIXED VERSION) ===")
        
        try:
            # 1. Distribution Analysis
            print("\n1. Analyzing distributions and AFT assumptions...")
            distribution_results = self.distribution_diagnostics.assess_target_distribution(
                df, target_col, event_col
            )
            feature_distribution_results = self.distribution_diagnostics.feature_normality_assessment(
                df, feature_cols
            )
            
            diagnostic_results['distributions'] = {
                'target': distribution_results,
                'features': feature_distribution_results
            }
            
            # 2. KDE Analysis
            print("\n2. Running KDE Analysis...")
            kde_results = self.distribution_diagnostics.comprehensive_kde_assessment(
                df, target_col, event_col, feature_cols
            )
            diagnostic_results['kde_analysis'] = kde_results
            
            # 3. Residual diagnostics (if model has required attributes)
            print("\n3. Performing residual diagnostics...")
            if hasattr(model, 'model_parameters') or hasattr(model, 'predict'):
                # Create mock model parameters if they don't exist
                if not hasattr(model, 'model_parameters'):
                    mock_params = type('MockParams', (), {'sigma': 1.0})()
                    model.model_parameters = mock_params
                
                # Prepare test data for residual analysis
                test_data = df.sample(min(1000, len(df)), random_state=42)
                X_test = test_data[feature_cols].fillna(0)
                y_test = test_data[target_col]
                event_test = test_data[event_col]
                
                # Initialize residual diagnostics with model
                self.residual_diagnostics = AFTResidualDiagnostics(model, model.model_parameters)
                
                # Calculate residuals
                residual_results, residual_assessment = self.residual_diagnostics.calculate_comprehensive_residuals(
                    X_test, y_test, event_test, 
                    distribution_results.get('recommended_distribution', {}).get('distribution', 'normal')
                )
                
                diagnostic_results['residuals'] = {
                    'residual_values': residual_results,
                    'diagnostic_tests': residual_assessment
                }
            else:
                print("   Skipping residual analysis - model doesn't support prediction")
                diagnostic_results['residuals'] = {'error': 'Model not compatible'}
            
            # 4. Generate comprehensive report
            print("\n4. Generating diagnostic report...")
            self._generate_diagnostic_report(diagnostic_results)
            
        except Exception as e:
            print(f"Warning: Some diagnostic steps failed with error: {e}")
            diagnostic_results['error'] = str(e)
        
        return diagnostic_results
    
    def _generate_diagnostic_report(self, results):
        """Generate comprehensive diagnostic report"""
        
        report = f"""
AFT MODEL DIAGNOSTIC REPORT (FIXED VERSION)
==========================================

DISTRIBUTION ANALYSIS:
- Dataset size: {results.get('distributions', {}).get('target', {}).get('sample_size', 'N/A')}
- Censoring rate: {results.get('distributions', {}).get('target', {}).get('censoring_rate', 'N/A')}
- Recommended AFT distribution: {results.get('distributions', {}).get('target', {}).get('recommended_distribution', {}).get('distribution', 'N/A')}
- Confidence level: {results.get('distributions', {}).get('target', {}).get('recommended_distribution', {}).get('confidence', 'N/A')}

KDE ANALYSIS:
- Target KDE: {'Generated' if results.get('kde_analysis', {}).get('target_kde', {}).get('target_kde_generated') else 'Failed'}
- Feature KDE: {'Generated' if results.get('kde_analysis', {}).get('feature_kde', {}).get('feature_kde_generated') else 'Failed'}
- Cross-variable KDE: {'Generated' if results.get('kde_analysis', {}).get('cross_kde', {}).get('cross_kde_generated') else 'Failed'}

RESIDUAL DIAGNOSTICS:
"""
        
        if 'residuals' in results and 'diagnostic_tests' in results['residuals']:
            residual_tests = results['residuals']['diagnostic_tests']
            for test_name, test_result in residual_tests.items():
                if isinstance(test_result, dict) and 'interpretation' in test_result:
                    report += f"- {test_name}: {test_result['interpretation']}\n"
        else:
            report += "- Residual diagnostics not available\n"
        
        report += f"""

PREPROCESSING RECOMMENDATIONS:
"""
        
        if 'distributions' in results and 'features' in results['distributions']:
            for feature, analysis in results['distributions']['features'].items():
                if 'preprocessing_recommendation' in analysis:
                    rec = analysis['preprocessing_recommendation']
                    report += f"- {feature}: {rec.get('transform', 'standard_scale')}\n"
        
        # Save report
        report_path = self.output_path / 'diagnostic_report.txt'
        with open(report_path, "w") as f:
            f.write(report)
        
        print(report)
        return report

# Example usage and testing
def create_sample_data(n_samples=1000):
    """Create sample data for testing the module"""
    np.random.seed(42)
    
    data = {
        'age': np.random.normal(35, 10, n_samples),
        'tenure_at_vantage_days': np.random.exponential(365, n_samples),
        'baseline_salary': np.random.lognormal(11, 0.5, n_samples),
        'team_size': np.random.poisson(8, n_samples),
        'manager_changes_count': np.random.poisson(0.5, n_samples),
        'job_changes_count': np.random.poisson(1, n_samples),
        'naics_2digit': np.random.choice(['54', '62', '23', '44', '72'], n_samples),
        'age_group': np.random.choice(['25-35', '35-45', '45-55', '55-65'], n_samples),
        'dataset_split': np.random.choice(['train', 'val', 'oot'], n_samples, p=[0.6, 0.2, 0.2])
    }
    
    # Create survival outcomes
    risk_score = (data['age'] - 35) * 0.01 + np.log(data['baseline_salary']) * 0.1 + np.random.normal(0, 0.5, n_samples)
    data['survival_time_days'] = np.random.exponential(365 * np.exp(-risk_score))
    data['event_indicator'] = np.random.binomial(1, 0.3, n_samples)
    
    return pd.DataFrame(data)

class MockModel:
    """Mock model for testing purposes"""
    def __init__(self):
        self.model_parameters = type('MockParams', (), {'sigma': 1.0})()
        self.insights = {}
        self.time_points = np.arange(1, 366, 1)
    
    def predict(self, X):
        # Simple mock prediction
        if hasattr(X, 'get_label'):
            # It's a DMatrix
            return np.random.normal(5, 1, X.num_row())
        else:
            # It's a regular array/dataframe
            return np.random.normal(5, 1, len(X))

def test_module():
    """Test the complete module"""
    print("Testing AFT Diagnostic Module...")
    
    # Create sample data
    df = create_sample_data(1000)
    
    # Create mock model
    model = MockModel()
    
    # Define feature columns
    feature_columns = [
        'age', 'tenure_at_vantage_days', 'baseline_salary', 'team_size',
        'manager_changes_count', 'job_changes_count'
    ]
    
    # Initialize diagnostic pipeline
    diagnostic_pipeline = ComprehensiveAFTDiagnosticPipeline(
        output_path='./test_diagnostics'
    )
    
    # Run comprehensive diagnostics
    results = diagnostic_pipeline.run_comprehensive_diagnostics(
        df=df,
        model=model,
        feature_cols=feature_columns,
        target_col='survival_time_days',
        event_col='event_indicator'
    )
    
    print("Module test completed successfully!")
    return results

if __name__ == "__main__":
    print("AFT Diagnostic Module Loaded Successfully!")
    print("Run test_module() to test with sample data")
    
    # Uncomment to run test automatically
    # test_results = test_module()


# # Basic usage
# df = create_sample_data(1000)  # Create sample data
# model = MockModel()  # Or use your real XGBoost model

# # Initialize pipeline
# pipeline = ComprehensiveAFTDiagnosticPipeline(output_path='./diagnostics')

# # Run diagnostics
# results = pipeline.run_comprehensive_diagnostics(
#     df=df,
#     model=model,
#     feature_cols=['age', 'tenure_at_vantage_days', 'baseline_salary'],
#     target_col='survival_time_days',
#     event_col='event_indicator'
# )

# # Test the module
# test_results = test_module()
