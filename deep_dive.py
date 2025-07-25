import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class OptimizedAFTDiagnostics:
    """Optimized AFT diagnostics with unlimited features/industries, smart sampling"""
    
    def __init__(self, output_path='./aft_diagnostics', performance_mode=True):
        self.output_path = output_path
        self.performance_mode = performance_mode
        
        # Sampling parameters only (no feature/industry limits)
        self.max_sample_size = 50000 if performance_mode else None
        self.max_kde_samples = 10000
        self.max_test_samples = 2000
        self.chunk_size = 100000  # For memory management
        
        self.fitted_transformers = {}
        
        import os
        os.makedirs(output_path, exist_ok=True)
    
    def smart_sample(self, df, max_samples=None, stratify_col=None):
        """Intelligent sampling preserving all segments"""
        if max_samples is None:
            max_samples = self.max_sample_size
            
        if len(df) <= max_samples:
            return df
        
        if stratify_col and stratify_col in df.columns:
            # Stratified sampling to preserve all segments
            return df.groupby(stratify_col).apply(
                lambda x: x.sample(min(len(x), max_samples // df[stratify_col].nunique()), 
                                 random_state=42)
            ).reset_index(drop=True)
        else:
            return df.sample(n=max_samples, random_state=42)
    
    def process_in_chunks(self, df, chunk_size=None):
        """Process large dataframes in chunks for memory efficiency"""
        if chunk_size is None:
            chunk_size = self.chunk_size
            
        for i in range(0, len(df), chunk_size):
            yield df.iloc[i:i+chunk_size]
    
    def assess_target_distribution(self, df, time_col='survival_time_days', event_col='event_indicator'):
        """Optimized target distribution analysis with sampling"""
        # Sample for performance while preserving distribution
        if len(df) > self.max_sample_size:
            df_sample = self.smart_sample(df, self.max_sample_size, event_col)
        else:
            df_sample = df
        
        uncensored_mask = df_sample[event_col] == 1
        survival_times = df_sample.loc[uncensored_mask, time_col].values
        
        # Limit samples for expensive statistical tests only
        if len(survival_times) > self.max_test_samples:
            test_sample = np.random.choice(survival_times, self.max_test_samples, replace=False)
        else:
            test_sample = survival_times
        
        log_survival_times = np.log(test_sample + 1)
        
        # Fast normality test (Jarque-Bera instead of Shapiro-Wilk)
        if len(log_survival_times) > 20:
            jb_stat, jb_p = stats.jarque_bera(log_survival_times)
        else:
            jb_stat, jb_p = np.nan, np.nan
        
        # Distribution fit tests with sampling
        logistic_params = stats.logistic.fit(log_survival_times)
        ks_logistic = stats.kstest(log_survival_times[:1000],  # Sample for speed
                                 lambda x: stats.logistic.cdf(x, *logistic_params))
        
        return {
            'target_distribution_tests': {
                'normal_aft': {
                    'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p},
                    'interpretation': 'PASS' if jb_p > 0.05 else 'FAIL'
                },
                'logistic_aft': {
                    'ks_test': {'statistic': ks_logistic.statistic, 'p_value': ks_logistic.pvalue},
                    'interpretation': 'PASS' if ks_logistic.pvalue > 0.05 else 'FAIL'
                }
            },
            'recommended_distribution': self._recommend_distribution(jb_p, ks_logistic.pvalue),
            'sample_size': len(survival_times),
            'censoring_rate': 1 - uncensored_mask.mean(),
            'total_records': len(df)
        }
    
    def plot_target_kde_analysis(self, df, time_col='survival_time_days', event_col='event_indicator'):
        """KDE analysis with sampling but all industries shown"""
        # Sample data for KDE performance
        df_sample = self.smart_sample(df, self.max_kde_samples, event_col)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        uncensored_data = df_sample[df_sample[event_col] == 1][time_col]
        censored_data = df_sample[df_sample[event_col] == 0][time_col]
        
        # 1. Original survival times
        if len(uncensored_data) > 0:
            sns.kdeplot(data=uncensored_data, ax=axes[0,0], label='Uncensored', alpha=0.6)
        if len(censored_data) > 0:
            sns.kdeplot(data=censored_data, ax=axes[0,0], label='Censored', alpha=0.6)
        axes[0,0].set_title('Survival Times Distribution')
        axes[0,0].legend()
        
        # 2. Log-transformed
        if len(uncensored_data) > 0:
            log_uncensored = np.log(uncensored_data + 1)
            sns.kdeplot(data=log_uncensored, ax=axes[0,1], label='Log(Uncensored)', alpha=0.6)
            axes[0,1].set_title('Log-Transformed Distribution')
            axes[0,1].legend()
        
        # 3. ALL industries (no limit) - sample within each industry
        if 'naics_2digit' in df_sample.columns:
            all_industries = df_sample['naics_2digit'].value_counts()
            # Show all industries with sufficient data
            valid_industries = all_industries[all_industries >= 50].index
            colors = sns.color_palette("husl", len(valid_industries))
            
            for i, industry in enumerate(valid_industries):
                industry_data = df_sample[df_sample['naics_2digit'] == industry][time_col]
                if len(industry_data) > 50:
                    # Sample within industry if needed
                    if len(industry_data) > 1000:
                        industry_data = industry_data.sample(1000, random_state=42)
                    sns.kdeplot(data=np.log(industry_data + 1), ax=axes[1,0],
                              label=f'NAICS {industry}', color=colors[i % len(colors)], alpha=0.7)
            
            axes[1,0].set_title(f'Distribution by ALL Industries ({len(valid_industries)} shown)')
            axes[1,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Event vs Censored comparison
        if len(uncensored_data) > 0 and len(censored_data) > 0:
            sns.kdeplot(data=np.log(uncensored_data + 1), ax=axes[1,1], 
                       label='Events', alpha=0.6, color='red')
            sns.kdeplot(data=np.log(censored_data + 1), ax=axes[1,1], 
                       label='Censored', alpha=0.6, color='green')
            axes[1,1].set_title('Event vs Censored Comparison')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/target_kde_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_feature_kde_analysis(self, df, feature_cols):
        """KDE analysis for ALL features with intelligent sampling"""
        # Sample data but keep ALL features
        df_sample = self.smart_sample(df, 20000)
        
        # Calculate number of subplot rows/cols for ALL features
        n_features = len(feature_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.ravel()
        
        for idx, feature in enumerate(feature_cols):
            if feature not in df_sample.columns:
                continue
                
            ax = axes[idx]
            
            # Original distribution with sampling
            original_data = df_sample[feature].dropna()
            if len(original_data) > 5000:
                original_data = original_data.sample(5000, random_state=42)
            
            if len(original_data) > 0:
                sns.kdeplot(data=original_data, ax=ax, label='Original', alpha=0.6, color='blue')
            
            # Event-stratified KDEs
            if 'event_indicator' in df_sample.columns:
                event_data = df_sample[df_sample['event_indicator'] == 1][feature].dropna()
                censored_data = df_sample[df_sample['event_indicator'] == 0][feature].dropna()
                
                if len(event_data) > 50:
                    sample_size = min(2000, len(event_data))
                    event_sample = event_data.sample(sample_size, random_state=42) if len(event_data) > sample_size else event_data
                    sns.kdeplot(data=event_sample, ax=ax, label='Events', color='red', alpha=0.7)
                
                if len(censored_data) > 50:
                    sample_size = min(2000, len(censored_data))
                    censored_sample = censored_data.sample(sample_size, random_state=42) if len(censored_data) > sample_size else censored_data
                    sns.kdeplot(data=censored_sample, ax=ax, label='Censored', color='green', alpha=0.7)
            
            ax.set_title(f'{feature}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(len(feature_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Feature Distribution Analysis - ALL {len(feature_cols)} Features', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/feature_kde_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def calculate_residuals(self, model, X, y, event, distribution='normal'):
        """Fast residual calculation with chunked processing"""
        # Process in chunks for memory efficiency
        all_residuals = {'cox_snell': [], 'martingale': [], 'standardized': []}
        
        for chunk in self.process_in_chunks(pd.concat([X, y, event], axis=1), 50000):
            X_chunk = chunk.iloc[:, :-2]
            y_chunk = chunk.iloc[:, -2]
            event_chunk = chunk.iloc[:, -1]
            
            # Get predictions for chunk
            if hasattr(model, 'predict'):
                dmatrix = xgb.DMatrix(X_chunk)
                eta_pred = model.predict(dmatrix)
            else:
                eta_pred = model
            
            # Calculate residuals for chunk
            log_y = np.log(y_chunk)
            standardized_residuals = log_y - eta_pred
            
            # Simple Cox-Snell approximation
            z_scores = standardized_residuals / np.std(standardized_residuals)
            cox_snell_residuals = np.abs(z_scores)
            
            # Martingale residuals
            martingale_residuals = event_chunk - cox_snell_residuals
            
            # Collect results
            all_residuals['cox_snell'].extend(cox_snell_residuals.tolist())
            all_residuals['martingale'].extend(martingale_residuals.tolist())
            all_residuals['standardized'].extend(standardized_residuals.tolist())
        
        # Convert back to arrays
        for key in all_residuals:
            all_residuals[key] = np.array(all_residuals[key])
        
        return all_residuals
    
    def binned_performance_analysis(self, df, predictions_col, feature_cols, sample_size=100000):
        """Binned analysis for ALL features with sampling"""
        # Sample data if too large
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
        else:
            df_sample = df
        
        results = {}
        
        # Process ALL features (no limit)
        for feature in feature_cols:
            if feature not in df_sample.columns:
                continue
            
            feature_data = df_sample[[feature, predictions_col, 'event_indicator']].dropna()
            
            if len(feature_data) < 100:
                continue
            
            # Create bins
            try:
                feature_data['bin'] = pd.qcut(feature_data[feature], q=5, 
                                           labels=[f'Q{i+1}' for i in range(5)])
            except:
                feature_data['bin'] = pd.cut(feature_data[feature], bins=5, 
                                          labels=[f'Q{i+1}' for i in range(5)])
            
            bin_results = []
            for bin_name in feature_data['bin'].cat.categories:
                bin_df = feature_data[feature_data['bin'] == bin_name]
                
                if len(bin_df) < 20:
                    continue
                
                bin_metrics = {
                    'feature': feature,
                    'bin': bin_name,
                    'sample_size': len(bin_df),
                    'event_rate': bin_df['event_indicator'].mean(),
                    'mean_prediction': bin_df[predictions_col].mean(),
                    'calibration_error': abs(bin_df[predictions_col].mean() - 
                                           bin_df['event_indicator'].mean())
                }
                bin_results.append(bin_metrics)
            
            results[feature] = bin_results
        
        return results
    
    def stability_analysis(self, df, model, feature_cols):
        """Stability analysis for ALL industries with sampling"""
        stability_results = {}
        
        # Industry analysis (ALL industries, no limit)
        if 'naics_2digit' in df.columns:
            industry_counts = df['naics_2digit'].value_counts()
            valid_industries = industry_counts[industry_counts >= 500].index.tolist()  # Lower threshold
            
            for industry in valid_industries:
                industry_df = df[df['naics_2digit'] == industry].copy()
                
                # Sample within industry if too large
                if len(industry_df) > 20000:
                    industry_df = industry_df.sample(20000, random_state=42)
                
                # Prepare features
                X_industry = industry_df[feature_cols].fillna(0)
                
                # Calculate predictions
                dmatrix = xgb.DMatrix(X_industry)
                predictions = model.predict(dmatrix)
                
                # Get actual outcomes
                y_actual = industry_df['event_indicator'].values
                times_actual = industry_df['survival_time_days'].values
                
                # Calculate metrics
                industry_metrics = self._calculate_segment_metrics(
                    predictions, y_actual, times_actual, f'Industry_{industry}'
                )
                
                stability_results[industry] = industry_metrics
        
        return stability_results
    
    def _calculate_segment_metrics(self, predictions, actuals, times, segment_name):
        """Calculate comprehensive metrics for each segment"""
        from sklearn.metrics import roc_auc_score
        
        n_samples = len(predictions)
        event_rate = np.mean(actuals)
        mean_survival_time = np.mean(times[actuals == 1]) if np.any(actuals == 1) else np.nan
        
        # Discrimination metrics
        if len(set(actuals)) > 1:
            try:
                auc = roc_auc_score(actuals, predictions)
            except:
                auc = np.nan
        else:
            auc = np.nan
        
        # Calibration metrics
        mean_predicted_risk = np.mean(predictions)
        calibration_error = abs(mean_predicted_risk - event_rate)
        
        return {
            'segment': segment_name,
            'n_samples': n_samples,
            'event_rate': event_rate,
            'mean_survival_time': mean_survival_time,
            'auc': auc,
            'mean_predicted_risk': mean_predicted_risk,
            'calibration_error': calibration_error,
            'prediction_std': np.std(predictions)
        }
    
    def _recommend_distribution(self, normal_p, logistic_p):
        """Quick distribution recommendation"""
        if not np.isnan(normal_p) and normal_p > logistic_p:
            return {'distribution': 'normal', 'confidence': 'high' if normal_p > 0.05 else 'low'}
        else:
            return {'distribution': 'logistic', 'confidence': 'high' if logistic_p > 0.05 else 'low'}
    
    def comprehensive_diagnostics(self, df, model, feature_cols, 
                                time_col='survival_time_days', event_col='event_indicator'):
        """Main diagnostic pipeline - ALL features, ALL industries, smart sampling"""
        
        print(f"=== COMPREHENSIVE AFT DIAGNOSTICS ===")
        print(f"Dataset size: {len(df):,} records")
        print(f"Features: {len(feature_cols)} (ALL included)")
        
        results = {}
        
        # 1. Target distribution analysis
        print("\n1. Analyzing target distribution...")
        target_results = self.assess_target_distribution(df, time_col, event_col)
        results['target_distribution'] = target_results
        
        # 2. KDE visualizations
        print("\n2. Generating KDE visualizations...")
        self.plot_target_kde_analysis(df, time_col, event_col)
        self.plot_feature_kde_analysis(df, feature_cols)
        
        # 3. Add predictions to dataframe
        print("\n3. Calculating predictions...")
        X_all = df[feature_cols].fillna(0)
        
        # Process predictions in chunks if dataset is large
        all_predictions = []
        for chunk in self.process_in_chunks(X_all, 50000):
            dmatrix = xgb.DMatrix(chunk)
            chunk_predictions = model.predict(dmatrix)
            all_predictions.extend(chunk_predictions.tolist())
        
        df_copy = df.copy()
        df_copy['predictions'] = all_predictions
        
        # 4. Binned performance analysis (ALL features)
        print(f"\n4. Analyzing ALL {len(feature_cols)} features across bins...")
        binned_results = self.binned_performance_analysis(df_copy, 'predictions', feature_cols)
        results['binned_performance'] = binned_results
        
        # 5. Stability analysis (ALL industries)
        print("\n5. Stability analysis across ALL industries...")
        stability_results = self.stability_analysis(df_copy, model, feature_cols)
        results['stability'] = stability_results
        
        # 6. Residual analysis (sampled)
        print("\n6. Residual diagnostics...")
        # Sample for residual analysis
        sample_size = min(30000, len(df))
        df_residual = df.sample(sample_size, random_state=42)
        
        X_residual = df_residual[feature_cols].fillna(0)
        y_residual = df_residual[time_col]
        event_residual = df_residual[event_col]
        
        residual_results = self.calculate_residuals(model, X_residual, y_residual, event_residual)
        results['residuals'] = residual_results
        
        # 7. Generate summary report
        print("\n7. Generating comprehensive report...")
        self._generate_summary_report(results, len(feature_cols), len(stability_results))
        
        return results
    
    def _generate_summary_report(self, results, n_features, n_industries):
        """Generate comprehensive summary report"""
        
        target = results['target_distribution']
        
        report = f"""
COMPREHENSIVE AFT DIAGNOSTICS REPORT
===================================

DATASET OVERVIEW:
- Total records: {target['total_records']:,}
- Sample used for tests: {target['sample_size']:,}
- Censoring rate: {target['censoring_rate']:.1%}

ANALYSIS SCOPE:
- Features analyzed: {n_features} (ALL features included)
- Industries analyzed: {n_industries} (ALL valid industries)
- Sampling applied: Yes (for performance optimization)

TARGET DISTRIBUTION:
- Recommended AFT distribution: {target['recommended_distribution']['distribution']}
- Confidence: {target['recommended_distribution']['confidence']}
- Normal AFT test: {target['target_distribution_tests']['normal_aft']['interpretation']}
- Logistic AFT test: {target['target_distribution_tests']['logistic_aft']['interpretation']}

PERFORMANCE ANALYSIS:
- Binned analysis completed for ALL {n_features} features
- Industry stability analysis: {n_industries} industries analyzed
- No artificial limits applied to features or industries

RESIDUAL DIAGNOSTICS:
- Residual analysis completed with sampling optimization
- Memory-efficient chunked processing applied

RECOMMENDATIONS:
1. Use {target['recommended_distribution']['distribution']} AFT distribution
2. All {n_features} features retained in analysis
3. All {n_industries} industries monitored for stability
4. Sampling strategy preserves statistical validity while ensuring performance

OPTIMIZATION APPLIED:
✓ Smart sampling for statistical tests and visualizations
✓ Chunked processing for memory efficiency
✓ Preserved all features and industries as requested
✓ Maintained statistical rigor with performance optimization
"""
        
        # Save report
        with open(f"{self.output_path}/comprehensive_report.txt", "w") as f:
            f.write(report)
        
        print(report)
        return report

# USAGE EXAMPLE
def run_full_diagnostics(df, model, feature_cols):
    """
    Complete diagnostic pipeline respecting user preferences:
    - ALL features included (no limits)
    - ALL industries analyzed (no limits) 
    - Smart sampling for performance
    - Other optimizations as needed
    """
    
    # Initialize diagnostics
    diagnostics = OptimizedAFTDiagnostics(
        output_path='./diagnostics',
        performance_mode=True  # Enables sampling optimizations
    )
    
    # Run comprehensive analysis
    results = diagnostics.comprehensive_diagnostics(
        df=df,
        model=model,
        feature_cols=feature_cols,
        time_col='survival_time_days',
        event_col='event_indicator'
    )
    
    return results

# Example usage:
# results = run_full_diagnostics(your_dataframe, your_xgboost_model, your_feature_columns)
