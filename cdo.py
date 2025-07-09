"""
Combines Kaplan-Meier insights with XGBoost AFT predictive modeling
Focus: Baseline + Industry + Demographics + Temporal + Predictive Validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Professional plotting configuration
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class CDOSurvivalAnalysis:
    """Complete survival analysis for CDO presentation"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = self._prepare_data(data)
        self.kmf = KaplanMeierFitter()
        self.insights = {}
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare analysis dataset with flexible business segments"""
        df = df.copy()
        
        # NAICS 2-digit grouping
        df['naics_2digit'] = df['naics_cd'].astype(str).str[:2]
        
        # Flexible age grouping (handles age > 100)
        df['age_group'] = pd.cut(df['age'], 
                                bins=[0, 25, 35, 45, 55, 65, np.inf], 
                                labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'],
                                include_lowest=True)
        
        # Flexible tenure grouping (handles very long tenure)
        tenure_years = df['tenure_at_vantage_days'] / 365.25
        df['tenure_group'] = pd.cut(tenure_years,
                                   bins=[0, 0.5, 1, 2, 3, 5, np.inf],
                                   labels=['<6mo', '6mo-1yr', '1-2yr', '2-3yr', '3-5yr', '5yr+'],
                                   include_lowest=True)
        
        # Flexible salary grouping (handles very high salaries)
        df['salary_group'] = pd.cut(df['baseline_salary'], 
                                   bins=[0, 40000, 60000, 80000, 120000, 200000, np.inf],
                                   labels=['<40K', '40-60K', '60-80K', '80-120K', '120-200K', '200K+'],
                                   include_lowest=True)
        
        return df
    
    def analyze_baseline(self) -> Dict:
        """Generate population baseline with key business metrics"""
        print("Analyzing baseline retention patterns...")
        
        self.kmf.fit(self.data['survival_time_days'], self.data['event_indicator'])
        
        # Key business metrics
        metrics = {
            'population_size': len(self.data),
            'event_rate': self.data['event_indicator'].mean(),
            'median_survival': self.kmf.median_survival_time_,
            'retention_30d': self.kmf.survival_function_at_times(30).iloc[0],
            'retention_90d': self.kmf.survival_function_at_times(90).iloc[0],
            'retention_180d': self.kmf.survival_function_at_times(180).iloc[0],
            'retention_365d': self.kmf.survival_function_at_times(365).iloc[0]
        }
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        self.kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='navy')
        
        # Add key milestones
        milestones = [30, 90, 180, 365]
        colors = ['red', 'orange', 'green', 'purple']
        for day, color in zip(milestones, colors):
            retention = self.kmf.survival_function_at_times(day).iloc[0]
            ax.axvline(x=day, color=color, linestyle='--', alpha=0.7)
            ax.text(day, retention + 0.02, f'{day}d\n{retention:.1%}', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title('Employee Retention Baseline - ADP Population', fontsize=16, fontweight='bold')
        ax.set_xlabel('Days Since Assignment Start', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig('baseline_survival.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"BASELINE INSIGHTS:")
        print(f"  Population: {metrics['population_size']:,} employees")
        print(f"  Event rate: {metrics['event_rate']:.1%}")
        print(f"  Median survival: {metrics['median_survival']:.0f} days ({metrics['median_survival']/30.44:.1f} months)")
        print(f"  90-day retention: {metrics['retention_90d']:.1%}")
        print(f"  1-year retention: {metrics['retention_365d']:.1%}")
        
        self.insights['baseline'] = metrics
        return metrics
    
    def analyze_industries(self, top_n: int = 10) -> Dict:
        """Analyze retention by industry (NAICS 2-digit)"""
        print("Analyzing industry retention patterns...")
        
        # Get top industries by volume
        industry_counts = self.data['naics_2digit'].value_counts()
        top_industries = industry_counts.head(top_n).index.tolist()
        
        # Filter for meaningful sample sizes
        valid_industries = []
        for industry in top_industries:
            if industry_counts[industry] >= 5000:  # Minimum sample size
                valid_industries.append(industry)
        
        industry_data = self.data[self.data['naics_2digit'].isin(valid_industries)]
        industry_metrics = {}
        
        fig, ax = plt.subplots(figsize=(14, 10))
        colors = sns.color_palette("husl", len(valid_industries))
        
        for i, industry in enumerate(valid_industries):
            subset = industry_data[industry_data['naics_2digit'] == industry]
            
            kmf = KaplanMeierFitter()
            kmf.fit(subset['survival_time_days'], subset['event_indicator'])
            
            industry_metrics[industry] = {
                'sample_size': len(subset),
                'event_rate': subset['event_indicator'].mean(),
                'median_survival': kmf.median_survival_time_,
                'retention_90d': kmf.survival_function_at_times(90).iloc[0],
                'retention_365d': kmf.survival_function_at_times(365).iloc[0]
            }
            
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i], 
                                     label=f'NAICS {industry} (n={len(subset):,})')
        
        ax.set_title(f'Retention by Industry - Top {len(valid_industries)} Industries', fontsize=16, fontweight='bold')
        ax.set_xlabel('Days Since Assignment Start', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('industry_survival.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Industry ranking
        ranked = sorted(industry_metrics.items(), 
                       key=lambda x: x[1]['retention_365d'], reverse=True)
        
        print(f"INDUSTRY PERFORMANCE RANKING:")
        print("Top Performers:")
        for i, (industry, metrics) in enumerate(ranked[:3]):
            print(f"  {i+1}. NAICS {industry}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})")
        
        print("Attention Needed:")
        for i, (industry, metrics) in enumerate(ranked[-3:]):
            print(f"  NAICS {industry}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})")
        
        performance_gap = ranked[0][1]['retention_365d'] - ranked[-1][1]['retention_365d']
        print(f"\nIndustry opportunity: {performance_gap:.1%} retention gap")
        
        self.insights['industry'] = {'metrics': industry_metrics, 'ranked': ranked}
        return industry_metrics
    
    def analyze_demographics(self) -> Dict:
        """Analyze retention by key demographic segments"""
        print("Analyzing demographic retention patterns...")
        
        demographics = {
            'age_group': 'Age Segments',
            'tenure_group': 'Tenure Segments', 
            'salary_group': 'Salary Segments'
        }
        
        demo_insights = {}
        
        for demo_col, title in demographics.items():
            demo_data = self.data[self.data[demo_col].notna()]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            category_metrics = {}
            
            # Get categories sorted by logical order
            categories = demo_data[demo_col].cat.categories if hasattr(demo_data[demo_col], 'cat') else demo_data[demo_col].unique()
            
            for category in categories:
                subset = demo_data[demo_data[demo_col] == category]
                
                if len(subset) < 1000:  # Skip small segments
                    continue
                
                kmf = KaplanMeierFitter()
                kmf.fit(subset['survival_time_days'], subset['event_indicator'])
                
                category_metrics[str(category)] = {
                    'sample_size': len(subset),
                    'event_rate': subset['event_indicator'].mean(),
                    'median_survival': kmf.median_survival_time_,
                    'retention_365d': kmf.survival_function_at_times(365).iloc[0]
                }
                
                kmf.plot_survival_function(ax=ax, ci_show=False, 
                                         label=f'{category} (n={len(subset):,})')
            
            ax.set_title(f'Retention by {title}', fontsize=16, fontweight='bold')
            ax.set_xlabel('Days Since Assignment Start', fontsize=12)
            ax.set_ylabel('Survival Probability', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{demo_col}_survival.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Print top insights
            if category_metrics:
                ranked_categories = sorted(category_metrics.items(), 
                                         key=lambda x: x[1]['retention_365d'], reverse=True)
                
                print(f"\n{title.upper()} INSIGHTS:")
                print("Best performing:")
                for category, metrics in ranked_categories[:2]:
                    print(f"  {category}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})")
                
                print("Needs attention:")
                for category, metrics in ranked_categories[-2:]:
                    print(f"  {category}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})")
            
            demo_insights[demo_col] = category_metrics
        
        self.insights['demographics'] = demo_insights
        return demo_insights
    
    def analyze_temporal_trends(self) -> Dict:
        """Compare retention trends across time periods"""
        print("Analyzing temporal retention trends...")
        
        if 'dataset_split' not in self.data.columns:
            print("Dataset split not available")
            return {}
        
        # Compare 2023 vs 2024 cohorts
        cohort_2023 = self.data[self.data['dataset_split'].isin(['train', 'val'])]
        cohort_2024 = self.data[self.data['dataset_split'] == 'oot']
        
        if len(cohort_2024) < 5000:
            print("Insufficient 2024 data for temporal analysis")
            return {}
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Fit survival curves
        kmf_2023 = KaplanMeierFitter()
        kmf_2023.fit(cohort_2023['survival_time_days'], cohort_2023['event_indicator'])
        
        kmf_2024 = KaplanMeierFitter()
        kmf_2024.fit(cohort_2024['survival_time_days'], cohort_2024['event_indicator'])
        
        # Plot comparison
        kmf_2023.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='blue',
                                       label=f'2023 Cohort (n={len(cohort_2023):,})')
        kmf_2024.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color='red',
                                       label=f'2024 Cohort (n={len(cohort_2024):,})')
        
        ax.set_title('Retention Trends: 2023 vs 2024 Cohorts', fontsize=16, fontweight='bold')
        ax.set_xlabel('Days Since Assignment Start', fontsize=12)
        ax.set_ylabel('Survival Probability', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('temporal_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate metrics
        trend_metrics = {
            '2023_retention_365d': kmf_2023.survival_function_at_times(365).iloc[0],
            '2024_retention_365d': kmf_2024.survival_function_at_times(365).iloc[0],
            '2023_median_survival': kmf_2023.median_survival_time_,
            '2024_median_survival': kmf_2024.median_survival_time_
        }
        
        retention_change = trend_metrics['2024_retention_365d'] - trend_metrics['2023_retention_365d']
        trend_direction = "improving" if retention_change > 0 else "declining"
        
        print(f"TEMPORAL TRENDS:")
        print(f"  2023 retention: {trend_metrics['2023_retention_365d']:.1%}")
        print(f"  2024 retention: {trend_metrics['2024_retention_365d']:.1%}")
        print(f"  Change: {retention_change:+.1%} ({trend_direction})")
        
        self.insights['temporal'] = trend_metrics
        return trend_metrics
    
    def build_predictive_model(self) -> Dict:
        """Build XGBoost AFT model for predictive validation"""
        print("Building XGBoost AFT predictive model...")
        
        # Prepare features
        feature_columns = ['age', 'tenure_at_vantage_days', 'baseline_salary']
        
        # Add encoded categorical features
        if 'naics_2digit' in self.data.columns:
            le_naics = LabelEncoder()
            self.data['naics_encoded'] = le_naics.fit_transform(self.data['naics_2digit'].astype(str))
            feature_columns.append('naics_encoded')
        
        # Prepare model dataset
        model_data = self.data[feature_columns + ['survival_time_days', 'event_indicator']].dropna()
        
        # Train-test split
        np.random.seed(42)
        train_idx = np.random.choice(len(model_data), size=int(0.8 * len(model_data)), replace=False)
        test_idx = np.setdiff1d(np.arange(len(model_data)), train_idx)
        
        train_data = model_data.iloc[train_idx]
        test_data = model_data.iloc[test_idx]
        
        # Prepare XGBoost data
        X_train = train_data[feature_columns]
        y_train = train_data['survival_time_days']
        event_train = train_data['event_indicator']
        
        X_test = test_data[feature_columns]
        y_test = test_data['survival_time_days']
        event_test = test_data['event_indicator']
        
        # Train XGBoost AFT model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Set survival information
        dtrain.set_float_info('label_lower_bound', y_train.values)
        dtrain.set_uint_info('label_right_censored', (1 - event_train).values.astype(np.uint32))
        
        # Model parameters
        params = {
            'objective': 'survival:aft',
            'aft_loss_distribution': 'weibull',
            'max_depth': 4,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # Train model
        model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        
        # Generate predictions
        train_pred = model.predict(dtrain)
        test_pred = model.predict(dtest)
        
        # Calculate performance
        c_index_train = concordance_index(y_train, train_pred, event_train)
        c_index_test = concordance_index(y_test, test_pred, event_test)
        
        # Feature importance
        feature_importance = model.get_score(importance_type='gain')
        importance_df = pd.DataFrame([
            {'feature': feature_columns[int(f[1:])], 'importance': score}
            for f, score in feature_importance.items()
        ]).sort_values('importance', ascending=False)
        
        # Risk segmentation
        risk_percentiles = np.percentile(test_pred, [20, 50, 80])
        
        # Visualize feature importance
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        ax.set_title('Feature Importance - XGBoost AFT Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        model_results = {
            'train_size': len(train_data),
            'test_size': len(test_data),
            'c_index_train': c_index_train,
            'c_index_test': c_index_test,
            'feature_importance': importance_df,
            'risk_percentiles': risk_percentiles
        }
        
        print(f"MODEL PERFORMANCE:")
        print(f"  Training C-index: {c_index_train:.3f}")
        print(f"  Test C-index: {c_index_test:.3f}")
        print(f"  Performance: {'Strong' if c_index_test > 0.65 else 'Moderate' if c_index_test > 0.55 else 'Baseline'}")
        
        print(f"\nTOP FEATURES:")
        for _, row in importance_df.head(3).iterrows():
            print(f"  {row['feature']}: {row['importance']:.1f}")
        
        print(f"\nRISK SEGMENTATION:")
        print(f"  High risk (20%): < {risk_percentiles[0]:.0f} days predicted survival")
        print(f"  Medium risk (60%): {risk_percentiles[0]:.0f} - {risk_percentiles[2]:.0f} days")
        print(f"  Low risk (20%): > {risk_percentiles[2]:.0f} days predicted survival")
        
        self.insights['model'] = model_results
        return model_results
    
    def generate_executive_summary(self) -> Dict:
        """Generate comprehensive executive summary"""
        print("\nEXECUTIVE SUMMARY")
        print("=" * 50)
        
        baseline = self.insights['baseline']
        
        summary = {
            'population_size': f"{baseline['population_size']:,}",
            'median_survival_days': f"{baseline['median_survival']:.0f}",
            'median_survival_months': f"{baseline['median_survival']/30.44:.1f}",
            'retention_90d': f"{baseline['retention_90d']:.1%}",
            'retention_365d': f"{baseline['retention_365d']:.1%}",
            'event_rate': f"{baseline['event_rate']:.1%}"
        }
        
        # Add key business opportunities
        opportunities = []
        
        if 'industry' in self.insights:
            ranked = self.insights['industry']['ranked']
            gap = ranked[0][1]['retention_365d'] - ranked[-1][1]['retention_365d']
            opportunities.append(f"Industry performance gap: {gap:.1%} improvement opportunity")
        
        if 'temporal' in self.insights:
            temporal = self.insights['temporal']
            change = temporal['2024_retention_365d'] - temporal['2023_retention_365d']
            direction = "improving" if change > 0 else "declining"
            opportunities.append(f"Retention {direction} {abs(change):.1%} year-over-year")
        
        if 'model' in self.insights:
            model = self.insights['model']
            opportunities.append(f"Predictive model achieves {model['c_index_test']:.3f} C-index")
        
        summary['key_opportunities'] = opportunities
        
        print(f"Population analyzed: {summary['population_size']} employees")
        print(f"Median assignment duration: {summary['median_survival_days']} days ({summary['median_survival_months']} months)")
        print(f"90-day retention: {summary['retention_90d']}")
        print(f"1-year retention: {summary['retention_365d']}")
        print(f"Event rate: {summary['event_rate']}")
        
        print(f"\nKey business opportunities:")
        for opp in opportunities:
            print(f"  - {opp}")
        
        return summary
    
    def run_complete_analysis(self) -> Dict:
        """Execute complete survival analysis sequence"""
        print("STARTING COMPLETE CDO SURVIVAL ANALYSIS")
        print("=" * 60)
        
        # Execute full analysis
        self.analyze_baseline()
        self.analyze_industries()
        self.analyze_demographics()
        self.analyze_temporal_trends()
        self.build_predictive_model()
        
        # Generate executive summary
        summary = self.generate_executive_summary()
        
        return {'insights': self.insights, 'summary': summary}


def run_cdo_survival_analysis(employee_data: pd.DataFrame) -> Tuple[CDOSurvivalAnalysis, Dict]:
    """
    Execute complete CDO survival analysis combining Kaplan-Meier and XGBoost AFT
    
    Args:
        employee_data: Employee-level dataset with survival outcomes
    
    Returns:
        Tuple of (analyzer, executive_summary)
    """
    analyzer = CDOSurvivalAnalysis(employee_data)
    results = analyzer.run_complete_analysis()
    
    return analyzer, results['summary']


# Example usage for CDO meeting
if __name__ == "__main__":
    # Single command execution
    # analyzer, summary = run_cdo_survival_analysis(employee_level_data)
    
    # CDO presentation metrics
    # print(f"Population: {summary['population_size']}")
    # print(f"Median duration: {summary['median_survival_months']} months")
    # print(f"Retention rate: {summary['retention_365d']}")
    pass
