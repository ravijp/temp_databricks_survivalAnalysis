# Enhanced XGBoost AFT Survival Analysis
# Combines Kaplan-Meier insights with XGBoost AFT predictive modeling

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from scipy import stats, interpolate
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore")

# Professional plotting configuration
plt.style.use("default")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 11

class SurvivalAnalysis:
    """Complete KM XGBoost AFT survival analysis"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = self._prepare_data(data)
        self.kmf = KaplanMeierFitter()
        self.insights = {}
        self.enhanced_results = {}
        self.survival_curves = None
        self.calibration_metrics = {}
        self.time_points = None
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare analysis dataset with flexible business segments"""
        df = df.copy()
        
        # NAICS 2-digit grouping
        df["naics_2digit"] = df["naics_cd"].astype(str).str[:2]
        
        # Age grouping
        vantage_dt, birth_dt = pd.to_datetime(df["vantage_date"], errors='coerce'), pd.to_datetime(df["birth_dt"], errors='coerce')
        birth_dt = birth_dt.where(birth_dt.dt.year.between(1900, 2020), pd.NaT)
        df['age'] = (vantage_dt - birth_dt).dt.days / 365.25
        
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
            include_lowest=True,
        )
        
        # Tenure grouping
        df["tenure_at_vantage_days"] = np.where(df["tenure_at_vantage_days"] > 365 * 100, 365 * 100, df["tenure_at_vantage_days"])
        tenure_years = df["tenure_at_vantage_days"] / 365.25
        
        df["tenure_group"] = pd.cut(
            tenure_years,
            bins=[0, 0.5, 1, 2, 3, 5, np.inf],
            labels=["<6mo", "6mo-1yr", "1-2yr", "2-3yr", "3-5yr", "5yr+"],
            include_lowest=True,
        )
        
        # Salary grouping
        df["salary_group"] = pd.cut(
            df["baseline_salary"],
            bins=[0, 40000, 60000, 80000, 120000, 200000, np.inf],
            labels=["<40K", "40-60K", "60-80K", "80-120K", "120-200K", "200K+"],
            include_lowest=True,
        )
        
        return df
        
    def analyze_baseline(self) -> Dict:
        """Generate population baseline with key business metrics using train val split"""
        print("Analyzing baseline retention patterns...")
        
        # Use train val split for primary insights
        train_data = self.data[self.data["dataset_split"].isin(["train", "val"])]
        
        self.kmf.fit(train_data["survival_time_days"], train_data["event_indicator"])
        
        # Key business metrics
        metrics = {
            "population_size": len(train_data),
            "total_population": len(self.data),
            "event_rate": train_data["event_indicator"].mean(),
            "median_survival": self.kmf.median_survival_time_,
            "retention_30d": self.kmf.survival_function_at_times(30).iloc[0],
            "retention_90d": self.kmf.survival_function_at_times(90).iloc[0],
            "retention_180d": self.kmf.survival_function_at_times(180).iloc[0],
            "retention_365d": self.kmf.survival_function_at_times(365).iloc[0],
        }
        
        # Visualization
        fig, ax = plt.subplots(figsize=(15, 10))
        self.kmf.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color="navy")
        
        # Add key milestones
        milestones = [30, 90, 180, 365]
        colors = ["red", "orange", "green", "purple"]
        for day, color in zip(milestones, colors):
            retention = self.kmf.survival_function_at_times(day).iloc[0]
            ax.axvline(x=day, color=color, linestyle="--", alpha=0.7)
            ax.text(
                day,
                retention + 0.02,
                f"{day}d\n{retention:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
            
        ax.set_title(
            "Employee Retention Baseline - ADP Population",
            fontsize=17,
            fontweight="bold",
        )
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig("baseline_survival.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        print(f"BASELINE INSIGHTS:")
        print(f"   Train + val population: {metrics['population_size']:,} employees")
        print(f"   Total population: {metrics['total_population']:,} employees")
        print(f"   Event rate: {metrics['event_rate']:.1%}")
        print(
            f"   Median survival: {metrics['median_survival']:.0f} days ({metrics['median_survival']/30.44:.1f} months)"
        )
        print(f"   90-day retention: {metrics['retention_90d']:.1%}")
        print(f"   1-year retention: {metrics['retention_365d']:.1%}")
        
        self.insights["baseline"] = metrics
        return metrics
        
    def analyze_industries(self, top_n: int = 10) -> Dict:
        """Analyze retention by industry (NAICS 2-digit) using train val split"""
        print("Analyzing industry retention patterns...")
        
        train_data = self.data[self.data["dataset_split"].isin(["train", "val"])]
        
        # Get top industries by volume
        industry_counts = train_data["naics_2digit"].value_counts()
        top_industries = industry_counts.head(top_n).index.tolist()
        
        # Filter for meaningful sample sizes
        valid_industries = []
        for industry in top_industries:
            if industry_counts[industry] >= 1000:
                valid_industries.append(industry)
                
        industry_data = train_data[train_data["naics_2digit"].isin(valid_industries)]
        industry_metrics = {}
        
        fig, ax = plt.subplots(figsize=(14, 10))
        colors = sns.color_palette("husl", len(valid_industries))
        
        for i, industry in enumerate(valid_industries):
            subset = industry_data[industry_data["naics_2digit"] == industry]
            
            kmf = KaplanMeierFitter()
            kmf.fit(subset["survival_time_days"], subset["event_indicator"])
            
            industry_metrics[industry] = {
                "sample_size": len(subset),
                "event_rate": subset["event_indicator"].mean(),
                "median_survival": kmf.median_survival_time_,
                "retention_90d": kmf.survival_function_at_times(90).iloc[0],
                "retention_365d": kmf.survival_function_at_times(365).iloc[0],
            }
            
            kmf.plot_survival_function(
                ax=ax,
                ci_show=False,
                color=colors[i],
                label=f"NAICS {industry} (n={len(subset):,})",
            )
            
        ax.set_title(
            f"Retention by Industry - Top {len(valid_industries)} Industries",
            fontsize=17,
            fontweight="bold",
        )
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("industry_survival.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        # Industry ranking
        ranked = sorted(
            industry_metrics.items(), key=lambda x: x[1]["retention_365d"], reverse=True
        )
        
        print(f"INDUSTRY PERFORMANCE RANKING:")
        print("Top Performers:")
        for i, (industry, metrics) in enumerate(ranked[:3]):
            print(
                f"   {i+1}. NAICS {industry}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})"
            )
            
        print("Attention Needed:")
        for i, (industry, metrics) in enumerate(ranked[-3:]):
            print(
                f"   NAICS {industry}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})"
            )
            
        performance_gap = (
            ranked[0][1]["retention_365d"] - ranked[-1][1]["retention_365d"]
        )
        print(f"\nIndustry opportunity: {performance_gap:.1%} retention gap")
        
        self.insights["industry"] = {"metrics": industry_metrics, "ranked": ranked}
        return industry_metrics
        
    def analyze_demographics(self) -> Dict:
        """Analyze retention by key demographic segments using train val split"""
        print("Analyzing demographic retention patterns...")
        
        train_data = self.data[self.data["dataset_split"].isin(["train", "val"])]
        
        demographics = {
            "age_group": "Age Segments",
            "tenure_group": "Tenure Segments",
            "salary_group": "Salary Segments",
        }
        
        demo_insights = {}
        
        for demo_col, title in demographics.items():
            demo_data = train_data[train_data[demo_col].notna()]
            
            fig, ax = plt.subplots(figsize=(15, 10))
            category_metrics = {}
            
            categories = (
                demo_data[demo_col].cat.categories
                if hasattr(demo_data[demo_col], "cat")
                else demo_data[demo_col].unique()
            )
            
            for category in categories:
                subset = demo_data[demo_data[demo_col] == category]
                
                if len(subset) < 500:
                    continue
                    
                kmf = KaplanMeierFitter()
                kmf.fit(subset["survival_time_days"], subset["event_indicator"])
                
                category_metrics[str(category)] = {
                    "sample_size": len(subset),
                    "event_rate": subset["event_indicator"].mean(),
                    "median_survival": kmf.median_survival_time_,
                    "retention_365d": kmf.survival_function_at_times(365).iloc[0],
                }
                
                kmf.plot_survival_function(
                    ax=ax, ci_show=False, label=f"{category} (n={len(subset):,})"
                )
                
            ax.set_title(f"Retention by {title}", fontsize=17, fontweight="bold")
            ax.set_xlabel("Days Since Assignment Start", fontsize=14)
            ax.set_ylabel("Survival Probability", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{demo_col}_survival.png", dpi=300, bbox_inches="tight")
            plt.show()
            
            if category_metrics:
                ranked_categories = sorted(
                    category_metrics.items(),
                    key=lambda x: x[1]["retention_365d"],
                    reverse=True,
                )
                
                print(f"\n{title.upper()} INSIGHTS:")
                print("Best performing:")
                for category, metrics in ranked_categories[:2]:
                    print(
                        f"   {category}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})"
                    )
                    
                print("Needs attention:")
                for category, metrics in ranked_categories[-2:]:
                    print(
                        f"   {category}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})"
                    )
                    
            demo_insights[demo_col] = category_metrics
            
        self.insights["demographics"] = demo_insights
        return demo_insights
        
    def analyze_temporal_trends(self) -> Dict:
        """Compare retention trends: 2023 (train+val) vs 2024 (oot)"""
        print("Analyzing temporal retention trends...")
        
        cohort_2023 = self.data[self.data["dataset_split"].isin(["train", "val"])]
        cohort_2024 = self.data[self.data["dataset_split"] == "oot"]
        
        if len(cohort_2024) < 1000:
            print("Insufficient 2024 data for temporal analysis")
            return {}
            
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Fit survival curves
        kmf_2023 = KaplanMeierFitter()
        kmf_2023.fit(cohort_2023["survival_time_days"], cohort_2023["event_indicator"])
        
        kmf_2024 = KaplanMeierFitter()
        kmf_2024.fit(cohort_2024["survival_time_days"], cohort_2024["event_indicator"])
        
        # Plot comparison
        kmf_2023.plot_survival_function(
            ax=ax,
            ci_show=False,
            linewidth=3,
            color="blue",
            label=f"2023 Cohort (n={len(cohort_2023):,})",
        )
        kmf_2024.plot_survival_function(
            ax=ax,
            ci_show=False,
            linewidth=3,
            color="red",
            label=f"2024 Cohort (n={len(cohort_2024):,})",
        )
        
        ax.set_title(
            "Temporal Retention Trends: 2023 vs 2024", fontsize=17, fontweight="bold"
        )
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("temporal_trends.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        # Calculate metrics
        trend_metrics = {
            "2023_size": len(cohort_2023),
            "2024_size": len(cohort_2024),
            "2023_retention_90d": kmf_2023.survival_function_at_times(90).iloc[0],
            "2024_retention_90d": kmf_2024.survival_function_at_times(90).iloc[0],
            "2023_retention_365d": kmf_2023.survival_function_at_times(365).iloc[0],
            "2024_retention_365d": kmf_2024.survival_function_at_times(365).iloc[0],
            "2023_median_survival": kmf_2023.median_survival_time_,
            "2024_median_survival": kmf_2024.median_survival_time_,
        }
        
        # Calculate changes
        retention_change_90d = (
            trend_metrics["2024_retention_90d"] - trend_metrics["2023_retention_90d"]
        )
        retention_change_365d = (
            trend_metrics["2024_retention_365d"] - trend_metrics["2023_retention_365d"]
        )
        
        print(f"TEMPORAL TRENDS:")
        print(f"   2023 cohort: {trend_metrics['2023_size']:,} employees")
        print(f"   2024 cohort: {trend_metrics['2024_size']:,} employees")
        print(f"   90-day retention change: {retention_change_90d:+.1%}")
        print(f"   1-year retention change: {retention_change_365d:+.1%}")
        
        trend_metrics["retention_change_90d"] = retention_change_90d
        trend_metrics["retention_change_365d"] = retention_change_365d
        
        self.insights["temporal"] = trend_metrics
        return trend_metrics
        
    def build_predictive_model(self) -> Dict:
        """Build XGBoost AFT model with proper categorical handling"""
        print("Building XGBoost AFT predictive model...")
        
        train_data = self.data[self.data['dataset_split'] == 'train']
        val_data = self.data[self.data['dataset_split'] == 'val']
        
        if len(val_data) < 1000:
            print("Insufficient validation data - using random split from train")
            np.random.seed(42)
            train_idx = np.random.choice(len(train_data), size=int(0.8 * len(train_data)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(train_data)), train_idx)
            val_data = train_data.iloc[val_idx]
            train_data = train_data.iloc[train_idx]
            
        # Define features
        numeric_features = [
            'age', 'tenure_at_vantage_days', 'team_size', 'baseline_salary',
            'team_avg_comp', 'salary_growth_ratio', 'manager_changes_count'
        ]
        
        categorical_features = [
            'gender_cd', 'pay_rt_type_cd', 'full_tm_part_tm_cd', 'fscl_actv_ind'
        ]
        
        # Filter features that exist in data
        available_numeric = [f for f in numeric_features if f in train_data.columns]
        available_categorical = [f for f in categorical_features if f in train_data.columns]
        
        # Copy data
        train_data = train_data.copy()
        val_data = val_data.copy()
        
        # Handle categorical features
        label_encoders = {}
        for cat_feature in available_categorical:
            train_cats = train_data[cat_feature].fillna('MISSING').astype(str)
            val_cats = val_data[cat_feature].fillna('MISSING').astype(str)
            
            all_categories = set(train_cats.unique()) | set(val_cats.unique())
            category_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
            
            train_data[f'{cat_feature}_encoded'] = train_cats.map(category_mapping)
            val_data[f'{cat_feature}_encoded'] = val_cats.map(category_mapping)
            
            label_encoders[cat_feature] = category_mapping
            
        # NAICS encoding
        if 'naics_2digit' in train_data.columns:
            train_naics = train_data['naics_2digit'].fillna('MISSING').astype(str)
            val_naics = val_data['naics_2digit'].fillna('MISSING').astype(str)
            
            all_naics = set(train_naics.unique()) | set(val_naics.unique())
            naics_mapping = {naics: idx for idx, naics in enumerate(all_naics)}
            
            train_data['naics_encoded'] = train_naics.map(naics_mapping)
            val_data['naics_encoded'] = val_naics.map(naics_mapping)
            
            available_numeric.append('naics_encoded')
            label_encoders['naics_2digit'] = naics_mapping
            
        # Create feature list
        encoded_categorical = [f'{cat}_encoded' for cat in available_categorical]
        feature_columns = available_numeric + encoded_categorical
        
        # Handle missing values
        for col in feature_columns:
            if col in train_data.columns and train_data[col].isna().any():
                if train_data[col].dtype in [np.float64, np.int64]:
                    fill_value = train_data[col].median()
                    train_data[col] = train_data[col].fillna(fill_value)
                    val_data[col] = val_data[col].fillna(fill_value)
                else:
                    fill_value = train_data[col].mode().iloc[0]
                    train_data[col] = train_data[col].fillna(fill_value)
                    val_data[col] = val_data[col].fillna(fill_value)
                    
        # Create datasets
        model_columns = feature_columns + ['survival_time_days', 'event_indicator']
        train_model_data = train_data[model_columns].dropna()
        val_model_data = val_data[model_columns].dropna()
        
        # Extract features and targets
        X_train = train_model_data[feature_columns]
        y_train = train_model_data['survival_time_days']
        event_train = train_model_data['event_indicator']
        
        X_val = val_model_data[feature_columns]
        y_val = val_model_data['survival_time_days']
        event_val = val_model_data['event_indicator']
        
        # Train XGBoost AFT model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        dtrain.set_float_info('label_lower_bound', y_train.values)
        dtrain.set_float_info('label_upper_bound', y_train.values)
        
        dval.set_float_info('label_lower_bound', y_val.values)
        dval.set_float_info('label_upper_bound', y_val.values)
        
        # Model parameters
        params = {
            'objective': 'survival:aft',
            'aft_loss_distribution': 'normal',
            'max_depth': 4,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'seed': 42
        }
        
        # Train model
        model = xgb.train(params, dtrain, num_boost_round=100, verbose_eval=False)
        
        # Store model
        self.model = model
        self.model_data = {
            'X_train': X_train, 'y_train': y_train, 'event_train': event_train,
            'X_val': X_val, 'y_val': y_val, 'event_val': event_val
        }
        
        # Generate predictions
        train_pred = model.predict(dtrain)
        val_pred = model.predict(dval)
        
        # Calculate performance
        c_index_train = concordance_index(y_train, train_pred, event_train)
        c_index_val = concordance_index(y_val, val_pred, event_val)
        
        # Feature importance
        feature_importance = model.get_score(importance_type='gain')
        if feature_importance and list(feature_importance.keys())[0].startswith("f"):
            importance_df = pd.DataFrame([
                {'feature': feature_columns[int(f[1:])], 'importance': score}
                for f, score in feature_importance.items()
            ]).sort_values('importance', ascending=False)
        else:
            importance_df = pd.DataFrame([
                {'feature': f, 'importance': score}
                for f, score in feature_importance.items()
            ]).sort_values('importance', ascending=False)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
        ax.set_title('Feature Importance - XGBoost AFT Model', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        model_results = {
            'train_size': len(train_model_data),
            'val_size': len(val_model_data),
            'c_index_train': c_index_train,
            'c_index_val': c_index_val,
            'feature_importance': importance_df,
            'features_used': feature_columns,
            'label_encoders': label_encoders
        }
        
        print(f"MODEL PERFORMANCE:")
        print(f"   Training size: {len(train_model_data):,}")
        print(f"   Validation size: {len(val_model_data):,}")
        print(f"   Training C-index: {c_index_train:.3f}")
        print(f"   Validation C-index: {c_index_val:.3f}")
        print(f"   Features used: {len(feature_columns)}")
        
        print(f"\nTOP FEATURES:")
        for _, row in importance_df.head(3).iterrows():
            print(f"   {row['feature']}: {row['importance']:.1f}")
        
        # Add prediction diagnostics
        print(f"\nPREDICTION DIAGNOSTICS:")
        print(f"   Raw predictions - Mean: {val_pred.mean():.3f}, Std: {val_pred.std():.3f}")
        print(f"   Raw predictions - Min: {val_pred.min():.3f}, Max: {val_pred.max():.3f}")
        print(f"   Actual survival times - Mean: {y_val.mean():.1f}, Std: {y_val.std():.1f}")
        print(f"   Event rate: {event_val.mean():.3f}")
        
        # Check correlation between predictions and actual survival
        from scipy.stats import pearsonr, spearmanr
        corr_pearson, p_pearson = pearsonr(val_pred, y_val)
        corr_spearman, p_spearman = spearmanr(val_pred, y_val)
        
        print(f"   Pearson correlation with survival time: {corr_pearson:.3f} (p={p_pearson:.3f})")
        print(f"   Spearman correlation with survival time: {corr_spearman:.3f} (p={p_spearman:.3f})")
        
        # Check if model is just predicting the mean
        prediction_range = val_pred.max() - val_pred.min()
        prediction_cv = val_pred.std() / val_pred.mean() if val_pred.mean() != 0 else 0
        
        print(f"   Prediction range: {prediction_range:.3f}")
        print(f"   Prediction coefficient of variation: {prediction_cv:.3f}")
        
        if val_pred.std() < 0.1:
            print("   WARNING: Low prediction variance - model may not be learning well")
        
        if abs(corr_pearson) < 0.1:
            print("   WARNING: Low correlation with survival time - model may not be predictive")
        
        if prediction_cv < 0.1:
            print("   WARNING: Low prediction variability - model may be predicting near-constant values")
            
        self.insights['model'] = model_results
        return model_results
    
    def diagnose_model_learning(self):
        """Comprehensive model learning diagnostics"""
        if not hasattr(self, 'model') or not hasattr(self, 'model_data'):
            print("Model not trained yet")
            return
        
        print("\nMODEL LEARNING DIAGNOSTICS:")
        print("="*50)
        
        X_train = self.model_data['X_train']
        y_train = self.model_data['y_train']
        X_val = self.model_data['X_val']
        y_val = self.model_data['y_val']
        
        # Get predictions
        dtrain = xgb.DMatrix(X_train)
        dval = xgb.DMatrix(X_val)
        
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        # Check if model is just predicting the mean
        train_mean = y_train.mean()
        val_mean = y_val.mean()
        
        print(f"1. PREDICTION ANALYSIS:")
        print(f"   Training predictions - Mean: {train_pred.mean():.3f}, Std: {train_pred.std():.3f}")
        print(f"   Validation predictions - Mean: {val_pred.mean():.3f}, Std: {val_pred.std():.3f}")
        print(f"   Training target mean: {train_mean:.3f}")
        print(f"   Validation target mean: {val_mean:.3f}")
        
        # Check if predictions are close to target mean
        train_diff_from_mean = abs(train_pred.mean() - train_mean)
        val_diff_from_mean = abs(val_pred.mean() - val_mean)
        
        print(f"   Train pred vs target mean difference: {train_diff_from_mean:.3f}")
        print(f"   Val pred vs target mean difference: {val_diff_from_mean:.3f}")
        
        # Check prediction variance
        if train_pred.std() < 10:
            print("   WARNING: Very low training prediction variance")
        if val_pred.std() < 10:
            print("   WARNING: Very low validation prediction variance")
        
        # Check if all predictions are identical
        if len(np.unique(train_pred)) == 1:
            print("   ERROR: All training predictions are identical!")
        if len(np.unique(val_pred)) == 1:
            print("   ERROR: All validation predictions are identical!")
        
        print(f"\n2. FEATURE ANALYSIS:")
        # Check feature variance
        feature_vars = X_train.var()
        zero_var_features = feature_vars[feature_vars == 0].index.tolist()
        low_var_features = feature_vars[feature_vars < 0.01].index.tolist()
        
        if zero_var_features:
            print(f"   Zero variance features: {zero_var_features}")
        if low_var_features:
            print(f"   Low variance features: {low_var_features}")
        
        # Check for constant features
        constant_features = []
        for col in X_train.columns:
            if X_train[col].nunique() == 1:
                constant_features.append(col)
        
        if constant_features:
            print(f"   Constant features: {constant_features}")
        
        print(f"\n3. MODEL COMPLEXITY:")
        # Check if model is using features
        feature_importance = self.model.get_score(importance_type='gain')
        if not feature_importance:
            print("   WARNING: No feature importance found - model may not be using features")
        else:
            print(f"   Number of features with non-zero importance: {len(feature_importance)}")
            print(f"   Total features: {len(X_train.columns)}")
        
        return {
            'train_pred_std': train_pred.std(),
            'val_pred_std': val_pred.std(),
            'unique_train_preds': len(np.unique(train_pred)),
            'unique_val_preds': len(np.unique(val_pred)),
            'zero_var_features': zero_var_features,
            'constant_features': constant_features
        }
    
    def get_improvement_recommendations(self):
        """Provide recommendations based on model diagnostics"""
        if not hasattr(self, 'model'):
            return []
        
        recommendations = []
        
        # Check survival curve issues
        if hasattr(self, 'survival_curves') and self.survival_curves is not None:
            final_survival = self.survival_curves[:, -1]
            if final_survival.std() < 0.01:
                recommendations.append({
                    'issue': 'Identical Survival Curves',
                    'severity': 'High',
                    'recommendation': 'Model predictions lack variance. Consider: 1) Adding more features, 2) Reducing regularization, 3) Increasing model complexity, 4) Checking for data quality issues'
                })
        
        # Check Gini coefficient
        if 'lorenz_curve' in self.enhanced_results:
            gini = self.enhanced_results['lorenz_curve']['gini_coefficient']
            if gini < 0:
                recommendations.append({
                    'issue': 'Negative Gini Coefficient',
                    'severity': 'Critical',
                    'recommendation': 'Model is performing worse than random. Check: 1) Target variable definition, 2) Feature-target relationships, 3) Data leakage, 4) Model hyperparameters'
                })
            elif gini < 0.1:
                recommendations.append({
                    'issue': 'Low Gini Coefficient',
                    'severity': 'High',
                    'recommendation': 'Poor model discrimination. Consider: 1) Feature engineering, 2) Different model architecture, 3) More training data, 4) Ensemble methods'
                })
        
        # Check model learning
        X_val = self.model_data['X_val']
        dval = xgb.DMatrix(X_val)
        val_pred = self.model.predict(dval)
        
        if val_pred.std() < 10:
            recommendations.append({
                'issue': 'Low Prediction Variance',
                'severity': 'High',
                'recommendation': 'Model predictions have low variance. Try: 1) Reduce regularization (increase eta, reduce lambda), 2) Increase max_depth, 3) Add more diverse features'
            })
        
        return recommendations
    
    def generate_individual_survival_curves(self, X_test, time_points=None):
        """Generate individual survival curves with proper variance"""
        if time_points is None:
            time_points = np.arange(1, 366, 1)
        
        # Get predictions
        dtest = xgb.DMatrix(X_test)
        predictions = self.model.predict(dtest)
        
        print(f"Raw predictions - Mean: {predictions.mean():.3f}, Std: {predictions.std():.3f}")
        print(f"Raw predictions - Min: {predictions.min():.3f}, Max: {predictions.max():.3f}")
        
        # For XGBoost AFT, predictions are log-scale parameters
        # Convert to individual scale parameters for each employee
        survival_curves = []
        scales = []
        
        for i, pred in enumerate(predictions):
            # Use prediction directly as log-scale parameter
            scale = np.exp(pred)
            
            # Ensure reasonable bounds but preserve variance
            scale = np.clip(scale, 30, 2000)
            scales.append(scale)
            
            # Generate exponential survival curve: S(t) = exp(-t/scale)
            curve = np.exp(-time_points / scale)
            survival_curves.append(curve)
        
        self.survival_curves = np.array(survival_curves)
        self.time_points = time_points
        
        # Validate curves
        final_survival = self.survival_curves[:, -1]
        scales_array = np.array(scales)
        
        print(f"Scale parameters - Mean: {scales_array.mean():.1f}, Std: {scales_array.std():.1f}")
        print(f"Scale parameters - Min: {scales_array.min():.1f}, Max: {scales_array.max():.1f}")
        print(f"Final survival - Mean: {final_survival.mean():.3f}, Std: {final_survival.std():.3f}")
        
        if final_survival.std() < 0.01:
            print("WARNING: Very low survival curve variance - check model predictions")
        
        return self.survival_curves
    
    def plot_individual_survival_curves(self, n_curves=5):
        """Plot individual survival curves with detailed diagnostics"""
        if self.survival_curves is None:
            print("No survival curves generated yet")
            return
        
        plt.figure(figsize=(12, 8))
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        print(f"Plotting {min(n_curves, len(self.survival_curves))} survival curves:")
        
        for i in range(min(n_curves, len(self.survival_curves))):
            curve = self.survival_curves[i]
            
            # Print diagnostics for each curve
            curve_slope = curve[0] - curve[-1]  # Should be positive (declining)
            print(f"   Employee {i+1}: Initial={curve[0]:.3f}, Final={curve[-1]:.3f}, Slope={curve_slope:.3f}")
            
            plt.plot(self.time_points, curve, color=colors[i], linewidth=2,
                    label=f'Employee {i+1} (final: {curve[-1]:.3f})')
        
        # Add mean curve
        mean_curve = np.mean(self.survival_curves, axis=0)
        plt.plot(self.time_points, mean_curve, 'k--', linewidth=3, alpha=0.7, label='Mean')
        
        # Add some statistics
        final_survivals = self.survival_curves[:, -1]
        print(f"Final survival statistics:")
        print(f"   Mean: {final_survivals.mean():.3f}")
        print(f"   Std: {final_survivals.std():.3f}")
        print(f"   Min: {final_survivals.min():.3f}")
        print(f"   Max: {final_survivals.max():.3f}")
        
        if final_survivals.std() < 0.01:
            print("   WARNING: Very low variance in final survival probabilities")
        
        plt.xlabel('Time (days)')
        plt.ylabel('Survival Probability')
        plt.title('Individual Employee Survival Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 365)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
    
    def calculate_lorenz_curve(self, y_true, y_pred_raw):
        """Calculate Lorenz curve using raw predictions"""
        # For XGBoost AFT: higher predictions = longer survival time = lower risk
        # So we use predictions directly as survival time predictions
        y_pred_risk = 1.0 / (y_pred_raw + 1e-6)  # Inverse relationship: higher survival time = lower risk
        
        df = pd.DataFrame({
            'event': y_true,
            'predicted_risk': y_pred_risk
        })
        
        print(f"Risk scores - Mean: {y_pred_risk.mean():.6f}, Std: {y_pred_risk.std():.6f}")
        print(f"Risk scores - Min: {y_pred_risk.min():.6f}, Max: {y_pred_risk.max():.6f}")
        
        # Sort by predicted risk (descending - highest risk first)
        df = df.sort_values('predicted_risk', ascending=False).reset_index(drop=True)
        
        # Create deciles
        n_deciles = 10
        decile_size = len(df) // n_deciles
        
        cumulative_events = []
        cumulative_population = []
        
        for i in range(n_deciles):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size if i < n_deciles - 1 else len(df)
            
            # Events in this decile and all previous
            events_so_far = df.iloc[:end_idx]['event'].sum()
            population_so_far = end_idx
            
            cumulative_events.append(events_so_far)
            cumulative_population.append(population_so_far)
        
        # Convert to proportions
        total_events = df['event'].sum()
        total_population = len(df)
        
        if total_events == 0:
            print("WARNING: No events for Lorenz curve")
            return 0.0
        
        cumulative_events_prop = np.array(cumulative_events) / total_events
        cumulative_population_prop = np.array(cumulative_population) / total_population
        
        # Calculate Gini coefficient using the area under the curve
        # Perfect model would have area = 0.5, random model has area = 0.5
        # Gini = 1 - 2 * area_under_curve
        
        # Add origin point
        x_vals = np.concatenate([[0], cumulative_population_prop])
        y_vals = np.concatenate([[0], cumulative_events_prop])
        
        area_under_curve = np.trapz(y_vals, x_vals)
        gini = 1 - 2 * area_under_curve
        
        print(f"Total events: {total_events}, Area under curve: {area_under_curve:.4f}")
        
        self.enhanced_results['lorenz_curve'] = {
            'cumulative_events': cumulative_events_prop,
            'cumulative_population': cumulative_population_prop,
            'gini_coefficient': gini
        }
        
        return gini
    
    def plot_lorenz_curve(self):
        """Plot Lorenz curve"""
        if 'lorenz_curve' not in self.enhanced_results:
            print("Lorenz curve not calculated yet")
            return
        
        results = self.enhanced_results['lorenz_curve']
        
        plt.figure(figsize=(10, 8))
        plt.plot(results['cumulative_population'], results['cumulative_events'], 
                'ro-', linewidth=2, markersize=8, label='Model')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Model')
        
        plt.xlabel('Cumulative % of Population')
        plt.ylabel('Cumulative % of Events')
        plt.title(f'Lorenz Curve - Gini: {results["gini_coefficient"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if results['gini_coefficient'] < 0.05:
            plt.text(0.5, 0.2, 'WARNING: Low Gini - Poor Discrimination', 
                    ha='center', fontsize=12, color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def calculate_time_dependent_cindex(self, y_true, events, predictions, time_horizons=[30, 90, 180, 365]):
        """Calculate C-index at different time horizons"""
        c_indices = {}
        
        for horizon in time_horizons:
            mask = (y_true >= horizon) | (events == 1)
            
            if mask.sum() < 100:
                continue
            
            y_subset = y_true[mask]
            events_subset = events[mask]
            pred_subset = predictions[mask]
            
            y_horizon = np.minimum(y_subset, horizon)
            events_horizon = np.where(y_subset <= horizon, events_subset, 0)
            
            c_index = concordance_index(y_horizon, pred_subset, events_horizon)
            c_indices[horizon] = c_index
        
        self.enhanced_results['time_dependent_cindex'] = c_indices
        
        print("Time-dependent C-index:")
        for horizon, c_index in c_indices.items():
            print(f"  {horizon} days: {c_index:.3f}")
        
        return c_indices
    
    def run_enhanced_analysis(self) -> Dict:
        """Run enhanced analysis with comprehensive diagnostics"""
        print("RUNNING ENHANCED SURVIVAL ANALYSIS")
        print("=" * 50)
        
        # Run basic analysis
        self.analyze_baseline()
        self.analyze_industries()
        self.analyze_demographics()
        self.analyze_temporal_trends()
        self.build_predictive_model()
        
        if 'model' not in self.insights:
            print("Model not trained - cannot run enhanced analysis")
            return {'insights': self.insights}
        
        # Add comprehensive model diagnostics
        model_diagnostics = self.diagnose_model_learning()
        
        # Get validation data
        X_val = self.model_data['X_val']
        y_val = self.model_data['y_val']
        event_val = self.model_data['event_val']
        
        print("\n1. Generating survival curves...")
        self.generate_individual_survival_curves(X_val)
        self.plot_individual_survival_curves()
        
        print("\n2. Calculating Lorenz curve...")
        raw_predictions = self.model.predict(xgb.DMatrix(X_val))
        gini = self.calculate_lorenz_curve(event_val, raw_predictions)
        self.plot_lorenz_curve()
        
        print("\n3. Time-dependent performance...")
        self.calculate_time_dependent_cindex(y_val, event_val, raw_predictions)
        
        # Generate summary with recommendations
        enhanced_summary = self.generate_enhanced_summary()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETE")
        print("="*50)
        
        return {
            'insights': self.insights,
            'enhanced_results': self.enhanced_results,
            'enhanced_summary': enhanced_summary,
            'model_diagnostics': model_diagnostics
        }
    
    def generate_enhanced_summary(self) -> Dict:
        """Generate enhanced summary"""
        baseline = self.insights.get("baseline", {})
        
        summary = {
            "population_size": baseline.get('population_size', 'N/A'),
            "retention_365d": baseline.get('retention_365d', 'N/A'),
            "model_c_index": self.insights.get('model', {}).get('c_index_val', 'N/A'),
            "gini_coefficient": self.enhanced_results.get('lorenz_curve', {}).get('gini_coefficient', 'N/A'),
            "time_dependent_cindex": self.enhanced_results.get('time_dependent_cindex', {}),
        }
        
        print("\nENHANCED SUMMARY:")
        print(f"Population: {summary['population_size']}")
        print(f"Model C-index: {summary['model_c_index']}")
        print(f"Gini Coefficient: {summary['gini_coefficient']}")
        
        return summary
    
    def generate_executive_summary(self) -> Dict:
        """Generate executive summary"""
        print("\nEXECUTIVE SUMMARY")
        print("=" * 50)
        
        baseline = self.insights["baseline"]
        
        summary = {
            "train_population_size": f"{baseline['population_size']:,}",
            "total_population_size": f"{baseline['total_population']:,}",
            "retention_365d": f"{baseline['retention_365d']:.1%}",
            "event_rate": f"{baseline['event_rate']:.1%}",
        }
        
        print(f"Population: {summary['train_population_size']} employees")
        print(f"1-year retention: {summary['retention_365d']}")
        print(f"Event rate: {summary['event_rate']}")
        
        return summary
    
    def run_complete_analysis(self) -> Dict:
        """Execute complete analysis"""
        print("STARTING COMPLETE SURVIVAL ANALYSIS")
        print("=" * 60)
        
        self.analyze_baseline()
        self.analyze_industries()
        self.analyze_demographics()
        self.analyze_temporal_trends()
        self.build_predictive_model()
        
        summary = self.generate_executive_summary()
        
        return {"insights": self.insights, "summary": summary}

# Usage
if __name__ == "__main__":
    # Example usage:
    # 1. Load your employee data
    # data = df.select(*final_cols).toPandas()
    
    # 2. Initialize analyzer
    # analyzer = SurvivalAnalysis(data)
    
    # 3. Run enhanced analysis with diagnostics
    # results = analyzer.run_enhanced_analysis()
    
    # 4. Check recommendations
    # recommendations = results['enhanced_summary']['recommendations']
    # for rec in recommendations:
    #     print(f"Issue: {rec['issue']}")
    #     print(f"Recommendation: {rec['recommendation']}")
    
    # 5. Or run original analysis
    # results = analyzer.run_complete_analysis()
    
    pass
