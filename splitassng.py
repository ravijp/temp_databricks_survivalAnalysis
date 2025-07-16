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
        
        # Flexible age grouping (handles age > 100)
        vantage_dt, birth_dt = pd.to_datetime(df["vantage_date"], errors='coerce'), pd.to_datetime(df["birth_dt"], errors='coerce')
        birth_dt = birth_dt.where(birth_dt.dt.year.between(1900, 2020), pd.NaT)
        df['age'] = (vantage_dt - birth_dt).dt.days / 365.25
        
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
            include_lowest=True,
        )
        
        # Flexible tenure grouping (handles very long tenure)
        df["tenure_at_vantage_days"] = np.where(df["tenure_at_vantage_days"] > 365 * 100, 365 * 100, df["tenure_at_vantage_days"])
        tenure_years = df["tenure_at_vantage_days"] / 365.25
        
        df["tenure_group"] = pd.cut(
            tenure_years,
            bins=[0, 0.5, 1, 2, 3, 5, np.inf],
            labels=["<6mo", "6mo-1yr", "1-2yr", "2-3yr", "3-5yr", "5yr+"],
            include_lowest=True,
        )
        
        # Flexible salary grouping (handles very high salaries)
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
        
        # Use train split for industry insights
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
        
        # Use train val split for demographic insights
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
            
            # Get categories sorted by logical order
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
            
            # Print top insights
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
        
        # Use dataset split to define temporal cohorts
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
        
        # Use proper train/val splits
        train_data = self.data[self.data['dataset_split'] == 'train']
        val_data = self.data[self.data['dataset_split'] == 'val']
        
        if len(val_data) < 1000:
            print("Insufficient validation data - using random split from train")
            np.random.seed(42)
            train_idx = np.random.choice(len(train_data), size=int(0.8 * len(train_data)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(train_data)), train_idx)
            val_data = train_data.iloc[val_idx]
            train_data = train_data.iloc[train_idx]
            
        # Define numeric features
        numeric_features = [
            'age', 'tenure_at_vantage_days', 'team_size', 'baseline_salary',
            'team_avg_comp', 'salary_growth_ratio', 'manager_changes_count'
        ]
        
        # Define categorical features
        categorical_features = [
            'gender_cd', 'pay_rt_type_cd', 'full_tm_part_tm_cd', 'fscl_actv_ind'
        ]
        
        # Filter features that actually exist in data
        available_numeric = [f for f in numeric_features if f in train_data.columns]
        available_categorical = [f for f in categorical_features if f in train_data.columns]
        
        # Copy data to avoid modifying original
        train_data = train_data.copy()
        val_data = val_data.copy()
        
        # Handle categorical features with LabelEncoder
        label_encoders = {}
        for cat_feature in available_categorical:
            # Prepare data
            train_cats = train_data[cat_feature].fillna('MISSING').astype(str)
            val_cats = val_data[cat_feature].fillna('MISSING').astype(str)
            
            # Get unique categories in both train and validation sets
            all_categories = set(train_cats.unique()) | set(val_cats.unique())
            
            # Create a mapping dictionary
            category_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
            
            # Transform data using the mapping
            train_data[f'{cat_feature}_encoded'] = train_cats.map(category_mapping)
            val_data[f'{cat_feature}_encoded'] = val_cats.map(category_mapping)
            
            # Store the mapping
            label_encoders[cat_feature] = category_mapping
            
        # NAICS encoding
        if 'naics_2digit' in train_data.columns:
            train_naics = train_data['naics_2digit'].fillna('MISSING').astype(str)
            val_naics = val_data['naics_2digit'].fillna('MISSING').astype(str)
            
            # Get all unique NAICS codes
            all_naics = set(train_naics.unique()) | set(val_naics.unique())
            
            # Create mapping
            naics_mapping = {naics: idx for idx, naics in enumerate(all_naics)}
            
            # Apply mapping
            train_data['naics_encoded'] = train_naics.map(naics_mapping)
            val_data['naics_encoded'] = val_naics.map(naics_mapping)
            
            available_numeric.append('naics_encoded')
            label_encoders['naics_2digit'] = naics_mapping
            
        # Create final feature list
        encoded_categorical = [f'{cat}_encoded' for cat in available_categorical]
        feature_columns = available_numeric + encoded_categorical
        
        # Handle missing values
        for col in feature_columns:
            if col in train_data.columns and train_data[col].isna().any():
                if train_data[col].dtype in [np.float64, np.int64]:
                    # Fill numeric columns with median
                    fill_value = train_data[col].median()
                    train_data[col] = train_data[col].fillna(fill_value)
                    val_data[col] = val_data[col].fillna(fill_value)
                else:
                    # Fill categorical columns with mode
                    fill_value = train_data[col].mode().iloc[0]
                    train_data[col] = train_data[col].fillna(fill_value)
                    val_data[col] = val_data[col].fillna(fill_value)
                    
        # Create model datasets
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
        
        # Set survival information
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
        
        # Store model for later use
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
        
        # Risk segmentation
        risk_percentiles = np.percentile(val_pred, [20, 50, 80])
        importance_df.to_csv('./feature_importance.csv', index=False)
        
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
            'risk_percentiles': risk_percentiles,
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
            
        self.insights['model'] = model_results
        return model_results
    
    def generate_individual_survival_curves(self, X_test, time_points=None, distribution='weibull'):
        """Generate proper individual survival curves"""
        if time_points is None:
            time_points = np.arange(1, 366, 1)
        
        # Get XGBoost AFT predictions
        dtest = xgb.DMatrix(X_test)
        log_predictions = self.model.predict(dtest)
        
        # Ensure sufficient variance
        if log_predictions.std() < 0.1:
            print("WARNING: Low prediction variance - model may not be learning")
        
        # Transform to survival times
        scale_predictions = np.exp(log_predictions)
        
        # Proper scale parameter estimation
        scale_param = max(log_predictions.std(), 0.5)
        
        survival_curves = []
        for pred in scale_predictions:
            if distribution == 'weibull':
                lambda_param = pred
                k_param = np.clip(1.0 / scale_param, 0.5, 5.0)
                curve = np.exp(-((time_points / lambda_param) ** k_param))
            else:  # lognormal
                mu_param = np.log(pred)
                sigma_param = max(scale_param, 0.3)
                curve = 1 - stats.norm.cdf((np.log(time_points) - mu_param) / sigma_param)
            
            # Ensure valid survival curve
            curve = np.clip(curve, 0, 1)
            
            # Force monotonically decreasing
            for i in range(1, len(curve)):
                if curve[i] > curve[i-1]:
                    curve[i] = curve[i-1]
            
            survival_curves.append(curve)
        
        self.survival_curves = np.array(survival_curves)
        self.time_points = time_points
        
        # Validation
        self._validate_survival_curves()
        
        return self.survival_curves
    
    def _validate_survival_curves(self):
        """Validate survival curves are reasonable"""
        if self.survival_curves is None:
            return
        
        # Check variance
        curve_variance = np.var(self.survival_curves, axis=0)
        mean_variance = np.mean(curve_variance)
        
        if mean_variance < 0.001:
            print("WARNING: Survival curves too similar - poor discrimination")
        
        # Check final survival spread
        final_survival = self.survival_curves[:, -1]
        if final_survival.std() < 0.05:
            print("WARNING: Low final survival variance")
        
        print(f"Survival curve validation: variance={mean_variance:.4f}, final_std={final_survival.std():.3f}")
    
    def plot_individual_survival_curves(self, n_curves=5):
        """Plot individual survival curves with validation"""
        if self.survival_curves is None:
            print("No survival curves generated yet")
            return
        
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'orange', 'green', 'red', 'purple']
        
        for i in range(min(n_curves, len(self.survival_curves))):
            curve = self.survival_curves[i]
            
            # Check if curve is flat
            curve_slope = abs(curve[0] - curve[-1])
            if curve_slope < 0.01:
                print(f"WARNING: Employee {i+1} has flat curve (slope: {curve_slope:.4f})")
            
            plt.plot(self.time_points, curve, color=colors[i], linewidth=2,
                    label=f'Employee {i+1} (final: {curve[-1]:.3f})')
        
        # Add mean curve
        mean_curve = np.mean(self.survival_curves, axis=0)
        plt.plot(self.time_points, mean_curve, 'k--', linewidth=3, alpha=0.7, label='Mean')
        
        plt.xlabel('Time (days)')
        plt.ylabel('Survival Probability')
        plt.title('Individual Employee Survival Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 365)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
    
    def calculate_lorenz_curve(self, y_true, y_pred_survival):
        """Proper Lorenz curve calculation"""
        # Convert survival probability to risk
        y_pred_risk = 1 - y_pred_survival
        
        # Create dataframe
        df = pd.DataFrame({
            'event': y_true,
            'predicted_risk': y_pred_risk
        })
        
        # Sort by predicted risk
        df = df.sort_values('predicted_risk', ascending=False).reset_index(drop=True)
        
        # Create deciles
        df['decile'] = pd.qcut(range(len(df)), 10, labels=False)
        
        # Calculate decile events
        decile_events = df.groupby('decile')['event'].sum()
        total_events = df['event'].sum()
        
        # Cumulative percentages
        cumulative_events = decile_events.cumsum() / total_events
        cumulative_population = np.arange(1, 11) / 10
        
        # Calculate Gini coefficient
        gini = 1 - 2 * np.trapz(cumulative_events, cumulative_population)
        
        # Store results
        self.enhanced_results['lorenz_curve'] = {
            'cumulative_events': cumulative_events,
            'cumulative_population': cumulative_population,
            'gini_coefficient': gini
        }
        
        return gini
    
    def plot_lorenz_curve(self):
        """Plot Lorenz curve with validation"""
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
        
        # Validation
        if results['gini_coefficient'] < 0.05:
            plt.text(0.5, 0.2, 'WARNING: Low Gini - Poor Discrimination', 
                    ha='center', fontsize=12, color='red',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def calculate_ipcw_brier_score(self, y_true, events, survival_matrix, time_points):
        """Proper IPCW Brier score calculation"""
        from lifelines import KaplanMeierFitter
        
        # Fit censoring distribution
        kmf_censor = KaplanMeierFitter()
        kmf_censor.fit(y_true, 1 - events)
        
        brier_scores = []
        
        for i, t in enumerate(time_points):
            if i >= survival_matrix.shape[1]:
                break
            
            # Observed events at time t
            observed_events = ((y_true <= t) & (events == 1)).astype(int)
            
            # Predicted survival probabilities at time t
            predicted_survival = survival_matrix[:, i]
            predicted_events = 1 - predicted_survival
            
            # Calculate IPCW weights
            weights = []
            for j in range(len(y_true)):
                if events[j] == 1 and y_true[j] <= t:
                    # Event occurred before t
                    censor_prob = kmf_censor.survival_function_at_times(y_true[j]).values[0]
                    weight = 1.0 / max(censor_prob, 0.01)
                elif y_true[j] > t:
                    # Still at risk at time t
                    censor_prob = kmf_censor.survival_function_at_times(t).values[0]
                    weight = 1.0 / max(censor_prob, 0.01)
                else:
                    # Censored before t
                    weight = 0.0
                
                weights.append(weight)
            
            weights = np.array(weights)
            
            # Calculate weighted Brier score
            brier_t = np.average((observed_events - predicted_events) ** 2, weights=weights)
            
            # Validate result
            if brier_t < 0.01 or brier_t > 1.0:
                print(f"WARNING: Suspicious Brier score {brier_t:.4f} at time {t}")
            
            brier_scores.append(brier_t)
        
        # Integrate over time
        valid_times = time_points[:len(brier_scores)]
        ibs = np.trapz(brier_scores, valid_times) / (valid_times[-1] - valid_times[0])
        
        # Store results
        self.enhanced_results['ipcw_results'] = {
            'brier_scores': brier_scores,
            'time_points': valid_times,
            'integrated_brier_score': ibs
        }
        
        print(f"IPCW Integrated Brier Score: {ibs:.4f}")
        
        # Validation
        if ibs < 0.01:
            print("WARNING: IBS suspiciously low - check calculation")
        elif ibs > 0.5:
            print("WARNING: IBS very high - poor model performance")
        
        return ibs
    
    def calculate_time_dependent_cindex(self, y_true, events, predictions, time_horizons=[30, 90, 180, 365]):
        """Calculate C-index at different time horizons"""
        from lifelines.utils import concordance_index
        
        c_indices = {}
        
        for horizon in time_horizons:
            # Filter to observations with at least 'horizon' days of follow-up
            mask = (y_true >= horizon) | (events == 1)
            
            if mask.sum() < 100:
                continue
            
            y_subset = y_true[mask]
            events_subset = events[mask]
            pred_subset = predictions[mask]
            
            # For this horizon, events after horizon are censored
            y_horizon = np.minimum(y_subset, horizon)
            events_horizon = np.where(y_subset <= horizon, events_subset, 0)
            
            c_index = concordance_index(y_horizon, pred_subset, events_horizon)
            c_indices[horizon] = c_index
        
        self.enhanced_results['time_dependent_cindex'] = c_indices
        
        print("Time-dependent C-index:")
        for horizon, c_index in c_indices.items():
            print(f"  {horizon} days: {c_index:.3f}")
        
        return c_indices
    
    def assess_calibration(self, y_true, events, survival_probs, time_point=365):
        """Assess model calibration at specific time point"""
        # Create risk buckets
        risk_scores = 1 - survival_probs
        n_bins = 10
        
        # Equal-frequency binning
        bin_edges = np.percentile(risk_scores, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] = 1.0
        
        predicted_risks = []
        observed_risks = []
        bin_counts = []
        
        for i in range(n_bins):
            lower = bin_edges[i]
            upper = bin_edges[i + 1]
            
            if i == 0:
                mask = (risk_scores >= lower) & (risk_scores <= upper)
            else:
                mask = (risk_scores > lower) & (risk_scores <= upper)
            
            if mask.sum() == 0:
                continue
            
            # Predicted risk in this bin
            pred_risk = risk_scores[mask].mean()
            
            # Observed risk in this bin
            y_bin = y_true[mask]
            events_bin = events[mask]
            
            # Calculate observed risk at time_point
            obs_risk = ((y_bin <= time_point) & (events_bin == 1)).mean()
            
            predicted_risks.append(pred_risk)
            observed_risks.append(obs_risk)
            bin_counts.append(mask.sum())
        
        # Calculate calibration metrics
        calibration_error = np.mean(np.abs(np.array(predicted_risks) - np.array(observed_risks)))
        
        self.calibration_metrics = {
            'predicted_risks': predicted_risks,
            'observed_risks': observed_risks,
            'bin_counts': bin_counts,
            'calibration_error': calibration_error,
            'time_point': time_point
        }
        
        print(f"Calibration Error at {time_point} days: {calibration_error:.4f}")
        
        return calibration_error
    
    def plot_calibration_curve(self):
        """Plot calibration curve"""
        if not self.calibration_metrics:
            print("Calibration not assessed yet")
            return
        
        metrics = self.calibration_metrics
        
        plt.figure(figsize=(10, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
        plt.scatter(metrics['predicted_risks'], metrics['observed_risks'], 
                   s=100, alpha=0.7, label='Model')
        
        plt.xlabel('Predicted Risk')
        plt.ylabel('Observed Risk')
        plt.title(f'Calibration Curve at {metrics["time_point"]} days\n'
                 f'Calibration Error: {metrics["calibration_error"]:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.show()
    
    def create_risk_segments(self, survival_probs, method='tertiles'):
        """Create High/Medium/Low risk segments for business use"""
        risk_scores = 1 - survival_probs
        
        if method == 'tertiles':
            # Equal-size groups
            low_threshold = np.percentile(risk_scores, 33.33)
            high_threshold = np.percentile(risk_scores, 66.67)
            
            risk_segments = np.where(risk_scores <= low_threshold, 'Low',
                                   np.where(risk_scores <= high_threshold, 'Medium', 'High'))
        
        elif method == 'business_thresholds':
            # Business-defined thresholds
            low_threshold = 0.2
            high_threshold = 0.5
            
            risk_segments = np.where(risk_scores <= low_threshold, 'Low',
                                   np.where(risk_scores <= high_threshold, 'Medium', 'High'))
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Store results
        self.enhanced_results['risk_segments'] = {
            'segments': risk_segments,
            'risk_scores': risk_scores,
            'thresholds': {'low': low_threshold, 'high': high_threshold}
        }
        
        # Summary statistics
        segment_counts = pd.Series(risk_segments).value_counts()
        print("Risk Segment Distribution:")
        for segment, count in segment_counts.items():
            print(f"  {segment}: {count} ({count/len(risk_segments)*100:.1f}%)")
        
        return risk_segments
    
    def validate_risk_segments(self, y_true, events, time_point=365):
        """Validate risk segments have different outcomes"""
        if 'risk_segments' not in self.enhanced_results:
            print("Risk segments not created yet")
            return
        
        segments = self.enhanced_results['risk_segments']['segments']
        
        # Calculate actual outcomes by segment
        segment_outcomes = {}
        
        for segment in ['Low', 'Medium', 'High']:
            mask = segments == segment
            if mask.sum() == 0:
                continue
            
            y_segment = y_true[mask]
            events_segment = events[mask]
            
            # Calculate outcome rate
            outcome_rate = ((y_segment <= time_point) & (events_segment == 1)).mean()
            segment_outcomes[segment] = outcome_rate
        
        # Store results
        self.enhanced_results['segment_validation'] = segment_outcomes
        
        print(f"Actual Outcome Rates at {time_point} days:")
        for segment, rate in segment_outcomes.items():
            print(f"  {segment} Risk: {rate:.3f}")
        
        return segment_outcomes
    
    def run_enhanced_analysis(self) -> Dict:
        """Enhanced analysis with all improvements"""
        print("RUNNING ENHANCED SURVIVAL ANALYSIS")
        print("=" * 50)
        
        # Run existing analysis first
        self.analyze_baseline()
        self.analyze_industries()
        self.analyze_demographics()
        self.analyze_temporal_trends()
        self.build_predictive_model()
        
        if 'model' not in self.insights:
            print("Model not trained - cannot run enhanced analysis")
            return {'insights': self.insights}
        
        # Get test data
        X_val = self.model_data['X_val']
        y_val = self.model_data['y_val']
        event_val = self.model_data['event_val']
        
        print("\n1. Generating proper survival curves...")
        self.generate_individual_survival_curves(X_val)
        self.plot_individual_survival_curves()
        
        print("\n2. Calculating Lorenz curve...")
        # Generate survival probabilities at 1 year
        mean_survival = np.mean(self.survival_curves[:, -1])
        gini = self.calculate_lorenz_curve(event_val, 
                                         np.full(len(event_val), mean_survival))
        self.plot_lorenz_curve()
        
        print("\n3. Calculating IPCW metrics...")
        ibs = self.calculate_ipcw_brier_score(y_val, event_val, 
                                            self.survival_curves, 
                                            self.time_points[::10])
        
        print("\n4. Time-dependent performance...")
        self.calculate_time_dependent_cindex(y_val, event_val, 
                                           self.model.predict(xgb.DMatrix(X_val)))
        
        print("\n5. Assessing calibration...")
        survival_1year = self.survival_curves[:, -1]
        self.assess_calibration(y_val, event_val, survival_1year)
        self.plot_calibration_curve()
        
        print("\n6. Creating risk segments...")
        risk_segments = self.create_risk_segments(survival_1year)
        self.validate_risk_segments(y_val, event_val)
        
        # Generate enhanced summary
        enhanced_summary = self.generate_enhanced_summary()
        
        return {
            'insights': self.insights,
            'enhanced_results': self.enhanced_results,
            'enhanced_summary': enhanced_summary
        }
    
    def generate_enhanced_summary(self) -> Dict:
        """Generate enhanced executive summary"""
        baseline = self.insights.get("baseline", {})
        
        summary = {
            # Existing metrics
            "population_size": baseline.get('population_size', 'N/A'),
            "retention_365d": baseline.get('retention_365d', 'N/A'),
            "model_c_index": self.insights.get('model', {}).get('c_index_val', 'N/A'),
            
            # Enhanced metrics
            "gini_coefficient": self.enhanced_results.get('lorenz_curve', {}).get('gini_coefficient', 'N/A'),
            "ipcw_brier_score": self.enhanced_results.get('ipcw_results', {}).get('integrated_brier_score', 'N/A'),
            "calibration_error": self.calibration_metrics.get('calibration_error', 'N/A'),
            "time_dependent_cindex": self.enhanced_results.get('time_dependent_cindex', {}),
            
            # Business metrics
            "risk_segments": self.enhanced_results.get('risk_segments', {}),
            "segment_validation": self.enhanced_results.get('segment_validation', {}),
        }
        
        print("\nENHANCED EXECUTIVE SUMMARY")
        print("=" * 50)
        print(f"Population: {summary['population_size']}")
        print(f"Model C-index: {summary['model_c_index']}")
        print(f"Gini Coefficient: {summary['gini_coefficient']}")
        print(f"IPCW Brier Score: {summary['ipcw_brier_score']}")
        print(f"Calibration Error: {summary['calibration_error']}")
        
        return summary
    
    def generate_executive_summary(self) -> Dict:
        """Generate comprehensive executive summary"""
        print("\nEXECUTIVE SUMMARY")
        print("=" * 50)
        
        baseline = self.insights["baseline"]
        
        summary = {
            "train_population_size": f"{baseline['population_size']:,}",
            "total_population_size": f"{baseline['total_population']:,}",
            "median_survival_days": f"{baseline['median_survival']:.0f}",
            "median_survival_months": f"{baseline['median_survival']/30.44:.1f}",
            "retention_90d": f"{baseline['retention_90d']:.1%}",
            "retention_365d": f"{baseline['retention_365d']:.1%}",
            "event_rate": f"{baseline['event_rate']:.1%}",
        }
        
        # Add key business opportunities
        opportunities = []
        
        if "industry" in self.insights:
            ranked = self.insights["industry"]["ranked"]
            if ranked:
                gap = ranked[0][1]["retention_365d"] - ranked[-1][1]["retention_365d"]
                opportunities.append(
                    f"Industry performance gap: {gap:.1%} improvement opportunity"
                )
                
        if "temporal" in self.insights:
            temporal = self.insights["temporal"]
            if "retention_change_365d" in temporal:
                change = temporal["retention_change_365d"]
                direction = "improving" if change > 0 else "declining"
                opportunities.append(
                    f"Retention {direction} {abs(change):.1%} year-over-year"
                )
                
        if "model" in self.insights:
            model = self.insights["model"]
            opportunities.append(
                f"Predictive model achieves {model['c_index_val']:.3f} C-index"
            )
            
        # Add enhanced metrics
        if 'enhanced_results' in self.__dict__:
            if 'lorenz_curve' in self.enhanced_results:
                gini = self.enhanced_results['lorenz_curve']['gini_coefficient']
                opportunities.append(f"Model discrimination: Gini = {gini:.3f}")
            
            if 'ipcw_results' in self.enhanced_results:
                ibs = self.enhanced_results['ipcw_results']['integrated_brier_score']
                opportunities.append(f"Model calibration: IBS = {ibs:.3f}")
        
        summary["key_opportunities"] = opportunities
        
        print(
            f"Analysis based on train population: {summary['train_population_size']} employees"
        )
        print(
            f"Total population available: {summary['total_population_size']} employees"
        )
        print(
            f"Median assignment duration: {summary['median_survival_days']} days ({summary['median_survival_months']} months)"
        )
        print(f"90-day retention: {summary['retention_90d']}")
        print(f"1-year retention: {summary['retention_365d']}")
        print(f"Event rate: {summary['event_rate']}")
        
        print(f"\nKey business opportunities:")
        for opp in opportunities:
            print(f"   - {opp}")
            
        return summary
    
    def run_complete_analysis(self) -> Dict:
        """Execute complete survival analysis sequence"""
        print("STARTING COMPLETE SURVIVAL ANALYSIS")
        print("=" * 60)
        
        # Execute full analysis
        self.analyze_baseline()
        self.analyze_industries()
        self.analyze_demographics()
        self.analyze_temporal_trends()
        self.build_predictive_model()
        
        # Generate executive summary
        summary = self.generate_executive_summary()
        
        return {"insights": self.insights, "summary": summary}

# Usage example for the enhanced analysis
if __name__ == "__main__":
    # Load your data
    # data = df.select(*final_cols).toPandas()
    
    # Initialize analyzer
    # analyzer = SurvivalAnalysis(data)
    
    # Run enhanced analysis
    # results = analyzer.run_enhanced_analysis()
    
    # Or run original analysis
    # results = analyzer.run_complete_analysis()
    
    pass
