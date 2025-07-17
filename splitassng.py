# Enhanced XGBoost AFT Survival Analysis
# Advanced implementation with proper AFT handling and comprehensive Kaplan-Meier insights

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
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings("ignore")

plt.style.use("default")
sns.set_palette("Set2")
plt.rcParams["figure.figsize"] = (15, 10)
plt.rcParams["font.size"] = 11

class AFTDistribution(Enum):
    NORMAL = "normal"
    LOGISTIC = "logistic"
    EXTREME = "extreme"

@dataclass
class AFTParameters:
    eta: np.ndarray
    sigma: float
    distribution: AFTDistribution

class EnhancedSurvivalAnalysis:
    """
    Advanced XGBoost AFT survival analysis with comprehensive Kaplan-Meier insights
    
    Provides proper AFT implementation with comprehensive validation and 
    business-focused analytics including industry, demographic, and temporal analysis.
    """
    
    def __init__(self, data: pd.DataFrame, aft_distribution: AFTDistribution = AFTDistribution.NORMAL):
        self.data = self._prepare_data(data)
        self.aft_distribution = aft_distribution
        self.kmf = KaplanMeierFitter()
        self.insights = {}
        self.enhanced_results = {}
        self.model_parameters = None
        self.survival_curves = None
        self.calibration_metrics = {}
        self.time_points = None
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare analysis dataset with business segments"""
        df = df.copy()
        
        df["naics_2digit"] = df["naics_cd"].astype(str).str[:2]
        
        vantage_dt = pd.to_datetime(df["vantage_date"], errors='coerce')
        birth_dt = pd.to_datetime(df["birth_dt"], errors='coerce')
        birth_dt = birth_dt.where(birth_dt.dt.year.between(1900, 2020), pd.NaT)
        df['age'] = (vantage_dt - birth_dt).dt.days / 365.25
        
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"],
            include_lowest=True,
        )
        
        df["tenure_at_vantage_days"] = np.clip(df["tenure_at_vantage_days"], 0, 365 * 50)
        tenure_years = df["tenure_at_vantage_days"] / 365.25
        
        df["tenure_group"] = pd.cut(
            tenure_years,
            bins=[0, 0.5, 1, 2, 3, 5, np.inf],
            labels=["<6mo", "6mo-1yr", "1-2yr", "2-3yr", "3-5yr", "5yr+"],
            include_lowest=True,
        )
        
        df["salary_group"] = pd.cut(
            df["baseline_salary"],
            bins=[0, 40000, 60000, 80000, 120000, 200000, np.inf],
            labels=["<40K", "40-60K", "60-80K", "80-120K", "120-200K", "200K+"],
            include_lowest=True,
        )
        
        return df
    
    def analyze_baseline(self) -> Dict:
        """Generate population baseline metrics"""
        print("Analyzing baseline retention patterns...")
        
        train_data = self.data[self.data["dataset_split"].isin(["train", "val"])]
        
        self.kmf.fit(train_data["survival_time_days"], train_data["event_indicator"])
        
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
        
        fig, ax = plt.subplots(figsize=(15, 10))
        self.kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=3, color="navy")
        
        milestones = [30, 90, 180, 365]
        colors = ["red", "orange", "green", "purple"]
        for day, color in zip(milestones, colors):
            retention = self.kmf.survival_function_at_times(day).iloc[0]
            ax.axvline(x=day, color=color, linestyle="--", alpha=0.7)
            ax.text(day, retention + 0.02, f"{day}d\n{retention:.1%}",
                   ha="center", va="bottom", fontsize=11, fontweight="bold")
        
        ax.set_title("Employee Retention Baseline - ADP Population", 
                    fontsize=17, fontweight="bold")
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig("baseline_survival_enhanced.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        print(f"BASELINE INSIGHTS:")
        print(f"   Train + val population: {metrics['population_size']:,} employees")
        print(f"   Event rate: {metrics['event_rate']:.1%}")
        print(f"   Median survival: {metrics['median_survival']:.0f} days")
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
    
    def build_enhanced_predictive_model(self) -> Dict:
        """Build enhanced XGBoost AFT model"""
        print("Building enhanced XGBoost AFT predictive model...")
        
        train_data = self.data[self.data['dataset_split'] == 'train']
        val_data = self.data[self.data['dataset_split'] == 'val']
        
        if len(val_data) < 1000:
            print("Insufficient validation data - using random split from train")
            np.random.seed(42)
            train_idx = np.random.choice(len(train_data), size=int(0.8 * len(train_data)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(train_data)), train_idx)
            val_data = train_data.iloc[val_idx]
            train_data = train_data.iloc[train_idx]
        
        numeric_features = [
            'age', 'tenure_at_vantage_days', 'team_size', 'baseline_salary',
            'team_avg_comp', 'salary_growth_ratio', 'manager_changes_count'
        ]
        
        categorical_features = [
            'gender_cd', 'pay_rt_type_cd', 'full_tm_part_tm_cd', 'fscl_actv_ind'
        ]
        
        available_numeric = [f for f in numeric_features if f in train_data.columns]
        available_categorical = [f for f in categorical_features if f in train_data.columns]
        
        train_data, val_data, feature_columns, label_encoders = self._process_features(
            train_data, val_data, available_numeric, available_categorical
        )
        
        model_columns = feature_columns + ['survival_time_days', 'event_indicator']
        train_model_data = train_data[model_columns].dropna()
        val_model_data = val_data[model_columns].dropna()
        
        X_train = train_model_data[feature_columns]
        y_train = train_model_data['survival_time_days']
        event_train = train_model_data['event_indicator']
        
        X_val = val_model_data[feature_columns]
        y_val = val_model_data['survival_time_days']
        event_val = val_model_data['event_indicator']
        
        # Train AFT model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        dtrain.set_float_info('label_lower_bound', y_train.values)
        dtrain.set_float_info('label_upper_bound', y_train.values)
        dval.set_float_info('label_lower_bound', y_val.values)
        dval.set_float_info('label_upper_bound', y_val.values)
        
        params = {
            'objective': 'survival:aft',
            'aft_loss_distribution': self.aft_distribution.value,
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'seed': 42,
            'verbosity': 0
        }
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params, dtrain, 
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        self.model = model
        self.model_data = {
            'X_train': X_train, 'y_train': y_train, 'event_train': event_train,
            'X_val': X_val, 'y_val': y_val, 'event_val': event_val,
            'feature_columns': feature_columns, 'label_encoders': label_encoders
        }
        
        # Estimate AFT parameters
        self.model_parameters = self._estimate_aft_parameters()
        
        # Calculate performance metrics
        train_pred = model.predict(dtrain)
        val_pred = model.predict(dval)
        
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
        
        # Enhanced diagnostics
        prediction_stats = {
            'train_pred_mean': train_pred.mean(),
            'train_pred_std': train_pred.std(),
            'val_pred_mean': val_pred.mean(),
            'val_pred_std': val_pred.std(),
            'prediction_range': val_pred.max() - val_pred.min(),
            'unique_predictions': len(np.unique(val_pred))
        }
        
        performance_metrics = {
            'train_size': len(train_model_data),
            'val_size': len(val_model_data),
            'c_index_train': c_index_train,
            'c_index_val': c_index_val,
            'feature_importance': importance_df,
            'prediction_stats': prediction_stats,
            'aft_distribution': self.aft_distribution.value
        }
        
        self._plot_feature_importance(importance_df)
        
        print(f"MODEL PERFORMANCE:")
        print(f"   Training size: {len(train_model_data):,}")
        print(f"   Validation size: {len(val_model_data):,}")
        print(f"   C-index (validation): {c_index_val:.3f}")
        print(f"   AFT distribution: {self.aft_distribution.value}")
        if self.model_parameters:
            print(f"   Scale parameter: {self.model_parameters.sigma:.3f}")
        
        self.insights['enhanced_model'] = performance_metrics
        return performance_metrics
    
    def _process_features(self, train_data: pd.DataFrame, val_data: pd.DataFrame,
                         numeric_features: List[str], categorical_features: List[str]) -> Tuple:
        """Process features with encoding and validation"""
        train_data = train_data.copy()
        val_data = val_data.copy()
        
        label_encoders = {}
        for cat_feature in categorical_features:
            train_cats = train_data[cat_feature].fillna('MISSING').astype(str)
            val_cats = val_data[cat_feature].fillna('MISSING').astype(str)
            
            all_categories = set(train_cats.unique()) | set(val_cats.unique())
            category_mapping = {cat: idx for idx, cat in enumerate(sorted(all_categories))}
            
            train_data[f'{cat_feature}_encoded'] = train_cats.map(category_mapping)
            val_data[f'{cat_feature}_encoded'] = val_cats.map(category_mapping)
            
            label_encoders[cat_feature] = category_mapping
        
        if 'naics_2digit' in train_data.columns:
            train_naics = train_data['naics_2digit'].fillna('MISSING').astype(str)
            val_naics = val_data['naics_2digit'].fillna('MISSING').astype(str)
            
            all_naics = set(train_naics.unique()) | set(val_naics.unique())
            naics_mapping = {naics: idx for idx, naics in enumerate(sorted(all_naics))}
            
            train_data['naics_encoded'] = train_naics.map(naics_mapping)
            val_data['naics_encoded'] = val_naics.map(naics_mapping)
            
            numeric_features = numeric_features + ['naics_encoded']
            label_encoders['naics_2digit'] = naics_mapping
        
        encoded_categorical = [f'{cat}_encoded' for cat in categorical_features]
        feature_columns = numeric_features + encoded_categorical
        
        for col in feature_columns:
            if col in train_data.columns:
                if train_data[col].dtype in [np.float64, np.int64]:
                    fill_value = train_data[col].median()
                else:
                    fill_value = train_data[col].mode().iloc[0] if not train_data[col].mode().empty else 0
                
                train_data[col] = train_data[col].fillna(fill_value)
                val_data[col] = val_data[col].fillna(fill_value)
        
        return train_data, val_data, feature_columns, label_encoders
    
    def _estimate_aft_parameters(self) -> AFTParameters:
        """Estimate AFT scale parameter from training residuals"""
        X_train = self.model_data['X_train']
        y_train = self.model_data['y_train']
        
        dtrain = xgb.DMatrix(X_train)
        eta_predictions = self.model.predict(dtrain)
        
        log_residuals = np.log(y_train) - eta_predictions
        
        if self.aft_distribution == AFTDistribution.NORMAL:
            sigma = np.std(log_residuals)
        elif self.aft_distribution == AFTDistribution.LOGISTIC:
            sigma = np.std(log_residuals) * np.sqrt(3) / np.pi
        elif self.aft_distribution == AFTDistribution.EXTREME:
            sigma = np.std(log_residuals) * np.sqrt(6) / np.pi
        else:
            sigma = np.std(log_residuals)
        
        sigma = np.clip(sigma, 0.1, 5.0)
        
        print(f"AFT Parameter Estimation:")
        print(f"   Location parameters - Mean: {eta_predictions.mean():.3f}, Std: {eta_predictions.std():.3f}")
        print(f"   Scale parameter: {sigma:.3f}")
        
        return AFTParameters(
            eta=eta_predictions,
            sigma=sigma,
            distribution=self.aft_distribution
        )
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature', ax=ax)
        ax.set_title('Feature Importance - Enhanced XGBoost AFT Model', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('enhanced_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_survival_curves(self, X_test: pd.DataFrame, time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate mathematically correct survival curves for AFT models"""
        if time_points is None:
            time_points = np.arange(1, 366, 1)
        
        if self.model_parameters is None:
            raise ValueError("Model parameters not estimated. Run build_enhanced_predictive_model() first.")
        
        dtest = xgb.DMatrix(X_test)
        eta_predictions = self.model.predict(dtest)
        
        print(f"AFT Survival Curve Generation:")
        print(f"   Location parameters - Mean: {eta_predictions.mean():.3f}, Std: {eta_predictions.std():.3f}")
        print(f"   Scale parameter: {self.model_parameters.sigma:.3f}")
        print(f"   Distribution: {self.aft_distribution.value}")
        
        survival_curves = []
        
        for eta in eta_predictions:
            if self.aft_distribution == AFTDistribution.NORMAL:
                log_times = np.log(time_points)
                z_scores = (log_times - eta) / self.model_parameters.sigma
                survival_probs = 1 - stats.norm.cdf(z_scores)
                
            elif self.aft_distribution == AFTDistribution.LOGISTIC:
                log_times = np.log(time_points)
                z_scores = (log_times - eta) / self.model_parameters.sigma
                survival_probs = 1 / (1 + np.exp(z_scores))
                
            elif self.aft_distribution == AFTDistribution.EXTREME:
                log_times = np.log(time_points)
                z_scores = (log_times - eta) / self.model_parameters.sigma
                survival_probs = np.exp(-np.exp(z_scores))
                
            else:
                raise ValueError(f"Unsupported AFT distribution: {self.aft_distribution}")
            
            survival_probs = np.clip(survival_probs, 0, 1)
            survival_curves.append(survival_probs)
        
        self.survival_curves = np.array(survival_curves)
        self.time_points = time_points
        
        final_survival = self.survival_curves[:, -1]
        print(f"   Final survival probabilities - Mean: {final_survival.mean():.3f}, Std: {final_survival.std():.3f}")
        
        return self.survival_curves
    
    def calculate_risk_scores(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate mathematically correct risk scores for AFT models"""
        risk_scores = -predictions
        risk_scores = risk_scores - risk_scores.min()
        if risk_scores.max() > 0:
            risk_scores = risk_scores / risk_scores.max()
        
        return risk_scores
    
    def calculate_enhanced_lorenz_curve(self, y_true: np.ndarray, y_pred_raw: np.ndarray) -> float:
        """Calculate Lorenz curve with comprehensive decile analysis"""
        risk_scores = self.calculate_risk_scores(y_pred_raw)
        
        df = pd.DataFrame({
            'event': y_true,
            'predicted_risk': risk_scores,
            'raw_prediction': y_pred_raw
        })
        
        print(f"Enhanced Lorenz Curve Calculation:")
        print(f"   Risk scores - Mean: {risk_scores.mean():.3f}, Std: {risk_scores.std():.3f}")
        
        df = df.sort_values('predicted_risk', ascending=False).reset_index(drop=True)
        
        # Calculate decile analysis
        decile_analysis = self._calculate_decile_analysis(df)
        
        # Standard Lorenz curve calculation
        n_deciles = 10
        decile_size = len(df) // n_deciles
        
        cumulative_events = []
        cumulative_population = []
        
        for i in range(n_deciles):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size if i < n_deciles - 1 else len(df)
            
            events_so_far = df.iloc[:end_idx]['event'].sum()
            population_so_far = end_idx
            
            cumulative_events.append(events_so_far)
            cumulative_population.append(population_so_far)
        
        total_events = df['event'].sum()
        total_population = len(df)
        
        if total_events == 0:
            print("WARNING: No events for Lorenz curve calculation")
            return 0.0
        
        cumulative_events_prop = np.array(cumulative_events) / total_events
        cumulative_population_prop = np.array(cumulative_population) / total_population
        
        x_vals = np.concatenate([[0], cumulative_population_prop])
        y_vals = np.concatenate([[0], cumulative_events_prop])
        
        area_under_curve = np.trapz(y_vals, x_vals)
        gini_coefficient = 2 * area_under_curve - 1
        
        print(f"   Gini coefficient: {gini_coefficient:.4f}")
        
        self.enhanced_results['enhanced_lorenz_curve'] = {
            'cumulative_events': cumulative_events_prop,
            'cumulative_population': cumulative_population_prop,
            'gini_coefficient': gini_coefficient,
            'decile_analysis': decile_analysis
        }
        
        return gini_coefficient
    
    def _calculate_decile_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive decile analysis table"""
        n_deciles = 10
        decile_size = len(df) // n_deciles
        total_events = df['event'].sum()
        total_population = len(df)
        random_capture_rate = 1 / n_deciles
        
        decile_results = []
        
        for i in range(n_deciles):
            start_idx = i * decile_size
            end_idx = (i + 1) * decile_size if i < n_deciles - 1 else len(df)
            
            decile_df = df.iloc[start_idx:end_idx]
            
            decile_population = len(decile_df)
            decile_events = decile_df['event'].sum()
            decile_event_rate = decile_events / decile_population if decile_population > 0 else 0
            
            # Cumulative metrics
            cumulative_events = df.iloc[:end_idx]['event'].sum()
            cumulative_population = end_idx
            cumulative_capture_rate = cumulative_events / total_events if total_events > 0 else 0
            
            # Business metrics
            capture_rate = decile_events / total_events if total_events > 0 else 0
            lift = capture_rate / random_capture_rate if random_capture_rate > 0 else 0
            
            # Risk concentration
            avg_risk_score = decile_df['predicted_risk'].mean()
            risk_concentration = avg_risk_score / df['predicted_risk'].mean() if df['predicted_risk'].mean() > 0 else 0
            
            decile_results.append({
                'decile': i + 1,
                'population': decile_population,
                'population_pct': decile_population / total_population * 100,
                'events': decile_events,
                'event_rate': decile_event_rate,
                'capture_rate': capture_rate,
                'cumulative_capture_rate': cumulative_capture_rate,
                'lift': lift,
                'avg_risk_score': avg_risk_score,
                'risk_concentration': risk_concentration
            })
        
        return pd.DataFrame(decile_results)
    
    def plot_enhanced_lorenz_analysis(self):
        """Plot comprehensive Lorenz analysis with decile table"""
        if 'enhanced_lorenz_curve' not in self.enhanced_results:
            print("Enhanced Lorenz curve not calculated yet")
            return
        
        results = self.enhanced_results['enhanced_lorenz_curve']
        decile_df = results['decile_analysis']
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Lorenz curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(results['cumulative_population'], results['cumulative_events'], 
                'ro-', linewidth=3, markersize=8, label='Model')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        ax1.fill_between(results['cumulative_population'], 
                        results['cumulative_events'],
                        results['cumulative_population'],
                        alpha=0.3, color='blue')
        
        ax1.set_xlabel('Cumulative % Population')
        ax1.set_ylabel('Cumulative % Events')
        ax1.set_title(f'Lorenz Curve\nGini: {results["gini_coefficient"]:.4f}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Capture rate by decile
        ax2 = fig.add_subplot(gs[0, 1])
        bars = ax2.bar(decile_df['decile'], decile_df['capture_rate'] * 100, 
                      color='skyblue', alpha=0.7)
        ax2.axhline(y=10, color='red', linestyle='--', label='Random (10%)')
        ax2.set_xlabel('Decile')
        ax2.set_ylabel('Capture Rate (%)')
        ax2.set_title('Event Capture Rate by Decile')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, decile_df['capture_rate'] * 100):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
        
        # Plot 3: Lift by decile
        ax3 = fig.add_subplot(gs[0, 2])
        bars = ax3.bar(decile_df['decile'], decile_df['lift'], 
                      color='lightgreen', alpha=0.7)
        ax3.axhline(y=1, color='red', linestyle='--', label='Random (1.0)')
        ax3.set_xlabel('Decile')
        ax3.set_ylabel('Lift')
        ax3.set_title('Lift by Decile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, decile_df['lift']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Event rate by decile
        ax4 = fig.add_subplot(gs[1, 0])
        bars = ax4.bar(decile_df['decile'], decile_df['event_rate'] * 100, 
                      color='orange', alpha=0.7)
        overall_event_rate = decile_df['events'].sum() / decile_df['population'].sum() * 100
        ax4.axhline(y=overall_event_rate, color='red', linestyle='--', 
                   label=f'Overall ({overall_event_rate:.1f}%)')
        ax4.set_xlabel('Decile')
        ax4.set_ylabel('Event Rate (%)')
        ax4.set_title('Event Rate by Decile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cumulative capture rate
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.plot(decile_df['decile'], decile_df['cumulative_capture_rate'] * 100, 
                'go-', linewidth=3, markersize=8)
        ax5.plot(decile_df['decile'], decile_df['decile'] * 10, 
                'r--', linewidth=2, label='Random')
        ax5.set_xlabel('Decile')
        ax5.set_ylabel('Cumulative Capture Rate (%)')
        ax5.set_title('Cumulative Event Capture')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Risk concentration
        ax6 = fig.add_subplot(gs[1, 2])
        bars = ax6.bar(decile_df['decile'], decile_df['risk_concentration'], 
                      color='purple', alpha=0.7)
        ax6.axhline(y=1, color='red', linestyle='--', label='Average (1.0)')
        ax6.set_xlabel('Decile')
        ax6.set_ylabel('Risk Concentration')
        ax6.set_title('Risk Score Concentration by Decile')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Table: Decile Analysis
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('tight')
        ax7.axis('off')
        
        table_data = decile_df.copy()
        table_data['population_pct'] = table_data['population_pct'].round(1)
        table_data['event_rate'] = (table_data['event_rate'] * 100).round(1)
        table_data['capture_rate'] = (table_data['capture_rate'] * 100).round(1)
        table_data['cumulative_capture_rate'] = (table_data['cumulative_capture_rate'] * 100).round(1)
        table_data['lift'] = table_data['lift'].round(2)
        table_data['risk_concentration'] = table_data['risk_concentration'].round(2)
        
        table_data.columns = ['Decile', 'Population', 'Pop %', 'Events', 'Event Rate %', 
                             'Capture Rate %', 'Cumulative Capture %', 'Lift', 'Risk Concentration']
        
        table = ax7.table(cellText=table_data.values, colLabels=table_data.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        # Color code the table
        for i in range(len(table_data)):
            if table_data.iloc[i]['Lift'] > 2:
                table[(i+1, 7)].set_facecolor('#90EE90')  # Light green for high lift
            elif table_data.iloc[i]['Lift'] > 1.5:
                table[(i+1, 7)].set_facecolor('#FFFFE0')  # Light yellow for moderate lift
        
        ax7.set_title('Comprehensive Decile Analysis Table', fontsize=14, fontweight='bold', pad=20)
        
        plt.suptitle('Enhanced Lorenz Analysis - Decile Performance Overview', 
                    fontsize=16, fontweight='bold')
        plt.savefig('enhanced_lorenz_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary insights
        print("\nDECILE ANALYSIS INSIGHTS:")
        print("=" * 40)
        top_decile = decile_df.iloc[0]
        print(f"Top decile captures {top_decile['capture_rate']*100:.1f}% of all events")
        print(f"Top decile lift: {top_decile['lift']:.1f}x better than random")
        print(f"Top decile event rate: {top_decile['event_rate']*100:.1f}%")
        
        top_3_deciles = decile_df.head(3)
        print(f"Top 3 deciles capture {top_3_deciles['capture_rate'].sum()*100:.1f}% of all events")
        print(f"Top 3 deciles represent {top_3_deciles['population_pct'].sum():.1f}% of population")
    
    def plot_enhanced_survival_curves(self, n_curves: int = 10) -> None:
        """Plot individual survival curves with enhanced visualization"""
        if self.survival_curves is None:
            print("No survival curves generated yet")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_curves))
        
        for i in range(min(n_curves, len(self.survival_curves))):
            curve = self.survival_curves[i]
            ax1.plot(self.time_points, curve, color=colors[i], linewidth=2, alpha=0.7,
                    label=f'Employee {i+1}' if i < 5 else '')
        
        mean_curve = np.mean(self.survival_curves, axis=0)
        std_curve = np.std(self.survival_curves, axis=0)
        
        ax1.plot(self.time_points, mean_curve, 'k-', linewidth=4, label='Mean')
        ax1.fill_between(self.time_points, 
                        mean_curve - 1.96 * std_curve / np.sqrt(len(self.survival_curves)),
                        mean_curve + 1.96 * std_curve / np.sqrt(len(self.survival_curves)),
                        alpha=0.3, color='gray', label='95% CI')
        
        ax1.set_xlabel('Time (days)', fontsize=12)
        ax1.set_ylabel('Survival Probability', fontsize=12)
        ax1.set_title(f'Individual Survival Curves (n={min(n_curves, len(self.survival_curves))})', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 365)
        ax1.set_ylim(0, 1)
        
        final_survivals = self.survival_curves[:, -1]
        ax2.hist(final_survivals, bins=50, alpha=0.7, density=True, color='skyblue')
        ax2.axvline(final_survivals.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {final_survivals.mean():.3f}')
        ax2.axvline(final_survivals.median(), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {final_survivals.median():.3f}')
        
        ax2.set_xlabel('Final Survival Probability', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        ax2.set_title('Distribution of 1-Year Survival Probabilities', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('enhanced_survival_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Survival Curve Statistics:")
        print(f"   Mean final survival: {final_survivals.mean():.3f}")
        print(f"   Std final survival: {final_survivals.std():.3f}")
        print(f"   Range: [{final_survivals.min():.3f}, {final_survivals.max():.3f}]")
        print(f"   Coefficient of variation: {final_survivals.std() / final_survivals.mean():.3f}")
    
    def validate_enhanced_model(self) -> Dict:
        """Comprehensive model validation"""
        print("\nCOMPREHENSIVE MODEL VALIDATION")
        print("=" * 40)
        
        X_val = self.model_data['X_val']
        y_val = self.model_data['y_val']
        event_val = self.model_data['event_val']
        
        dval = xgb.DMatrix(X_val)
        predictions = self.model.predict(dval)
        
        print("1. PREDICTION DISTRIBUTION:")
        print(f"   Mean: {predictions.mean():.3f}")
        print(f"   Std: {predictions.std():.3f}")
        print(f"   Range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        
        print("\n2. DIRECTIONAL VALIDATION:")
        high_pred_mask = predictions >= np.median(predictions)
        low_pred_mask = predictions < np.median(predictions)
        
        high_survival = y_val[high_pred_mask].mean()
        low_survival = y_val[low_pred_mask].mean()
        high_event_rate = event_val[high_pred_mask].mean()
        low_event_rate = event_val[low_pred_mask].mean()
        
        direction_correct = high_survival > low_survival
        event_direction_correct = high_event_rate < low_event_rate
        
        print(f"   High predictions longer survival: {direction_correct}")
        print(f"   High predictions lower event rate: {event_direction_correct}")
        
        print("\n3. SURVIVAL CURVE VALIDATION:")
        if self.survival_curves is not None:
            final_survival = self.survival_curves[:, -1]
            print(f"   Final survival std: {final_survival.std():.3f}")
            print(f"   Adequate variance: {final_survival.std() > 0.01}")
        
        print("\n4. RISK RANKING VALIDATION:")
        if 'enhanced_lorenz_curve' in self.enhanced_results:
            decile_df = self.enhanced_results['enhanced_lorenz_curve']['decile_analysis']
            top_decile_event_rate = decile_df.iloc[0]['event_rate']
            bottom_decile_event_rate = decile_df.iloc[-1]['event_rate']
            ranking_correct = top_decile_event_rate > bottom_decile_event_rate
            
            print(f"   Top decile event rate: {top_decile_event_rate:.3f}")
            print(f"   Bottom decile event rate: {bottom_decile_event_rate:.3f}")
            print(f"   Risk ranking correct: {ranking_correct}")
        
        return {
            'direction_validation': {
                'survival_direction_correct': direction_correct,
                'event_direction_correct': event_direction_correct
            },
            'survival_curve_variance': final_survival.std() if self.survival_curves is not None else None,
            'model_approach': 'AFT'
        }
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive enhanced survival analysis"""
        print("RUNNING COMPREHENSIVE ENHANCED SURVIVAL ANALYSIS")
        print("=" * 60)
        
        print("\n1. Baseline Analysis...")
        self.analyze_baseline()
        
        print("\n2. Industry Analysis...")
        self.analyze_industries()
        
        print("\n3. Demographic Analysis...")
        self.analyze_demographics()
        
        print("\n4. Temporal Trends Analysis...")
        self.analyze_temporal_trends()
        
        print("\n5. Enhanced Predictive Modeling...")
        self.build_enhanced_predictive_model()
        
        print("\n6. Generating Survival Curves...")
        X_val = self.model_data['X_val']
        self.generate_survival_curves(X_val)
        self.plot_enhanced_survival_curves()
        
        print("\n7. Enhanced Risk Assessment...")
        y_val = self.model_data['y_val']
        event_val = self.model_data['event_val']
        raw_predictions = self.model.predict(xgb.DMatrix(X_val))
        
        gini = self.calculate_enhanced_lorenz_curve(event_val, raw_predictions)
        self.plot_enhanced_lorenz_analysis()
        
        print("\n8. Model Validation...")
        validation_results = self.validate_enhanced_model()
        
        print("\n9. Generating Summary...")
        summary = self.generate_comprehensive_summary()
        
        print("\n" + "="*60)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*60)
        
        return {
            'insights': self.insights,
            'enhanced_results': self.enhanced_results,
            'validation_results': validation_results,
            'summary': summary,
            'model_parameters': self.model_parameters
        }
    
    def generate_comprehensive_summary(self) -> Dict:
        """Generate comprehensive summary"""
        baseline = self.insights.get("baseline", {})
        model_results = self.insights.get("enhanced_model", {})
        
        summary = {
            "population_size": baseline.get('population_size', 'N/A'),
            "retention_365d": baseline.get('retention_365d', 'N/A'),
            "model_c_index": model_results.get('c_index_val', 'N/A'),
            "aft_distribution": self.aft_distribution.value,
            "scale_parameter": self.model_parameters.sigma if self.model_parameters else 'N/A',
            "gini_coefficient": self.enhanced_results.get('enhanced_lorenz_curve', {}).get('gini_coefficient', 'N/A')
        }
        
        print("\nCOMPREHENSIVE SUMMARY:")
        print("=" * 30)
        print(f"Population: {summary['population_size']}")
        print(f"1-year retention: {summary['retention_365d']}")
        print(f"Model C-index: {summary['model_c_index']}")
        print(f"AFT distribution: {summary['aft_distribution']}")
        print(f"Scale parameter: {summary['scale_parameter']}")
        print(f"Gini coefficient: {summary['gini_coefficient']}")
        
        return summary

def run_analysis_demo():
    """Demonstration of enhanced analysis capabilities"""
    print("ENHANCED SURVIVAL ANALYSIS CAPABILITIES")
    print("=" * 50)
    
    capabilities = [
        "Comprehensive Kaplan-Meier analysis (baseline, industry, demographics, temporal)",
        "Proper AFT distribution handling (normal/logistic/extreme)",
        "Mathematically correct risk scoring",
        "Comprehensive decile analysis with business metrics",
        "Enhanced survival curve generation",
        "Robust model validation framework",
        "Creative Lorenz curve visualizations",
        "Distribution flexibility for different data patterns"
    ]
    
    print("\nKEY CAPABILITIES:")
    for cap in capabilities:
        print(f"   - {cap}")
    
    print("\nUSAGE EXAMPLE:")
    print("   analyzer = EnhancedSurvivalAnalysis(data, AFTDistribution.NORMAL)")
    print("   results = analyzer.run_comprehensive_analysis()")
    
    return "Enhanced analysis ready for deployment"

if __name__ == "__main__":
    demo_result = run_analysis_demo()
    print(f"\nResult: {demo_result}")
