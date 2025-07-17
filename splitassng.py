# Enhanced XGBoost AFT Survival Analysis - Expert Implementation with Complete KM Analysis
# Fixed feature preprocessing and comprehensive Kaplan-Meier insights

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from scipy import stats
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
    feature_stats: Dict

class FeatureProcessor:
    """Expert-level feature preprocessing with proper scaling validation"""
    
    def __init__(self):
        self.scaler = None
        self.encoders = {}
        self.feature_stats = {}
        self.processed_features = []
        
    def fit_transform_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Fixed feature preprocessing with proper validation"""
        
        train_processed = train_df.copy()
        val_processed = val_df.copy()
        
        # Define feature categories
        continuous_features = []
        categorical_features = []
        
        # 1. Log-transform highly skewed features
        log_features = ['baseline_salary', 'team_avg_comp', 'tenure_at_vantage_days', 'salary_growth_ratio']
        for feature in log_features:
            if feature in train_df.columns:
                train_vals = np.maximum(train_df[feature].fillna(1), 1)
                val_vals = np.maximum(val_df[feature].fillna(1), 1)
                
                train_processed[f'{feature}_log'] = np.log(train_vals)
                val_processed[f'{feature}_log'] = np.log(val_vals)
                continuous_features.append(f'{feature}_log')
                
                print(f"   {feature}_log: mean={train_processed[f'{feature}_log'].mean():.3f}, std={train_processed[f'{feature}_log'].std():.3f}")
        
        # 2. Add other continuous features
        other_continuous = ['age', 'team_size', 'manager_changes_count']
        for feature in other_continuous:
            if feature in train_df.columns:
                train_processed[feature] = train_df[feature].fillna(train_df[feature].median())
                val_processed[feature] = val_df[feature].fillna(train_df[feature].median())
                continuous_features.append(feature)
                
                print(f"   {feature}: mean={train_processed[feature].mean():.3f}, std={train_processed[feature].std():.3f}")
        
        # 3. Process categorical features
        cat_features = ['gender_cd', 'pay_rt_type_cd', 'full_tm_part_tm_cd', 'fscl_actv_ind', 'naics_2digit']
        for feature in cat_features:
            if feature in train_df.columns:
                train_cats = train_processed[feature].fillna('MISSING').astype(str)
                val_cats = val_processed[feature].fillna('MISSING').astype(str)
                
                all_categories = sorted(set(train_cats.unique()) | set(val_cats.unique()))
                category_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
                
                encoded_feature = f'{feature}_encoded'
                train_processed[encoded_feature] = train_cats.map(category_mapping)
                val_processed[encoded_feature] = val_cats.map(category_mapping)
                categorical_features.append(encoded_feature)
                
                self.encoders[feature] = category_mapping
        
        # 4. CRITICAL: Apply StandardScaler to ALL features (continuous + categorical)
        all_features = continuous_features + categorical_features
        
        print(f"\nFeature preprocessing before scaling:")
        print(f"   Continuous features: {len(continuous_features)}")
        print(f"   Categorical features: {len(categorical_features)}")
        print(f"   Total features: {len(all_features)}")
        
        if all_features:
            # Initialize and fit scaler
            self.scaler = StandardScaler()
            
            # Prepare data for scaling
            train_feature_data = train_processed[all_features].fillna(0)
            val_feature_data = val_processed[all_features].fillna(0)
            
            # Fit scaler on training data
            self.scaler.fit(train_feature_data)
            
            # Transform both datasets
            train_scaled = self.scaler.transform(train_feature_data)
            val_scaled = self.scaler.transform(val_feature_data)
            
            # Replace original features with scaled versions
            train_processed[all_features] = train_scaled
            val_processed[all_features] = val_scaled
            
            # VALIDATION: Check scaling worked
            print(f"\nScaling validation:")
            for i, feature in enumerate(all_features):
                mean_val = train_processed[feature].mean()
                std_val = train_processed[feature].std()
                print(f"   {feature}: mean={mean_val:.6f}, std={std_val:.6f}")
                
                if abs(mean_val) > 1e-10:
                    print(f"   WARNING: {feature} mean not close to 0!")
                if abs(std_val - 1.0) > 0.1:
                    print(f"   WARNING: {feature} std not close to 1!")
        
        self.processed_features = all_features
        return train_processed, val_processed, all_features

class EnhancedSurvivalAnalysis:
    """Expert-level XGBoost AFT with complete Kaplan-Meier analysis"""
    
    def __init__(self, data: pd.DataFrame, aft_distribution: AFTDistribution = AFTDistribution.NORMAL):
        self.data = self._prepare_data(data)
        self.aft_distribution = aft_distribution
        self.kmf = KaplanMeierFitter()
        self.insights = {}
        self.enhanced_results = {}
        self.model_parameters = None
        self.survival_curves = None
        self.time_points = None
        self.feature_processor = FeatureProcessor()
        
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare analysis dataset with business segments"""
        df = df.copy()
        
        df["naics_2digit"] = df["naics_cd"].astype(str).str[:2]
        
        vantage_dt = pd.to_datetime(df["vantage_date"], errors='coerce')
        birth_dt = pd.to_datetime(df["birth_dt"], errors='coerce')
        birth_dt = birth_dt.where(birth_dt.dt.year.between(1900, 2020), pd.NaT)
        df['age'] = (vantage_dt - birth_dt).dt.days / 365.25
        
        df["age_group"] = pd.cut(
            df["age"], bins=[0, 25, 35, 45, 55, 65, np.inf],
            labels=["<25", "25-35", "35-45", "45-55", "55-65", "65+"], include_lowest=True
        )
        
        df["tenure_at_vantage_days"] = np.clip(df["tenure_at_vantage_days"], 0, 365 * 50)
        tenure_years = df["tenure_at_vantage_days"] / 365.25
        
        df["tenure_group"] = pd.cut(
            tenure_years, bins=[0, 0.5, 1, 2, 3, 5, np.inf],
            labels=["<6mo", "6mo-1yr", "1-2yr", "2-3yr", "3-5yr", "5yr+"], include_lowest=True
        )
        
        df["salary_group"] = pd.cut(
            df["baseline_salary"], bins=[0, 40000, 60000, 80000, 120000, 200000, np.inf],
            labels=["<40K", "40-60K", "60-80K", "80-120K", "120-200K", "200K+"], include_lowest=True
        )
        
        return df
    
    def analyze_baseline(self) -> Dict:
        """Generate population baseline with key business metrics"""
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
        """Analyze retention by industry (NAICS 2-digit)"""
        print("Analyzing industry retention patterns...")
        
        train_data = self.data[self.data["dataset_split"].isin(["train", "val"])]
        
        # Get top industries by volume
        industry_counts = train_data["naics_2digit"].value_counts()
        top_industries = industry_counts.head(top_n).index.tolist()
        
        # Filter for meaningful sample sizes
        valid_industries = [industry for industry in top_industries if industry_counts[industry] >= 1000]
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
            
            kmf.plot_survival_function(ax=ax, ci_show=False, color=colors[i],
                                     label=f"NAICS {industry} (n={len(subset):,})")
        
        ax.set_title(f"Retention by Industry - Top {len(valid_industries)} Industries",
                    fontsize=17, fontweight="bold")
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("industry_survival.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        ranked = sorted(industry_metrics.items(), key=lambda x: x[1]["retention_365d"], reverse=True)
        
        print(f"INDUSTRY PERFORMANCE RANKING:")
        print("Top Performers:")
        for i, (industry, metrics) in enumerate(ranked[:3]):
            print(f"   {i+1}. NAICS {industry}: {metrics['retention_365d']:.1%} retention (n={metrics['sample_size']:,})")
        
        self.insights["industry"] = {"metrics": industry_metrics, "ranked": ranked}
        return industry_metrics
    
    def analyze_demographics(self) -> Dict:
        """Analyze retention by key demographic segments"""
        print("Analyzing demographic retention patterns...")
        
        train_data = self.data[self.data["dataset_split"].isin(["train", "val"])]
        
        demographics = {
            "age_group": "Age Segments",
            "tenure_group": "Tenure Segments", 
            "salary_group": "Salary Segments",
        }
        
        demo_insights = {}
        
        for demo_col, title in demographics.items():
            if demo_col not in train_data.columns:
                continue
                
            demo_data = train_data[train_data[demo_col].notna()]
            
            fig, ax = plt.subplots(figsize=(15, 10))
            category_metrics = {}
            
            categories = (demo_data[demo_col].cat.categories if hasattr(demo_data[demo_col], "cat") 
                         else demo_data[demo_col].unique())
            
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
                
                kmf.plot_survival_function(ax=ax, ci_show=False, 
                                         label=f"{category} (n={len(subset):,})")
            
            ax.set_title(f"Retention by {title}", fontsize=17, fontweight="bold")
            ax.set_xlabel("Days Since Assignment Start", fontsize=14)
            ax.set_ylabel("Survival Probability", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{demo_col}_survival.png", dpi=300, bbox_inches="tight")
            plt.show()
            
            demo_insights[demo_col] = category_metrics
        
        self.insights["demographics"] = demo_insights
        return demo_insights
    
    def analyze_temporal_trends(self) -> Dict:
        """Compare retention trends: 2023 vs 2024"""
        print("Analyzing temporal retention trends...")
        
        cohort_2023 = self.data[self.data["dataset_split"].isin(["train", "val"])]
        cohort_2024 = self.data[self.data["dataset_split"] == "oot"]
        
        if len(cohort_2024) < 1000:
            print("Insufficient 2024 data for temporal analysis")
            return {}
        
        fig, ax = plt.subplots(figsize=(15, 10))
        
        kmf_2023 = KaplanMeierFitter()
        kmf_2023.fit(cohort_2023["survival_time_days"], cohort_2023["event_indicator"])
        
        kmf_2024 = KaplanMeierFitter()
        kmf_2024.fit(cohort_2024["survival_time_days"], cohort_2024["event_indicator"])
        
        kmf_2023.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color="blue",
                                       label=f"2023 Cohort (n={len(cohort_2023):,})")
        kmf_2024.plot_survival_function(ax=ax, ci_show=False, linewidth=3, color="red",
                                       label=f"2024 Cohort (n={len(cohort_2024):,})")
        
        ax.set_title("Temporal Retention Trends: 2023 vs 2024", fontsize=17, fontweight="bold")
        ax.set_xlabel("Days Since Assignment Start", fontsize=14)
        ax.set_ylabel("Survival Probability", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("temporal_trends.png", dpi=300, bbox_inches="tight")
        plt.show()
        
        trend_metrics = {
            "2023_size": len(cohort_2023),
            "2024_size": len(cohort_2024),
            "2023_retention_90d": kmf_2023.survival_function_at_times(90).iloc[0],
            "2024_retention_90d": kmf_2024.survival_function_at_times(90).iloc[0],
            "2023_retention_365d": kmf_2023.survival_function_at_times(365).iloc[0],
            "2024_retention_365d": kmf_2024.survival_function_at_times(365).iloc[0],
        }
        
        self.insights["temporal"] = trend_metrics
        return trend_metrics
    
    def build_enhanced_predictive_model(self) -> Dict:
        """Build enhanced XGBoost AFT model with fixed preprocessing"""
        print("Building enhanced XGBoost AFT model with fixed preprocessing...")
        
        train_data = self.data[self.data['dataset_split'] == 'train']
        val_data = self.data[self.data['dataset_split'] == 'val']
        
        if len(val_data) < 1000:
            print("Insufficient validation data - using random split from train")
            np.random.seed(42)
            train_idx = np.random.choice(len(train_data), size=int(0.8 * len(train_data)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(train_data)), train_idx)
            val_data = train_data.iloc[val_idx]
            train_data = train_data.iloc[train_idx]
        
        # Fixed feature preprocessing
        train_processed, val_processed, feature_columns = self.feature_processor.fit_transform_features(
            train_data, val_data
        )
        
        # Prepare model datasets
        model_columns = feature_columns + ['survival_time_days', 'event_indicator']
        train_model_data = train_processed[model_columns].dropna()
        val_model_data = val_processed[model_columns].dropna()
        
        X_train = train_model_data[feature_columns]
        y_train = train_model_data['survival_time_days']
        event_train = train_model_data['event_indicator']
        
        X_val = val_model_data[feature_columns]
        y_val = val_model_data['survival_time_days']
        event_val = val_model_data['event_indicator']
        
        # Final validation check
        print(f"\nFinal feature validation:")
        print(f"   Feature matrix shape: {X_train.shape}")
        print(f"   Overall feature mean: {X_train.mean().mean():.6f}")
        print(f"   Overall feature std: {X_train.std().mean():.6f}")
        print(f"   Target range: [{y_train.min():.1f}, {y_train.max():.1f}] days")
        print(f"   Log target range: [{np.log(y_train.min()):.3f}, {np.log(y_train.max()):.3f}]")
        
        # Train AFT model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        dtrain.set_float_info('label_lower_bound', y_train.values)
        dtrain.set_float_info('label_upper_bound', y_train.values)
        dval.set_float_info('label_lower_bound', y_val.values)
        dval.set_float_info('label_upper_bound', y_val.values)
        
        # Conservative hyperparameters
        params = {
            'objective': 'survival:aft',
            'aft_loss_distribution': self.aft_distribution.value,
            'max_depth': 3,      # Very conservative
            'eta': 0.01,         # Very low learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1.0,    # High regularization
            'reg_lambda': 5.0,   # High regularization
            'seed': 42,
            'verbosity': 0
        }
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(params, dtrain, num_boost_round=200, evals=evals,
                         early_stopping_rounds=20, verbose_eval=False)
        
        self.model = model
        self.model_data = {
            'X_train': X_train, 'y_train': y_train, 'event_train': event_train,
            'X_val': X_val, 'y_val': y_val, 'event_val': event_val,
            'feature_columns': feature_columns
        }
        
        # AFT parameter estimation
        self.model_parameters = self._estimate_aft_parameters_expert()
        
        # Performance metrics
        train_pred = model.predict(dtrain)
        val_pred = model.predict(dval)
        
        c_index_train = concordance_index(y_train, train_pred, event_train)
        c_index_val = concordance_index(y_val, val_pred, event_val)
        
        performance_metrics = {
            'train_size': len(train_model_data),
            'val_size': len(val_model_data),
            'c_index_train': c_index_train,
            'c_index_val': c_index_val,
            'prediction_stats': {
                'train_pred_mean': train_pred.mean(),
                'train_pred_std': train_pred.std(),
                'val_pred_mean': val_pred.mean(),
                'val_pred_std': val_pred.std(),
            }
        }
        
        print(f"\nFIXED MODEL PERFORMANCE:")
        print(f"   C-index (validation): {c_index_val:.3f}")
        print(f"   Prediction mean: {val_pred.mean():.3f} (log-scale)")
        print(f"   Prediction std: {val_pred.std():.3f} (log-scale)")
        print(f"   Expected survival range: [{np.exp(val_pred.min()):.1f}, {np.exp(val_pred.max()):.1f}] days")
        
        self.insights['enhanced_model'] = performance_metrics
        return performance_metrics
    
    def _estimate_aft_parameters_expert(self) -> AFTParameters:
        """Expert AFT parameter estimation with proper diagnostics"""
        X_train = self.model_data['X_train']
        y_train = self.model_data['y_train']
        
        dtrain = xgb.DMatrix(X_train)
        eta_predictions = self.model.predict(dtrain)
        
        log_actual = np.log(y_train)
        log_residuals = log_actual - eta_predictions
        
        # Distribution-specific sigma
        if self.aft_distribution == AFTDistribution.NORMAL:
            sigma = np.std(log_residuals)
        elif self.aft_distribution == AFTDistribution.LOGISTIC:
            sigma = np.std(log_residuals) * np.sqrt(3) / np.pi
        elif self.aft_distribution == AFTDistribution.EXTREME:
            sigma = np.std(log_residuals) * np.sqrt(6) / np.pi
        else:
            sigma = np.std(log_residuals)
        
        print(f"\nExpert AFT Parameter Estimation:")
        print(f"   Prediction range: [{eta_predictions.min():.3f}, {eta_predictions.max():.3f}]")
        print(f"   Residual std: {log_residuals.std():.3f}")
        print(f"   Estimated sigma: {sigma:.3f}")
        
        return AFTParameters(
            eta=eta_predictions,
            sigma=sigma,
            distribution=self.aft_distribution,
            feature_stats=self.feature_processor.feature_stats
        )
    
    def generate_survival_curves(self, X_test: pd.DataFrame, time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate survival curves from AFT model"""
        if time_points is None:
            time_points = np.arange(1, 366, 1)
        
        if self.model_parameters is None:
            raise ValueError("Model parameters not estimated")
        
        dtest = xgb.DMatrix(X_test)
        eta_predictions = self.model.predict(dtest)
        
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
            
            survival_probs = np.clip(survival_probs, 1e-6, 1.0 - 1e-6)
            survival_curves.append(survival_probs)
        
        self.survival_curves = np.array(survival_curves)
        self.time_points = time_points
        
        final_survival = self.survival_curves[:, -1]
        print(f"\nSurvival curve statistics:")
        print(f"   Final survival (365d): {final_survival.mean():.3f} ± {final_survival.std():.3f}")
        
        return self.survival_curves
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run complete analysis with all Kaplan-Meier components"""
        print("EXPERT COMPREHENSIVE SURVIVAL ANALYSIS")
        print("=" * 50)
        
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
        
        summary = {
            'insights': self.insights,
            'enhanced_results': self.enhanced_results,
            'model_parameters': self.model_parameters
        }
        
        return summary

# Usage
if __name__ == "__main__":
    print("Expert XGBoost AFT with Complete Kaplan-Meier Analysis - Ready")
