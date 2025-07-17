# Enhanced XGBoost AFT Survival Analysis - Expert Implementation
# Proper preprocessing pipeline with principled AFT parameter estimation

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
    scaler: StandardScaler
    feature_stats: Dict

class FeatureProcessor:
    """Expert-level feature preprocessing for survival analysis"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    def identify_feature_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Intelligently categorize features by type and required preprocessing"""
        
        # Features requiring log transformation (highly skewed, multiplicative effects)
        log_transform_features = [
            'baseline_salary', 'team_avg_comp', 'tenure_at_vantage_days',
            'salary_growth_ratio', 'manager_changes_count'
        ]
        
        # Features requiring standard scaling (continuous, various scales)
        standard_scale_features = [
            'age', 'team_size'
        ]
        
        # Categorical features (proper encoding needed)
        categorical_features = [
            'gender_cd', 'pay_rt_type_cd', 'full_tm_part_tm_cd', 
            'fscl_actv_ind', 'naics_2digit'
        ]
        
        # Filter to only include features that exist in the dataset
        available_features = {
            'log_transform': [f for f in log_transform_features if f in df.columns],
            'standard_scale': [f for f in standard_scale_features if f in df.columns],
            'categorical': [f for f in categorical_features if f in df.columns]
        }
        
        return available_features
    
    def fit_transform_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """Fit preprocessing pipeline on training data and transform both sets"""
        
        train_processed = train_df.copy()
        val_processed = val_df.copy()
        
        feature_types = self.identify_feature_types(train_df)
        
        # 1. Log transformation for skewed features
        for feature in feature_types['log_transform']:
            if feature in train_df.columns:
                # Add small constant to handle zeros, then log transform
                train_values = train_df[feature].fillna(1)
                val_values = val_df[feature].fillna(1)
                
                # Ensure positive values
                train_values = np.maximum(train_values, 1)
                val_values = np.maximum(val_values, 1)
                
                train_processed[f'{feature}_log'] = np.log(train_values)
                val_processed[f'{feature}_log'] = np.log(val_values)
                
                # Store stats for validation
                self.feature_stats[f'{feature}_log'] = {
                    'mean': train_processed[f'{feature}_log'].mean(),
                    'std': train_processed[f'{feature}_log'].std(),
                    'min': train_processed[f'{feature}_log'].min(),
                    'max': train_processed[f'{feature}_log'].max()
                }
        
        # 2. Standard scaling for continuous features
        continuous_features = (
            [f'{f}_log' for f in feature_types['log_transform'] if f in train_df.columns] +
            feature_types['standard_scale']
        )
        
        if continuous_features:
            scaler = StandardScaler()
            
            # Fit on training data
            train_continuous = train_processed[continuous_features].fillna(0)
            scaler.fit(train_continuous)
            
            # Transform both sets
            train_processed[continuous_features] = scaler.transform(train_continuous)
            val_continuous = val_processed[continuous_features].fillna(0)
            val_processed[continuous_features] = scaler.transform(val_continuous)
            
            self.scalers['continuous'] = scaler
        
        # 3. Categorical encoding
        for feature in feature_types['categorical']:
            if feature in train_df.columns:
                # Combine train and validation to get all categories
                train_cats = train_processed[feature].fillna('MISSING').astype(str)
                val_cats = val_processed[feature].fillna('MISSING').astype(str)
                
                all_categories = sorted(set(train_cats.unique()) | set(val_cats.unique()))
                category_mapping = {cat: idx for idx, cat in enumerate(all_categories)}
                
                train_processed[f'{feature}_encoded'] = train_cats.map(category_mapping)
                val_processed[f'{feature}_encoded'] = val_cats.map(category_mapping)
                
                self.encoders[feature] = category_mapping
        
        # 4. Create final feature list
        final_features = (
            continuous_features +
            [f'{f}_encoded' for f in feature_types['categorical'] if f in train_df.columns]
        )
        
        # 5. Validation and quality checks
        print("Feature Preprocessing Summary:")
        print(f"   Log-transformed features: {len([f for f in feature_types['log_transform'] if f in train_df.columns])}")
        print(f"   Standard-scaled features: {len(continuous_features)}")
        print(f"   Categorical features: {len(feature_types['categorical'])}")
        print(f"   Final feature count: {len(final_features)}")
        
        # Check for any remaining issues
        train_final = train_processed[final_features]
        val_final = val_processed[final_features]
        
        for col in final_features:
            if train_final[col].isnull().any() or val_final[col].isnull().any():
                print(f"   WARNING: {col} still has null values")
            
            train_std = train_final[col].std()
            if train_std < 1e-6:
                print(f"   WARNING: {col} has very low variance: {train_std}")
        
        return train_processed, val_processed, final_features

class EnhancedSurvivalAnalysis:
    """
    Expert-level XGBoost AFT survival analysis with principled preprocessing
    """
    
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
        
        self.insights["baseline"] = metrics
        return metrics
    
    def build_enhanced_predictive_model(self) -> Dict:
        """Build enhanced XGBoost AFT model with expert preprocessing"""
        print("Building enhanced XGBoost AFT model with principled preprocessing...")
        
        train_data = self.data[self.data['dataset_split'] == 'train']
        val_data = self.data[self.data['dataset_split'] == 'val']
        
        if len(val_data) < 1000:
            print("Insufficient validation data - using random split from train")
            np.random.seed(42)
            train_idx = np.random.choice(len(train_data), size=int(0.8 * len(train_data)), replace=False)
            val_idx = np.setdiff1d(np.arange(len(train_data)), train_idx)
            val_data = train_data.iloc[val_idx]
            train_data = train_data.iloc[train_idx]
        
        # Expert feature preprocessing
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
        
        # Validate feature preprocessing
        print(f"Feature preprocessing validation:")
        print(f"   Training features shape: {X_train.shape}")
        print(f"   Feature means: {X_train.mean().abs().max():.3f} (should be ~0)")
        print(f"   Feature stds: {X_train.std().mean():.3f} (should be ~1)")
        
        # Train AFT model with expert hyperparameters
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        dtrain.set_float_info('label_lower_bound', y_train.values)
        dtrain.set_float_info('label_upper_bound', y_train.values)
        dval.set_float_info('label_lower_bound', y_val.values)
        dval.set_float_info('label_upper_bound', y_val.values)
        
        # Expert hyperparameter configuration
        params = {
            'objective': 'survival:aft',
            'aft_loss_distribution': self.aft_distribution.value,
            'max_depth': 4,  # Reduced for survival data
            'eta': 0.03,     # Much lower learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.5,    # Increased L1 regularization
            'reg_lambda': 2.0,   # Increased L2 regularization
            'seed': 42,
            'verbosity': 0
        }
        
        evals = [(dtrain, 'train'), (dval, 'val')]
        model = xgb.train(
            params, dtrain, 
            num_boost_round=500,  # Reduced iterations
            evals=evals,
            early_stopping_rounds=30,
            verbose_eval=False
        )
        
        self.model = model
        self.model_data = {
            'X_train': X_train, 'y_train': y_train, 'event_train': event_train,
            'X_val': X_val, 'y_val': y_val, 'event_val': event_val,
            'feature_columns': feature_columns
        }
        
        # Expert AFT parameter estimation
        self.model_parameters = self._estimate_aft_parameters_expert()
        
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
        
        performance_metrics = {
            'train_size': len(train_model_data),
            'val_size': len(val_model_data),
            'c_index_train': c_index_train,
            'c_index_val': c_index_val,
            'feature_importance': importance_df,
            'aft_distribution': self.aft_distribution.value,
            'prediction_stats': {
                'train_pred_mean': train_pred.mean(),
                'train_pred_std': train_pred.std(),
                'val_pred_mean': val_pred.mean(),
                'val_pred_std': val_pred.std(),
            }
        }
        
        self._plot_feature_importance(importance_df)
        
        print(f"EXPERT MODEL PERFORMANCE:")
        print(f"   Training size: {len(train_model_data):,}")
        print(f"   Validation size: {len(val_model_data):,}")
        print(f"   C-index (validation): {c_index_val:.3f}")
        print(f"   Prediction mean: {val_pred.mean():.3f} (log-scale)")
        print(f"   Prediction std: {val_pred.std():.3f} (log-scale)")
        print(f"   Scale parameter: {self.model_parameters.sigma:.3f}")
        
        self.insights['enhanced_model'] = performance_metrics
        return performance_metrics
    
    def _estimate_aft_parameters_expert(self) -> AFTParameters:
        """Expert AFT parameter estimation - clean and principled"""
        X_train = self.model_data['X_train']
        y_train = self.model_data['y_train']
        
        dtrain = xgb.DMatrix(X_train)
        eta_predictions = self.model.predict(dtrain)
        
        # Natural log-space residuals
        log_actual = np.log(y_train)
        log_residuals = log_actual - eta_predictions
        
        # Distribution-specific scale parameter estimation
        if self.aft_distribution == AFTDistribution.NORMAL:
            sigma = np.std(log_residuals)
        elif self.aft_distribution == AFTDistribution.LOGISTIC:
            sigma = np.std(log_residuals) * np.sqrt(3) / np.pi
        elif self.aft_distribution == AFTDistribution.EXTREME:
            sigma = np.std(log_residuals) * np.sqrt(6) / np.pi
        else:
            sigma = np.std(log_residuals)
        
        print(f"Expert AFT Parameter Estimation:")
        print(f"   Prediction range: [{eta_predictions.min():.3f}, {eta_predictions.max():.3f}]")
        print(f"   Residual std: {log_residuals.std():.3f}")
        print(f"   Estimated sigma: {sigma:.3f}")
        
        return AFTParameters(
            eta=eta_predictions,
            sigma=sigma,
            distribution=self.aft_distribution,
            scaler=self.feature_processor.scalers.get('continuous'),
            feature_stats=self.feature_processor.feature_stats
        )
    
    def _plot_feature_importance(self, importance_df: pd.DataFrame):
        """Plot feature importance"""
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(data=importance_df.head(15), x='importance', y='feature', ax=ax)
        ax.set_title('Feature Importance - Expert XGBoost AFT Model', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        plt.tight_layout()
        plt.savefig('expert_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_survival_curves(self, X_test: pd.DataFrame, time_points: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate mathematically correct survival curves"""
        if time_points is None:
            time_points = np.arange(1, 366, 1)
        
        if self.model_parameters is None:
            raise ValueError("Model parameters not estimated. Run build_enhanced_predictive_model() first.")
        
        dtest = xgb.DMatrix(X_test)
        eta_predictions = self.model.predict(dtest)
        
        print(f"Expert Survival Curve Generation:")
        print(f"   Predictions (η) - Mean: {eta_predictions.mean():.3f}, Std: {eta_predictions.std():.3f}")
        print(f"   Expected survival times: [{np.exp(eta_predictions.min()):.1f}, {np.exp(eta_predictions.max()):.1f}] days")
        
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
            
            survival_probs = np.clip(survival_probs, 1e-6, 1.0 - 1e-6)
            survival_curves.append(survival_probs)
        
        self.survival_curves = np.array(survival_curves)
        self.time_points = time_points
        
        # Natural diagnostics
        final_survival = self.survival_curves[:, -1]
        initial_survival = self.survival_curves[:, 0]
        
        print(f"   Survival Curve Statistics:")
        print(f"     Final survival (365d) - Mean: {final_survival.mean():.3f}, Std: {final_survival.std():.3f}")
        print(f"     Survival decline: {(initial_survival.mean() - final_survival.mean()):.3f}")
        
        return self.survival_curves
    
    def calculate_risk_scores(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate risk scores from AFT predictions"""
        # Higher eta = longer survival = lower risk
        risk_scores = -predictions
        
        # Normalize to [0, 1] range
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
            return 0.0
        
        cumulative_events_prop = np.array(cumulative_events) / total_events
        cumulative_population_prop = np.array(cumulative_population) / total_population
        
        x_vals = np.concatenate([[0], cumulative_population_prop])
        y_vals = np.concatenate([[0], cumulative_events_prop])
        
        area_under_curve = np.trapz(y_vals, x_vals)
        gini_coefficient = 2 * area_under_curve - 1
        
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
            
            cumulative_events = df.iloc[:end_idx]['event'].sum()
            cumulative_population = end_idx
            cumulative_capture_rate = cumulative_events / total_events if total_events > 0 else 0
            
            capture_rate = decile_events / total_events if total_events > 0 else 0
            lift = capture_rate / random_capture_rate if random_capture_rate > 0 else 0
            
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
    
    def validate_enhanced_model(self) -> Dict:
        """Comprehensive model validation"""
        print("\nEXPERT MODEL VALIDATION")
        print("=" * 30)
        
        X_val = self.model_data['X_val']
        y_val = self.model_data['y_val']
        event_val = self.model_data['event_val']
        
        dval = xgb.DMatrix(X_val)
        predictions = self.model.predict(dval)
        
        # Core validation metrics
        from scipy.stats import pearsonr, spearmanr
        corr_pearson, p_pearson = pearsonr(predictions, y_val)
        corr_spearman, p_spearman = spearmanr(predictions, y_val)
        
        # Directional validation
        high_pred_mask = predictions >= np.median(predictions)
        low_pred_mask = predictions < np.median(predictions)
        
        high_survival = y_val[high_pred_mask].mean()
        low_survival = y_val[low_pred_mask].mean()
        high_event_rate = event_val[high_pred_mask].mean()
        low_event_rate = event_val[low_pred_mask].mean()
        
        direction_correct = high_survival > low_survival
        event_direction_correct = high_event_rate < low_event_rate
        
        print(f"Prediction Statistics:")
        print(f"   Mean: {predictions.mean():.3f}, Std: {predictions.std():.3f}")
        print(f"   Correlation with actual: {corr_pearson:.3f}")
        print(f"   Directional validation: {direction_correct}")
        print(f"   Event direction: {event_direction_correct}")
        
        # Overall assessment
        issues = []
        if predictions.std() < 0.1:
            issues.append("Low prediction variance")
        if not direction_correct:
            issues.append("Incorrect directional relationship")
        if abs(corr_pearson) < 0.2:
            issues.append("Low correlation with actual survival")
        
        print(f"\nValidation Status: {'PASS' if not issues else 'ISSUES: ' + ', '.join(issues)}")
        
        return {
            'prediction_stats': {
                'mean': predictions.mean(),
                'std': predictions.std(),
                'correlation_pearson': corr_pearson,
                'correlation_spearman': corr_spearman
            },
            'direction_validation': {
                'survival_direction_correct': direction_correct,
                'event_direction_correct': event_direction_correct
            },
            'issues': issues,
            'overall_status': len(issues) == 0
        }
    
    def run_comprehensive_analysis(self) -> Dict:
        """Run comprehensive enhanced survival analysis"""
        print("EXPERT COMPREHENSIVE SURVIVAL ANALYSIS")
        print("=" * 50)
        
        print("\n1. Baseline Analysis...")
        self.analyze_baseline()
        
        print("\n2. Enhanced Predictive Modeling...")
        self.build_enhanced_predictive_model()
        
        print("\n3. Generating Survival Curves...")
        X_val = self.model_data['X_val']
        self.generate_survival_curves(X_val)
        
        print("\n4. Risk Assessment...")
        y_val = self.model_data['y_val']
        event_val = self.model_data['event_val']
        raw_predictions = self.model.predict(xgb.DMatrix(X_val))
        
        gini = self.calculate_enhanced_lorenz_curve(event_val, raw_predictions)
        
        print("\n5. Model Validation...")
        validation_results = self.validate_enhanced_model()
        
        print("\n6. Summary...")
        summary = self.generate_comprehensive_summary()
        
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
        
        print("\nEXPERT SUMMARY:")
        print("=" * 20)
        print(f"Population: {summary['population_size']}")
        print(f"1-year retention: {summary['retention_365d']}")
        print(f"Model C-index: {summary['model_c_index']}")
        print(f"AFT distribution: {summary['aft_distribution']}")
        print(f"Scale parameter: {summary['scale_parameter']}")
        print(f"Gini coefficient: {summary['gini_coefficient']}")
        
        return summary

# Usage example
if __name__ == "__main__":
    # data = your_prepared_data
    # analyzer = EnhancedSurvivalAnalysis(data, AFTDistribution.NORMAL)
    # results = analyzer.run_comprehensive_analysis()
    print("Expert XGBoost AFT Survival Analysis - Ready for deployment")
