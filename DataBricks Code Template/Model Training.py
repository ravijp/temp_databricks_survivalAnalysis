# XGBoost AFT Model Training with OOS/OOT Validation
# Trains survival analysis model with Out-of-Sample and Out-of-Time validation

# =============================================================================
# CONFIGURATION PARAMETERS (MODIFY THESE AS NEEDED)
# =============================================================================

# Model performance thresholds
MIN_CONCORDANCE_INDEX = 0.7  # Minimum acceptable concordance index
MIN_TRAINING_SAMPLES = 1000  # Minimum samples needed for training
MIN_EVENTS = 100  # Minimum events needed for survival analysis

# Employee identifier (must match data processing pipeline)
EMPLOYEE_ID_FIELD = "employee_id"

# Feature selection (based on causal impact for HR decisions)
FEATURE_COLUMNS = [
    # Salary and compensation features
    "log_salary", "salary_clean",
    
    # Tenure and experience features  
    "tenure_days", "time_to_event",
    
    # Department indicators (will be added dynamically)
    # "dept_engineering", "dept_sales", etc.
    
    # Industry and region features
    "has_industry_data",
    
    # Add more features based on your specific data
]

# Categorical features to encode
CATEGORICAL_FEATURES = ["department_clean", "job_title_clean", "tenure_bucket"]

# Date configuration for OOS/OOT validation
TRAINING_CUTOFF_MONTHS = 18  # Use data up to 18 months ago for training
VALIDATION_CUTOFF_MONTHS = 12  # Use 12-18 months ago for OOS validation  
TEST_CUTOFF_MONTHS = 6  # Use 6-12 months ago for OOT validation

# XGBoost parameters
XGBOOST_PARAMS = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik", 
    "aft_loss_distribution": "normal",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 200,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0
}

# Hyperparameter tuning (set to True for automatic tuning)
ENABLE_HYPERPARAMETER_TUNING = False  # Set to True for production
HYPEROPT_MAX_EVALS = 30  # Number of hyperparameter combinations to try

# Data and model locations
SURVIVAL_FEATURES_TABLE = "hr_analytics.processed.survival_features"
MODEL_REGISTRY_NAME = "employee_survival_xgboost_aft"
EXPERIMENT_NAME = "/Shared/employee_survival_production"

# Output locations
MODEL_VALIDATION_TABLE = "hr_analytics.models.validation_results"
FEATURE_IMPORTANCE_TABLE = "hr_analytics.models.feature_importance"

# =============================================================================
# IMPORTS
# =============================================================================

import mlflow
import mlflow.xgboost
import mlflow.sklearn
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Hyperparameter tuning imports
if ENABLE_HYPERPARAMETER_TUNING:
    from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, Trials

# Set MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)

print("🎯 Starting XGBoost AFT Model Training with OOS/OOT Validation")
print(f"📊 Minimum concordance index required: {MIN_CONCORDANCE_INDEX}")
print(f"🔬 Experiment: {EXPERIMENT_NAME}")
print(f"📋 Model registry: {MODEL_REGISTRY_NAME}")

# =============================================================================
# 1. ADVANCED DATA LOADING WITH TEMPORAL SPLITS
# =============================================================================

def load_survival_data_with_temporal_splits():
    """Load survival data and create temporal splits for OOS/OOT validation"""
    
    print("\n📊 Loading survival data with temporal splits...")
    
    # Load processed survival features
    df = spark.table(SURVIVAL_FEATURES_TABLE).toPandas()
    
    if len(df) < MIN_TRAINING_SAMPLES:
        raise Exception(f"❌ Insufficient data: {len(df)} samples (need {MIN_TRAINING_SAMPLES})")
    
    # Convert date columns
    df['processing_date'] = pd.to_datetime(df['processing_date'])
    df['vantage_date_used'] = pd.to_datetime(df['vantage_date_used'])
    
    # Calculate temporal split dates
    current_date = datetime.now()
    training_cutoff = current_date - timedelta(days=TRAINING_CUTOFF_MONTHS * 30)
    validation_cutoff = current_date - timedelta(days=VALIDATION_CUTOFF_MONTHS * 30)
    test_cutoff = current_date - timedelta(days=TEST_CUTOFF_MONTHS * 30)
    
    print(f"📅 Temporal split dates:")
    print(f"   - Training data: Before {training_cutoff.strftime('%Y-%m-%d')}")
    print(f"   - OOS validation: {validation_cutoff.strftime('%Y-%m-%d')} to {training_cutoff.strftime('%Y-%m-%d')}")
    print(f"   - OOT test: {test_cutoff.strftime('%Y-%m-%d')} to {validation_cutoff.strftime('%Y-%m-%d')}")
    
    # Create temporal splits
    train_df = df[df['processing_date'] < training_cutoff].copy()
    val_df = df[(df['processing_date'] >= validation_cutoff) & 
                (df['processing_date'] < training_cutoff)].copy()
    test_df = df[(df['processing_date'] >= test_cutoff) & 
                 (df['processing_date'] < validation_cutoff)].copy()
    
    print(f"📊 Dataset splits:")
    print(f"   - Training: {len(train_df):,} samples")
    print(f"   - OOS Validation: {len(val_df):,} samples") 
    print(f"   - OOT Test: {len(test_df):,} samples")
    
    # Validate minimum requirements
    train_events = train_df['event_observed'].sum()
    val_events = val_df['event_observed'].sum()
    test_events = test_df['event_observed'].sum()
    
    print(f"📊 Event counts:")
    print(f"   - Training events: {train_events}")
    print(f"   - Validation events: {val_events}")
    print(f"   - Test events: {test_events}")
    
    if train_events < MIN_EVENTS:
        raise Exception(f"❌ Insufficient training events: {train_events} (need {MIN_EVENTS})")
    
    if val_events < 10:
        print("⚠️ Warning: Low validation events, results may be unreliable")
    
    if test_events < 10:
        print("⚠️ Warning: Low test events, results may be unreliable")
    
    return train_df, val_df, test_df, (training_cutoff, validation_cutoff, test_cutoff)

# =============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# =============================================================================

def prepare_features_for_modeling(train_df, val_df, test_df):
    """Prepare features for XGBoost AFT modeling with proper encoding"""
    
    print("\n🔧 Preparing features for modeling...")
    
    # Dynamically detect department columns
    dept_columns = [col for col in train_df.columns if col.startswith('dept_')]
    
    # Build final feature list
    base_features = ["log_salary", "salary_clean", "tenure_days", "has_industry_data"]
    all_features = base_features + dept_columns
    
    # Add any other numeric features that exist
    numeric_features = []
    for col in train_df.columns:
        if train_df[col].dtype in ['int64', 'float64'] and col not in all_features:
            if col not in [EMPLOYEE_ID_FIELD, 'event_observed', 'time_to_event']:
                numeric_features.append(col)
    
    all_features.extend(numeric_features)
    
    print(f"🎯 Selected features ({len(all_features)}):")
    print(f"   - Base features: {base_features}")
    print(f"   - Department features: {dept_columns}")
    print(f"   - Additional numeric: {numeric_features}")
    
    # Prepare feature matrices
    def prepare_feature_matrix(df, feature_list):
        X = df[feature_list].fillna(0)  # Fill NaN with 0
        y = df['time_to_event'].values
        events = df['event_observed'].values
        return X, y, events
    
    X_train, y_train, events_train = prepare_feature_matrix(train_df, all_features)
    X_val, y_val, events_val = prepare_feature_matrix(val_df, all_features)
    X_test, y_test, events_test = prepare_feature_matrix(test_df, all_features)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✅ Feature preparation completed")
    print(f"   - Training features shape: {X_train_scaled.shape}")
    print(f"   - Validation features shape: {X_val_scaled.shape}")
    print(f"   - Test features shape: {X_test_scaled.shape}")
    
    return {
        'X_train': X_train_scaled, 'y_train': y_train, 'events_train': events_train,
        'X_val': X_val_scaled, 'y_val': y_val, 'events_val': events_val,
        'X_test': X_test_scaled, 'y_test': y_test, 'events_test': events_test,
        'feature_names': all_features,
        'scaler': scaler
    }

# =============================================================================
# 3. ADVANCED XGBOOST AFT MODEL CLASS
# =============================================================================

class XGBoostAFTAdvanced:
    """Advanced XGBoost AFT model with comprehensive evaluation"""
    
    def __init__(self, params=None):
        self.params = params or XGBOOST_PARAMS.copy()
        self.model = None
        self.feature_names = None
        self.scaler = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train XGBoost AFT model with optional validation"""
        
        print("🎯 Training XGBoost AFT model...")
        
        # Prepare evaluation set if validation data provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Initialize and train model
        self.model = xgb.XGBRegressor(**self.params)
        
        # Train with early stopping if validation set provided
        if len(eval_set) > 1:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='rmse',  # XGBoost doesn't natively support AFT eval metrics
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("✅ Model training completed")
        return self.model
    
    def predict_survival_time(self, X):
        """Predict survival time (time to event)"""
        return self.model.predict(X)
    
    def predict_risk_score(self, X):
        """Predict risk score (higher = more likely to terminate)"""
        survival_times = self.predict_survival_time(X)
        # Convert to risk score (inverse relationship)
        max_time = np.percentile(survival_times, 95)  # Use 95th percentile as max
        risk_scores = 1 - (survival_times / max_time)
        return np.clip(risk_scores, 0, 1)  # Ensure 0-1 range
    
    def evaluate_model(self, X, y_true, events, dataset_name=""):
        """Comprehensive model evaluation"""
        
        # Predictions
        y_pred = self.predict_survival_time(X)
        risk_scores = self.predict_risk_score(X)
        
        # Survival-specific metrics
        concordance = concordance_index(y_true, -risk_scores, events)
        
        # Regression metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = np.mean(np.abs(y_true - y_pred))
        
        # Event-specific metrics
        event_indices = events == 1
        if np.sum(event_indices) > 0:
            event_concordance = concordance_index(
                y_true[event_indices], -risk_scores[event_indices], 
                events[event_indices]
            )
        else:
            event_concordance = np.nan
        
        metrics = {
            f'{dataset_name}_concordance_index': concordance,
            f'{dataset_name}_event_concordance': event_concordance,
            f'{dataset_name}_rmse_days': rmse,
            f'{dataset_name}_mae_days': mae,
            f'{dataset_name}_mean_survival_time': np.mean(y_pred),
            f'{dataset_name}_mean_risk_score': np.mean(risk_scores)
        }
        
        return metrics, y_pred, risk_scores

# =============================================================================
# 4. HYPERPARAMETER TUNING (OPTIONAL)
# =============================================================================

def hyperparameter_tuning(prepared_data):
    """Perform hyperparameter tuning using Hyperopt"""
    
    if not ENABLE_HYPERPARAMETER_TUNING:
        print("⚠️ Hyperparameter tuning disabled, using default parameters")
        return XGBOOST_PARAMS
    
    print(f"\n🔧 Starting hyperparameter tuning ({HYPEROPT_MAX_EVALS} evaluations)...")
    
    def objective(params):
        with mlflow.start_run(nested=True):
            # Convert hyperopt params to XGBoost format
            xgb_params = {
                'objective': 'survival:aft',
                'eval_metric': 'aft-nloglik',
                'aft_loss_distribution': 'normal',
                'max_depth': int(params['max_depth']),
                'learning_rate': params['learning_rate'],
                'n_estimators': int(params['n_estimators']),
                'subsample': params['subsample'],
                'colsample_bytree': params['colsample_bytree'],
                'random_state': 42,
                'verbosity': 0
            }
            
            # Train model
            model = XGBoostAFTAdvanced(xgb_params)
            model.train(
                prepared_data['X_train'], prepared_data['y_train'],
                prepared_data['X_val'], prepared_data['y_val']
            )
            
            # Evaluate on validation set
            val_metrics, _, _ = model.evaluate_model(
                prepared_data['X_val'], prepared_data['y_val'], 
                prepared_data['events_val'], 'val'
            )
            
            # Log parameters and metrics
            mlflow.log_params(xgb_params)
            mlflow.log_metrics(val_metrics)
            
            # Return negative concordance (hyperopt minimizes)
            return {'loss': -val_metrics['val_concordance_index'], 'status': STATUS_OK}
    
    # Define search space
    search_space = {
        'max_depth': hp.choice('max_depth', [4, 6, 8, 10]),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'n_estimators': hp.choice('n_estimators', [100, 200, 300]),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
    }
    
    # Run optimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=HYPEROPT_MAX_EVALS,
        trials=trials
    )
    
    # Convert best params back to XGBoost format
    best_params = XGBOOST_PARAMS.copy()
    best_params.update({
        'max_depth': [4, 6, 8, 10][best['max_depth']],
        'learning_rate': best['learning_rate'],
        'n_estimators': [100, 200, 300][best['n_estimators']],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree']
    })
    
    print(f"✅ Best hyperparameters found: {best_params}")
    return best_params

# =============================================================================
# 5. MAIN TRAINING PIPELINE
# =============================================================================

def train_survival_model():
    """Main function to train the survival model with OOS/OOT validation"""
    
    print("🚀 Starting comprehensive model training pipeline...")
    
    with mlflow.start_run(run_name=f"survival_model_training_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        try:
            # Step 1: Load data with temporal splits
            train_df, val_df, test_df, split_dates = load_survival_data_with_temporal_splits()
            
            # Log dataset information
            mlflow.log_param("training_samples", len(train_df))
            mlflow.log_param("validation_samples", len(val_df))
            mlflow.log_param("test_samples", len(test_df))
            mlflow.log_param("training_events", train_df['event_observed'].sum())
            mlflow.log_param("validation_events", val_df['event_observed'].sum())
            mlflow.log_param("test_events", test_df['event_observed'].sum())
            
            # Step 2: Prepare features
            prepared_data = prepare_features_for_modeling(train_df, val_df, test_df)
            mlflow.log_param("num_features", len(prepared_data['feature_names']))
            
            # Step 3: Hyperparameter tuning (optional)
            best_params = hyperparameter_tuning(prepared_data)
            mlflow.log_params(best_params)
            
            # Step 4: Train final model
            print("\n🎯 Training final model with best parameters...")
            final_model = XGBoostAFTAdvanced(best_params)
            final_model.feature_names = prepared_data['feature_names']
            final_model.scaler = prepared_data['scaler']
            
            final_model.train(
                prepared_data['X_train'], prepared_data['y_train'],
                prepared_data['X_val'], prepared_data['y_val']
            )
            
            # Step 5: Comprehensive evaluation
            print("\n📊 Evaluating model performance...")
            
            # Training performance
            train_metrics, train_pred, train_risk = final_model.evaluate_model(
                prepared_data['X_train'], prepared_data['y_train'], 
                prepared_data['events_train'], 'train'
            )
            
            # OOS validation performance
            val_metrics, val_pred, val_risk = final_model.evaluate_model(
                prepared_data['X_val'], prepared_data['y_val'], 
                prepared_data['events_val'], 'val'
            )
            
            # OOT test performance
            test_metrics, test_pred, test_risk = final_model.evaluate_model(
                prepared_data['X_test'], prepared_data['y_test'], 
                prepared_data['events_test'], 'test'
            )
            
            # Combine all metrics
            all_metrics = {**train_metrics, **val_metrics, **test_metrics}
            mlflow.log_metrics(all_metrics)
            
            # Step 6: Model validation against threshold
            test_concordance = test_metrics['test_concordance_index']
            val_concordance = val_metrics['val_concordance_index']
            
            print(f"\n📊 Model Performance Summary:")
            print(f"   - Training Concordance: {train_metrics['train_concordance_index']:.4f}")
            print(f"   - Validation Concordance: {val_concordance:.4f}")
            print(f"   - Test Concordance: {test_concordance:.4f}")
            print(f"   - Required Threshold: {MIN_CONCORDANCE_INDEX}")
            
            # Check if model meets threshold
            model_passed = test_concordance >= MIN_CONCORDANCE_INDEX
            mlflow.log_param("model_passed_threshold", model_passed)
            mlflow.log_param("concordance_threshold", MIN_CONCORDANCE_INDEX)
            
            if not model_passed:
                print(f"❌ Model failed to meet concordance threshold: {test_concordance:.4f} < {MIN_CONCORDANCE_INDEX}")
                print("🔄 Consider:")
                print("   - Collecting more data")
                print("   - Adding more features")
                print("   - Tuning hyperparameters further")
                print("   - Adjusting the threshold")
                mlflow.log_param("model_quality", "failed")
                return None
            else:
                print(f"✅ Model meets performance threshold!")
                mlflow.log_param("model_quality", "passed")
            
            # Step 7: Generate model insights
            print("\n📈 Generating model insights...")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': prepared_data['feature_names'],
                'importance': final_model.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            feature_importance_spark = spark.createDataFrame(feature_importance)
            (feature_importance_spark
             .write
             .format("delta")
             .mode("overwrite")
             .saveAsTable(FEATURE_IMPORTANCE_TABLE))
            
            # Visualizations
            plt.figure(figsize=(12, 8))
            
            # Feature importance plot
            plt.subplot(2, 2, 1)
            top_features = feature_importance.head(10)
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            
            # Survival time distribution
            plt.subplot(2, 2, 2)
            plt.hist(test_pred, bins=30, alpha=0.7, label='Predicted')
            plt.hist(prepared_data['y_test'], bins=30, alpha=0.7, label='Actual')
            plt.xlabel('Survival Time (days)')
            plt.ylabel('Frequency')
            plt.title('Predicted vs Actual Survival Times')
            plt.legend()
            
            # Risk score distribution
            plt.subplot(2, 2, 3)
            plt.hist(test_risk, bins=30, alpha=0.7)
            plt.xlabel('Risk Score')
            plt.ylabel('Frequency')
            plt.title('Risk Score Distribution')
            
            # Concordance by risk tier
            plt.subplot(2, 2, 4)
            risk_tiers = pd.cut(test_risk, bins=[0, 0.33, 0.67, 1.0], labels=['Low', 'Medium', 'High'])
            tier_concordance = []
            for tier in ['Low', 'Medium', 'High']:
                tier_mask = risk_tiers == tier
                if np.sum(tier_mask) > 0:
                    tier_conc = concordance_index(
                        prepared_data['y_test'][tier_mask], 
                        -test_risk[tier_mask], 
                        prepared_data['events_test'][tier_mask]
                    )
                    tier_concordance.append(tier_conc)
                else:
                    tier_concordance.append(0)
            
            plt.bar(['Low', 'Medium', 'High'], tier_concordance)
            plt.ylabel('Concordance Index')
            plt.title('Concordance by Risk Tier')
            plt.ylim(0, 1)
            
            plt.tight_layout()
            mlflow.log_figure(plt.gcf(), "model_performance_summary.png")
            plt.close()
            
            # Step 8: Save model to registry
            print("\n💾 Saving model to registry...")
            
            # Create a wrapper class for the complete model
            class CompleteAFTModel:
                def __init__(self, model, scaler, feature_names):
                    self.model = model
                    self.scaler = scaler
                    self.feature_names = feature_names
                
                def predict_survival_time(self, X):
                    if isinstance(X, pd.DataFrame):
                        X = X[self.feature_names].fillna(0)
                    X_scaled = self.scaler.transform(X)
                    return self.model.predict(X_scaled)
                
                def predict_risk_score(self, X):
                    survival_times = self.predict_survival_time(X)
                    max_time = np.percentile(survival_times, 95)
                    risk_scores = 1 - (survival_times / max_time)
                    return np.clip(risk_scores, 0, 1)
            
            complete_model = CompleteAFTModel(
                final_model.model, 
                final_model.scaler, 
                final_model.feature_names
            )
            
            # Log model
            mlflow.sklearn.log_model(
                sk_model=complete_model,
                artifact_path="survival_model",
                registered_model_name=MODEL_REGISTRY_NAME
            )
            
            # Save validation results
            validation_results = pd.DataFrame([{
                'model_name': MODEL_REGISTRY_NAME,
                'training_date': datetime.now(),
                'train_concordance': train_metrics['train_concordance_index'],
                'val_concordance': val_concordance,
                'test_concordance': test_concordance,
                'model_passed': model_passed,
                'threshold_used': MIN_CONCORDANCE_INDEX,
                'num_features': len(prepared_data['feature_names']),
                'training_samples': len(train_df),
                'test_samples': len(test_df)
            }])
            
            validation_spark = spark.createDataFrame(validation_results)
            (validation_spark
             .write
             .format("delta")
             .mode("append")
             .saveAsTable(MODEL_VALIDATION_TABLE))
            
            print("✅ Model saved successfully!")
            
            # Promote to production if passed
            if model_passed:
                client = mlflow.tracking.MlflowClient()
                latest_version = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=["None"])[0]
                
                client.transition_model_version_stage(
                    name=MODEL_REGISTRY_NAME,
                    version=latest_version.version,
                    stage="Production"
                )
                
                print(f"✅ Model version {latest_version.version} promoted to Production")
                mlflow.log_param("promoted_to_production", True)
            
            print("\n🎉 Model training completed successfully!")
            return final_model
            
        except Exception as e:
            print(f"❌ Model training failed: {str(e)}")
            mlflow.log_param("training_error", str(e))
            raise

# Run the training
if __name__ == "__main__":
    model = train_survival_model()
    if model is None:
        raise Exception("Model training failed or model did not meet quality threshold")
        