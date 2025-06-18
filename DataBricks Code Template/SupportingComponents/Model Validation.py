# Model Validation Gate - Ensures model meets performance criteria before deployment
# This notebook validates the trained model and blocks deployment if performance is inadequate

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Model validation parameters (configurable)
MIN_CONCORDANCE_THRESHOLD = float(dbutils.widgets.get("min_concordance_threshold") if "min_concordance_threshold" in [w.name for w in dbutils.widgets.getAll()] else 0.7)
MODEL_NAME = dbutils.widgets.get("model_name") if "model_name" in [w.name for w in dbutils.widgets.getAll()] else "employee_survival_xgboost_aft"
MODEL_STAGE = dbutils.widgets.get("model_stage") if "model_stage" in [w.name for w in dbutils.widgets.getAll()] else "Production"

# Validation criteria
MAX_RMSE_DAYS = 365  # Maximum acceptable RMSE in days
MIN_TRAINING_SAMPLES = 1000  # Minimum samples model should be trained on

print("🔍 Model Validation Gate - Performance Check")
print(f"🎯 Model: {MODEL_NAME} ({MODEL_STAGE})")
print(f"📊 Minimum concordance required: {MIN_CONCORDANCE_THRESHOLD}")
print(f"📊 Maximum RMSE allowed: {MAX_RMSE_DAYS} days")

# =============================================================================
# IMPORTS
# =============================================================================

import mlflow
import mlflow.sklearn
from datetime import datetime
import pandas as pd

# =============================================================================
# MODEL VALIDATION
# =============================================================================

def validate_model_performance():
    """Validate the model meets performance requirements"""
    
    print("\n🔍 Starting Model Validation Gate...")
    
    try:
        # Load model from registry
        model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
        print(f"📋 Loading model: {model_uri}")
        
        # Get model metadata
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(MODEL_NAME, stages=[MODEL_STAGE])
        
        if not model_versions:
            raise Exception(f"No model found in {MODEL_STAGE} stage")
        
        model_version = model_versions[0]
        print(f"✅ Found model version: {model_version.version}")
        
        # Get the training run to access metrics
        run_id = model_version.run_id
        run = client.get_run(run_id)
        
        print(f"📊 Training run: {run_id}")
        
        # Extract key metrics
        metrics = run.data.metrics
        params = run.data.params
        
        # Critical metrics to check
        test_concordance = metrics.get('test_concordance_index')
        val_concordance = metrics.get('val_concordance_index') 
        train_concordance = metrics.get('train_concordance_index')
        test_rmse = metrics.get('test_rmse_days')
        training_samples = params.get('training_samples')
        
        print(f"\n📊 Model Performance Metrics:")
        print(f"   - Test Concordance Index: {test_concordance:.4f}" if test_concordance else "   - Test Concordance Index: Not available")
        print(f"   - Validation Concordance Index: {val_concordance:.4f}" if val_concordance else "   - Validation Concordance Index: Not available")
        print(f"   - Training Concordance Index: {train_concordance:.4f}" if train_concordance else "   - Training Concordance Index: Not available")
        print(f"   - Test RMSE: {test_rmse:.1f} days" if test_rmse else "   - Test RMSE: Not available")
        print(f"   - Training Samples: {training_samples}" if training_samples else "   - Training Samples: Not available")
        
        # Validation checks
        validation_issues = []
        
        # Check 1: Concordance index threshold
        if test_concordance is None:
            validation_issues.append("Test concordance index not available")
        elif test_concordance < MIN_CONCORDANCE_THRESHOLD:
            validation_issues.append(f"Test concordance {test_concordance:.4f} below threshold {MIN_CONCORDANCE_THRESHOLD}")
        
        # Check 2: RMSE reasonable
        if test_rmse is not None and test_rmse > MAX_RMSE_DAYS:
            validation_issues.append(f"Test RMSE {test_rmse:.1f} days exceeds maximum {MAX_RMSE_DAYS} days")
        
        # Check 3: Sufficient training data
        if training_samples is not None:
            try:
                sample_count = int(training_samples)
                if sample_count < MIN_TRAINING_SAMPLES:
                    validation_issues.append(f"Training samples {sample_count} below minimum {MIN_TRAINING_SAMPLES}")
            except:
                pass
        
        # Check 4: Model overfitting (large gap between train and test)
        if test_concordance is not None and train_concordance is not None:
            concordance_gap = train_concordance - test_concordance
            if concordance_gap > 0.1:  # 10% difference indicates potential overfitting
                validation_issues.append(f"Potential overfitting: concordance gap {concordance_gap:.4f}")
        
        # Check 5: Model underfitting (all performances poor)
        if val_concordance is not None and val_concordance < 0.55:
            validation_issues.append(f"Model appears to be underfitting: validation concordance {val_concordance:.4f}")
        
        # Additional model health checks
        model_age_days = (datetime.now().timestamp() * 1000 - model_version.creation_timestamp) / (1000 * 60 * 60 * 24)
        if model_age_days > 45:  # Model older than 45 days
            print(f"⚠️ Warning: Model is {model_age_days:.0f} days old")
        
        # Summary
        print(f"\n📊 Validation Summary:")
        print(f"   - Validation issues: {len(validation_issues)}")
        print(f"   - Model age: {model_age_days:.0f} days")
        
        if validation_issues:
            print(f"\n❌ VALIDATION ISSUES:")
            for issue in validation_issues:
                print(f"   - {issue}")
        
        # Load and test model functionality
        try:
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Model loaded successfully")
            
            # Quick functionality test
            if hasattr(model, 'predict_survival_time') and hasattr(model, 'predict_risk_score'):
                print(f"✅ Model has required prediction methods")
            else:
                validation_issues.append("Model missing required prediction methods")
                
        except Exception as e:
            validation_issues.append(f"Model loading failed: {str(e)}")
        
        # Create validation report
        validation_report = {
            "validation_timestamp": datetime.now(),
            "model_name": MODEL_NAME,
            "model_version": model_version.version,
            "model_stage": MODEL_STAGE,
            "test_concordance_index": test_concordance,
            "val_concordance_index": val_concordance,
            "train_concordance_index": train_concordance,
            "test_rmse_days": test_rmse,
            "training_samples": training_samples,
            "validation_issues_count": len(validation_issues),
            "validation_issues": validation_issues,
            "model_age_days": model_age_days,
            "validation_passed": len(validation_issues) == 0,
            "concordance_threshold": MIN_CONCORDANCE_THRESHOLD
        }
        
        # Save validation report
        try:
            validation_df = spark.createDataFrame([validation_report])
            (validation_df
             .write
             .format("delta")
             .mode("append")
             .option("mergeSchema", "true")
             .saveAsTable("hr_analytics.models.validation_gate_results"))
            
            print(f"✅ Validation report saved")
        except Exception as e:
            print(f"⚠️ Failed to save validation report: {str(e)}")
        
        # Decision
        if len(validation_issues) == 0:
            print(f"\n✅ MODEL VALIDATION GATE PASSED")
            print(f"   - Model {MODEL_NAME} v{model_version.version} meets all performance criteria")
            print(f"   - Concordance index: {test_concordance:.4f} (threshold: {MIN_CONCORDANCE_THRESHOLD})")
            return True
        else:
            print(f"\n❌ MODEL VALIDATION GATE FAILED")
            print(f"   - {len(validation_issues)} validation issues found")
            print(f"   - Model does not meet performance criteria")
            
            # Provide recommendations
            print(f"\n💡 RECOMMENDATIONS:")
            if test_concordance and test_concordance < MIN_CONCORDANCE_THRESHOLD:
                print(f"   - Improve model performance (current: {test_concordance:.4f}, need: {MIN_CONCORDANCE_THRESHOLD})")
                print(f"   - Consider feature engineering or hyperparameter tuning")
            if any("overfitting" in issue for issue in validation_issues):
                print(f"   - Address overfitting with regularization or more data")
            if any("underfitting" in issue for issue in validation_issues):
                print(f"   - Increase model complexity or improve features")
                
            raise Exception(f"MODEL VALIDATION FAILED: {'; '.join(validation_issues[:3])}")
        
    except Exception as e:
        print(f"❌ Model validation failed: {str(e)}")
        raise

# Run validation
success = validate_model_performance()
if not success:
    dbutils.notebook.exit("FAILED")
    