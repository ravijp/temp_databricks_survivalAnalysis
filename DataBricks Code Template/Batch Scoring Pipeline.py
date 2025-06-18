# Batch Scoring Pipeline for Employee Risk Prediction
# Scores active employees with survival analysis model for HR decision making

# =============================================================================
# CONFIGURATION PARAMETERS (MODIFY THESE AS NEEDED)
# =============================================================================

# Employee identifier (must match other components)
EMPLOYEE_ID_FIELD = "employee_id"

# Model configuration
MODEL_REGISTRY_NAME = "employee_survival_xgboost_aft"
MODEL_STAGE = "Production"  # or "Staging" for testing

# Active employee identification
ACTIVE_EMPLOYEE_CRITERIA = {
    "has_end_date": False,  # No end_date means still employed
    "min_tenure_days": 30,  # Minimum 30 days tenure to score
    "max_tenure_days": 3650,  # Maximum 10 years (data quality check)
    "exclude_status": ["TERMINATED", "RESIGNED", "INACTIVE"]  # Exclude these statuses
}

# Risk tier configuration
RISK_TIERS = {
    "low": {"min": 0.0, "max": 0.33, "label": "Low Risk"},
    "medium": {"min": 0.33, "max": 0.67, "label": "Medium Risk"}, 
    "high": {"min": 0.67, "max": 1.0, "label": "High Risk"}
}

# Business thresholds for alerts
ALERT_THRESHOLDS = {
    "high_risk_percentage": 0.15,  # Alert if >15% employees are high risk
    "department_high_risk_count": 5,  # Alert if department has >5 high risk employees
    "manager_high_risk_count": 3  # Alert if manager has >3 high risk direct reports
}

# Data sources
SURVIVAL_FEATURES_TABLE = "hr_analytics.processed.survival_features"
PROCESSED_EMPLOYEE_TABLE = "hr_analytics.processed.employee_master"

# Output tables
EMPLOYEE_SCORES_TABLE = "hr_analytics.predictions.employee_risk_scores"
DEPARTMENT_SUMMARY_TABLE = "hr_analytics.predictions.department_risk_summary"
MANAGER_ALERTS_TABLE = "hr_analytics.predictions.manager_risk_alerts"
SCORING_METADATA_TABLE = "hr_analytics.predictions.scoring_metadata"

# Client reporting configuration (for external HR partners)
CLIENT_REPORTING = {
    "enable_individual_predictions": True,  # Provide employee-level scores
    "enable_department_summaries": True,    # Provide department-level summaries
    "enable_trend_analysis": True,          # Include historical trends
    "anonymize_employee_ids": False,        # Set to True if privacy required
    "include_confidence_intervals": True    # Include prediction confidence
}

# =============================================================================
# IMPORTS
# =============================================================================

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("🎯 Starting Batch Scoring Pipeline for Employee Risk Prediction")
print(f"📊 Model: {MODEL_REGISTRY_NAME} ({MODEL_STAGE})")
print(f"🎯 Target: Employee-level predictions for HR decision making")
print(f"📋 Risk tiers: {len(RISK_TIERS)} levels")

# =============================================================================
# 1. MODEL LOADING AND VALIDATION
# =============================================================================

def load_and_validate_model():
    """Load production model and validate it's working"""
    
    print("\n🔧 Loading production model...")
    
    try:
        # Load model from registry
        model_uri = f"models:/{MODEL_REGISTRY_NAME}/{MODEL_STAGE}"
        model = mlflow.sklearn.load_model(model_uri)
        
        print(f"✅ Successfully loaded model: {model_uri}")
        
        # Get model metadata
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(MODEL_REGISTRY_NAME, stages=[MODEL_STAGE])[0]
        
        model_metadata = {
            "model_name": MODEL_REGISTRY_NAME,
            "model_version": model_version.version,
            "model_stage": MODEL_STAGE,
            "model_uri": model_uri,
            "creation_timestamp": model_version.creation_timestamp,
            "last_updated_timestamp": model_version.last_updated_timestamp
        }
        
        print(f"📊 Model metadata:")
        print(f"   - Version: {model_version.version}")
        print(f"   - Created: {datetime.fromtimestamp(model_metadata['creation_timestamp']/1000)}")
        
        # Basic model validation with dummy data
        try:
            # Create dummy feature vector matching expected features
            if hasattr(model, 'feature_names'):
                dummy_features = pd.DataFrame(
                    np.random.randn(1, len(model.feature_names)),
                    columns=model.feature_names
                )
            else:
                # Fallback to common features
                dummy_features = pd.DataFrame({
                    'log_salary': [10.5],
                    'salary_clean': [50000],
                    'tenure_days': [365],
                    'has_industry_data': [1]
                })
            
            # Test prediction
            test_survival = model.predict_survival_time(dummy_features)
            test_risk = model.predict_risk_score(dummy_features)
            
            print(f"✅ Model validation passed")
            print(f"   - Test survival prediction: {test_survival[0]:.1f} days")
            print(f"   - Test risk score: {test_risk[0]:.3f}")
            
        except Exception as e:
            raise Exception(f"Model validation failed: {str(e)}")
        
        return model, model_metadata
        
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        raise

# =============================================================================
# 2. ACTIVE EMPLOYEE IDENTIFICATION
# =============================================================================

def identify_active_employees():
    """Identify all active employees eligible for scoring"""
    
    print("\n👥 Identifying active employees for scoring...")
    
    # Load processed employee data
    employee_df = spark.table(PROCESSED_EMPLOYEE_TABLE)
    
    print(f"📊 Total employees in master table: {employee_df.count():,}")
    
    # Apply active employee criteria
    active_filter = (
        # No end date (still employed)
        (col("end_date").isNull() | (col("end_date") >= current_date())) &
        
        # Minimum tenure
        (col("tenure_days") >= ACTIVE_EMPLOYEE_CRITERIA["min_tenure_days"]) &
        
        # Maximum tenure (data quality)
        (col("tenure_days") <= ACTIVE_EMPLOYEE_CRITERIA["max_tenure_days"]) &
        
        # Not null employee ID
        (col(EMPLOYEE_ID_FIELD).isNotNull())
    )
    
    # Additional status filtering if employment_status exists
    if "employment_status" in employee_df.columns:
        active_filter = active_filter & (
            ~upper(col("employment_status")).isin(ACTIVE_EMPLOYEE_CRITERIA["exclude_status"])
        )
    
    active_employees = employee_df.filter(active_filter)
    
    active_count = active_employees.count()
    print(f"✅ Found {active_count:,} active employees for scoring")
    
    if active_count == 0:
        raise Exception("❌ No active employees found for scoring")
    
    # Show sample of active employees (for validation)
    print("📋 Sample of active employees:")
    sample_df = active_employees.select(
        EMPLOYEE_ID_FIELD, "department_clean", "tenure_days", "salary_clean"
    ).limit(5).toPandas()
    
    for _, row in sample_df.iterrows():
        print(f"   - {row[EMPLOYEE_ID_FIELD]}: {row['department_clean']}, {row['tenure_days']} days, ${row['salary_clean']:,.0f}")
    
    return active_employees

# =============================================================================
# 3. FEATURE PREPARATION FOR SCORING
# =============================================================================

def prepare_scoring_features(active_employees, model):
    """Prepare features for model scoring"""
    
    print("\n🔧 Preparing features for scoring...")
    
    # Get model's expected features
    if hasattr(model, 'feature_names'):
        expected_features = model.feature_names
        print(f"📋 Model expects {len(expected_features)} features")
    else:
        # Fallback to common survival analysis features
        expected_features = [
            'log_salary', 'salary_clean', 'tenure_days', 'has_industry_data'
        ]
        print(f"⚠️ Using fallback features: {expected_features}")
    
    # Convert to Pandas for easier feature manipulation
    active_pandas = active_employees.toPandas()
    
    # Ensure all expected features exist
    missing_features = []
    for feature in expected_features:
        if feature not in active_pandas.columns:
            missing_features.append(feature)
    
    if missing_features:
        print(f"⚠️ Missing features, will create defaults: {missing_features}")
        
        # Create default values for missing features
        for feature in missing_features:
            if 'dept_' in feature:
                active_pandas[feature] = 0  # Department indicators default to 0
            elif 'has_' in feature or 'is_' in feature:
                active_pandas[feature] = 0  # Boolean features default to 0
            else:
                active_pandas[feature] = active_pandas.get(feature.replace('_clean', ''), 0)
    
    # Create feature matrix
    feature_matrix = active_pandas[expected_features].fillna(0)
    
    print(f"✅ Feature matrix prepared: {feature_matrix.shape}")
    print(f"   - Features: {list(feature_matrix.columns)}")
    
    # Validate feature matrix
    if feature_matrix.isnull().any().any():
        print("⚠️ Warning: NaN values detected in feature matrix")
        feature_matrix = feature_matrix.fillna(0)
    
    return feature_matrix, active_pandas

# =============================================================================
# 4. BATCH PREDICTION EXECUTION
# =============================================================================

def execute_batch_predictions(model, feature_matrix, employee_data):
    """Execute batch predictions for all active employees"""
    
    print("\n🔮 Executing batch predictions...")
    
    try:
        # Make predictions
        print("   - Predicting survival times...")
        survival_times = model.predict_survival_time(feature_matrix)
        
        print("   - Calculating risk scores...")
        risk_scores = model.predict_risk_score(feature_matrix)
        
        print(f"✅ Predictions completed for {len(survival_times):,} employees")
        
        # Create results dataframe
        results = employee_data[[
            EMPLOYEE_ID_FIELD, 'department_clean', 'job_title_clean', 
            'tenure_days', 'salary_clean'
        ]].copy()
        
        results['predicted_survival_days'] = survival_times
        results['risk_score'] = risk_scores
        
        # Assign risk tiers
        def assign_risk_tier(score):
            for tier, config in RISK_TIERS.items():
                if config['min'] <= score < config['max']:
                    return config['label']
            return RISK_TIERS['high']['label']  # Default to high if >= 1.0
        
        results['risk_tier'] = results['risk_score'].apply(assign_risk_tier)
        
        # Add scoring metadata
        results['scoring_date'] = datetime.now()
        results['model_version'] = model.model_name if hasattr(model, 'model_name') else MODEL_REGISTRY_NAME
        results['scoring_run_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Data quality validation
        print("\n📊 Prediction quality validation:")
        
        # Check for reasonable predictions
        valid_survival = (results['predicted_survival_days'] > 0) & (results['predicted_survival_days'] < 5000)
        valid_risk = (results['risk_score'] >= 0) & (results['risk_score'] <= 1)
        
        invalid_survival = (~valid_survival).sum()
        invalid_risk = (~valid_risk).sum()
        
        if invalid_survival > 0:
            print(f"⚠️ Warning: {invalid_survival} invalid survival predictions")
            results.loc[~valid_survival, 'predicted_survival_days'] = results['predicted_survival_days'].median()
        
        if invalid_risk > 0:
            print(f"⚠️ Warning: {invalid_risk} invalid risk scores")
            results.loc[~valid_risk, 'risk_score'] = results['risk_score'].median()
        
        # Summary statistics
        print(f"📊 Prediction summary:")
        print(f"   - Average survival time: {results['predicted_survival_days'].mean():.0f} days")
        print(f"   - Average risk score: {results['risk_score'].mean():.3f}")
        
        risk_distribution = results['risk_tier'].value_counts()
        print(f"   - Risk distribution:")
        for tier, count in risk_distribution.items():
            percentage = (count / len(results)) * 100
            print(f"     • {tier}: {count:,} ({percentage:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Prediction execution failed: {str(e)}")
        raise

# =============================================================================
# 5. BUSINESS INTELLIGENCE AND ALERTS
# =============================================================================

def generate_business_insights(results):
    """Generate business insights and alerts for HR partners"""
    
    print("\n📈 Generating business insights and alerts...")
    
    # Department-level analysis
    dept_summary = results.groupby('department_clean').agg({
        EMPLOYEE_ID_FIELD: 'count',
        'risk_score': ['mean', 'std'],
        'predicted_survival_days': 'mean',
        'salary_clean': 'mean'
    }).round(3)
    
    dept_summary.columns = ['employee_count', 'avg_risk_score', 'std_risk_score', 
                           'avg_predicted_survival', 'avg_salary']
    
    # Add risk tier counts by department
    risk_by_dept = results.groupby(['department_clean', 'risk_tier']).size().unstack(fill_value=0)
    dept_summary = dept_summary.join(risk_by_dept, how='left').fillna(0)
    
    # Calculate high risk percentage by department
    dept_summary['high_risk_percentage'] = (
        dept_summary.get('High Risk', 0) / dept_summary['employee_count']
    )
    
    print(f"✅ Department analysis completed for {len(dept_summary)} departments")
    
    # Identify alerts
    alerts = []
    
    # Overall high risk alert
    overall_high_risk_pct = (results['risk_tier'] == 'High Risk').mean()
    if overall_high_risk_pct > ALERT_THRESHOLDS['high_risk_percentage']:
        alerts.append({
            'alert_type': 'high_risk_overall',
            'alert_level': 'WARNING',
            'message': f"High overall risk: {overall_high_risk_pct:.1%} of employees are high risk",
            'affected_count': (results['risk_tier'] == 'High Risk').sum(),
            'threshold': ALERT_THRESHOLDS['high_risk_percentage']
        })
    
    # Department-specific alerts
    high_risk_departments = dept_summary[
        dept_summary['high_risk_percentage'] > ALERT_THRESHOLDS['high_risk_percentage']
    ]
    
    for dept_name, dept_data in high_risk_departments.iterrows():
        alerts.append({
            'alert_type': 'high_risk_department',
            'alert_level': 'WARNING',
            'department': dept_name,
            'message': f"Department {dept_name}: {dept_data['high_risk_percentage']:.1%} high risk",
            'affected_count': int(dept_data.get('High Risk', 0)),
            'total_employees': int(dept_data['employee_count'])
        })
    
    # Top risk employees (for management attention)
    top_risk_employees = results.nlargest(20, 'risk_score')[[
        EMPLOYEE_ID_FIELD, 'department_clean', 'risk_score', 'risk_tier', 
        'predicted_survival_days'
    ]]
    
    print(f"🚨 Generated {len(alerts)} alerts")
    
    return dept_summary, alerts, top_risk_employees

# =============================================================================
# 6. DATA PERSISTENCE AND REPORTING
# =============================================================================

def save_scoring_results(results, dept_summary, alerts, model_metadata):
    """Save all scoring results to Delta tables"""
    
    print("\n💾 Saving scoring results...")
    
    try:
        # 1. Save employee-level predictions
        results_spark = spark.createDataFrame(results)
        (results_spark
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .partitionBy("scoring_date")
         .saveAsTable(EMPLOYEE_SCORES_TABLE))
        
        print(f"✅ Employee scores saved to: {EMPLOYEE_SCORES_TABLE}")
        
        # 2. Save department summary
        dept_summary_reset = dept_summary.reset_index()
        dept_summary_reset['scoring_date'] = datetime.now()
        dept_summary_spark = spark.createDataFrame(dept_summary_reset)
        
        (dept_summary_spark
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .saveAsTable(DEPARTMENT_SUMMARY_TABLE))
        
        print(f"✅ Department summary saved to: {DEPARTMENT_SUMMARY_TABLE}")
        
        # 3. Save alerts if any
        if alerts:
            alerts_df = pd.DataFrame(alerts)
            alerts_df['scoring_date'] = datetime.now()
            alerts_spark = spark.createDataFrame(alerts_df)
            
            (alerts_spark
             .write
             .format("delta")
             .mode("append")
             .option("mergeSchema", "true")
             .saveAsTable(MANAGER_ALERTS_TABLE))
            
            print(f"✅ Alerts saved to: {MANAGER_ALERTS_TABLE}")
        
        # 4. Save scoring metadata
        scoring_metadata = pd.DataFrame([{
            'scoring_run_id': results['scoring_run_id'].iloc[0],
            'scoring_date': datetime.now(),
            'model_name': model_metadata['model_name'],
            'model_version': model_metadata['model_version'],
            'model_stage': model_metadata['model_stage'],
            'employees_scored': len(results),
            'high_risk_count': (results['risk_tier'] == 'High Risk').sum(),
            'medium_risk_count': (results['risk_tier'] == 'Medium Risk').sum(),
            'low_risk_count': (results['risk_tier'] == 'Low Risk').sum(),
            'avg_risk_score': results['risk_score'].mean(),
            'alerts_generated': len(alerts)
        }])
        
        metadata_spark = spark.createDataFrame(scoring_metadata)
        (metadata_spark
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .saveAsTable(SCORING_METADATA_TABLE))
        
        print(f"✅ Scoring metadata saved to: {SCORING_METADATA_TABLE}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {str(e)}")
        raise

def generate_client_report(results, dept_summary, alerts):
    """Generate summary report for HR client partners"""
    
    print("\n📋 Generating client report...")
    
    # Executive summary
    total_employees = len(results)
    high_risk_count = (results['risk_tier'] == 'High Risk').sum()
    medium_risk_count = (results['risk_tier'] == 'Medium Risk').sum()
    low_risk_count = (results['risk_tier'] == 'Low Risk').sum()
    
    avg_survival_time = results['predicted_survival_days'].mean()
    avg_risk_score = results['risk_score'].mean()
    
    report = f"""
    
    ================================================================================
    📊 EMPLOYEE TURNOVER RISK ANALYSIS REPORT
    ================================================================================
    
    📅 Scoring Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    👥 Total Employees Analyzed: {total_employees:,}
    
    🎯 RISK DISTRIBUTION:
    • High Risk (Likely to leave within 6 months):  {high_risk_count:,} ({high_risk_count/total_employees:.1%})
    • Medium Risk (Moderate retention concern):      {medium_risk_count:,} ({medium_risk_count/total_employees:.1%})
    • Low Risk (Strong retention likelihood):       {low_risk_count:,} ({low_risk_count/total_employees:.1%})
    
    📈 KEY METRICS:
    • Average Risk Score: {avg_risk_score:.3f} (0=lowest risk, 1=highest risk)
    • Average Predicted Tenure: {avg_survival_time/365:.1f} years
    
    🚨 ALERTS GENERATED: {len(alerts)}
    """
    
    if alerts:
        report += "\n    🚨 ATTENTION REQUIRED:\n"
        for alert in alerts[:5]:  # Show top 5 alerts
            report += f"    • {alert['message']}\n"
    
    # Department insights
    report += f"""
    
    📊 DEPARTMENT ANALYSIS:
    Top 5 Departments by Risk:
    """
    
    top_risk_depts = dept_summary.nlargest(5, 'high_risk_percentage')
    for dept_name, dept_data in top_risk_depts.iterrows():
        report += f"    • {dept_name}: {dept_data['high_risk_percentage']:.1%} high risk ({dept_data.get('High Risk', 0):.0f}/{dept_data['employee_count']:.0f} employees)\n"
    
    # Actionable recommendations
    report += f"""
    
    💡 RECOMMENDED ACTIONS:
    """
    
    if high_risk_count > 0:
        report += f"    • Immediate attention: Review {min(high_risk_count, 20)} highest risk employees\n"
        report += f"    • Retention interviews: Schedule 1:1s with high-risk employees\n"
        report += f"    • Succession planning: Prepare backfill plans for critical high-risk roles\n"
    
    if len(alerts) > 0:
        report += f"    • Department focus: Address departments with elevated risk levels\n"
    
    report += f"    • Trend monitoring: Re-run analysis in 2 weeks to track changes\n"
    
    report += f"""
    
    📋 DATA TABLES CREATED:
    • Employee-level predictions: {EMPLOYEE_SCORES_TABLE}
    • Department summaries: {DEPARTMENT_SUMMARY_TABLE}
    • Risk alerts: {MANAGER_ALERTS_TABLE}
    
    ================================================================================
    """
    
    print(report)
    return report

# =============================================================================
# 7. MAIN BATCH SCORING PIPELINE
# =============================================================================

def run_batch_scoring_pipeline():
    """Main function to execute the complete batch scoring pipeline"""
    
    print("🚀 Starting Batch Scoring Pipeline...")
    start_time = datetime.now()
    
    try:
        # Step 1: Load and validate model
        model, model_metadata = load_and_validate_model()
        
        # Step 2: Identify active employees
        active_employees = identify_active_employees()
        
        # Step 3: Prepare features for scoring
        feature_matrix, employee_data = prepare_scoring_features(active_employees, model)
        
        # Step 4: Execute batch predictions
        results = execute_batch_predictions(model, feature_matrix, employee_data)
        
        # Step 5: Generate business insights
        dept_summary, alerts, top_risk_employees = generate_business_insights(results)
        
        # Step 6: Save all results
        save_scoring_results(results, dept_summary, alerts, model_metadata)
        
        # Step 7: Generate client report
        client_report = generate_client_report(results, dept_summary, alerts)
        
        # Pipeline completion summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🎉 BATCH SCORING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏱️ Total execution time: {duration}")
        print(f"👥 Employees scored: {len(results):,}")
        print(f"🚨 Alerts generated: {len(alerts)}")
        print(f"📊 Departments analyzed: {len(dept_summary)}")
        print(f"🎯 High risk employees: {(results['risk_tier'] == 'High Risk').sum():,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Batch scoring pipeline failed: {str(e)}")
        return False

# Run the pipeline
if __name__ == "__main__":
    success = run_batch_scoring_pipeline()
    if not success:
        raise Exception("Batch scoring pipeline failed")
    