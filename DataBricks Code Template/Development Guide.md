# HR Survival Analysis Pipeline - Complete Deployment Guide

## 📋 Overview

This guide provides step-by-step instructions for deploying the complete HR Survival Analysis Pipeline in Databricks. The pipeline includes data discovery, complex data processing, XGBoost AFT model training with OOS/OOT validation, batch scoring, and business reporting.

## 🎯 Architecture Summary

```
Monthly HR Data Sources → Data Discovery & Profiling → Complex Data Processing → 
Feature Engineering → XGBoost AFT Training → Model Validation → Batch Scoring → 
Business Reporting → Cleanup & Notifications
```

## 📦 Prerequisites

### Platform Requirements
- **Databricks Runtime**: 13.3.x-scala2.12 or higher
- **Cluster Configuration**: 
  - Data Processing: i3.xlarge (2 workers)
  - ML Training: i3.2xlarge (2 workers)
- **Unity Catalog**: Enabled (recommended)
- **Delta Lakes**: Enabled
- **MLflow**: Enabled

### Required Libraries
```python
# Install via cluster libraries or notebook
%pip install xgboost==1.7.6
%pip install lifelines==0.27.7
%pip install hyperopt==0.2.7
%pip install scikit-learn==1.3.0
%pip install pandas==1.5.3
```

### Permissions Required
- **Workspace Admin**: For workflow creation
- **Cluster Management**: For auto-scaling clusters
- **Unity Catalog Admin**: For table governance
- **MLflow Admin**: For model registry

## 🏗️ Infrastructure Setup

### Step 1: Create Database Structure
```sql
-- Create database hierarchy
CREATE DATABASE IF NOT EXISTS hr_analytics;
CREATE DATABASE IF NOT EXISTS hr_analytics.discovery;
CREATE DATABASE IF NOT EXISTS hr_analytics.processed;
CREATE DATABASE IF NOT EXISTS hr_analytics.predictions;
CREATE DATABASE IF NOT EXISTS hr_analytics.models;
CREATE DATABASE IF NOT EXISTS hr_analytics.reports;
CREATE DATABASE IF NOT EXISTS hr_analytics.maintenance;
CREATE DATABASE IF NOT EXISTS hr_analytics.notifications;
CREATE DATABASE IF NOT EXISTS hr_analytics.audit;
```

### Step 2: Setup Mount Points (if using cloud storage)
```python
# Mount your HR data source
dbutils.fs.mount(
    source="gs://your-hr-data-bucket",  # or s3://... or abfss://...
    mount_point="/mnt/hr-data",
    extra_configs={"your.auth.config": "value"}
)

# Mount exports location
dbutils.fs.mount(
    source="gs://your-exports-bucket",
    mount_point="/mnt/exports"
)
```

### Step 3: Configure Unity Catalog Tables
```sql
-- Register external tables if needed
CREATE TABLE hr_analytics.discovery.table_catalog
USING DELTA
LOCATION '/mnt/data/discovery/table_catalog';
```

## 📝 Configuration Setup

### Step 1: Update Configuration Parameters

For each notebook, update the configuration parameters at the top:

#### Data Discovery & Profiling (`01_data_discovery_profiling`)
```python
# Update these with your actual values
EMPLOYEE_ID_FIELD = "your_employee_id_field"  # e.g., "emp_id", "employee_number"
HR_TABLES = [
    "your_catalog.your_schema.employee_monthly",
    "your_catalog.your_schema.naics_codes",
    "your_catalog.your_schema.region_mappings",
    # Add all your HR tables here
]
```

#### Complex Data Processing (`02_complex_data_processing`)
```python
# Update field mappings
EMPLOYEE_ID_FIELD = "your_employee_id_field"
START_DATE_FIELD = "your_start_date_field"  # e.g., "employment_start_date"
END_DATE_FIELD = "your_end_date_field"      # e.g., "employment_end_date"

# Update table configuration
TABLE_CONFIG = {
    "employee_monthly": {
        "table_name": "your_catalog.your_schema.employee_monthly",
        "key_field": EMPLOYEE_ID_FIELD,
        "required_fields": [EMPLOYEE_ID_FIELD, START_DATE_FIELD, END_DATE_FIELD],
        "optional_fields": ["department", "job_title", "salary", "employment_status"]
    },
    # Update with your actual table configurations
}
```

#### Model Training (`04_xgboost_aft_training`)
```python
# Update model parameters
MIN_CONCORDANCE_INDEX = 0.7  # Adjust based on your quality requirements
FEATURE_COLUMNS = [
    # Update with features available in your data
    "log_salary", "tenure_days", "department_indicators"
]
```

#### Batch Scoring (`06_batch_scoring`)
```python
# Update business rules
RISK_TIERS = {
    "low": {"min": 0.0, "max": 0.33, "label": "Low Risk"},
    "medium": {"min": 0.33, "max": 0.67, "label": "Medium Risk"}, 
    "high": {"min": 0.67, "max": 1.0, "label": "High Risk"}
}

ALERT_THRESHOLDS = {
    "high_risk_percentage": 0.15,  # Adjust based on your business needs
}
```

### Step 2: Setup Email Notifications
Update notification recipients in `09_success_notification`:
```python
NOTIFICATION_RECIPIENTS = {
    "hr_leadership": ["your-hr-leadership@company.com"],
    "hr_analytics": ["your-hr-analytics@company.com"],  
    "data_team": ["your-data-team@company.com"],
    "ml_team": ["your-ml-team@company.com"],
    "clients": ["your-client-contact@company.com"]
}
```

## 🚀 Deployment Steps

### Step 1: Upload Notebooks
1. Create workspace folder: `/Shared/hr_survival_pipeline/`
2. Upload all notebook components:
   - `01_data_discovery_profiling`
   - `02_complex_data_processing`
   - `03_data_quality_gate`
   - `04_xgboost_aft_training`
   - `05_model_validation_gate`
   - `06_batch_scoring`
   - `07_business_reporting`
   - `08_pipeline_cleanup`
   - `09_success_notification`

### Step 2: Create Job Clusters
Create the following job clusters via the Databricks UI:

#### Data Processing Cluster
```json
{
  "cluster_name": "hr-data-processing",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "i3.xlarge",
  "num_workers": 2,
  "autotermination_minutes": 60,
  "spark_conf": {
    "spark.databricks.delta.preview.enabled": "true",
    "spark.sql.adaptive.enabled": "true"
  }
}
```

#### ML Training Cluster
```json
{
  "cluster_name": "hr-ml-training",
  "spark_version": "13.3.x-scala2.12",
  "node_type_id": "i3.2xlarge",
  "num_workers": 2,
  "autotermination_minutes": 30
}
```

### Step 3: Install Required Libraries
For each cluster, install:
- PyPI: `xgboost==1.7.6`
- PyPI: `lifelines==0.27.7`
- PyPI: `hyperopt==0.2.7`
- PyPI: `scikit-learn==1.3.0`

### Step 4: Create Databricks Workflow
1. Go to **Workflows** in Databricks
2. Click **Create Job**
3. Import the workflow JSON from `monthly_workflow_orchestration`
4. Update cluster references to match your created clusters
5. Update notebook paths to match your uploaded locations

### Step 5: Initial Data Validation
Before scheduling, run individual components manually:

```python
# Test data discovery
%run /Shared/hr_survival_pipeline/01_data_discovery_profiling

# Test data processing
%run /Shared/hr_survival_pipeline/02_complex_data_processing

# Validate data quality
%run /Shared/hr_survival_pipeline/03_data_quality_gate
```

### Step 6: Model Training Validation
```python
# Run initial model training
%run /Shared/hr_survival_pipeline/04_xgboost_aft_training

# Check model registry
import mlflow
client = mlflow.tracking.MlflowClient()
models = client.search_registered_models()
print([model.name for model in models])
```

### Step 7: End-to-End Testing
1. **Manual Run**: Execute the complete workflow manually
2. **Validate Outputs**: Check all created tables and model registry
3. **Test Notifications**: Verify notification system works
4. **Performance Check**: Monitor execution time and costs

## 📊 Monitoring & Maintenance

### Key Metrics to Monitor
1. **Data Quality**: Event rates, missing values, schema changes
2. **Model Performance**: Concordance index, prediction distribution
3. **Pipeline Health**: Execution time, failure rates, resource usage
4. **Business Impact**: High-risk employee counts, departmental trends

### Monitoring Queries
```sql
-- Check latest pipeline run
SELECT * FROM hr_analytics.audit.pipeline_summary 
ORDER BY completion_timestamp DESC LIMIT 1;

-- Monitor data quality trends
SELECT 
    validation_timestamp,
    total_records,
    event_rate,
    quality_issues_count
FROM hr_analytics.processed.data_quality_report
ORDER BY validation_timestamp DESC LIMIT 10;

-- Track model performance over time
SELECT 
    training_date,
    test_concordance,
    model_passed,
    training_samples
FROM hr_analytics.models.validation_results
ORDER BY training_date DESC LIMIT 10;

-- Monitor high-risk employee trends
SELECT 
    scoring_date,
    total_employees,
    high_risk_count,
    high_risk_count::float / total_employees as high_risk_rate
FROM hr_analytics.predictions.scoring_metadata
ORDER BY scoring_date DESC LIMIT 6;
```

### Automated Monitoring Setup
Create alerts in Databricks for:
- Pipeline failures
- Data quality degradation
- Model performance drops
- Unusual high-risk percentages

## 🔧 Troubleshooting

### Common Issues & Solutions

#### 1. Data Quality Gate Failures
**Symptoms**: Pipeline stops at data quality validation
**Solutions**:
- Check data source availability
- Verify field mappings in configuration
- Review minimum threshold settings
- Validate date formats and ranges

#### 2. Model Training Failures
**Symptoms**: Low concordance index, training errors
**Solutions**:
- Increase minimum training samples
- Review feature engineering logic
- Check for data leakage
- Adjust hyperparameter ranges

#### 3. Batch Scoring Issues
**Symptoms**: Scoring failures, unexpected risk distributions
**Solutions**:
- Verify model registry access
- Check feature consistency between training and scoring
- Validate active employee identification logic
- Review risk tier thresholds

#### 4. Performance Issues
**Symptoms**: Long execution times, high costs
**Solutions**:
- Optimize cluster sizes
- Enable Delta table optimization
- Review data partitioning strategy
- Consider caching intermediate results

### Debugging Commands
```python
# Check table accessibility
tables_to_check = [
    "hr_analytics.processed.survival_features",
    "hr_analytics.predictions.employee_risk_scores"
]

for table in tables_to_check:
    try:
        count = spark.table(table).count()
        print(f"✅ {table}: {count:,} records")
    except Exception as e:
        print(f"❌ {table}: {str(e)}")

# Check model registry
import mlflow
client = mlflow.tracking.MlflowClient()
try:
    models = client.get_latest_versions("employee_survival_xgboost_aft", stages=["Production"])
    print(f"✅ Production model: Version {models[0].version}" if models else "❌ No production model")
except Exception as e:
    print(f"❌ Model registry error: {str(e)}")

# Check recent pipeline runs
recent_runs = spark.sql("""
    SELECT completion_timestamp, pipeline_status, employees_analyzed
    FROM hr_analytics.audit.pipeline_summary 
    ORDER BY completion_timestamp DESC LIMIT 5
""").show()
```

## 📈 Optimization Tips

### Performance Optimization
1. **Cluster Sizing**: Start small, scale based on actual usage
2. **Delta Optimization**: Enable auto-optimize and auto-compact
3. **Z-Ordering**: Implement on frequently queried columns
4. **Caching**: Cache intermediate datasets for complex transformations

### Cost Optimization
1. **Spot Instances**: Use for non-critical workloads
2. **Auto-termination**: Set appropriate timeouts
3. **Resource Monitoring**: Monitor DBU usage patterns
4. **Data Retention**: Implement appropriate retention policies

### Model Optimization
1. **Feature Selection**: Regularly review feature importance
2. **Hyperparameter Tuning**: Enable periodic retuning
3. **Model Validation**: Implement robust validation frameworks
4. **A/B Testing**: Compare model versions in production

## 🔒 Security & Compliance

### Data Privacy
- Employee ID anonymization options
- PII handling procedures
- Data access audit trails
- Retention policy compliance

### Access Control
- Unity Catalog permissions
- Workspace access controls
- Model registry permissions
- Export data security

## 📅 Maintenance Schedule

### Monthly (Automated)
- Pipeline execution (25th of each month)
- Data quality validation
- Model performance monitoring
- Business report generation

### Quarterly (Manual)
- Model retraining with expanded features
- Hyperparameter optimization
- Performance benchmark review
- Cost optimization analysis

### Annually (Manual)
- Complete architecture review
- Security audit
- Compliance validation
- Technology stack updates

## 🎯 Success Criteria

### Technical KPIs
- ✅ Pipeline success rate > 95%
- ✅ Model concordance index > 0.70
- ✅ Data quality score > 80%
- ✅ End-to-end execution < 6 hours

### Business KPIs
- ✅ High-risk employee identification accuracy
- ✅ Retention intervention success rate > 30%
- ✅ Cost avoidance from prevented turnover
- ✅ HR partner satisfaction scores

## 📞 Support & Contact

### Technical Support
- **Data Engineering**: data-team@company.com
- **ML Engineering**: ml-team@company.com
- **Databricks Support**: Via your enterprise support channel

### Business Support
- **HR Analytics**: hr-analytics@company.com
- **Business Users**: hr-leadership@company.com

---

## 🎉 Deployment Checklist

Before going live, ensure:

- [ ] All configuration parameters updated
- [ ] Data sources accessible and validated
- [ ] Clusters created and libraries installed
- [ ] Notebooks uploaded and tested individually
- [ ] Workflow created and configured
- [ ] Email notifications configured
- [ ] Manual end-to-end test completed successfully
- [ ] Monitoring dashboards setup
- [ ] Documentation reviewed with stakeholders
- [ ] Support contacts established
- [ ] Backup and recovery procedures defined

**Once all items are checked, you're ready to enable the monthly schedule!**

---

*This deployment guide provides comprehensive instructions for implementing the HR Survival Analysis Pipeline. For additional support or customization requirements, contact the development team.*