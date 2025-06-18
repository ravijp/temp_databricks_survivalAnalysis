# Data Quality Gate - Critical validation before model training
# This notebook performs comprehensive data quality checks and fails the pipeline if critical issues are found

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Data quality thresholds (configurable)
MIN_EVENTS = int(dbutils.widgets.get("min_events") if "min_events" in [w.name for w in dbutils.widgets.getAll()] else 100)
MAX_EVENT_RATE = float(dbutils.widgets.get("max_event_rate") if "max_event_rate" in [w.name for w in dbutils.widgets.getAll()] else 0.5)
MIN_EVENT_RATE = float(dbutils.widgets.get("min_event_rate") if "min_event_rate" in [w.name for w in dbutils.widgets.getAll()] else 0.05)
FAIL_ON_QA_ISSUES = dbutils.widgets.get("fail_on_quality_issues") if "fail_on_quality_issues" in [w.name for w in dbutils.widgets.getAll()] else "true"

# Table locations
SURVIVAL_FEATURES_TABLE = "hr_analytics.processed.survival_features"
DATA_QUALITY_REPORT_TABLE = "hr_analytics.processed.data_quality_report"

print("🔍 Data Quality Gate - Critical Pipeline Validation")
print(f"📊 Minimum events required: {MIN_EVENTS}")
print(f"📊 Event rate range: {MIN_EVENT_RATE:.3f} - {MAX_EVENT_RATE:.3f}")
print(f"🚨 Fail on issues: {FAIL_ON_QA_ISSUES}")

# =============================================================================
# IMPORTS
# =============================================================================

from pyspark.sql.functions import *
from datetime import datetime
import sys

# =============================================================================
# DATA QUALITY VALIDATION
# =============================================================================

def run_data_quality_gate():
    """Run comprehensive data quality validation"""
    
    print("\n🔍 Starting Data Quality Gate validation...")
    
    # Load survival features
    try:
        df = spark.table(SURVIVAL_FEATURES_TABLE)
        total_records = df.count()
        print(f"📊 Total records: {total_records:,}")
    except Exception as e:
        print(f"❌ CRITICAL: Cannot load survival features table: {str(e)}")
        if FAIL_ON_QA_ISSUES.lower() == "true":
            raise Exception("DATA QUALITY GATE FAILED: Cannot load data")
        return False
    
    # Initialize quality checks
    quality_issues = []
    warnings = []
    
    # Check 1: Minimum record count
    if total_records < 500:
        quality_issues.append(f"Too few records: {total_records} (need at least 500)")
    
    # Check 2: Event analysis
    event_count = df.filter(col("event_observed") == 1).count()
    event_rate = event_count / total_records if total_records > 0 else 0
    
    print(f"📊 Events: {event_count:,} ({event_rate:.3f} rate)")
    
    if event_count < MIN_EVENTS:
        quality_issues.append(f"Too few events: {event_count} (need at least {MIN_EVENTS})")
    
    if event_rate > MAX_EVENT_RATE:
        quality_issues.append(f"Event rate too high: {event_rate:.3f} (max {MAX_EVENT_RATE})")
    
    if event_rate < MIN_EVENT_RATE:
        quality_issues.append(f"Event rate too low: {event_rate:.3f} (min {MIN_EVENT_RATE})")
    
    # Check 3: Required fields
    required_fields = ["employee_id", "time_to_event", "event_observed"]
    for field in required_fields:
        null_count = df.filter(col(field).isNull()).count()
        if null_count > 0:
            quality_issues.append(f"Missing {field}: {null_count} records")
    
    # Check 4: Data ranges
    # Tenure should be reasonable
    tenure_stats = df.select(
        min("time_to_event").alias("min_tenure"),
        max("time_to_event").alias("max_tenure"),
        avg("time_to_event").alias("avg_tenure")
    ).collect()[0]
    
    if tenure_stats.min_tenure < 0:
        quality_issues.append(f"Negative tenure detected: {tenure_stats.min_tenure}")
    
    if tenure_stats.max_tenure > 5000:  # > ~13 years
        warnings.append(f"Very high tenure detected: {tenure_stats.max_tenure} days")
    
    # Check 5: Feature completeness
    feature_columns = [col for col in df.columns if col.startswith(('log_', 'dept_', 'has_'))]
    
    for col_name in feature_columns:
        null_pct = df.filter(col(col_name).isNull()).count() / total_records
        if null_pct > 0.5:
            warnings.append(f"High null rate in {col_name}: {null_pct:.1%}")
    
    # Check 6: Recent data availability
    latest_date = df.select(max("processing_date")).collect()[0][0]
    if latest_date:
        days_old = (datetime.now().date() - latest_date).days
        if days_old > 7:
            warnings.append(f"Data is {days_old} days old")
    
    # Summary
    print(f"\n📊 Data Quality Summary:")
    print(f"   - Total records: {total_records:,}")
    print(f"   - Event count: {event_count:,}")
    print(f"   - Event rate: {event_rate:.3f}")
    print(f"   - Average tenure: {tenure_stats.avg_tenure:.0f} days")
    print(f"   - Quality issues: {len(quality_issues)}")
    print(f"   - Warnings: {len(warnings)}")
    
    # Report issues
    if quality_issues:
        print(f"\n❌ QUALITY ISSUES FOUND:")
        for issue in quality_issues:
            print(f"   - {issue}")
    
    if warnings:
        print(f"\n⚠️ WARNINGS:")
        for warning in warnings:
            print(f"   - {warning}")
    
    # Create quality report
    quality_report = {
        "validation_timestamp": datetime.now(),
        "total_records": total_records,
        "event_count": event_count,
        "event_rate": event_rate,
        "avg_tenure_days": float(tenure_stats.avg_tenure),
        "quality_issues_count": len(quality_issues),
        "warnings_count": len(warnings),
        "quality_issues": quality_issues,
        "warnings": warnings,
        "gate_passed": len(quality_issues) == 0
    }
    
    # Save quality report
    try:
        quality_df = spark.createDataFrame([quality_report])
        (quality_df
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .saveAsTable(DATA_QUALITY_REPORT_TABLE))
        
        print(f"✅ Quality report saved to: {DATA_QUALITY_REPORT_TABLE}")
    except Exception as e:
        print(f"⚠️ Failed to save quality report: {str(e)}")
    
    # Decision logic
    if len(quality_issues) == 0:
        print("\n✅ DATA QUALITY GATE PASSED")
        return True
    else:
        print(f"\n❌ DATA QUALITY GATE FAILED: {len(quality_issues)} critical issues")
        
        if FAIL_ON_QA_ISSUES.lower() == "true":
            raise Exception(f"DATA QUALITY GATE FAILED: {'; '.join(quality_issues)}")
        else:
            print("⚠️ Proceeding despite quality issues (fail_on_quality_issues=false)")
            return False

# Run the gate
success = run_data_quality_gate()
if not success and FAIL_ON_QA_ISSUES.lower() == "true":
    dbutils.notebook.exit("FAILED")
    