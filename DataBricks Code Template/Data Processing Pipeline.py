
# Complex Multi-Table Data Processing Pipeline
# Processes multiple HR tables into survival analysis ready dataset

# =============================================================================
# CONFIGURATION PARAMETERS (MODIFY THESE AS NEEDED)
# =============================================================================

# Employee identifier (configurable)
EMPLOYEE_ID_FIELD = "employee_id"  # Change to your actual employee ID field name

# Date field names (configurable for start-stop format)
START_DATE_FIELD = "start_date"  # Employment start date
END_DATE_FIELD = "end_date"      # Employment end date (null for active employees)
HIRE_DATE_FIELD = "hire_date"    # Original hire date

# Table mappings (update with your actual table names and key fields)
TABLE_CONFIG = {
    "employee_monthly": {
        "table_name": "catalog.hr_schema.employee_monthly",
        "key_field": EMPLOYEE_ID_FIELD,
        "required_fields": [EMPLOYEE_ID_FIELD, START_DATE_FIELD, END_DATE_FIELD],
        "optional_fields": ["department", "job_title", "salary", "employment_status"]
    },
    "naics": {
        "table_name": "catalog.reference.naics",
        "key_field": "naics_code",
        "required_fields": ["naics_code", "naics_description"],
        "join_field": "naics_code"  # Field to join with employee data
    },
    "region_mappings": {
        "table_name": "catalog.reference.region_mappings",
        "key_field": "region_code",
        "required_fields": ["region_code", "region_name"],
        "join_field": "region_code"
    }
    # Add more tables as needed
}

# Termination criteria (configurable)
VOLUNTARY_TERMINATION_CODES = ["VOLUNTARY", "RESIGNATION", "QUIT"]  # Update with your actual codes
EXCLUDE_EMPLOYMENT_TYPES = ["CONTRACTOR", "INTERN", "TEMP"]  # Exclude these types

# Feature engineering parameters
MIN_TENURE_DAYS = 30  # Minimum tenure to include in analysis
MAX_TENURE_DAYS = 3650  # Maximum reasonable tenure (10 years)
VANTAGE_DATE_MONTHS_BACK = 12  # Look back 12 months from vantage point

# Output table locations
PROCESSED_EMPLOYEE_TABLE = "hr_analytics.processed.employee_master"
SURVIVAL_FEATURES_TABLE = "hr_analytics.processed.survival_features"
DATA_QUALITY_REPORT_TABLE = "hr_analytics.processed.data_quality_report"

# =============================================================================
# IMPORTS
# =============================================================================

from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pandas as pd
from datetime import datetime, timedelta
import warnings

print("🏗️ Starting Complex Data Processing Pipeline...")
print(f"🎯 Employee ID Field: {EMPLOYEE_ID_FIELD}")
print(f"📅 Date Fields: {START_DATE_FIELD}, {END_DATE_FIELD}")
print(f"📊 Processing {len(TABLE_CONFIG)} configured tables")

# =============================================================================
# 1. DATA VALIDATION & LOADING
# =============================================================================

def validate_and_load_table(table_config, table_key):
    """Validate and load a single table with error handling"""
    
    table_name = table_config["table_name"]
    required_fields = table_config["required_fields"]
    
    print(f"📋 Loading and validating: {table_name}")
    
    try:
        # Load table
        df = spark.table(table_name)
        existing_columns = df.columns
        
        # Check required fields
        missing_fields = [field for field in required_fields if field not in existing_columns]
        if missing_fields:
            print(f"  ⚠️ Missing required fields in {table_name}: {missing_fields}")
            return None, f"Missing fields: {missing_fields}"
        
        # Basic validation
        row_count = df.count()
        if row_count == 0:
            print(f"  ⚠️ Table {table_name} is empty")
            return None, "Empty table"
        
        print(f"  ✅ {table_name}: {row_count:,} rows, {len(existing_columns)} columns")
        
        # Add metadata
        df_with_meta = df.withColumn("source_table", lit(table_key))
        
        return df_with_meta, None
        
    except Exception as e:
        error_msg = f"Error loading {table_name}: {str(e)}"
        print(f"  ❌ {error_msg}")
        return None, error_msg

def load_all_tables():
    """Load and validate all configured tables"""
    
    print("\n📊 Loading all configured tables...")
    
    loaded_tables = {}
    load_errors = {}
    
    for table_key, table_config in TABLE_CONFIG.items():
        df, error = validate_and_load_table(table_config, table_key)
        if df is not None:
            loaded_tables[table_key] = df
        else:
            load_errors[table_key] = error
    
    print(f"\n✅ Successfully loaded: {len(loaded_tables)} tables")
    print(f"❌ Failed to load: {len(load_errors)} tables")
    
    if load_errors:
        print("❌ Load errors:")
        for table, error in load_errors.items():
            print(f"   - {table}: {error}")
    
    return loaded_tables, load_errors

# =============================================================================
# 2. EMPLOYEE MASTER DATA CREATION
# =============================================================================

def create_employee_master(employee_monthly_df):
    """Create master employee record from monthly data"""
    
    print("\n👥 Creating employee master records...")
    
    # Handle start-stop format data
    # Assumption: Each row represents an employment period
    
    # Clean and prepare data
    cleaned_df = (employee_monthly_df
                  .filter(col(EMPLOYEE_ID_FIELD).isNotNull())
                  .filter(col(START_DATE_FIELD).isNotNull())
                  .withColumn(START_DATE_FIELD, col(START_DATE_FIELD).cast("date"))
                  .withColumn(END_DATE_FIELD, col(END_DATE_FIELD).cast("date")))
    
    # Exclude unwanted employment types if employment_status exists
    if "employment_status" in cleaned_df.columns:
        cleaned_df = cleaned_df.filter(
            ~upper(col("employment_status")).isin(EXCLUDE_EMPLOYMENT_TYPES)
        )
    
    # Calculate tenure for each employment period
    current_date = datetime.now().date()
    
    tenure_df = (cleaned_df
                 .withColumn("period_end_date", 
                           coalesce(col(END_DATE_FIELD), lit(current_date)))
                 .withColumn("tenure_days", 
                           datediff(col("period_end_date"), col(START_DATE_FIELD)))
                 .filter(col("tenure_days") >= MIN_TENURE_DAYS)
                 .filter(col("tenure_days") <= MAX_TENURE_DAYS))
    
    # Identify termination events (voluntary only)
    termination_df = tenure_df.withColumn(
        "is_voluntary_termination",
        when(col(END_DATE_FIELD).isNotNull(), 1).otherwise(0)
    )
    
    # Filter for voluntary terminations if employment_status exists
    if "employment_status" in termination_df.columns:
        termination_df = termination_df.withColumn(
            "is_voluntary_termination",
            when(
                (col(END_DATE_FIELD).isNotNull()) & 
                (upper(col("employment_status")).isin(VOLUNTARY_TERMINATION_CODES)),
                1
            ).otherwise(0)
        )
    
    # For employees with multiple periods, get the most recent/relevant period
    window_spec = Window.partitionBy(EMPLOYEE_ID_FIELD).orderBy(col(START_DATE_FIELD).desc())
    
    master_df = (termination_df
                 .withColumn("row_number", row_number().over(window_spec))
                 .filter(col("row_number") == 1)
                 .drop("row_number"))
    
    print(f"✅ Created master records for {master_df.count():,} employees")
    
    return master_df

# =============================================================================
# 3. REFERENCE DATA INTEGRATION
# =============================================================================

def integrate_reference_data(master_df, loaded_tables):
    """Integrate reference data (NAICS, region mappings) with employee data"""
    
    print("\n🔗 Integrating reference data...")
    
    enriched_df = master_df
    
    # Integrate NAICS data if available
    if "naics" in loaded_tables:
        naics_df = loaded_tables["naics"]
        
        # Assume employee data has naics_code field to join on
        if "naics_code" in master_df.columns:
            enriched_df = (enriched_df
                          .join(naics_df.select("naics_code", "naics_description"), 
                               on="naics_code", how="left"))
            print("  ✅ Integrated NAICS industry data")
        else:
            print("  ⚠️ No naics_code field found in employee data")
    
    # Integrate region data if available
    if "region_mappings" in loaded_tables:
        region_df = loaded_tables["region_mappings"]
        
        # Assume employee data has region_code field to join on
        if "region_code" in master_df.columns:
            enriched_df = (enriched_df
                          .join(region_df.select("region_code", "region_name"), 
                               on="region_code", how="left"))
            print("  ✅ Integrated region mapping data")
        else:
            print("  ⚠️ No region_code field found in employee data")
    
    return enriched_df

# =============================================================================
# 4. SURVIVAL ANALYSIS FEATURE ENGINEERING
# =============================================================================

def create_survival_features(enriched_df):
    """Create features specifically for survival analysis"""
    
    print("\n🎯 Creating survival analysis features...")
    
    # Calculate vantage date (12 months back from current date)
    current_date = datetime.now().date()
    vantage_date = current_date - timedelta(days=VANTAGE_DATE_MONTHS_BACK * 30)
    
    # Create survival analysis specific columns
    survival_df = (enriched_df
                   .withColumn("vantage_date", lit(vantage_date))
                   .withColumn("current_date", lit(current_date))
                   
                   # Target variable: observed event (1 if terminated, 0 if censored)
                   .withColumn("event_observed", col("is_voluntary_termination"))
                   
                   # Time to event (tenure in days)
                   .withColumn("time_to_event", col("tenure_days"))
                   
                   # Create categorical features
                   .withColumn("department_clean", 
                             when(col("department").isNull(), "Unknown")
                             .otherwise(upper(trim(col("department")))))
                   
                   .withColumn("job_title_clean",
                             when(col("job_title").isNull(), "Unknown")
                             .otherwise(upper(trim(col("job_title")))))
                   
                   # Salary features
                   .withColumn("salary_clean", 
                             when(col("salary").isNull(), 0)
                             .when(col("salary") <= 0, 0)
                             .otherwise(col("salary")))
                   
                   .withColumn("log_salary", 
                             when(col("salary_clean") > 0, log(col("salary_clean")))
                             .otherwise(0))
                   
                   # Tenure buckets
                   .withColumn("tenure_bucket",
                             when(col("tenure_days") <= 90, "0-3_months")
                             .when(col("tenure_days") <= 365, "3-12_months")
                             .when(col("tenure_days") <= 730, "1-2_years")
                             .otherwise("2+_years"))
                   
                   # Industry features (if NAICS available)
                   .withColumn("has_industry_data", 
                             when(col("naics_description").isNotNull(), 1).otherwise(0))
                   
                   # Processing metadata
                   .withColumn("processing_date", lit(current_date))
                   .withColumn("vantage_date_used", lit(vantage_date)))
    
    # One-hot encode department (limit to top departments to avoid too many features)
    dept_counts = (survival_df
                   .groupBy("department_clean")
                   .count()
                   .orderBy(col("count").desc())
                   .limit(10)
                   .select("department_clean")
                   .collect())
    
    top_departments = [row.department_clean for row in dept_counts]
    
    # Create department indicator variables
    for dept in top_departments:
        survival_df = survival_df.withColumn(
            f"dept_{dept.lower().replace(' ', '_').replace('-', '_')}",
            when(col("department_clean") == dept, 1).otherwise(0)
        )
    
    print(f"✅ Created survival features with {len(top_departments)} department indicators")
    
    return survival_df

# =============================================================================
# 5. DATA QUALITY VALIDATION
# =============================================================================

def validate_survival_data(survival_df):
    """Validate the final survival analysis dataset"""
    
    print("\n✅ Validating survival analysis dataset...")
    
    # Basic counts
    total_records = survival_df.count()
    event_count = survival_df.filter(col("event_observed") == 1).count()
    censored_count = total_records - event_count
    
    # Data quality checks
    quality_issues = []
    
    # Check for minimum events
    if event_count < 50:
        quality_issues.append(f"Too few events: {event_count} (need at least 50)")
    
    # Check event rate
    event_rate = event_count / total_records if total_records > 0 else 0
    if event_rate < 0.05 or event_rate > 0.5:
        quality_issues.append(f"Unusual event rate: {event_rate:.3f} (expect 0.05-0.50)")
    
    # Check for missing critical data
    null_employees = survival_df.filter(col(EMPLOYEE_ID_FIELD).isNull()).count()
    if null_employees > 0:
        quality_issues.append(f"Missing employee IDs: {null_employees}")
    
    null_tenure = survival_df.filter(col("time_to_event").isNull()).count()
    if null_tenure > 0:
        quality_issues.append(f"Missing tenure data: {null_tenure}")
    
    # Create quality report
    quality_report = {
        "total_records": total_records,
        "event_count": event_count,
        "censored_count": censored_count,
        "event_rate": event_rate,
        "quality_issues": quality_issues,
        "validation_timestamp": datetime.now()
    }
    
    print(f"📊 Validation Results:")
    print(f"   - Total records: {total_records:,}")
    print(f"   - Events (terminations): {event_count:,}")
    print(f"   - Censored (active): {censored_count:,}")
    print(f"   - Event rate: {event_rate:.3f}")
    
    if quality_issues:
        print(f"⚠️ Quality Issues Found:")
        for issue in quality_issues:
            print(f"   - {issue}")
        
        # Decide whether to proceed
        critical_issues = [issue for issue in quality_issues 
                         if "Too few events" in issue or "Missing employee IDs" in issue]
        
        if critical_issues:
            print("❌ Critical data quality issues found. Cannot proceed.")
            return False, quality_report
        else:
            print("⚠️ Minor quality issues found. Proceeding with warnings.")
    else:
        print("✅ All data quality checks passed!")
    
    return True, quality_report

# =============================================================================
# 6. MAIN PIPELINE EXECUTION
# =============================================================================

def run_data_processing_pipeline():
    """Main function to run the complete data processing pipeline"""
    
    print("🚀 Starting complete data processing pipeline...")
    start_time = datetime.now()
    
    try:
        # Step 1: Load all tables
        loaded_tables, load_errors = load_all_tables()
        
        if "employee_monthly" not in loaded_tables:
            raise Exception("❌ Cannot proceed without employee_monthly table")
        
        # Step 2: Create employee master
        master_df = create_employee_master(loaded_tables["employee_monthly"])
        
        # Step 3: Integrate reference data
        enriched_df = integrate_reference_data(master_df, loaded_tables)
        
        # Step 4: Create survival features
        survival_df = create_survival_features(enriched_df)
        
        # Step 5: Validate data quality
        is_valid, quality_report = validate_survival_data(survival_df)
        
        if not is_valid:
            raise Exception("❌ Data quality validation failed")
        
        # Step 6: Save processed data
        print("\n💾 Saving processed data...")
        
        # Save employee master
        (enriched_df
         .write
         .format("delta")
         .mode("overwrite")
         .option("mergeSchema", "true")
         .saveAsTable(PROCESSED_EMPLOYEE_TABLE))
        
        print(f"✅ Employee master saved to: {PROCESSED_EMPLOYEE_TABLE}")
        
        # Save survival features
        (survival_df
         .write
         .format("delta")
         .mode("overwrite")
         .option("mergeSchema", "true")
         .saveAsTable(SURVIVAL_FEATURES_TABLE))
        
        print(f"✅ Survival features saved to: {SURVIVAL_FEATURES_TABLE}")
        
        # Save quality report
        quality_df = spark.createDataFrame([quality_report])
        (quality_df
         .write
         .format("delta")
         .mode("overwrite")
         .option("mergeSchema", "true")
         .saveAsTable(DATA_QUALITY_REPORT_TABLE))
        
        print(f"✅ Quality report saved to: {DATA_QUALITY_REPORT_TABLE}")
        
        # Pipeline summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🎉 DATA PROCESSING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"⏱️ Total execution time: {duration}")
        print(f"📊 Final dataset: {survival_df.count():,} employee records")
        print(f"🎯 Event rate: {quality_report['event_rate']:.3f}")
        print(f"📋 Tables created:")
        print(f"   - Employee master: {PROCESSED_EMPLOYEE_TABLE}")
        print(f"   - Survival features: {SURVIVAL_FEATURES_TABLE}")
        print(f"   - Quality report: {DATA_QUALITY_REPORT_TABLE}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        return False

# Run the pipeline
if __name__ == "__main__":
    success = run_data_processing_pipeline()
    if not success:
        raise Exception("Data processing pipeline failed")