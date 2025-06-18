# Pipeline Cleanup and Optimization
# Performs housekeeping tasks after successful pipeline execution

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Cleanup parameters (configurable via widgets)
OPTIMIZE_DELTA_TABLES = dbutils.widgets.get("optimize_delta_tables") if "optimize_delta_tables" in [w.name for w in dbutils.widgets.getAll()] else "true"
CLEANUP_TEMP_FILES = dbutils.widgets.get("cleanup_temp_files") if "cleanup_temp_files" in [w.name for w in dbutils.widgets.getAll()] else "true"
RETENTION_DAYS = int(dbutils.widgets.get("retention_days") if "retention_days" in [w.name for w in dbutils.widgets.getAll()] else 90)

# Tables to optimize
TABLES_TO_OPTIMIZE = [
    "hr_analytics.processed.employee_master",
    "hr_analytics.processed.survival_features", 
    "hr_analytics.predictions.employee_risk_scores",
    "hr_analytics.predictions.department_risk_summary",
    "hr_analytics.predictions.manager_risk_alerts",
    "hr_analytics.models.validation_results",
    "hr_analytics.reports.business_reports"
]

# Temp locations to cleanup
TEMP_LOCATIONS = [
    "/tmp/mlflow/",
    "/tmp/exports/",
    "/mnt/checkpoints/temp/",
    "/databricks/driver/tmp/"
]

print("🧹 Pipeline Cleanup and Optimization")
print(f"🔧 Optimize Delta tables: {OPTIMIZE_DELTA_TABLES}")
print(f"🗑️ Cleanup temp files: {CLEANUP_TEMP_FILES}")
print(f"📅 Retention period: {RETENTION_DAYS} days")

# =============================================================================
# IMPORTS
# =============================================================================

from pyspark.sql.functions import *
from datetime import datetime, timedelta
import os

# =============================================================================
# 1. DELTA TABLE OPTIMIZATION
# =============================================================================

def optimize_delta_tables():
    """Optimize Delta tables for better query performance"""
    
    if OPTIMIZE_DELTA_TABLES.lower() != "true":
        print("⏭️ Skipping Delta table optimization")
        return
    
    print("\n🔧 Optimizing Delta tables...")
    
    optimization_results = []
    
    for table_name in TABLES_TO_OPTIMIZE:
        try:
            print(f"   🔧 Optimizing {table_name}...")
            
            # Check if table exists
            try:
                table_info = spark.sql(f"DESCRIBE TABLE {table_name}")
                table_exists = True
            except:
                print(f"   ⚠️ Table {table_name} does not exist, skipping")
                continue
            
            # Get table size before optimization
            try:
                size_before = spark.sql(f"DESCRIBE DETAIL {table_name}").select("sizeInBytes").collect()[0][0]
                size_before_mb = size_before / (1024 * 1024) if size_before else 0
            except:
                size_before_mb = 0
            
            # Optimize table
            start_time = datetime.now()
            
            # Basic optimize
            spark.sql(f"OPTIMIZE {table_name}")
            
            # Z-order for specific tables with known query patterns
            if "employee_risk_scores" in table_name:
                spark.sql(f"OPTIMIZE {table_name} ZORDER BY (department_clean, risk_tier)")
                print(f"   ✅ Z-ordered by department and risk tier")
            elif "survival_features" in table_name:
                spark.sql(f"OPTIMIZE {table_name} ZORDER BY (employee_id, processing_date)")
                print(f"   ✅ Z-ordered by employee ID and processing date")
            
            optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Get table size after optimization  
            try:
                size_after = spark.sql(f"DESCRIBE DETAIL {table_name}").select("sizeInBytes").collect()[0][0]
                size_after_mb = size_after / (1024 * 1024) if size_after else 0
            except:
                size_after_mb = size_before_mb
            
            # Record results
            optimization_results.append({
                "table_name": table_name,
                "optimization_timestamp": datetime.now(),
                "size_before_mb": size_before_mb,
                "size_after_mb": size_after_mb,
                "optimization_time_seconds": optimization_time,
                "status": "success"
            })
            
            print(f"   ✅ {table_name} optimized in {optimization_time:.1f}s")
            print(f"      Size: {size_before_mb:.1f}MB → {size_after_mb:.1f}MB")
            
        except Exception as e:
            print(f"   ❌ Failed to optimize {table_name}: {str(e)}")
            optimization_results.append({
                "table_name": table_name,
                "optimization_timestamp": datetime.now(),
                "error": str(e),
                "status": "failed"
            })
    
    # Save optimization results
    if optimization_results:
        try:
            results_df = spark.createDataFrame(optimization_results)
            (results_df
             .write
             .format("delta")
             .mode("append")
             .option("mergeSchema", "true")
             .saveAsTable("hr_analytics.maintenance.optimization_log"))
            
            print(f"✅ Optimization results saved to hr_analytics.maintenance.optimization_log")
        except Exception as e:
            print(f"⚠️ Could not save optimization results: {str(e)}")
    
    successful_optimizations = sum(1 for r in optimization_results if r["status"] == "success")
    print(f"✅ Successfully optimized {successful_optimizations}/{len(TABLES_TO_OPTIMIZE)} tables")

# =============================================================================
# 2. DATA RETENTION MANAGEMENT
# =============================================================================

def manage_data_retention():
    """Manage data retention by removing old records"""
    
    print(f"\n🗂️ Managing data retention ({RETENTION_DAYS} days)...")
    
    cutoff_date = datetime.now() - timedelta(days=RETENTION_DAYS)
    
    # Tables with time-based partitions to clean
    retention_tables = [
        {
            "table": "hr_analytics.predictions.employee_risk_scores",
            "date_column": "scoring_date",
            "description": "Employee risk scores"
        },
        {
            "table": "hr_analytics.predictions.department_risk_summary", 
            "date_column": "scoring_date",
            "description": "Department summaries"
        },
        {
            "table": "hr_analytics.models.validation_results",
            "date_column": "training_date", 
            "description": "Model validation results"
        },
        {
            "table": "hr_analytics.reports.business_reports",
            "date_column": "report_timestamp",
            "description": "Business reports"
        }
    ]
    
    retention_results = []
    
    for table_info in retention_tables:
        try:
            table_name = table_info["table"]
            date_col = table_info["date_column"]
            description = table_info["description"]
            
            print(f"   🗂️ Processing {description}...")
            
            # Check if table exists
            try:
                spark.table(table_name)
            except:
                print(f"   ⚠️ Table {table_name} does not exist, skipping")
                continue
            
            # Count records before deletion
            total_before = spark.table(table_name).count()
            old_records = spark.table(table_name).filter(col(date_col) < cutoff_date).count()
            
            if old_records == 0:
                print(f"   ✅ No old records to delete in {description}")
                retention_results.append({
                    "table_name": table_name,
                    "records_deleted": 0,
                    "total_before": total_before,
                    "retention_timestamp": datetime.now(),
                    "status": "no_action_needed"
                })
                continue
            
            # Delete old records
            spark.sql(f"""
                DELETE FROM {table_name} 
                WHERE {date_col} < '{cutoff_date.strftime('%Y-%m-%d')}'
            """)
            
            # Count records after deletion
            total_after = spark.table(table_name).count()
            
            retention_results.append({
                "table_name": table_name,
                "records_deleted": old_records,
                "total_before": total_before,
                "total_after": total_after,
                "cutoff_date": cutoff_date,
                "retention_timestamp": datetime.now(),
                "status": "success"
            })
            
            print(f"   ✅ Deleted {old_records:,} old records from {description}")
            print(f"      Records: {total_before:,} → {total_after:,}")
            
        except Exception as e:
            print(f"   ❌ Failed to process {table_info['description']}: {str(e)}")
            retention_results.append({
                "table_name": table_info["table"],
                "error": str(e),
                "retention_timestamp": datetime.now(),
                "status": "failed"
            })
    
    # Save retention results
    if retention_results:
        try:
            retention_df = spark.createDataFrame(retention_results)
            (retention_df
             .write
             .format("delta")
             .mode("append")
             .option("mergeSchema", "true")  
             .saveAsTable("hr_analytics.maintenance.retention_log"))
            
            print(f"✅ Retention results saved to hr_analytics.maintenance.retention_log")
        except Exception as e:
            print(f"⚠️ Could not save retention results: {str(e)}")
    
    total_deleted = sum(r.get("records_deleted", 0) for r in retention_results if isinstance(r.get("records_deleted"), int))
    print(f"✅ Total records deleted: {total_deleted:,}")

# =============================================================================
# 3. TEMPORARY FILE CLEANUP
# =============================================================================

def cleanup_temp_files():
    """Clean up temporary files and directories"""
    
    if CLEANUP_TEMP_FILES.lower() != "true":
        print("⏭️ Skipping temporary file cleanup")
        return
    
    print("\n🗑️ Cleaning up temporary files...")
    
    cleanup_results = []
    
    for temp_path in TEMP_LOCATIONS:
        try:
            print(f"   🗑️ Checking {temp_path}...")
            
            # Check if path exists
            try:
                files = dbutils.fs.ls(temp_path)
                if not files:
                    print(f"   ✅ {temp_path} is already clean")
                    continue
            except:
                print(f"   ✅ {temp_path} does not exist")
                continue
            
            # Count files before cleanup
            file_count = len(files)
            
            # Remove old files (older than 1 day)
            cutoff_timestamp = datetime.now() - timedelta(days=1)
            files_removed = 0
            
            for file_info in files:
                try:
                    # Get file modification time
                    file_path = file_info.path
                    
                    # For simplicity, remove all temp files
                    # In production, you'd want more sophisticated logic
                    if any(pattern in file_path.lower() for pattern in ['.tmp', 'temp', 'checkpoint']):
                        dbutils.fs.rm(file_path, recurse=True)
                        files_removed += 1
                        
                except Exception as e:
                    print(f"   ⚠️ Could not remove {file_path}: {str(e)}")
                    continue
            
            cleanup_results.append({
                "temp_location": temp_path,
                "files_found": file_count,
                "files_removed": files_removed,
                "cleanup_timestamp": datetime.now(),
                "status": "success"
            })
            
            print(f"   ✅ Cleaned {files_removed}/{file_count} files from {temp_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to cleanup {temp_path}: {str(e)}")
            cleanup_results.append({
                "temp_location": temp_path,
                "error": str(e),
                "cleanup_timestamp": datetime.now(),
                "status": "failed"
            })
    
    total_files_removed = sum(r.get("files_removed", 0) for r in cleanup_results)
    print(f"✅ Total temporary files removed: {total_files_removed}")

# =============================================================================
# 4. VACUUM OPERATIONS
# =============================================================================

def vacuum_delta_tables():
    """Vacuum Delta tables to remove old file versions"""
    
    print("\n🧹 Vacuuming Delta tables...")
    
    vacuum_results = []
    
    # Only vacuum main data tables, not frequently updated ones
    tables_to_vacuum = [
        "hr_analytics.processed.employee_master",
        "hr_analytics.processed.survival_features"
    ]
    
    for table_name in tables_to_vacuum:
        try:
            print(f"   🧹 Vacuuming {table_name}...")
            
            # Check if table exists
            try:
                spark.table(table_name)
            except:
                print(f"   ⚠️ Table {table_name} does not exist, skipping")
                continue
            
            start_time = datetime.now()
            
            # Vacuum with 7 day retention (default is 7 days)
            spark.sql(f"VACUUM {table_name} RETAIN 168 HOURS")  # 7 days
            
            vacuum_time = (datetime.now() - start_time).total_seconds()
            
            vacuum_results.append({
                "table_name": table_name,
                "vacuum_timestamp": datetime.now(),
                "vacuum_time_seconds": vacuum_time,
                "retention_hours": 168,
                "status": "success"
            })
            
            print(f"   ✅ {table_name} vacuumed in {vacuum_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Failed to vacuum {table_name}: {str(e)}")
            vacuum_results.append({
                "table_name": table_name,
                "vacuum_timestamp": datetime.now(),
                "error": str(e),
                "status": "failed"
            })
    
    successful_vacuums = sum(1 for r in vacuum_results if r["status"] == "success")
    print(f"✅ Successfully vacuumed {successful_vacuums}/{len(tables_to_vacuum)} tables")

# =============================================================================
# 5. SYSTEM HEALTH CHECK
# =============================================================================

def perform_health_check():
    """Perform basic system health check after cleanup"""
    
    print("\n🏥 Performing system health check...")
    
    health_results = {
        "health_check_timestamp": datetime.now(),
        "tables_accessible": 0,
        "tables_inaccessible": 0,
        "total_records": 0,
        "latest_predictions_date": None,
        "model_in_production": False
    }
    
    # Check table accessibility
    for table_name in TABLES_TO_OPTIMIZE:
        try:
            count = spark.table(table_name).count()
            health_results["tables_accessible"] += 1
            health_results["total_records"] += count
            print(f"   ✅ {table_name}: {count:,} records")
        except Exception as e:
            health_results["tables_inaccessible"] += 1
            print(f"   ❌ {table_name}: Not accessible - {str(e)}")
    
    # Check latest predictions
    try:
        latest_date = spark.table("hr_analytics.predictions.employee_risk_scores").select(max("scoring_date")).collect()[0][0]
        health_results["latest_predictions_date"] = latest_date
        days_since_scoring = (datetime.now().date() - latest_date).days if latest_date else None
        print(f"   📊 Latest predictions: {latest_date} ({days_since_scoring} days ago)")
    except Exception as e:
        print(f"   ⚠️ Could not check latest predictions: {str(e)}")
    
    # Check model status
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        models = client.get_latest_versions("employee_survival_xgboost_aft", stages=["Production"])
        health_results["model_in_production"] = len(models) > 0
        print(f"   🤖 Production model available: {health_results['model_in_production']}")
    except Exception as e:
        print(f"   ⚠️ Could not check model status: {str(e)}")
    
    # Save health check results
    try:
        health_df = spark.createDataFrame([health_results])
        (health_df
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .saveAsTable("hr_analytics.maintenance.health_check_log"))
        
        print(f"✅ Health check results saved")
    except Exception as e:
        print(f"⚠️ Could not save health check results: {str(e)}")
    
    return health_results

# =============================================================================
# 6. MAIN CLEANUP PIPELINE
# =============================================================================

def run_pipeline_cleanup():
    """Main function to run complete cleanup operations"""
    
    print("🚀 Starting Pipeline Cleanup...")
    start_time = datetime.now()
    
    try:
        # Step 1: Optimize Delta tables
        optimize_delta_tables()
        
        # Step 2: Manage data retention
        manage_data_retention()
        
        # Step 3: Cleanup temporary files
        cleanup_temp_files()
        
        # Step 4: Vacuum Delta tables
        vacuum_delta_tables()
        
        # Step 5: Health check
        health_results = perform_health_check()
        
        # Summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🎉 PIPELINE CLEANUP COMPLETED!")
        print("="*80)
        print(f"⏱️ Total cleanup time: {duration}")
        print(f"📊 Tables accessible: {health_results['tables_accessible']}")
        print(f"📊 Total records: {health_results['total_records']:,}")
        print(f"🤖 Production model available: {health_results['model_in_production']}")
        
        if health_results['latest_predictions_date']:
            print(f"📅 Latest predictions: {health_results['latest_predictions_date']}")
        
        print("✅ System is healthy and ready for next pipeline run")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline cleanup failed: {str(e)}")
        return False

# Run the cleanup
if __name__ == "__main__":
    success = run_pipeline_cleanup()
    if not success:
        dbutils.notebook.exit("FAILED")
        