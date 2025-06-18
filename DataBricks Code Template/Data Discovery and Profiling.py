# Data Discovery & Profiling for HR Tables
# This notebook discovers and profiles HR tables to understand data structure and relationships

# =============================================================================
# CONFIGURATION PARAMETERS (MODIFY THESE AS NEEDED)
# =============================================================================

# Primary employee identifier field name (configurable)
EMPLOYEE_ID_FIELD = "employee_id"  # Change this to your actual employee ID field name

# Tables to analyze (provide your actual table list here)
HR_TABLES = [
    "catalog.hr_schema.employee_monthly",
    "catalog.reference.naics",
    "catalog.reference.region_mappings",
    # Add your actual table names here
    # "catalog.payroll.salary_history",
    # "catalog.hr.performance_reviews",
    # "catalog.org.department_structure"
]

# Table patterns to exclude from discovery
EXCLUDE_PATTERNS = [
    "_temp", "_tmp", "_test", "_backup", "_archive", "_staging"
]

# Data quality thresholds
MIN_RECORD_COUNT = 100
MAX_NULL_PERCENTAGE = 50
MIN_UNIQUE_VALUES = 2

# Output locations
DISCOVERY_RESULTS_TABLE = "hr_analytics.discovery.table_catalog"
DATA_QUALITY_TABLE = "hr_analytics.discovery.data_quality_metrics"
RELATIONSHIP_TABLE = "hr_analytics.discovery.table_relationships"

# =============================================================================
# IMPORTS
# =============================================================================

from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

print("🔍 Starting HR Data Discovery & Profiling...")
print(f"📋 Analyzing {len(HR_TABLES)} specified tables")
print(f"🎯 Looking for employee ID field: {EMPLOYEE_ID_FIELD}")

# =============================================================================
# 1. TABLE DISCOVERY & BASIC PROFILING
# =============================================================================

def discover_table_structure(table_name):
    """Discover basic structure and profile of a single table"""
    
    try:
        print(f"📊 Profiling table: {table_name}")
        
        # Get table basic info
        df = spark.table(table_name)
        
        # Basic metrics
        row_count = df.count()
        col_count = len(df.columns)
        
        # Column analysis
        column_info = []
        for col in df.columns:
            col_stats = df.select(
                count(col).alias("non_null_count"),
                countDistinct(col).alias("unique_count"),
                (count(col) / row_count * 100).alias("non_null_percentage")
            ).collect()[0]
            
            column_info.append({
                "column_name": col,
                "data_type": dict(df.dtypes)[col],
                "non_null_count": col_stats.non_null_count,
                "unique_count": col_stats.unique_count,
                "non_null_percentage": float(col_stats.non_null_percentage),
                "null_percentage": 100 - float(col_stats.non_null_percentage),
                "has_employee_id": EMPLOYEE_ID_FIELD.lower() in col.lower()
            })
        
        # Check if this might be an employee table
        employee_id_columns = [col for col in df.columns if EMPLOYEE_ID_FIELD.lower() in col.lower()]
        has_potential_employee_id = len(employee_id_columns) > 0
        
        # Look for date columns (important for start-stop format)
        date_columns = [col["column_name"] for col in column_info 
                       if "date" in col["data_type"].lower() or 
                          "timestamp" in col["data_type"].lower()]
        
        # Sample data for manual inspection
        sample_data = df.limit(5).toPandas().to_dict('records') if row_count > 0 else []
        
        result = {
            "table_name": table_name,
            "row_count": row_count,
            "column_count": col_count,
            "columns": column_info,
            "employee_id_columns": employee_id_columns,
            "has_potential_employee_id": has_potential_employee_id,
            "date_columns": date_columns,
            "sample_data": sample_data,
            "discovery_timestamp": datetime.now(),
            "data_quality_score": calculate_data_quality_score(column_info, row_count)
        }
        
        print(f"  ✅ {table_name}: {row_count:,} rows, {col_count} columns")
        if has_potential_employee_id:
            print(f"     🎯 Found potential employee ID columns: {employee_id_columns}")
        if date_columns:
            print(f"     📅 Found date columns: {date_columns}")
            
        return result
        
    except Exception as e:
        print(f"  ❌ Error profiling {table_name}: {str(e)}")
        return {
            "table_name": table_name,
            "error": str(e),
            "discovery_timestamp": datetime.now()
        }

def calculate_data_quality_score(column_info, row_count):
    """Calculate a simple data quality score (0-100)"""
    
    if row_count < MIN_RECORD_COUNT:
        return 0
    
    total_score = 0
    for col in column_info:
        # Penalize high null percentages
        null_penalty = max(0, col["null_percentage"] - 10) / 90 * 30  # Up to 30 points penalty
        
        # Reward reasonable unique values
        uniqueness_score = min(20, col["unique_count"] / max(1, row_count) * 100)
        
        col_score = max(0, 100 - null_penalty + uniqueness_score)
        total_score += col_score
    
    return min(100, total_score / len(column_info)) if column_info else 0

# =============================================================================
# 2. ANALYZE ALL TABLES
# =============================================================================

print("\n🔍 Discovering table structures...")

discovery_results = []
for table in HR_TABLES:
    # Skip tables matching exclude patterns
    if any(pattern in table.lower() for pattern in EXCLUDE_PATTERNS):
        print(f"⏭️ Skipping {table} (matches exclude pattern)")
        continue
    
    result = discover_table_structure(table)
    discovery_results.append(result)

# =============================================================================
# 3. IDENTIFY POTENTIAL RELATIONSHIPS
# =============================================================================

def identify_relationships(discovery_results):
    """Identify potential relationships between tables"""
    
    print("\n🔗 Identifying potential table relationships...")
    
    relationships = []
    
    # Find tables with employee ID columns
    employee_tables = [(r["table_name"], r["employee_id_columns"]) 
                      for r in discovery_results 
                      if r.get("has_potential_employee_id", False)]
    
    if employee_tables:
        print(f"📋 Found {len(employee_tables)} tables with potential employee IDs:")
        for table_name, emp_cols in employee_tables:
            print(f"   - {table_name}: {emp_cols}")
            
        # Create relationships based on common employee ID columns
        for i, (table1, cols1) in enumerate(employee_tables):
            for j, (table2, cols2) in enumerate(employee_tables[i+1:], i+1):
                common_cols = set(cols1) & set(cols2)
                if common_cols:
                    relationships.append({
                        "table1": table1,
                        "table2": table2,
                        "relationship_type": "employee_join",
                        "join_columns": list(common_cols),
                        "confidence": "high"
                    })
    
    # Look for reference tables (NAICS, region mappings, etc.)
    reference_tables = [r["table_name"] for r in discovery_results 
                       if any(ref in r["table_name"].lower() 
                             for ref in ["naics", "region", "mapping", "reference", "lookup"])]
    
    if reference_tables:
        print(f"📚 Found {len(reference_tables)} potential reference tables:")
        for ref_table in reference_tables:
            print(f"   - {ref_table}")
    
    return relationships

relationships = identify_relationships(discovery_results)

# =============================================================================
# 4. SAVE DISCOVERY RESULTS
# =============================================================================

print("\n💾 Saving discovery results...")

# Convert results to DataFrames and save
if discovery_results:
    # Main discovery results
    discovery_df_data = []
    column_df_data = []
    
    for result in discovery_results:
        if "error" not in result:
            # Main table info
            discovery_df_data.append({
                "table_name": result["table_name"],
                "row_count": result["row_count"],
                "column_count": result["column_count"],
                "has_potential_employee_id": result["has_potential_employee_id"],
                "employee_id_columns": ",".join(result["employee_id_columns"]),
                "date_columns": ",".join(result["date_columns"]),
                "data_quality_score": result["data_quality_score"],
                "discovery_timestamp": result["discovery_timestamp"]
            })
            
            # Column details
            for col in result["columns"]:
                column_df_data.append({
                    "table_name": result["table_name"],
                    "column_name": col["column_name"],
                    "data_type": col["data_type"],
                    "non_null_percentage": col["non_null_percentage"],
                    "unique_count": col["unique_count"],
                    "has_employee_id": col["has_employee_id"],
                    "discovery_timestamp": result["discovery_timestamp"]
                })
    
    # Create Spark DataFrames and save
    if discovery_df_data:
        discovery_spark_df = spark.createDataFrame(discovery_df_data)
        (discovery_spark_df
         .write
         .format("delta")
         .mode("overwrite")
         .option("mergeSchema", "true")
         .saveAsTable(DISCOVERY_RESULTS_TABLE))
        
        print(f"✅ Table catalog saved to {DISCOVERY_RESULTS_TABLE}")
    
    if column_df_data:
        columns_spark_df = spark.createDataFrame(column_df_data)
        (columns_spark_df
         .write
         .format("delta")
         .mode("overwrite")
         .option("mergeSchema", "true")
         .saveAsTable(DATA_QUALITY_TABLE))
        
        print(f"✅ Column details saved to {DATA_QUALITY_TABLE}")

# Save relationships
if relationships:
    relationships_df = spark.createDataFrame(relationships)
    (relationships_df
     .write
     .format("delta")
     .mode("overwrite")
     .option("mergeSchema", "true")
     .saveAsTable(RELATIONSHIP_TABLE))
    
    print(f"✅ Relationships saved to {RELATIONSHIP_TABLE}")

# =============================================================================
# 5. SUMMARY REPORT
# =============================================================================

print("\n" + "="*80)
print("📊 HR DATA DISCOVERY SUMMARY REPORT")
print("="*80)

valid_results = [r for r in discovery_results if "error" not in r]
error_results = [r for r in discovery_results if "error" in r]

print(f"✅ Successfully analyzed: {len(valid_results)} tables")
print(f"❌ Failed to analyze: {len(error_results)} tables")

if error_results:
    print("\n❌ FAILED TABLES:")
    for result in error_results:
        print(f"   - {result['table_name']}: {result['error']}")

if valid_results:
    print("\n📋 TABLE SUMMARY:")
    total_rows = sum(r["row_count"] for r in valid_results)
    avg_quality = sum(r["data_quality_score"] for r in valid_results) / len(valid_results)
    
    print(f"   - Total rows across all tables: {total_rows:,}")
    print(f"   - Average data quality score: {avg_quality:.1f}/100")
    
    # Show tables with employee IDs
    employee_tables = [r for r in valid_results if r["has_potential_employee_id"]]
    print(f"   - Tables with potential employee IDs: {len(employee_tables)}")
    
    # Show largest tables
    print("\n📊 LARGEST TABLES:")
    sorted_tables = sorted(valid_results, key=lambda x: x["row_count"], reverse=True)[:5]
    for table in sorted_tables:
        print(f"   - {table['table_name']}: {table['row_count']:,} rows")
    
    # Show data quality issues
    print("\n⚠️ DATA QUALITY ALERTS:")
    low_quality_tables = [r for r in valid_results if r["data_quality_score"] < 50]
    if low_quality_tables:
        for table in low_quality_tables:
            print(f"   - {table['table_name']}: Quality score {table['data_quality_score']:.1f}/100")
    else:
        print("   - No major data quality issues detected")

print("\n🎯 NEXT STEPS:")
print("1. Review the discovery results tables created")
print("2. Validate the employee ID column mappings")
print("3. Examine tables with date columns for start-stop format")
print("4. Proceed to complex data processing pipeline")

print(f"\n📊 Results saved to:")
print(f"   - Table catalog: {DISCOVERY_RESULTS_TABLE}")
print(f"   - Column details: {DATA_QUALITY_TABLE}")
print(f"   - Relationships: {RELATIONSHIP_TABLE}")

print("\n🎉 Data discovery completed successfully!")