# Databricks notebook source
# MAGIC %md
# MAGIC # ADP Employee Turnover Project - Table Access Testing
# MAGIC 
# MAGIC **Purpose:** Test access to Analytics Warehouse tables for survival analysis project
# MAGIC 
# MAGIC **Usage:**
# MAGIC 1. Add your table names to the `table_list` variable
# MAGIC 2. Run all cells to get comprehensive access report
# MAGIC 3. Share results with Blair/Dinesh for any access issues

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and Configuration

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import time
from datetime import datetime

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

print("🔧 Environment Setup Complete")
print(f"Spark Version: {spark.version}")
print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Table List Configuration
# MAGIC 
# MAGIC **Instructions:** Replace the sample table names below with your actual ADP Analytics Warehouse table names

# COMMAND ----------

# ADD YOUR TABLE NAMES HERE
# Format: "catalog.schema.table_name"
table_list = [
    # Example format - replace with your actual table names
    "your_catalog.your_schema.employee_data",
    "your_catalog.your_schema.termination_events", 
    "your_catalog.your_schema.salary_history",
    "your_catalog.your_schema.promotion_history",
    "your_catalog.your_schema.manager_assignments",
    "your_catalog.your_schema.department_mappings",
    "your_catalog.your_schema.naics_classifications",
    "your_catalog.your_schema.performance_ratings",
    "your_catalog.your_schema.overtime_records",
    "your_catalog.your_schema.remote_work_data"
]

print(f"📋 Testing access to {len(table_list)} tables:")
for i, table in enumerate(table_list, 1):
    print(f"  {i}. {table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Table Access Testing Functions

# COMMAND ----------

def test_table_exists(table_name):
    """
    Test if table exists and is accessible
    Returns: (exists, error_message)
    """
    try:
        # Try to get table info
        spark.sql(f"DESCRIBE TABLE {table_name}")
        return True, None
    except Exception as e:
        return False, str(e)

def test_table_read_access(table_name):
    """
    Test if we can read from the table
    Returns: (can_read, row_count, error_message)
    """
    try:
        # Try to count rows (lightweight operation)
        df = spark.sql(f"SELECT COUNT(*) as row_count FROM {table_name}")
        row_count = df.collect()[0]['row_count']
        return True, row_count, None
    except Exception as e:
        return False, None, str(e)

def get_table_schema(table_name):
    """
    Get table schema information
    Returns: (success, schema_info, error_message)
    """
    try:
        schema_df = spark.sql(f"DESCRIBE TABLE {table_name}")
        schema_info = schema_df.collect()
        return True, schema_info, None
    except Exception as e:
        return False, None, str(e)

def get_sample_data(table_name, limit=5):
    """
    Get sample data from table
    Returns: (success, sample_data, error_message)
    """
    try:
        sample_df = spark.sql(f"SELECT * FROM {table_name} LIMIT {limit}")
        sample_data = sample_df.collect()
        return True, sample_data, None
    except Exception as e:
        return False, None, str(e)

print("✅ Table testing functions defined")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comprehensive Table Access Test

# COMMAND ----------

def run_comprehensive_table_test(table_list):
    """
    Run comprehensive access test on all tables
    """
    
    results = []
    
    print("🧪 Starting Comprehensive Table Access Test")
    print("=" * 60)
    
    for i, table_name in enumerate(table_list, 1):
        print(f"\n📊 Testing Table {i}/{len(table_list)}: {table_name}")
        print("-" * 50)
        
        result = {
            'table_name': table_name,
            'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'exists': False,
            'readable': False,
            'row_count': None,
            'column_count': None,
            'columns': [],
            'sample_available': False,
            'errors': []
        }
        
        # Test 1: Table Exists
        exists, exist_error = test_table_exists(table_name)
        result['exists'] = exists
        
        if not exists:
            print(f"❌ Table does not exist or not accessible")
            print(f"   Error: {exist_error}")
            result['errors'].append(f"Existence check failed: {exist_error}")
            results.append(result)
            continue
        else:
            print("✅ Table exists and is accessible")
        
        # Test 2: Read Access
        can_read, row_count, read_error = test_table_read_access(table_name)
        result['readable'] = can_read
        result['row_count'] = row_count
        
        if not can_read:
            print(f"❌ Cannot read from table")
            print(f"   Error: {read_error}")
            result['errors'].append(f"Read access failed: {read_error}")
        else:
            print(f"✅ Read access confirmed - {row_count:,} rows")
        
        # Test 3: Schema Information
        schema_success, schema_info, schema_error = get_table_schema(table_name)
        if schema_success:
            columns = [row['col_name'] for row in schema_info if row['col_name'] and not row['col_name'].startswith('#')]
            result['columns'] = columns
            result['column_count'] = len(columns)
            print(f"✅ Schema retrieved - {len(columns)} columns")
            print(f"   Columns: {', '.join(columns[:5])}{'...' if len(columns) > 5 else ''}")
        else:
            print(f"⚠️  Could not retrieve schema: {schema_error}")
            result['errors'].append(f"Schema retrieval failed: {schema_error}")
        
        # Test 4: Sample Data (only if read access works)
        if can_read:
            sample_success, sample_data, sample_error = get_sample_data(table_name, 3)
            result['sample_available'] = sample_success
            if sample_success:
                print(f"✅ Sample data retrieved - {len(sample_data)} sample rows")
            else:
                print(f"⚠️  Could not retrieve sample data: {sample_error}")
                result['errors'].append(f"Sample data failed: {sample_error}")
        
        results.append(result)
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.5)
    
    return results

# Run the comprehensive test
test_results = run_comprehensive_table_test(table_list)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Test Results Summary

# COMMAND ----------

def generate_access_summary(test_results):
    """
    Generate comprehensive access summary
    """
    
    print("📋 TABLE ACCESS SUMMARY REPORT")
    print("=" * 60)
    print(f"Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total tables tested: {len(test_results)}")
    
    # Overall statistics
    accessible_count = sum(1 for r in test_results if r['exists'] and r['readable'])
    exists_count = sum(1 for r in test_results if r['exists'])
    readable_count = sum(1 for r in test_results if r['readable'])
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Tables that exist: {exists_count}/{len(test_results)} ({exists_count/len(test_results)*100:.1f}%)")
    print(f"  Tables readable: {readable_count}/{len(test_results)} ({readable_count/len(test_results)*100:.1f}%)")
    print(f"  Fully accessible: {accessible_count}/{len(test_results)} ({accessible_count/len(test_results)*100:.1f}%)")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    
    # Successful tables
    successful_tables = [r for r in test_results if r['exists'] and r['readable']]
    if successful_tables:
        print(f"\n✅ ACCESSIBLE TABLES ({len(successful_tables)}):")
        for result in successful_tables:
            row_count_str = f"{result['row_count']:,}" if result['row_count'] is not None else "Unknown"
            col_count_str = str(result['column_count']) if result['column_count'] is not None else "Unknown"
            print(f"  📊 {result['table_name']}")
            print(f"      Rows: {row_count_str} | Columns: {col_count_str}")
            if result['columns']:
                print(f"      Sample columns: {', '.join(result['columns'][:3])}{'...' if len(result['columns']) > 3 else ''}")
    
    # Failed tables
    failed_tables = [r for r in test_results if not (r['exists'] and r['readable'])]
    if failed_tables:
        print(f"\n❌ INACCESSIBLE TABLES ({len(failed_tables)}):")
        for result in failed_tables:
            print(f"  🚫 {result['table_name']}")
            if result['errors']:
                for error in result['errors']:
                    print(f"      Error: {error}")
    
    # Data availability summary for survival analysis
    print(f"\n🧬 SURVIVAL ANALYSIS DATA READINESS:")
    
    # Look for key tables needed for survival analysis
    key_table_indicators = [
        'employee', 'termination', 'salary', 'promotion', 
        'manager', 'department', 'performance', 'overtime'
    ]
    
    available_key_data = []
    for indicator in key_table_indicators:
        matching_tables = [r for r in successful_tables 
                          if any(indicator in r['table_name'].lower() 
                                for indicator in [indicator])]
        if matching_tables:
            available_key_data.append(f"{indicator.title()}: ✅")
        else:
            available_key_data.append(f"{indicator.title()}: ❌")
    
    for item in available_key_data:
        print(f"  {item}")
    
    return {
        'total_tables': len(test_results),
        'accessible_tables': accessible_count,
        'success_rate': accessible_count/len(test_results)*100,
        'successful_tables': successful_tables,
        'failed_tables': failed_tables
    }

# Generate summary
summary = generate_access_summary(test_results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Export Results for Team Sharing

# COMMAND ----------

# Create detailed results DataFrame for easy sharing
def create_results_dataframe(test_results):
    """
    Create a pandas DataFrame with test results for easy export/sharing
    """
    
    df_data = []
    for result in test_results:
        df_data.append({
            'Table_Name': result['table_name'],
            'Exists': '✅' if result['exists'] else '❌',
            'Readable': '✅' if result['readable'] else '❌',
            'Row_Count': result['row_count'],
            'Column_Count': result['column_count'],
            'Status': 'ACCESSIBLE' if (result['exists'] and result['readable']) else 'BLOCKED',
            'Error_Summary': '; '.join(result['errors']) if result['errors'] else 'None',
            'Test_Time': result['test_timestamp']
        })
    
    return pd.DataFrame(df_data)

# Create results DataFrame
results_df = create_results_dataframe(test_results)

print("📊 EXPORTABLE RESULTS TABLE:")
print("=" * 40)
display(results_df)

# Export to temporary view for easy querying
results_spark_df = spark.createDataFrame(results_df)
results_spark_df.createOrReplaceTempView("table_access_results")

print("\n💾 Results saved to temporary view: 'table_access_results'")
print("Use: SELECT * FROM table_access_results WHERE Status = 'BLOCKED' to see blocked tables")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Action Items and Next Steps

# COMMAND ----------

def generate_action_items(summary):
    """
    Generate specific action items based on test results
    """
    
    print("🎯 ACTION ITEMS AND NEXT STEPS")
    print("=" * 50)
    
    success_rate = summary['success_rate']
    
    if success_rate == 100:
        print("🎉 EXCELLENT: All tables accessible!")
        print("\n✅ IMMEDIATE ACTIONS:")
        print("  1. Begin data exploration phase")
        print("  2. Start feature engineering pipeline development")
        print("  3. Initialize survival analysis data preparation")
        
    elif success_rate >= 70:
        print("🟡 GOOD: Most tables accessible")
        print(f"   {summary['accessible_tables']}/{summary['total_tables']} tables available")
        print("\n✅ IMMEDIATE ACTIONS:")
        print("  1. Begin work with accessible tables")
        print("  2. Contact Blair/Dinesh for blocked table access")
        print("  3. Identify critical missing tables for survival analysis")
        
    elif success_rate >= 30:
        print("🟠 PARTIAL: Some tables accessible")
        print(f"   {summary['accessible_tables']}/{summary['total_tables']} tables available")
        print("\n⚠️  CRITICAL ACTIONS:")
        print("  1. ESCALATE: Contact Blair immediately for access resolution")
        print("  2. Identify minimum viable dataset for Week 1 progress")
        print("  3. Prepare alternative data sources if needed")
        
    else:
        print("🔴 CRITICAL: Most/all tables blocked")
        print(f"   Only {summary['accessible_tables']}/{summary['total_tables']} tables available")
        print("\n🚨 URGENT ACTIONS:")
        print("  1. IMMEDIATE ESCALATION: Contact Blair + Account Lead")
        print("  2. Schedule emergency access resolution meeting")
        print("  3. Prepare project timeline impact assessment")
    
    # Specific contacts and next steps
    print(f"\n📞 ESCALATION CONTACTS:")
    print("  1. Blair Christian (Director of Data Science) - Primary technical contact")
    print("  2. Dinesh Prodduturi (Data Engineering Manager) - Analytics Warehouse expert")
    print("  3. George Hatziemanuel (Product Manager) - For access authorization")
    
    print(f"\n📧 RECOMMENDED EMAIL TEMPLATE:")
    print("Subject: ADP Turnover Project - Table Access Status")
    print(f"Hi Blair/Dinesh,")
    print(f"")
    print(f"Table access test completed for ADP employee turnover project:")
    print(f"- Total tables tested: {summary['total_tables']}")
    print(f"- Accessible tables: {summary['accessible_tables']} ({success_rate:.1f}%)")
    print(f"- Status: {'Ready to proceed' if success_rate >= 70 else 'Need access resolution'}")
    print(f"")
    print(f"See attached results for detailed breakdown.")
    print(f"{'Ready to begin data exploration!' if success_rate >= 70 else 'Please advise on access resolution timeline.'}")
    print(f"")
    print(f"Best regards,")
    print(f"Ravi & Zenon Team")

# Generate action items
generate_action_items(summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save Results for Team Reference

# COMMAND ----------

# Save timestamp for reference
test_completion_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print("💾 TABLE ACCESS TEST COMPLETED")
print("=" * 40)
print(f"Completion time: {test_completion_time}")
print(f"Results available in: 'table_access_results' temp view")
print(f"Total tables tested: {len(test_results)}")
print(f"Success rate: {summary['success_rate']:.1f}%")

# Quick status for team standup
if summary['success_rate'] >= 70:
    status_emoji = "🟢"
    status_text = "GREEN - Ready to proceed"
elif summary['success_rate'] >= 30:
    status_emoji = "🟡"  
    status_text = "YELLOW - Partial access, needs resolution"
else:
    status_emoji = "🔴"
    status_text = "RED - Critical access issues"

print(f"\n{status_emoji} TEAM STATUS: {status_text}")
print("\n📋 Copy this summary for standup:")
print(f"Table Access Test: {summary['accessible_tables']}/{summary['total_tables']} tables accessible ({summary['success_rate']:.1f}%) - {status_text}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC 
# MAGIC ## Usage Instructions
# MAGIC 
# MAGIC 1. **Replace table names** in Section 2 with your actual ADP Analytics Warehouse table names
# MAGIC 2. **Run all cells** to execute comprehensive access testing
# MAGIC 3. **Check results** in Section 5 for detailed summary
# MAGIC 4. **Follow action items** in Section 7 based on your access status
# MAGIC 5. **Share results** with Blair/Dinesh if access issues found
# MAGIC 
# MAGIC **Table Name Format:** `catalog.schema.table_name`
# MAGIC 
# MAGIC **Example:** `adp_analytics.employee_data.current_employees`
