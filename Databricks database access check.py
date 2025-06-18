# ============================================================================
# DATABRICKS TABLE ACCESS CHECKER
# ============================================================================

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.utils import AnalysisException

# Initialize Spark (already available in Databricks)
# spark = SparkSession.builder.getOrCreate()  # Uncomment if needed

def check_table_access(table_list):
    """
    Check access to a list of tables and fetch single record from each
    
    Args:
        table_list: List of table names (can include catalog.schema.table format)
    """
    
    results = {}
    
    print("🔍 CHECKING TABLE ACCESS")
    print("=" * 50)
    
    for i, table_name in enumerate(table_list, 1):
        print(f"\n[{i}/{len(table_list)}] Checking: {table_name}")
        print("-" * 40)
        
        try:
            # Try to read one record from the table
            df = spark.table(table_name).limit(1)
            
            # Get record count to verify access
            count = df.count()
            
            if count > 0:
                # Fetch and display the record
                record = df.collect()[0]
                
                print(f"✅ ACCESS GRANTED")
                print(f"📊 Sample Record:")
                
                # Print each column and value
                for field in df.schema.fields:
                    col_name = field.name
                    col_value = record[col_name]
                    col_type = str(field.dataType)
                    print(f"   {col_name}: {col_value} ({col_type})")
                
                results[table_name] = {
                    'status': 'SUCCESS',
                    'record_count': count,
                    'columns': len(df.columns),
                    'schema': [f.name for f in df.schema.fields]
                }
                
            else:
                print(f"✅ ACCESS GRANTED (Empty Table)")
                print(f"📊 Columns: {df.columns}")
                
                results[table_name] = {
                    'status': 'SUCCESS_EMPTY', 
                    'record_count': 0,
                    'columns': len(df.columns),
                    'schema': [f.name for f in df.schema.fields]
                }
        
        except AnalysisException as e:
            error_msg = str(e)
            
            if "does not exist" in error_msg.lower():
                print(f"❌ TABLE NOT FOUND")
                results[table_name] = {'status': 'NOT_FOUND', 'error': 'Table does not exist'}
                
            elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                print(f"🚫 ACCESS DENIED")
                results[table_name] = {'status': 'ACCESS_DENIED', 'error': 'Insufficient permissions'}
                
            else:
                print(f"⚠️  ERROR: {error_msg}")
                results[table_name] = {'status': 'ERROR', 'error': error_msg}
        
        except Exception as e:
            print(f"⚠️  UNEXPECTED ERROR: {str(e)}")
            results[table_name] = {'status': 'UNEXPECTED_ERROR', 'error': str(e)}
    
    return results

def print_summary(results):
    """Print summary of table access check results"""
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY REPORT")  
    print("=" * 60)
    
    success_count = sum(1 for r in results.values() if r['status'] in ['SUCCESS', 'SUCCESS_EMPTY'])
    error_count = len(results) - success_count
    
    print(f"Total tables checked: {len(results)}")
    print(f"✅ Accessible: {success_count}")
    print(f"❌ Issues: {error_count}")
    
    if error_count > 0:
        print(f"\n🚫 TABLES WITH ISSUES:")
        for table, result in results.items():
            if result['status'] not in ['SUCCESS', 'SUCCESS_EMPTY']:
                status_emoji = {
                    'NOT_FOUND': '❓',
                    'ACCESS_DENIED': '🚫', 
                    'ERROR': '⚠️',
                    'UNEXPECTED_ERROR': '💥'
                }
                emoji = status_emoji.get(result['status'], '❌')
                print(f"   {emoji} {table}: {result['status']}")
    
    print(f"\n✅ ACCESSIBLE TABLES:")
    for table, result in results.items():
        if result['status'] in ['SUCCESS', 'SUCCESS_EMPTY']:
            col_count = result.get('columns', 0)
            rec_count = result.get('record_count', 0)
            print(f"   📊 {table}: {col_count} columns, {rec_count} records")

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Example 1: Simple table list
simple_tables = [
    'employees',
    'payroll', 
    'performance_reviews',
    'org_hierarchy'
]

# Example 2: Full catalog.schema.table format  
full_qualified_tables = [
    'analytics_warehouse.hr.employees',
    'analytics_warehouse.hr.compensation', 
    'analytics_warehouse.finance.payroll',
    'raw_data.adp.workforce_now',
    'processed.turnover.survival_data'
]

# Example 3: Mixed formats
mixed_tables = [
    'default.employees',
    'hr_analytics.employee_turnover',
    'analytics_warehouse.hr.compensation_history', 
    'some_catalog.some_schema.some_table',
    'non_existent_table'
]

# ============================================================================
# RUN THE CHECK
# ============================================================================

if __name__ == "__main__":
    
    # CUSTOMIZE THIS LIST FOR YOUR TABLES
    tables_to_check = [
        'analytics_warehouse.hr.employees',
        'analytics_warehouse.hr.compensation', 
        'analytics_warehouse.finance.payroll',
        'analytics_warehouse.org.hierarchy',
        'processed.turnover.monthly_data',
        'raw_data.adp.workforce_now',
        'some_non_existent_table'  # This will fail for testing
    ]
    
    print("🚀 Starting table access verification...")
    
    # Run the check
    results = check_table_access(tables_to_check)
    
    # Print summary
    print_summary(results)
    
    print(f"\n🎯 Next Steps:")
    print("1. Request access for any denied tables")
    print("2. Verify table names for any 'NOT_FOUND' tables") 
    print("3. Use accessible tables for your analysis")

# ============================================================================
# ALTERNATIVE: QUICK ONE-LINER FUNCTIONS
# ============================================================================

def quick_check(table_name):
    """Quick single table check"""
    try:
        record = spark.table(table_name).limit(1).collect()[0]
        print(f"✅ {table_name}: {dict(record.asDict())}")
        return True
    except Exception as e:
        print(f"❌ {table_name}: {str(e)}")
        return False

def batch_quick_check(table_list):
    """Quick batch check without detailed output"""
    print("🔍 QUICK ACCESS CHECK")
    print("-" * 25)
    
    accessible = []
    failed = []
    
    for table in table_list:
        if quick_check(table):
            accessible.append(table)
        else:
            failed.append(table)
    
    print(f"\n✅ Accessible ({len(accessible)}): {accessible}")
    print(f"❌ Failed ({len(failed)}): {failed}")
    
    return accessible, failed

# ============================================================================
# USAGE: Uncomment the approach you prefer
# ============================================================================

# Approach 1: Detailed check with full output
# results = check_table_access(your_table_list)

# Approach 2: Quick check for fast verification
# accessible, failed = batch_quick_check(your_table_list)

# Approach 3: Single table check
# quick_check('your_table_name')
