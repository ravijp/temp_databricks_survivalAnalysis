# Databricks notebook source
# MAGIC %md
# MAGIC # Comprehensive ADP Access Test - Find/Replace Aliases
# MAGIC 
# MAGIC **FIND AND REPLACE THESE ALIASES WITH YOUR ACTUAL ADP NAMES:**
# MAGIC 
# MAGIC | Alias | Replace With Your Actual Name |
# MAGIC |-------|------------------------------|
# MAGIC | `ADP_CATALOG_1` | your_catalog_name |
# MAGIC | `ADP_SCHEMA_1` | your_first_schema_name |
# MAGIC | `ADP_SCHEMA_2` | your_second_schema_name |
# MAGIC | `ADP_SCHEMA_3` | your_third_schema_name |
# MAGIC | `ADP_SCHEMA_4` | your_fourth_schema_name |
# MAGIC | `ADP_SCHEMA_5` | your_fifth_schema_name |
# MAGIC | `ADP_SCHEMA_6` | your_sixth_schema_name |
# MAGIC | `ADP_TABLE_1` | your_first_table_name |
# MAGIC | `ADP_TABLE_2` | your_second_table_name |
# MAGIC | `ADP_TABLE_3` | your_third_table_name |

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup and Basic Tests

# COMMAND ----------

import pandas as pd
from datetime import datetime
import traceback

# Environment information
env_info = [{
    'Component': 'Spark Version',
    'Value': spark.version,
    'Status': '✅ Working',
    'Timestamp': datetime.now().strftime('%H:%M:%S')
}, {
    'Component': 'Current User',
    'Value': spark.sparkContext.sparkUser(),
    'Status': '✅ Working', 
    'Timestamp': datetime.now().strftime('%H:%M:%S')
}, {
    'Component': 'Default Parallelism',
    'Value': str(spark.sparkContext.defaultParallelism),
    'Status': '✅ Working',
    'Timestamp': datetime.now().strftime('%H:%M:%S')
}]

display(pd.DataFrame(env_info))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Catalog Discovery and Access

# COMMAND ----------

# Get all available catalogs
try:
    catalogs_df = spark.sql("SHOW CATALOGS")
    display(catalogs_df)
    
    # Summary of catalogs
    catalog_count = catalogs_df.count()
    catalog_summary = [{
        'Metric': 'Total Catalogs Available',
        'Count': catalog_count,
        'Status': '✅ Catalog Access Working'
    }]
    display(pd.DataFrame(catalog_summary))
    
except Exception as e:
    error_info = [{
        'Error_Type': 'Catalog Access Failed',
        'Error_Message': str(e),
        'Status': '❌ Critical Issue'
    }]
    display(pd.DataFrame(error_info))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Current Location and Permissions

# COMMAND ----------

# Check current database context
try:
    current_info = []
    
    # Current catalog
    current_cat = spark.sql("SELECT current_catalog() as catalog").collect()[0]['catalog']
    current_info.append({
        'Setting': 'Current Catalog',
        'Value': current_cat,
        'Status': '✅ Accessible'
    })
    
    # Current schema  
    current_schema = spark.sql("SELECT current_schema() as schema").collect()[0]['schema']
    current_info.append({
        'Setting': 'Current Schema',
        'Value': current_schema, 
        'Status': '✅ Accessible'
    })
    
    # Current timestamp
    current_ts = spark.sql("SELECT current_timestamp() as ts").collect()[0]['ts']
    current_info.append({
        'Setting': 'Current Timestamp',
        'Value': str(current_ts),
        'Status': '✅ Working'
    })
    
    display(pd.DataFrame(current_info))
    
except Exception as e:
    error_info = [{
        'Error_Type': 'Current Context Failed',
        'Error_Message': str(e),
        'Status': '❌ Basic Access Issue'
    }]
    display(pd.DataFrame(error_info))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. ADP Schema Access Testing

# COMMAND ----------

# Define your ADP schemas here - REPLACE THESE ALIASES!
adp_schemas = [
    "ADP_SCHEMA_1",  # Replace with actual schema name
    "ADP_SCHEMA_2",  # Replace with actual schema name  
    "ADP_SCHEMA_3",  # Replace with actual schema name
    "ADP_SCHEMA_4",  # Replace with actual schema name
    "ADP_SCHEMA_5",  # Replace with actual schema name
    "ADP_SCHEMA_6"   # Replace with actual schema name
]

schema_results = []

for schema in adp_schemas:
    result = {
        'Schema_Name': schema,
        'Exists': 'Unknown',
        'USE_Permission': 'Unknown', 
        'Table_Count': 0,
        'Error_Details': 'None',
        'Status': 'Untested'
    }
    
    # Test 1: Schema Existence
    try:
        schema_info = spark.sql(f"DESCRIBE SCHEMA {schema}")
        result['Exists'] = '✅ Yes'
    except Exception as e:
        result['Exists'] = '❌ No'
        result['Error_Details'] = str(e)[:100]
    
    # Test 2: USE Permission (only if exists)
    if result['Exists'] == '✅ Yes':
        try:
            spark.sql(f"USE {schema}")
            result['USE_Permission'] = '✅ Yes'
            
            # Test 3: Table Listing (only if USE works)
            try:
                tables = spark.sql(f"SHOW TABLES IN {schema}")
                table_count = tables.count()
                result['Table_Count'] = table_count
                result['Status'] = '✅ Full Access'
            except Exception as e:
                result['Status'] = '⚠️ USE OK, Tables Failed'
                result['Error_Details'] = str(e)[:100]
                
        except Exception as e:
            result['USE_Permission'] = '❌ No'
            result['Status'] = '❌ Permission Denied'
            result['Error_Details'] = str(e)[:100]
    else:
        result['Status'] = '❌ Schema Not Found'
    
    schema_results.append(result)

# Display comprehensive schema results
display(pd.DataFrame(schema_results))

# COMMAND ----------

# MAGIC %md  
# MAGIC ## 5. Table Access Testing for Working Schemas

# COMMAND ----------

# Get schemas that have full access
working_schemas = [r['Schema_Name'] for r in schema_results if r['Status'] == '✅ Full Access']

table_test_results = []

for schema in working_schemas:
    try:
        # Get table list
        tables_df = spark.sql(f"SHOW TABLES IN {schema}")
        tables_list = [row['tableName'] for row in tables_df.collect()]
        
        # Test first 5 tables in each working schema
        for table_name in tables_list[:5]:
            full_table_name = f"{schema}.{table_name}"
            
            table_result = {
                'Schema': schema,
                'Table_Name': table_name,
                'Full_Name': full_table_name,
                'Row_Count': 0,
                'Column_Count': 0,
                'Access_Status': 'Unknown',
                'Error': 'None'
            }
            
            try:
                # Test row count
                count_df = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_table_name}")
                row_count = count_df.collect()[0]['cnt']
                table_result['Row_Count'] = row_count
                
                # Test column count
                desc_df = spark.sql(f"DESCRIBE TABLE {full_table_name}")
                col_count = desc_df.count()
                table_result['Column_Count'] = col_count
                
                table_result['Access_Status'] = '✅ Full Access'
                
            except Exception as e:
                table_result['Access_Status'] = '❌ Access Denied'
                table_result['Error'] = str(e)[:80]
            
            table_test_results.append(table_result)
            
    except Exception as e:
        error_result = {
            'Schema': schema,
            'Table_Name': 'Schema Error',
            'Full_Name': schema,
            'Row_Count': 0,
            'Column_Count': 0,
            'Access_Status': '❌ Schema Error',
            'Error': str(e)[:80]
        }
        table_test_results.append(error_result)

# Display table access results
if table_test_results:
    display(pd.DataFrame(table_test_results))
else:
    no_access_msg = [{
        'Message': 'No accessible schemas found for table testing',
        'Status': '❌ Critical Access Issue',
        'Action_Needed': 'Immediate escalation required'
    }]
    display(pd.DataFrame(no_access_msg))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Specific ADP Table Testing

# COMMAND ----------

# Test specific important tables - REPLACE THESE ALIASES!
important_tables = [
    "ADP_SCHEMA_1.ADP_TABLE_1",  # Replace with actual schema.table
    "ADP_SCHEMA_1.ADP_TABLE_2",  # Replace with actual schema.table
    "ADP_SCHEMA_2.ADP_TABLE_3",  # Replace with actual schema.table
]

specific_table_results = []

for table in important_tables:
    result = {
        'Table_Full_Name': table,
        'Exists': 'Unknown',
        'Row_Count': 0,
        'Sample_Columns': 'None',
        'Data_Types': 'None',
        'Status': 'Unknown',
        'Error': 'None'
    }
    
    try:
        # Test existence and basic info
        desc_result = spark.sql(f"DESCRIBE TABLE {table}")
        columns_info = desc_result.collect()
        
        result['Exists'] = '✅ Yes'
        result['Sample_Columns'] = ', '.join([col['col_name'] for col in columns_info[:5]])
        result['Data_Types'] = ', '.join([col['data_type'] for col in columns_info[:5]])
        
        # Test row count
        count_result = spark.sql(f"SELECT COUNT(*) as cnt FROM {table}")
        row_count = count_result.collect()[0]['cnt']
        result['Row_Count'] = row_count
        
        result['Status'] = '✅ Fully Accessible'
        
    except Exception as e:
        result['Exists'] = '❌ No'
        result['Status'] = '❌ Not Accessible'
        result['Error'] = str(e)[:100]
    
    specific_table_results.append(result)

display(pd.DataFrame(specific_table_results))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Alternative Catalog/Schema Exploration

# COMMAND ----------

# Try common catalog patterns
alternative_patterns = [
    "hive_metastore",
    "main", 
    "samples",
    "system",
    "default"
]

alternative_results = []

for catalog in alternative_patterns:
    result = {
        'Catalog_Name': catalog,
        'Accessible': 'Unknown',
        'Schema_Count': 0,
        'Sample_Schemas': 'None'
    }
    
    try:
        # Try to show schemas in this catalog
        schemas_df = spark.sql(f"SHOW SCHEMAS IN {catalog}")
        schema_list = [row['databaseName'] for row in schemas_df.collect()]
        
        result['Accessible'] = '✅ Yes'
        result['Schema_Count'] = len(schema_list)
        result['Sample_Schemas'] = ', '.join(schema_list[:3])
        
    except Exception as e:
        result['Accessible'] = '❌ No'
        result['Error'] = str(e)[:60]
    
    alternative_results.append(result)

display(pd.DataFrame(alternative_results))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Sample Data Preview (for accessible tables)

# COMMAND ----------

# Get sample data from first accessible table
accessible_tables = [r for r in table_test_results if r.get('Access_Status') == '✅ Full Access']

if accessible_tables:
    # Take first accessible table
    sample_table = accessible_tables[0]['Full_Name']
    
    try:
        # Get sample rows
        sample_df = spark.sql(f"SELECT * FROM {sample_table} LIMIT 3")
        display(sample_df)
        
        # Get table schema
        schema_df = spark.sql(f"DESCRIBE TABLE {sample_table}")
        display(schema_df)
        
    except Exception as e:
        error_msg = [{
            'Error': 'Sample data retrieval failed',
            'Table': sample_table,
            'Message': str(e)
        }]
        display(pd.DataFrame(error_msg))
else:
    no_sample_msg = [{
        'Message': 'No accessible tables found for sample data',
        'Status': '❌ No data access available'
    }]
    display(pd.DataFrame(no_sample_msg))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Comprehensive Access Summary

# COMMAND ----------

# Create comprehensive summary
summary_stats = []

# Schema access summary
total_schemas = len(schema_results)
accessible_schemas = len([r for r in schema_results if r['Status'] == '✅ Full Access'])
schema_success_rate = (accessible_schemas / total_schemas * 100) if total_schemas > 0 else 0

summary_stats.append({
    'Category': 'Schema Access',
    'Total_Tested': total_schemas,
    'Successful': accessible_schemas,
    'Success_Rate': f"{schema_success_rate:.1f}%",
    'Status': '✅ Good' if schema_success_rate >= 50 else '❌ Poor'
})

# Table access summary  
total_tables = len(table_test_results)
accessible_tables = len([r for r in table_test_results if r.get('Access_Status') == '✅ Full Access'])
table_success_rate = (accessible_tables / total_tables * 100) if total_tables > 0 else 0

summary_stats.append({
    'Category': 'Table Access',
    'Total_Tested': total_tables,
    'Successful': accessible_tables, 
    'Success_Rate': f"{table_success_rate:.1f}%",
    'Status': '✅ Good' if table_success_rate >= 50 else '❌ Poor'
})

# Overall data availability
total_data_sources = accessible_tables
data_status = 'READY' if total_data_sources >= 5 else 'LIMITED' if total_data_sources >= 1 else 'BLOCKED'

summary_stats.append({
    'Category': 'Overall Data Access',
    'Total_Tested': 'N/A',
    'Successful': total_data_sources,
    'Success_Rate': data_status,
    'Status': '✅ Ready' if data_status == 'READY' else '⚠️ Limited' if data_status == 'LIMITED' else '❌ Blocked'
})

display(pd.DataFrame(summary_stats))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Business Impact Assessment

# COMMAND ----------

# Assess impact on ADP project timeline
business_impact = []

# Data availability assessment
if accessible_tables >= 5:
    impact_level = "🟢 LOW IMPACT"
    timeline_status = "On Track"
    action_needed = "Proceed with data exploration"
elif accessible_tables >= 1:
    impact_level = "🟡 MEDIUM IMPACT" 
    timeline_status = "At Risk"
    action_needed = "Escalate for additional access, continue with available data"
else:
    impact_level = "🔴 HIGH IMPACT"
    timeline_status = "Blocked"
    action_needed = "EMERGENCY escalation required"

business_impact.append({
    'Assessment_Area': 'Data Availability',
    'Impact_Level': impact_level,
    'Timeline_Status': timeline_status,
    'Action_Required': action_needed
})

# Survival analysis readiness
survival_ready = accessible_tables >= 3  # Need multiple tables for survival analysis
survival_status = "Ready" if survival_ready else "Not Ready"
survival_action = "Begin survival analysis pipeline" if survival_ready else "Need more data sources"

business_impact.append({
    'Assessment_Area': 'Survival Analysis Readiness',
    'Impact_Level': '✅ Ready' if survival_ready else '❌ Not Ready',
    'Timeline_Status': survival_status,
    'Action_Required': survival_action
})

display(pd.DataFrame(business_impact))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Escalation Information

# COMMAND ----------

# Create escalation summary
escalation_info = []

# Critical issues found
critical_issues = [r for r in schema_results if '❌' in r['Status']]
permission_errors = [r for r in schema_results if 'Permission Denied' in r['Status']]

escalation_info.append({
    'Issue_Type': 'Schema Access Failures',
    'Count': len(critical_issues),
    'Severity': 'High' if len(critical_issues) > len(schema_results)/2 else 'Medium',
    'Escalation_Needed': 'Yes' if len(critical_issues) > 0 else 'No'
})

escalation_info.append({
    'Issue_Type': 'Permission Denied Errors',
    'Count': len(permission_errors),
    'Severity': 'Critical' if len(permission_errors) > 0 else 'None',
    'Escalation_Needed': 'Immediate' if len(permission_errors) > 0 else 'No'
})

display(pd.DataFrame(escalation_info))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Export Results Summary

# COMMAND ----------

# Create final results for team sharing
final_results = {
    'test_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_schemas_tested': len(schema_results),
    'accessible_schemas': len([r for r in schema_results if r['Status'] == '✅ Full Access']),
    'total_tables_tested': len(table_test_results),
    'accessible_tables': len([r for r in table_test_results if r.get('Access_Status') == '✅ Full Access']),
    'critical_issues': len([r for r in schema_results if '❌' in r['Status']]),
    'overall_status': data_status,
    'escalation_required': len(permission_errors) > 0
}

# Convert to display format
final_summary = []
for key, value in final_results.items():
    final_summary.append({
        'Metric': key.replace('_', ' ').title(),
        'Value': str(value),
        'For_Standup': f"{key}: {value}"
    })

display(pd.DataFrame(final_summary))

# COMMAND ----------

# Test completion message
completion_msg = [{
    'Status': 'TEST COMPLETED',
    'Total_Cells_Run': '12 cells completed successfully',
    'Results_Available': 'All results displayed above',
    'Next_Action': 'Review Section 9 (Summary) and Section 11 (Escalation) for key findings',
    'Escalation_Status': 'REQUIRED' if final_results['escalation_required'] else 'Optional'
}]

display(pd.DataFrame(completion_msg))
