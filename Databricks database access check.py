# Databricks notebook source
# MAGIC %md
# MAGIC # ADP Quick Access Diagnostic - CRITICAL ISSUE RESOLUTION
# MAGIC 
# MAGIC **Problem Identified:** Schema permission issues blocking table access
# MAGIC **Priority:** IMMEDIATE escalation required

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Basic Environment Check

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
from datetime import datetime

spark = SparkSession.builder.getOrCreate()

print("🔧 ENVIRONMENT DIAGNOSTIC")
print("=" * 40)
print(f"Timestamp: {datetime.now()}")
print(f"Spark Version: {spark.version}")
print(f"User: {spark.sparkContext.sparkUser()}")
print(f"App Name: {spark.sparkContext.appName}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Catalog/Schema Permissions Test

# COMMAND ----------

def test_basic_catalog_access():
    """
    Test basic catalog and schema access before testing individual tables
    """
    print("🔍 CATALOG ACCESS DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Can we list catalogs?
    try:
        catalogs = spark.sql("SHOW CATALOGS").collect()
        print("✅ Can list catalogs:")
        for catalog in catalogs:
            print(f"   - {catalog['catalog']}")
    except Exception as e:
        print(f"❌ Cannot list catalogs: {str(e)}")
        return False
    
    # Test 2: Can we list schemas in each catalog?
    print(f"\n🗂️  SCHEMA ACCESS TEST:")
    accessible_schemas = []
    
    for catalog in catalogs:
        catalog_name = catalog['catalog']
        try:
            schemas = spark.sql(f"SHOW SCHEMAS IN {catalog_name}").collect()
            print(f"✅ {catalog_name}: {len(schemas)} schemas")
            for schema in schemas[:3]:  # Show first 3
                schema_full = f"{catalog_name}.{schema['databaseName']}"
                accessible_schemas.append(schema_full)
                print(f"   - {schema_full}")
            if len(schemas) > 3:
                print(f"   ... and {len(schemas)-3} more")
        except Exception as e:
            print(f"❌ {catalog_name}: {str(e)[:100]}...")
    
    return accessible_schemas

accessible_schemas = test_basic_catalog_access()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Specific ADP Schema Access

# COMMAND ----------

def test_adp_specific_access():
    """
    Test access to likely ADP schema patterns
    """
    print("🎯 ADP-SPECIFIC SCHEMA ACCESS TEST")
    print("=" * 45)
    
    # Common ADP schema patterns (adjust based on your actual names)
    potential_adp_schemas = [
        "onedata_us_east_1_shared_prod.us_east_1_prd_ds_blue_raw",  # From your error
        "onedata_us_east_1_shared_prod.employee_data",
        "adp_analytics.employee_data", 
        "analytics.employee_data",
        "prod.employee_data",
        "shared_prod.analytics"
    ]
    
    working_schemas = []
    
    for schema in potential_adp_schemas:
        try:
            # Test USE SCHEMA permission
            spark.sql(f"USE {schema}")
            print(f"✅ {schema}: USE permission granted")
            
            # Test listing tables
            tables = spark.sql(f"SHOW TABLES IN {schema}").collect()
            print(f"   📊 {len(tables)} tables found")
            working_schemas.append(schema)
            
            # Show first few table names
            for table in tables[:3]:
                print(f"   - {table['tableName']}")
            if len(tables) > 3:
                print(f"   ... and {len(tables)-3} more")
                
        except Exception as e:
            error_msg = str(e)
            if "PERMISSION_DENIED" in error_msg:
                print(f"❌ {schema}: PERMISSION DENIED")
            elif "not found" in error_msg.lower():
                print(f"⚠️  {schema}: Schema not found")
            else:
                print(f"❌ {schema}: {error_msg[:80]}...")
    
    return working_schemas

working_schemas = test_adp_specific_access()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Quick Sample Table Test

# COMMAND ----------

def test_sample_tables(working_schemas):
    """
    Quick test of actual data access in working schemas
    """
    if not working_schemas:
        print("❌ No accessible schemas found - cannot test tables")
        return []
    
    print("📋 SAMPLE TABLE ACCESS TEST")
    print("=" * 40)
    
    accessible_tables = []
    
    for schema in working_schemas[:2]:  # Test first 2 working schemas
        try:
            tables = spark.sql(f"SHOW TABLES IN {schema}").collect()
            
            # Test first few tables
            for table in tables[:3]:
                table_name = f"{schema}.{table['tableName']}"
                try:
                    # Quick row count test
                    count_result = spark.sql(f"SELECT COUNT(*) as cnt FROM {table_name}").collect()
                    row_count = count_result[0]['cnt']
                    print(f"✅ {table_name}: {row_count:,} rows")
                    accessible_tables.append({
                        'table': table_name,
                        'rows': row_count,
                        'schema': schema
                    })
                except Exception as e:
                    print(f"❌ {table_name}: {str(e)[:60]}...")
                    
        except Exception as e:
            print(f"❌ Schema {schema}: {str(e)[:60]}...")
    
    return accessible_tables

accessible_tables = test_sample_tables(working_schemas)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Critical Issue Summary & Next Steps

# COMMAND ----------

def generate_critical_issue_summary(accessible_schemas, accessible_tables):
    """
    Generate immediate action summary for critical access issues
    """
    
    print("🚨 CRITICAL ISSUE SUMMARY")
    print("=" * 50)
    print(f"Test completed: {datetime.now().strftime('%H:%M:%S')}")
    
    # Status assessment
    if len(accessible_tables) >= 5:
        status = "🟡 PARTIAL ACCESS - Can proceed with available data"
        priority = "Medium - Continue development, escalate for full access"
    elif len(accessible_tables) >= 1:
        status = "🟠 LIMITED ACCESS - Minimal viable dataset"
        priority = "High - Immediate escalation needed"
    else:
        status = "🔴 NO DATA ACCESS - Project blocked"
        priority = "CRITICAL - Emergency escalation required"
    
    print(f"Status: {status}")
    print(f"Priority: {priority}")
    
    print(f"\n📊 ACCESS SUMMARY:")
    print(f"  Accessible schemas: {len(accessible_schemas)}")
    print(f"  Accessible tables: {len(accessible_tables)}")
    
    if accessible_schemas:
        print(f"\n✅ WORKING SCHEMAS:")
        for schema in accessible_schemas:
            print(f"  - {schema}")
    
    if accessible_tables:
        print(f"\n✅ ACCESSIBLE TABLES:")
        for table_info in accessible_tables:
            print(f"  - {table_info['table']}: {table_info['rows']:,} rows")
    
    # Immediate actions
    print(f"\n🎯 IMMEDIATE ACTIONS:")
    
    if len(accessible_tables) == 0:
        print("  1. 🚨 EMERGENCY: Contact Blair immediately")
        print("  2. 📧 CC: George + Account Lead on escalation")  
        print("  3. ⏰ Request: Emergency access resolution meeting")
        print("  4. 📋 Prepare: Timeline impact assessment")
        
    elif len(accessible_tables) < 5:
        print("  1. 📞 Contact Blair for additional schema access")
        print("  2. 🔄 Begin limited development with available tables")
        print("  3. 📋 Document minimum viable dataset requirements")
        print("  4. ⏱️  Set 24-hour resolution target for full access")
        
    else:
        print("  1. ✅ Begin data exploration with accessible tables")
        print("  2. 📞 Contact Blair for any missing critical tables")
        print("  3. 📋 Document accessible dataset for team")
    
    # Contact information
    print(f"\n📞 ESCALATION CONTACTS:")
    print("  Blair Christian - blair.christian@adp.com")
    print("  Dinesh Prodduturi - dinesh.prodduturi@adp.com") 
    print("  George Hatziemanuel - george.hatziemanuel@adp.com")
    
    # Email template
    print(f"\n📧 EMERGENCY EMAIL TEMPLATE:")
    print("Subject: URGENT - ADP Turnover Project Schema Access Issues")
    print("")
    print("Hi Blair,")
    print("")
    print(f"Critical access issue blocking ADP turnover project:")
    print(f"- Error: PERMISSION_DENIED on schema access")
    print(f"- Impact: {len(accessible_tables)} of expected tables accessible")
    print(f"- Timeline risk: Week 1 deliverables at risk")
    print("")
    print("Specific error from testing:")
    print("PERMISSION_DENIED: user does not have USE SCHEMA on Schema")
    print("")
    print("Request immediate resolution to maintain 4-week delivery timeline.")
    print("")
    print("Available for emergency call to resolve.")
    print("Ravi & Zenon Team")
    
    return {
        'status': status,
        'accessible_schemas': len(accessible_schemas),
        'accessible_tables': len(accessible_tables),
        'priority': priority
    }

# Generate summary
summary = generate_critical_issue_summary(accessible_schemas, accessible_tables)

# COMMAND ----------

print("⏰ QUICK DIAGNOSTIC COMPLETE")
print("Copy above summary for immediate Blair escalation")
print("Focus on schema permissions, not individual table access")
