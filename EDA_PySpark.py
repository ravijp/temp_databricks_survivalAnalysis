# DATABRICKS USAGE EXAMPLES FOR EMPLOYEE_MAIN_MONTHLY EDA
# ========================================================

# 1. IMMEDIATE QUICK INSPECTION (30 seconds)
# -------------------------------------------
def instant_table_check(table_name="employee_main_monthly"):
    """Get immediate insights in under 30 seconds"""
    df = spark.table(table_name)
    
    print(f"⚡ INSTANT TABLE CHECK: {table_name}")
    print("=" * 50)
    
    # Basic info
    print(f"📊 Dimensions: {df.count():,} rows × {len(df.columns)} columns")
    
    # Schema preview
    print(f"\n🏗️  Schema Preview:")
    for i, field in enumerate(df.schema.fields[:10], 1):
        print(f"  {i:2d}. {field.name:<25} | {str(field.dataType):<15}")
    if len(df.columns) > 10:
        print(f"     ... and {len(df.columns) - 10} more columns")
    
    # Sample data
    print(f"\n🔍 Sample Data:")
    df.show(3, truncate=True)
    
    return df

# Execute immediate check
df = instant_table_check("employee_main_monthly")


# 2. COMPREHENSIVE ANALYSIS (5-10 minutes)
# -----------------------------------------
# Run the full EDA analysis
results = analyze_employee_main_monthly(spark)


# 3. FOCUSED ANALYSIS FOR SURVIVAL MODELING
# ------------------------------------------
def survival_focused_eda(table_name="employee_main_monthly"):
    """EDA focused on survival analysis requirements"""
    df = spark.table(table_name)
    
    print("🎯 SURVIVAL ANALYSIS FOCUSED EDA")
    print("=" * 50)
    
    # 1. Identify key columns for survival analysis
    all_cols = [field.name.lower() for field in df.schema.fields]
    
    # Employee identifier
    id_candidates = [col for col in df.columns if any(term in col.lower() for term in ['id', 'employee', 'person', 'emp'])]
    print(f"👤 Employee ID candidates: {id_candidates}")
    
    # Time-related columns (critical for survival)
    time_candidates = [col for col in df.columns if any(term in col.lower() for term in 
                      ['date', 'time', 'month', 'year', 'period', 'start', 'end', 'hire', 'term'])]
    print(f"📅 Time-related columns: {time_candidates}")
    
    # Status/Event columns
    status_candidates = [col for col in df.columns if any(term in col.lower() for term in 
                        ['status', 'active', 'term', 'left', 'quit', 'exit', 'end', 'flag'])]
    print(f"🚦 Status/Event columns: {status_candidates}")
    
    # Potential covariates
    covariate_candidates = [col for col in df.columns if any(term in col.lower() for term in 
                           ['salary', 'wage', 'pay', 'manager', 'dept', 'title', 'role', 'location', 
                            'age', 'gender', 'tenure', 'performance', 'rating', 'hours', 'overtime'])]
    print(f"📊 Potential Covariates: {covariate_candidates}")
    
    # 2. Check for time-varying structure
    print(f"\n⏰ TIME-VARYING DATA CHECK")
    print("-" * 30)
    
    if id_candidates and time_candidates:
        # Check if we have multiple records per employee (time-varying)
        id_col = id_candidates[0]  # Use first ID candidate
        
        total_employees = df.select(id_col).distinct().count()
        total_records = df.count()
        avg_records_per_employee = total_records / total_employees
        
        print(f"Total unique employees: {total_employees:,}")
        print(f"Total records: {total_records:,}")
        print(f"Avg records per employee: {avg_records_per_employee:.2f}")
        
        if avg_records_per_employee > 1.5:
            print("✅ Time-varying data detected - suitable for survival analysis")
            
            # Show example of time-varying structure
            print(f"\n📋 Example time-varying structure:")
            sample_employee = df.select(id_col).distinct().limit(1).collect()[0][id_col]
            employee_records = df.filter(col(id_col) == sample_employee).orderBy(*time_candidates[:1])
            employee_records.show(10, truncate=False)
            
        else:
            print("⚠️  Appears to be cross-sectional data - may need transformation")
    
    # 3. Missing data assessment for key variables
    print(f"\n❌ MISSING DATA FOR KEY VARIABLES")
    print("-" * 40)
    
    key_cols = id_candidates + time_candidates + status_candidates
    for col_name in key_cols[:10]:  # Limit to avoid overwhelming output
        null_count = df.filter(col(col_name).isNull()).count()
        null_pct = (null_count / df.count()) * 100
        print(f"{col_name:<25}: {null_count:>8,} nulls ({null_pct:>5.1f}%)")
    
    return {
        'id_candidates': id_candidates,
        'time_candidates': time_candidates,
        'status_candidates': status_candidates,
        'covariate_candidates': covariate_candidates,
        'is_time_varying': avg_records_per_employee > 1.5 if id_candidates and time_candidates else False
    }

# Execute survival-focused analysis
survival_info = survival_focused_eda("employee_main_monthly")


# 4. COLUMN DEEP DIVE (for specific columns)
# -------------------------------------------
def deep_dive_column(table_name, column_name):
    """Deep dive into a specific column"""
    df = spark.table(table_name)
    
    print(f"🔬 DEEP DIVE: {column_name}")
    print("=" * 40)
    
    # Basic info
    col_type = dict(df.dtypes)[column_name]
    print(f"Data Type: {col_type}")
    
    # Null analysis
    total_count = df.count()
    null_count = df.filter(col(column_name).isNull()).count()
    print(f"Total records: {total_count:,}")
    print(f"Null records: {null_count:,} ({null_count/total_count*100:.2f}%)")
    
    # Type-specific analysis
    if col_type in ['string']:
        # Categorical analysis
        unique_count = df.select(column_name).distinct().count()
        print(f"Unique values: {unique_count:,}")
        
        if unique_count <= 50:
            print("\nValue distribution:")
            df.groupBy(column_name).count().orderBy(desc("count")).show(20)
        else:
            print("\nTop 10 values:")
            df.groupBy(column_name).count().orderBy(desc("count")).show(10)
            
    elif col_type in ['int', 'bigint', 'double', 'float']:
        # Numeric analysis
        stats = df.select(column_name).describe().collect()
        for stat in stats:
            print(f"{stat['summary']}: {stat[column_name]}")
            
        # Percentiles
        percentiles = df.select(column_name).na.drop().approxQuantile(column_name, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], 0.01)
        print(f"\nPercentiles:")
        print(f"1%: {percentiles[0]:.4f}, 5%: {percentiles[1]:.4f}, 25%: {percentiles[2]:.4f}")
        print(f"50%: {percentiles[3]:.4f}, 75%: {percentiles[4]:.4f}, 95%: {percentiles[5]:.4f}, 99%: {percentiles[6]:.4f}")
        
    elif col_type in ['date', 'timestamp']:
        # Date analysis
        date_stats = df.select(
            min(column_name).alias('min_date'),
            max(column_name).alias('max_date')
        ).collect()[0]
        
        print(f"Date range: {date_stats['min_date']} to {date_stats['max_date']}")
        
        # Show monthly distribution if it's a date
        print("\nMonthly distribution:")
        df.select(date_format(column_name, "yyyy-MM").alias("month")) \
          .groupBy("month") \
          .count() \
          .orderBy("month") \
          .show(20)

# Example usage for specific columns
# deep_dive_column("employee_main_monthly", "employee_id")  # Replace with actual column name


# 5. QUICK COMMANDS FOR IMMEDIATE USE
# ------------------------------------

# Get column names only
def get_columns(table_name="employee_main_monthly"):
    """Quick column list"""
    df = spark.table(table_name)
    return df.columns

# Show first few rows
def peek(table_name="employee_main_monthly", rows=5):
    """Quick peek at data"""
    spark.table(table_name).show(rows, truncate=False)

# Get table size
def table_size(table_name="employee_main_monthly"):
    """Quick size check"""
    df = spark.table(table_name)
    return f"{df.count():,} rows × {len(df.columns)} columns"

# Check if table exists
def table_exists(table_name):
    """Check if table exists"""
    try:
        spark.table(table_name).limit(1).collect()
        return True
    except:
        return False

# EXECUTION EXAMPLES
# ==================

print("🚀 READY TO ANALYZE employee_main_monthly")
print("=" * 50)
print("Available functions:")
print("1. instant_table_check() - 30 second overview")
print("2. comprehensive_table_eda() - Full 5-10 minute analysis")  
print("3. survival_focused_eda() - Survival analysis specific")
print("4. deep_dive_column() - Individual column analysis")
print("5. Quick utilities: get_columns(), peek(), table_size()")
print()
print("Example usage:")
print("# Quick start:")
print("df = instant_table_check('employee_main_monthly')")
print()
print("# Full analysis:")
print("results = comprehensive_table_eda(spark, 'employee_main_monthly')")
print()
print("# Survival focus:")
print("survival_info = survival_focused_eda('employee_main_monthly')")

# FOR IMMEDIATE EXECUTION - UNCOMMENT THE LINES BELOW:
# =====================================================

# Step 1: Quick check (30 seconds)
print("\n" + "="*60)
print("STEP 1: QUICK TABLE CHECK")
print("="*60)
df = instant_table_check("employee_main_monthly")

# Step 2: Survival-focused analysis (2-3 minutes)  
print("\n" + "="*60)
print("STEP 2: SURVIVAL ANALYSIS FOCUS")
print("="*60)
survival_info = survival_focused_eda("employee_main_monthly")

# Step 3: Full comprehensive analysis (5-10 minutes)
# Uncomment below when ready for full analysis:
# print("\n" + "="*60)
# print("STEP 3: COMPREHENSIVE ANALYSIS")  
# print("="*60)
# results = comprehensive_table_eda(spark, "employee_main_monthly")
