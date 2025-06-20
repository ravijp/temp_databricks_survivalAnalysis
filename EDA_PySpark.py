from pyspark.sql.functions import *
from pyspark.sql.types import *
import pandas as pd
import os
import time

def generate_edd(table_name, output_path="/dbfs/FileStore/edd", filter_condition=None, sample_rate=None):
    """
    Generate EDD CSV file for a table
    
    Parameters:
    table_name: Name of the table to analyze
    output_path: Directory to save CSV file
    filter_condition: Optional SQL WHERE condition (e.g., "year = 2024")
    sample_rate: Optional sampling rate between 0 and 1 (e.g., 0.1 for 10%)
    
    Returns: Path to generated CSV file
    """
    
    print(f"Generating EDD for: {table_name}")
    
    # Load table
    df = spark.table(table_name)
    
    # Apply filter if provided
    if filter_condition:
        df = df.filter(filter_condition)
        print(f"Applied filter: {filter_condition}")
    
    # Apply sampling if provided
    if sample_rate:
        df = df.sample(sample_rate, seed=42)
        print(f"Using {sample_rate*100}% sample")
    
    # Get row count
    total_rows = df.count()
    print(f"Analyzing {total_rows:,} rows")
    
    # Prepare results
    results = []
    columns = df.columns
    
    # Analyze each column
    for i, col_name in enumerate(columns, 1):
        print(f"Processing {i}/{len(columns)}: {col_name}")
        
        try:
            # Basic counts
            non_null_count = df.select(count(col(col_name))).collect()[0][0]
            null_count = total_rows - non_null_count
            unique_count = df.select(col_name).distinct().count()
            
            # Check if numeric
            col_type = dict(df.dtypes)[col_name]
            is_numeric = col_type in ['int', 'bigint', 'float', 'double', 'decimal']
            
            if is_numeric:
                # Numeric analysis
                try:
                    stats = df.select(col_name).na.drop().describe().collect()
                    stats_dict = {row['summary']: float(row[col_name]) for row in stats}
                    
                    percentiles = df.select(col_name).na.drop().approxQuantile(
                        col_name, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], 0.05)
                    
                    results.append({
                        'Field_Num': i,
                        'Field_Name': col_name,
                        'Type': 'Numeric',
                        'Num_Blanks': null_count,
                        'Num_Entries': non_null_count,
                        'Num_Unique': unique_count,
                        'Mean': round(stats_dict.get('mean', 0), 4),
                        'Stddev': round(stats_dict.get('stddev', 0), 4),
                        'Min': stats_dict.get('min', 0),
                        'P1': percentiles[0] if len(percentiles) > 0 else None,
                        'P5': percentiles[1] if len(percentiles) > 1 else None,
                        'P25': percentiles[2] if len(percentiles) > 2 else None,
                        'Median': percentiles[3] if len(percentiles) > 3 else None,
                        'P75': percentiles[4] if len(percentiles) > 4 else None,
                        'P95': percentiles[5] if len(percentiles) > 5 else None,
                        'P99': percentiles[6] if len(percentiles) > 6 else None,
                        'Max': stats_dict.get('max', 0)
                    })
                except:
                    # Fallback for numeric columns that fail
                    results.append({
                        'Field_Num': i,
                        'Field_Name': col_name,
                        'Type': 'Numeric',
                        'Num_Blanks': null_count,
                        'Num_Entries': non_null_count,
                        'Num_Unique': unique_count,
                        'Note': 'Stats_Error'
                    })
            else:
                # Categorical analysis
                result = {
                    'Field_Num': i,
                    'Field_Name': col_name,
                    'Type': 'Categorical',
                    'Num_Blanks': null_count,
                    'Num_Entries': non_null_count,
                    'Num_Unique': unique_count
                }
                
                # Get top values if not too many unique values
                if unique_count <= 1000 and non_null_count > 0:
                    try:
                        top_values = (df.select(col_name)
                                    .na.drop()
                                    .groupBy(col_name)
                                    .count()
                                    .orderBy(desc("count"))
                                    .limit(5)
                                    .collect())
                        
                        for j, row in enumerate(top_values):
                            result[f'Top_{j+1}'] = f"{row[col_name]}:{row['count']}"
                    except:
                        result['Note'] = 'Top_Values_Error'
                else:
                    result['Note'] = 'High_Cardinality' if unique_count > 1000 else 'No_Data'
                
                results.append(result)
        
        except Exception as e:
            # Error handling
            results.append({
                'Field_Num': i,
                'Field_Name': col_name,
                'Type': 'Error',
                'Num_Blanks': 0,
                'Num_Entries': 0,
                'Num_Unique': 0,
                'Error': str(e)
            })
    
    # Save to CSV
    os.makedirs(output_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{table_name.replace('.', '_')}_edd_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    pd.DataFrame(results).to_csv(filepath, index=False)
    
    print(f"EDD saved: {filepath}")
    return filepath

# Batch function for multiple tables
def batch_edd(table_list, output_path="/dbfs/FileStore/edd", sample_rate=None):
    """
    Generate EDD for multiple tables
    
    Parameters:
    table_list: List of table names
    output_path: Directory to save CSV files
    sample_rate: Optional sampling rate for all tables
    """
    
    print(f"Batch EDD for {len(table_list)} tables")
    results = {}
    
    for i, table in enumerate(table_list, 1):
        print(f"\n[{i}/{len(table_list)}] {table}")
        try:
            filepath = generate_edd(table, output_path, sample_rate=sample_rate)
            results[table] = filepath
        except Exception as e:
            print(f"ERROR: {e}")
            results[table] = f"ERROR: {e}"
    
    print(f"\nBatch complete:")
    for table, result in results.items():
        status = "SUCCESS" if result.endswith('.csv') else "FAILED"
        print(f"  {table}: {status}")
    
    return results

# Simple usage examples
if __name__ == "__main__":
    
    # Single table
    # generate_edd("employee_main_monthly")
    
    # Single table with filter
    # generate_edd("large_table", filter_condition="year >= 2024")
    
    # Single table with sampling
    # generate_edd("huge_table", sample_rate=0.01)
    
    # Multiple tables
    # tables = ["table1", "table2", "table3"]
    # batch_edd(tables)
    
    # Multiple tables with sampling
    # batch_edd(tables, sample_rate=0.1)
    
    print("EDD functions ready to use")
