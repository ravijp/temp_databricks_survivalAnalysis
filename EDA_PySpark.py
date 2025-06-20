from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, desc, lit, isnan, isnull
from pyspark.sql.types import DoubleType, StringType
import pandas as pd
import os
import time
from typing import List, Optional, Dict, Any

def generate_edd(table_name: str, output_path: str = "/dbfs/tmp/edd", 
                filter_condition: Optional[str] = None, 
                sample_threshold: int = 50_000_000) -> str:
    """
    Generate EDD following original logic - dynamic type detection during processing
    """
    
    print(f"Processing: {table_name}")
    start_time = time.time()
    
    # Load and prepare data
    df = spark.table(table_name)
    if filter_condition:
        df = df.filter(filter_condition)
    
    # Get table size and determine strategy
    total_rows = df.count()
    sample_rate = min(1.0, sample_threshold / max(total_rows, 1))
    
    if sample_rate < 1.0:
        df = df.sample(sample_rate, seed=42)
        print(f"Sampling {sample_rate:.3f} of {total_rows:,} rows")
        total_rows = df.count()
    
    df.cache()
    columns = df.columns
    print(f"Analyzing {total_rows:,} rows, {len(columns)} columns")
    
    # Convert all columns to string first (like original EDD reads CSV)
    df_str = df.select([col(c).cast(StringType()).alias(c) for c in columns])
    
    # Process each column following original EDD logic
    results = []
    
    for i, column_name in enumerate(columns, 1):
        print(f"Processing column {i}/{len(columns)}: {column_name}")
        
        # Get basic counts
        null_count = df_str.filter(col(column_name).isNull() | (col(column_name) == "")).count()
        non_null_count = total_rows - null_count
        
        if non_null_count == 0:
            # All null column
            results.append(_create_empty_column_result(i, column_name, null_count, 0, 0))
            continue
        
        # Try to determine if column is numeric by attempting conversion
        is_numeric, numeric_data = _test_numeric_conversion(df_str, column_name)
        
        if is_numeric and len(numeric_data) > 0:
            # Process as numeric
            result = _process_numeric_column(i, column_name, numeric_data, null_count, non_null_count)
        else:
            # Process as categorical  
            result = _process_categorical_column(df_str, i, column_name, null_count, non_null_count, total_rows)
        
        results.append(result)
    
    df.unpersist()
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{table_name.replace('.', '_')}_edd_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    pd.DataFrame(results).to_csv(filepath, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"EDD completed in {elapsed:.1f} minutes: {filepath}")
    
    return filepath

def _test_numeric_conversion(df_str, column_name: str) -> tuple[bool, List[float]]:
    """Test if column can be converted to numeric, following original EDD logic"""
    
    try:
        # Get non-null, non-empty values
        non_empty_df = df_str.filter((col(column_name).isNotNull()) & (col(column_name) != ""))
        
        # Try to convert to double - this will fail if any non-numeric values exist
        numeric_df = non_empty_df.select(col(column_name).cast(DoubleType()).alias("numeric_val"))
        
        # Check if conversion resulted in nulls (meaning non-numeric data exists)
        null_after_conversion = numeric_df.filter(col("numeric_val").isNull()).count()
        
        if null_after_conversion > 0:
            # Some values couldn't be converted - treat as categorical
            return False, []
        
        # All values converted successfully - collect numeric data
        numeric_values = [row.numeric_val for row in numeric_df.collect() if row.numeric_val is not None]
        
        return True, numeric_values
        
    except Exception as e:
        # Any error means not numeric
        return False, []

def _process_numeric_column(field_num: int, column_name: str, numeric_data: List[float], 
                          null_count: int, non_null_count: int) -> Dict:
    """Process numeric column following original EDD format"""
    
    if not numeric_data:
        return _create_empty_column_result(field_num, column_name, null_count, non_null_count, 0, col_type="Numeric")
    
    import numpy as np
    
    # Convert to numpy array for calculations (like original)
    arr = np.array(numeric_data)
    
    # Calculate statistics
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    
    # Calculate percentiles
    try:
        p1 = float(np.percentile(arr, 1))
        p5 = float(np.percentile(arr, 5))
        p25 = float(np.percentile(arr, 25))
        median = float(np.percentile(arr, 50))
        p75 = float(np.percentile(arr, 75))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
    except:
        p1 = p5 = p25 = median = p75 = p95 = p99 = None
    
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Numeric',
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': len(set(numeric_data)),
        'Stddev': round(std_val, 6) if not np.isnan(std_val) else None,
        'Mean_or_Top1': round(mean_val, 6) if not np.isnan(mean_val) else None,
        'Min_or_Top2': min_val,
        'P1_or_Top3': p1,
        'P5_or_Top4': p5,
        'P25_or_Top5': p25,
        'Median_or_Bot5': median,
        'P75_or_Bot4': p75,
        'P95_or_Bot3': p95,
        'P99_or_Bot2': p99,
        'Max_or_Bot1': max_val
    }

def _process_categorical_column(df_str, field_num: int, column_name: str, 
                              null_count: int, non_null_count: int, total_rows: int) -> Dict:
    """Process categorical column following original EDD format exactly"""
    
    # Get value counts
    try:
        value_counts = (df_str.filter((col(column_name).isNotNull()) & (col(column_name) != ""))
                       .groupBy(column_name)
                       .count()
                       .orderBy(desc("count"), col(column_name))  # Secondary sort for consistency
                       .collect())
        
        unique_count = len(value_counts)
        
        # Create value frequency list like original EDD
        cat_values = [(row[column_name], row['count']) for row in value_counts]
        
    except Exception as e:
        print(f"Error processing categorical column {column_name}: {e}")
        unique_count = 0
        cat_values = []
    
    # Handle special cases following original logic
    if unique_count == 1:
        # All same value
        val_str = f"{cat_values[0][0]}:{cat_values[0][1]}" if cat_values else "All_Same"
        cat_display = [val_str] + ["All_Same"] * 19
    elif unique_count == non_null_count:
        # All unique
        cat_display = ["All_Unique"] * 20
    elif unique_count == 0:
        # No data
        cat_display = ["No_Data"] * 20
    else:
        # Normal case - format as value:count
        cat_display = [f"{val}:{count}" for val, count in cat_values]
        # Pad to 20 elements
        cat_display.extend([""] * max(0, 20 - len(cat_display)))
    
    # Map to EDD format following original indexing
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Categorical',
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Stddev': None,
        'Mean_or_Top1': cat_display[0],
        'Min_or_Top2': cat_display[1] if len(cat_display) > 1 else cat_display[0],
        'P1_or_Top3': cat_display[2] if len(cat_display) > 2 else cat_display[0],
        'P5_or_Top4': cat_display[3] if len(cat_display) > 3 else cat_display[0],
        'P25_or_Top5': cat_display[4] if len(cat_display) > 4 else cat_display[0],
        'Median_or_Bot5': cat_display[-5] if len(cat_display) >= 5 else cat_display[0],
        'P75_or_Bot4': cat_display[-4] if len(cat_display) >= 4 else cat_display[0],
        'P95_or_Bot3': cat_display[-3] if len(cat_display) >= 3 else cat_display[0],
        'P99_or_Bot2': cat_display[-2] if len(cat_display) >= 2 else cat_display[0],
        'Max_or_Bot1': cat_display[-1] if len(cat_display) >= 1 else cat_display[0]
    }

def _create_empty_column_result(field_num: int, column_name: str, null_count: int, 
                              non_null_count: int, unique_count: int, col_type: str = "Categorical") -> Dict:
    """Create result for empty/all-null columns"""
    
    base_result = {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': col_type,
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Stddev': None,
        'Mean_or_Top1': 'No_Data',
        'Min_or_Top2': 'No_Data',
        'P1_or_Top3': 'No_Data',
        'P5_or_Top4': 'No_Data',
        'P25_or_Top5': 'No_Data',
        'Median_or_Bot5': 'No_Data',
        'P75_or_Bot4': 'No_Data',
        'P95_or_Bot3': 'No_Data',
        'P99_or_Bot2': 'No_Data',
        'Max_or_Bot1': 'No_Data'
    }
    
    if col_type == "Numeric":
        base_result.update({
            'Mean_or_Top1': None,
            'Min_or_Top2': None,
            'P1_or_Top3': None,
            'P5_or_Top4': None,
            'P25_or_Top5': None,
            'Median_or_Bot5': None,
            'P75_or_Bot4': None,
            'P95_or_Bot3': None,
            'P99_or_Bot2': None,
            'Max_or_Bot1': None
        })
    
    return base_result

def batch_edd(table_list: List[str], output_path: str = "/dbfs/tmp/edd", 
              sample_threshold: int = 50_000_000) -> Dict[str, str]:
    """Process multiple tables and generate batch summary"""
    
    print(f"Starting batch EDD for {len(table_list)} tables")
    batch_start = time.time()
    
    results = {}
    summary_data = []
    
    for i, table_name in enumerate(table_list, 1):
        print(f"\n[{i}/{len(table_list)}] {table_name}")
        
        try:
            start_time = time.time()
            filepath = generate_edd(table_name, output_path, sample_threshold=sample_threshold)
            elapsed = time.time() - start_time
            
            results[table_name] = filepath
            summary_data.append({
                'Table_Name': table_name,
                'Status': 'Success',
                'File_Path': filepath,
                'Processing_Time_Minutes': round(elapsed / 60, 2)
            })
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            results[table_name] = error_msg
            print(error_msg)
            
            summary_data.append({
                'Table_Name': table_name,
                'Status': 'Failed',
                'File_Path': '',
                'Processing_Time_Minutes': 0,
                'Error': str(e)
            })
    
    # Generate batch summary
    summary_file = os.path.join(output_path, f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    
    successful = sum(1 for r in results.values() if r.endswith('.csv'))
    total_time = (time.time() - batch_start) / 60
    
    print(f"\nBatch completed: {successful}/{len(table_list)} successful in {total_time:.1f} minutes")
    print(f"Summary saved: {summary_file}")
    
    return results

# Helper functions for file management
def list_edd_files(output_path: str = "/dbfs/tmp/edd") -> None:
    """List all EDD files in the output directory"""
    
    try:
        if os.path.exists(output_path):
            files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
            
            if files:
                print(f"EDD files in {output_path}:")
                for i, filename in enumerate(files, 1):
                    filepath = os.path.join(output_path, filename)
                    size_mb = os.path.getsize(filepath) / (1024*1024)
                    print(f"{i:2d}. {filename} ({size_mb:.2f} MB)")
                
                print(f"\nTo download: Use Data → DBFS → tmp → edd in Databricks UI")
            else:
                print("No EDD files found")
        else:
            print(f"Directory {output_path} does not exist")
            
    except Exception as e:
        print(f"Error listing files: {e}")

def display_edd_file(filepath: str, num_rows: int = 20) -> None:
    """Display EDD file contents in notebook"""
    
    try:
        df = pd.read_csv(filepath)
        
        print(f"EDD File: {os.path.basename(filepath)}")
        print(f"Columns analyzed: {len(df)}")
        print(f"File size: {os.path.getsize(filepath) / 1024:.1f} KB")
        print("-" * 80)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        display(df.head(num_rows))
        
        if len(df) > num_rows:
            print(f"\n... showing first {num_rows} of {len(df)} total columns")
            
    except Exception as e:
        print(f"Error displaying file: {e}")

# Usage examples
if __name__ == "__main__":
    print("EDD system ready. Key functions:")
    print("- generate_edd(table_name)")  
    print("- batch_edd(table_list)")
    print("- list_edd_files() - to see generated files")
    print("- display_edd_file(filepath) - to view EDD in notebook")
