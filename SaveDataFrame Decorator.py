import functools
import os
from datetime import datetime

def save_dataframe(
    save_path="/tmp/dataframe_snapshots",
    save_format="csv",
    when="after",
    include_timestamp=True,
    use_pandas_fallback=True,
    coalesce_partitions=1,
    max_records_to_save=10000,
    **save_options
):
    """
    Decorator to save PySpark DataFrames with record count limit.
    Supports both .transform(function) and .transform(function()) patterns.
    Fixed to handle Spark Connect DataFrames and decorator stacking.
    """
    def decorator(func):
        def _execute_with_save(func, df, *args, **kwargs):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
            func_name = func.__name__
            
            def _save_df(dataframe, suffix=""):
                # Check record count first
                try:
                    record_count = dataframe.count()
                    if record_count > max_records_to_save:
                        print(f"Skipping save for {func_name}{suffix} - {record_count} records exceeds limit of {max_records_to_save}")
                        return
                except Exception as e:
                    print(f"Warning: Could not count records for {func_name}: {e}")
                    print("Proceeding with save")
                
                # Build file path
                filename = f"{func_name}_{timestamp}{suffix}" if include_timestamp else f"{func_name}{suffix}"
                func_dir = os.path.join(save_path, func_name)
                file_path = os.path.join(func_dir, filename)
                
                # Create directory
                os.makedirs(func_dir, exist_ok=True)
                
                # Try Spark write first
                try:
                    df_to_save = dataframe.coalesce(coalesce_partitions) if coalesce_partitions else dataframe
                    
                    if save_format == "csv":
                        df_to_save.write.mode("overwrite").option("header", "true").csv(file_path, **save_options)
                    elif save_format == "parquet":
                        df_to_save.write.mode("overwrite").parquet(file_path, **save_options)
                    elif save_format == "json":
                        df_to_save.write.mode("overwrite").json(file_path, **save_options)
                    else:
                        df_to_save.write.mode("overwrite").format(save_format).save(file_path, **save_options)
                    
                    print(f"Saved {func_name}{suffix} to {file_path}")
                    
                except Exception as e:
                    if use_pandas_fallback:
                        try:
                            pandas_path = f"{file_path}.csv"
                            dataframe.toPandas().to_csv(pandas_path, index=False)
                            print(f"Saved {func_name}{suffix} to {pandas_path} (pandas fallback)")
                        except Exception as pandas_error:
                            print(f"Failed to save {func_name}{suffix}: Spark error: {e}, Pandas error: {pandas_error}")
                    else:
                        print(f"Failed to save {func_name}{suffix}: {e}")
            
            # Save before if requested
            if when in ["before", "both"]:
                _save_df(df, "_before")
            
            # Execute transformation
            result_df = func(df, *args, **kwargs)
            
            # Save after if requested
            if when in ["after", "both"]:
                suffix = "_after" if when == "both" else ""
                _save_df(result_df, suffix)
            
            return result_df
        
        def _is_dataframe(obj):
            """Check if object is any type of Spark DataFrame"""
            if obj is None:
                return False
            
            # Check for different DataFrame types
            obj_type = str(type(obj))
            dataframe_types = [
                'pyspark.sql.dataframe.DataFrame',
                'pyspark.sql.connect.dataframe.DataFrame',
                'DataFrame'  # catches any DataFrame subclass
            ]
            
            return any(df_type in obj_type for df_type in dataframe_types) or hasattr(obj, 'sql_ctx')
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Case 1: Called directly with DataFrame (transform(function))
            if args and _is_dataframe(args[0]):
                return _execute_with_save(func, args[0], *args[1:], **kwargs)
            
            # Case 2: Called with no args or non-DataFrame args (transform(function()))
            # Return a function that transform can call with DataFrame
            def transform_function(df):
                if not _is_dataframe(df):
                    raise TypeError(f"Expected DataFrame, got {type(df)}. "
                                  f"Make sure the function is being called with a DataFrame.")
                return _execute_with_save(func, df, *args, **kwargs)
            
            return transform_function
        
        return wrapper
    return decorator
