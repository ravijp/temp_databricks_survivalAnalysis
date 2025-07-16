def create_dataset_splits(
    self,
    active_employees_2023: DataFrame,
    active_employees_2024: DataFrame,
    train_ratio: float = 0.7,
    random_seed: int = 42,
) -> DataFrame:
    """
    Create temporal validation splits with proper person-level separation
    Ensures no person appears in both train and val (but allows train/val vs oot overlap)
    """
    
    # Use Spark's randomSplit - it handles the randomization correctly without caching!
    # randomSplit returns a list of DataFrames with the specified proportions
    train_val_splits = active_employees_2023.select(self.person_id_combined).randomSplit(
        [train_ratio, 1.0 - train_ratio], 
        seed=random_seed
    )
    
    train_persons = train_val_splits[0]
    val_persons = train_val_splits[1]
    
    # OOT population from 2024 (overlap with train/val is intentional)
    oot_persons = active_employees_2024.select(self.person_id_combined).distinct()
    
    # Create split assignments
    split_assignments = (
        train_persons.withColumn("dataset_split", lit("train"))
        .union(val_persons.withColumn("dataset_split", lit("val")))
        .union(oot_persons.withColumn("dataset_split", lit("oot")))
    )
    
    # Verify no person leakage between train and val
    train_val_overlap = train_persons.join(val_persons, self.person_id_combined, "inner").count()
    if train_val_overlap > 0:
        logger.error(f"ERROR: Found {train_val_overlap} persons in both train and val splits!")
        raise ValueError(f"Train/val overlap detected: {train_val_overlap} persons")
    
    logger.info("Dataset split summary:")
    split_summary = split_assignments.groupBy("dataset_split").count().collect()
    for row in split_summary:
        logger.info(f"  {row['dataset_split']}: {row['count']:,} persons")
        
    return split_assignments
