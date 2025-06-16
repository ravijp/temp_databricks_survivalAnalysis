# Real Survival Analysis Datasets Available Online for Employee Turnover Model Testing

## **Tier 1: Employee/HR-Specific Datasets (Highest Relevance)**

### **1. IBM HR Analytics Employee Attrition Dataset** ⭐⭐⭐⭐⭐
- **Source:** https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- **Records:** 1,470 employees with 35 columns
- **Event:** Employee attrition (voluntary termination)
- **Covariates:** Age, gender, job satisfaction, environment satisfaction, education field, job role, income, overtime, percentage salary hike, tenure, training times, years in current role, relationship status
- **Quality:** Clean, well-documented, business-relevant features
- **Format:** Cross-sectional (can be adapted for time-varying analysis)
- **ADP Relevance:** Perfect for initial model validation and business context understanding

### **2. Edward Babushkin's Employee Dataset**
- **Source:** Edward's Dropbox (mirror available)
- **Records:** Real call-center data from Russia
- **Features:** Personality traits (independence, self-control, anxiety, openness), demographic data
- **Context:** Call-center environment with high turnover rates
- **Use Case:** Survival analysis examples included
- **ADP Relevance:** ⭐⭐⭐⭐ (Real-world employee data with survival analysis applications)

### **3. Employee Turnover Dataset (Kaggle)**
- **Source:** https://www.kaggle.com/datasets/davinwijaya/employee-turnover
- **Description:** Originally used for Survival Analysis Model
- **Format:** Designed specifically for survival analysis
- **ADP Relevance:** ⭐⭐⭐⭐⭐ (Direct survival analysis application)

---

## **Tier 2: Business Churn Datasets (Adaptable for Employee Context)**

### **4. Telco Customer Churn Dataset** ⭐⭐⭐⭐
- **Source:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Records:** 7,043 customers with comprehensive service information
- **Features:** Gender, dependents, monthly charges, contract types, services subscribed
- **Survival Application:** Multiple studies have used this dataset for survival analysis with competing risks
- **Time Component:** Monthly charges and tenure data for time-varying analysis
- **Business Context:** Widely used for churn prediction with survival analysis methods

### **5. IBM Telco Customer Churn (Enhanced)**
- **Source:** IBM Cognos Analytics samples
- **Features:** Demographics, membership details, usage patterns, satisfaction scores, churn scores, Customer Lifetime Value (CLTV) index
- **Records:** Over 7,000 customers in California
- **Enhancement:** Includes satisfaction and CLTV metrics
- **Survival Relevance:** Recent research (2024) used this dataset specifically for survival analysis with competing risks

---

## **Tier 3: Built-in Survival Analysis Datasets (Model Validation)**

### **6. Lifelines Package Datasets** ⭐⭐⭐
**Source:** Python lifelines library (https://github.com/CamDavidsonPilon/lifelines)

#### **6a. Rossi Recidivism Dataset**
- **Records:** 432 convicts followed for one year after release
- **Features:** 9 variables including age, race, work experience, marital status, parole status, prior convictions
- **Time-Varying:** Weekly employment status, financial aid
- **Format:** Start-stop format available
- **Use Case:** Perfect for time-varying covariate practice

#### **6b. German Breast Cancer Study (GBSG2)**
- **Records:** 686 women with 8 clinical variables
- **Quality:** Clean, well-studied, established benchmarks
- **Use Case:** Validate Cox PH and AFT model implementations

#### **6c. Stanford Heart Transplant Dataset**
- **Records:** 172 patients with start-stop format
- **Features:** Age, year, surgery status, transplant status over time
- **Format:** Classic time-varying covariates example
- **Use Case:** Time-varying survival analysis validation

### **7. Additional Lifelines Datasets**
- **Canadian Senators:** Political tenure analysis
- **Load C. Botulinum:** Left and right censored data
- **Multicenter AIDS Cohort:** Medical survival analysis
- **Load Lymph Node:** Cancer recurrence study

---

## **Tier 4: Large-Scale Genomic Datasets (Advanced Testing)**

### **8. The Cancer Genome Atlas (TCGA)** ⭐⭐⭐⭐⭐
- **Source:** NCI Genomic Data Commons Data Portal
- **Scale:** Over 11,000 cancer patients across 33 cancer types
- **Data Types:** 2.5 petabytes of genomic, epigenomic, transcriptomic, and proteomic data
- **Clinical Data:** TCGA Clinical Data Resource (TCGA-CDR) with standardized survival endpoints
- **Validation:** Five-year survival frequencies strongly correlated with NCI-SEER data (R = 0.83)
- **Use Case:** Large-scale survival analysis with thousands of features

### **9. METABRIC Breast Cancer Dataset**
- **Source:** cBioPortal (https://www.cbioportal.org/)
- **Records:** 199+ TNBC cases with survival outcomes
- **Features:** Clinical and genomic data
- **Survival Application:** Extensively used for survival analysis research
- **Use Case:** Cross-validation with TCGA data

---

## **Implementation Framework for PySpark**

### **Dataset Access Code Templates**

```python
# Built-in Lifelines Datasets
from lifelines.datasets import (
    load_rossi, load_gbsg2, load_stanford_heart_transplants,
    load_waltons, load_canadian_senators
)

# Load with automatic format
rossi_df = load_rossi()
gbsg2_df = load_gbsg2()
stanford_df = load_stanford_heart_transplants()
```

```python
# Kaggle Datasets (requires kaggle API)
import kaggle

# Download IBM HR dataset
kaggle.api.dataset_download_files(
    'pavansubhasht/ibm-hr-analytics-attrition-dataset',
    path='./data/', unzip=True
)

# Download Telco churn dataset  
kaggle.api.dataset_download_files(
    'blastchar/telco-customer-churn',
    path='./data/', unzip=True
)
```

```python
# PySpark DataFrame Creation
from pyspark.sql import SparkSession
from pyspark.sql.types import *

spark = SparkSession.builder.appName("SurvivalAnalysis").getOrCreate()

# Define schema for survival data
survival_schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("start_time", DoubleType(), True),
    StructField("stop_time", DoubleType(), True),
    StructField("event", IntegerType(), True),
    StructField("treatment", IntegerType(), True),
    # Add covariates as needed
])

# Load data into Spark DataFrame
df_spark = spark.read.csv("path/to/dataset.csv", header=True, schema=survival_schema)
```

---

## **Recommended Testing Sequence for ADP Project**

### **Phase 1: Foundation (Week 1 - Before ADP Access)**
1. **IBM HR Analytics Dataset** - Understand business context and basic survival concepts
2. **Rossi Dataset** - Master start-stop format and time-varying covariates
3. **GBSG2 Dataset** - Validate Cox PH implementation against known results

### **Phase 2: Scale Testing (Week 1-2)**
1. **Telco Churn Dataset** - Test larger scale (7K+ records) and competing risks
2. **Stanford Heart Transplant** - Complex time-varying scenarios
3. **Custom Employee Simulation** - ADP-specific format practice

### **Phase 3: Production Readiness (Week 2-3)**
1. **TCGA Subset** - Large-scale survival analysis (1K+ patients, 100+ features)
2. **Multi-dataset Validation** - Cross-validation across different domains
3. **NAICS Stratification Testing** - Industry-specific survival patterns

---

## **Quality Assessment Framework**

### **Dataset Evaluation Metrics**
```python
def evaluate_dataset_quality(df, event_col, time_col):
    """Comprehensive dataset assessment for survival analysis"""
    
    metrics = {
        'sample_size': len(df),
        'event_rate': df[event_col].mean(),
        'censoring_rate': 1 - df[event_col].mean(),
        'time_range_months': df[time_col].max() - df[time_col].min(),
        'median_follow_up': df[time_col].median(),
        'missing_data_rate': df.isnull().sum().sum() / (len(df) * len(df.columns)),
        'feature_count': len(df.columns) - 2  # Exclude time and event
    }
    
    # Quality thresholds for employee turnover context
    quality_flags = {
        'sufficient_sample': metrics['sample_size'] >= 1000,
        'reasonable_event_rate': 0.1 <= metrics['event_rate'] <= 0.8,
        'adequate_follow_up': metrics['median_follow_up'] >= 6,  # 6+ months
        'low_missing_data': metrics['missing_data_rate'] < 0.15,
        'adequate_features': metrics['feature_count'] >= 5
    }
    
    return metrics, quality_flags
```

### **Survival Analysis Validation Pipeline**
```python
def validate_survival_implementation(df, known_results=None):
    """Validate survival analysis implementations against known benchmarks"""
    from lifelines import KaplanMeierFitter, CoxPHFitter
    
    # Kaplan-Meier validation
    kmf = KaplanMeierFitter()
    kmf.fit(df['duration'], df['event'])
    
    # Cox PH validation
    cph = CoxPHFitter()
    cph.fit(df, duration_col='duration', event_col='event')
    
    validation_results = {
        'median_survival': kmf.median_survival_time_,
        'concordance_index': cph.concordance_index_,
        'log_likelihood': cph.log_likelihood_,
        'aic': cph.AIC_partial_
    }
    
    if known_results:
        # Compare against published results
        for metric, expected in known_results.items():
            observed = validation_results[metric]
            relative_error = abs(observed - expected) / expected
            validation_results[f'{metric}_error'] = relative_error
            validation_results[f'{metric}_valid'] = relative_error < 0.05
    
    return validation_results
```

This comprehensive list provides real, accessible datasets that progress from basic survival analysis concepts to production-scale employee turnover modeling, perfectly aligned with your ADP project requirements and PySpark implementation needs.
