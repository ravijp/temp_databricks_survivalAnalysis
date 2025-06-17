# T-Learner, S-Learner, X-Learner: Complete Guide
## Meta-Learning Approaches for Causal Inference

---

## **1. Introduction to Meta-Learners**

### **What are Meta-Learners?**

Meta-learners are **machine learning approaches to causal inference** that use predictive models to estimate treatment effects. They're called "meta" because they use machine learning models as building blocks for causal estimation.

### **Common Framework**
All meta-learners aim to estimate:
- **ATE (Average Treatment Effect)**: Population-level treatment impact
- **CATE (Conditional Average Treatment Effect)**: Treatment effect for subgroups
- **ITE (Individual Treatment Effect)**: Treatment effect for specific individuals

### **Fundamental Equation**
```
ITE(x) = E[Y(1)|X=x] - E[Y(0)|X=x]

Where:
- Y(1) = potential outcome under treatment
- Y(0) = potential outcome under control  
- X = individual characteristics
- ITE(x) = treatment effect for someone with characteristics x
```

---

## **2. T-Learner (Two-Model Learner)**

### **Core Concept**
T-learner fits **separate models** for treated and control groups, then estimates treatment effects by comparing their predictions.

### **Detailed Algorithm**

#### **Step 1: Data Splitting**
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Split data by treatment assignment
treated_data = data[data['treatment'] == 1]
control_data = data[data['treatment'] == 0]

print(f"Treated group size: {len(treated_data)}")
print(f"Control group size: {len(control_data)}")
```

#### **Step 2: Model Training**
```python
# Define features (exclude treatment and outcome)
features = ['age', 'education', 'income', 'experience', 'location']

# Fit separate models
model_treated = RandomForestRegressor(n_estimators=100, random_state=42)
model_control = RandomForestRegressor(n_estimators=100, random_state=42)

# Train on respective subsets
model_treated.fit(treated_data[features], treated_data['outcome'])
model_control.fit(control_data[features], control_data['outcome'])
```

#### **Step 3: Counterfactual Prediction**
```python
# Predict outcomes for ALL individuals under both scenarios
y1_pred = model_treated.predict(data[features])  # Everyone as if treated
y0_pred = model_control.predict(data[features])  # Everyone as if control

# Calculate individual treatment effects
ite = y1_pred - y0_pred

# Calculate average treatment effect
ate = ite.mean()
print(f"Average Treatment Effect: {ate:.3f}")
```

#### **Step 4: Effect Analysis**
```python
# Analyze heterogeneous effects
high_effect_indices = ite > ite.quantile(0.8)
low_effect_indices = ite < ite.quantile(0.2)

print("High-effect individuals characteristics:")
print(data[high_effect_indices][features].describe())

print("Low-effect individuals characteristics:")  
print(data[low_effect_indices][features].describe())
```

### **T-Learner Advantages**

#### **1. Model Flexibility**
- Each model can capture different relationships in treated vs. control groups
- No assumption that treatment and control follow same patterns
- Can use different algorithms for each group if needed

#### **2. Treatment Effect Heterogeneity**
```python
# Example: Treatment effects vary by education level
education_effects = data.groupby('education_level')['ite'].mean()
print("Treatment effects by education:")
for edu_level, effect in education_effects.items():
    print(f"  {edu_level}: {effect:.3f}")
```

#### **3. No Feature Competition**
- Treatment doesn't compete with other features for model attention
- Each model focuses on predicting outcomes within its group
- Clearer interpretation of non-treatment feature effects

### **T-Learner Limitations**

#### **1. Data Efficiency Problem**
```python
# Demonstration of sample size reduction
total_n = len(data)
treated_n = len(treated_data)
control_n = len(control_data)

print(f"Total data: {total_n}")
print(f"Data for treated model: {treated_n} ({treated_n/total_n:.1%})")
print(f"Data for control model: {control_n} ({control_n/total_n:.1%})")
print("Each model uses only a subset of available data!")
```

#### **2. Extrapolation Risk**
```python
# Control model predicts for treated individuals (extrapolation)
# Treated model predicts for control individuals (extrapolation)
# This can be problematic if groups have different feature distributions

# Check overlap in feature distributions
import matplotlib.pyplot as plt

for feature in features:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(treated_data[feature], alpha=0.7, label='Treated')
    plt.hist(control_data[feature], alpha=0.7, label='Control')
    plt.title(f'{feature} Distribution')
    plt.legend()
    plt.show()
```

#### **3. Imbalanced Treatment Problems**
```python
# Example: 90% control, 10% treated
if len(treated_data) / len(data) < 0.2:
    print("WARNING: Treated group is very small!")
    print("T-learner may perform poorly due to limited treated data")
    print("Consider X-learner or other approaches")
```

### **Real-World T-Learner Applications**

#### **1. Marketing Campaign Effectiveness**
```python
# Business Context: Email marketing campaign
# Treatment: Received promotional email (1) vs. no email (0)
# Outcome: Purchase amount ($)
# Question: How much does email increase average purchase?

# Implementation
email_data = pd.read_csv('email_campaign_data.csv')
# Features: customer_age, past_purchases, loyalty_tier, region
# Treatment: received_email
# Outcome: purchase_amount

treated_customers = email_data[email_data['received_email'] == 1]
control_customers = email_data[email_data['received_email'] == 0]

# Business Insight Example:
# "Customers aged 25-35 with high loyalty show $45 average lift from email"
# "Customers aged 55+ show minimal response ($3 average lift)"
```

#### **2. Medical Treatment Analysis**
```python
# Business Context: New drug efficacy study
# Treatment: New drug (1) vs. placebo (0)  
# Outcome: Recovery time (days)
# Question: How much does the drug reduce recovery time?

medical_data = pd.read_csv('drug_trial_data.csv')
# Features: age, gender, severity, comorbidities
# Treatment: received_drug
# Outcome: recovery_days

# Business Insight Example:
# "Drug reduces recovery time by average 3.2 days"
# "Strongest effect in patients under 50 (4.1 days reduction)"
# "Minimal effect in severe cases (1.1 days reduction)"
```

#### **3. Educational Intervention**
```python
# Business Context: Online tutoring program
# Treatment: Received tutoring (1) vs. standard curriculum (0)
# Outcome: Test score improvement
# Question: How much does tutoring improve test scores?

education_data = pd.read_csv('tutoring_study.csv')
# Features: baseline_score, socioeconomic_status, school_type
# Treatment: received_tutoring  
# Outcome: score_improvement

# Business Insight Example:
# "Tutoring improves scores by average 12.5 points"
# "Largest effect for middle-performing students (18.3 points)"
# "Diminishing returns for high-performing students (6.2 points)"
```

---

## **3. S-Learner (Single-Model Learner)**

### **Core Concept**
S-learner fits **one model** that includes treatment as a feature, then estimates effects by comparing predictions with treatment on vs. off.

### **Detailed Algorithm**

#### **Step 1: Feature Engineering**
```python
# Include treatment as a feature
features_with_treatment = features + ['treatment']
print(f"Features for S-learner: {features_with_treatment}")

# No data splitting - use all data
print(f"Training data size: {len(data)} (full dataset)")
```

#### **Step 2: Single Model Training**
```python
from sklearn.ensemble import GradientBoostingRegressor

# Fit one model with treatment as feature
s_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
s_model.fit(data[features_with_treatment], data['outcome'])

# Check feature importance
feature_importance = pd.DataFrame({
    'feature': features_with_treatment,
    'importance': s_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature importance in S-learner:")
print(feature_importance)
```

#### **Step 3: Counterfactual Generation**
```python
# Create counterfactual datasets
data_treated = data.copy()
data_control = data.copy()

# Set treatment to 1 for everyone
data_treated['treatment'] = 1

# Set treatment to 0 for everyone  
data_control['treatment'] = 0

print("Created counterfactual datasets for effect estimation")
```

#### **Step 4: Treatment Effect Estimation**
```python
# Predict under both scenarios
y1_pred = s_model.predict(data_treated[features_with_treatment])
y0_pred = s_model.predict(data_control[features_with_treatment])

# Calculate effects
ite = y1_pred - y0_pred
ate = ite.mean()

print(f"S-learner Average Treatment Effect: {ate:.3f}")
print(f"Treatment effect standard deviation: {ite.std():.3f}")
```

### **S-Learner Advantages**

#### **1. Full Data Utilization**
```python
# Compare data usage
print("Data usage comparison:")
print(f"T-learner treated model: {len(treated_data)} samples")
print(f"T-learner control model: {len(control_data)} samples")
print(f"S-learner model: {len(data)} samples")
print(f"S-learner uses {len(data)/(len(treated_data)+len(control_data))*100:.1f}% more data")
```

#### **2. Shared Learning**
```python
# S-learner can learn patterns that apply to both groups
# Example: Age affects outcome similarly for treated and control groups
# This shared knowledge improves predictions for both scenarios
```

#### **3. Simplicity**
```python
# Single model vs. multiple models
# Easier hyperparameter tuning
# Simpler deployment
# Fewer models to maintain in production
```

#### **4. Stability with Small Samples**
```python
# When data is limited, S-learner often outperforms T-learner
# Especially helpful when treatment group is very small
if len(treated_data) < 100:
    print("Small treated group detected - S-learner may be preferred")
```

### **S-Learner Limitations**

#### **1. Treatment Dilution Problem**
```python
# Check if treatment feature is being ignored
treatment_importance = feature_importance[
    feature_importance['feature'] == 'treatment'
]['importance'].iloc[0]

if treatment_importance < 0.05:
    print("WARNING: Treatment feature has low importance!")
    print("Model may be ignoring treatment effect")
    print("Consider T-learner or feature engineering")
```

#### **2. Homogeneous Effect Assumption**
```python
# S-learner assumes treatment effect is relatively constant
# May miss important heterogeneity
ite_variance = ite.var()
if ite_variance > ate**2:  # High relative variance
    print("High treatment effect heterogeneity detected")
    print("S-learner may be missing important variation")
    print("Consider T-learner for better heterogeneity capture")
```

#### **3. Feature Competition**
```python
# Treatment competes with other features for model attention
# May be problematic when treatment effect is subtle
other_features_importance = feature_importance[
    feature_importance['feature'] != 'treatment'
]['importance'].sum()

competition_ratio = treatment_importance / other_features_importance
if competition_ratio < 0.1:
    print("Treatment feature being overshadowed by other features")
```

### **Real-World S-Learner Applications**

#### **1. Small-Scale A/B Tests**
```python
# Business Context: Website button color test
# Treatment: Blue button (1) vs. red button (0)
# Outcome: Click-through rate
# Sample size: 500 users (limited data)

button_data = pd.read_csv('button_test_data.csv')
# Small sample size makes S-learner attractive
# Simple treatment effect expected (homogeneous users)

# Business Insight Example:
# "Blue button increases CTR by 2.3 percentage points"
# "Effect is consistent across user segments"
```

#### **2. Homogeneous Population Study**
```python
# Business Context: Employee training program in single department
# Treatment: Received training (1) vs. no training (0)
# Outcome: Performance score
# Population: Similar roles, experience levels

training_data = pd.read_csv('dept_training_data.csv')
# Homogeneous population suggests consistent treatment effect
# S-learner works well when heterogeneity is limited

# Business Insight Example:
# "Training improves performance scores by 8.5 points on average"
# "Effect is consistent across the department"
```

#### **3. Resource-Constrained Projects**
```python
# Business Context: Quick analysis needed
# Treatment: New pricing strategy
# Outcome: Customer satisfaction
# Constraint: Limited time/resources for complex modeling

pricing_data = pd.read_csv('pricing_test_data.csv')
# S-learner preferred for simplicity and speed
# Single model easier to implement and explain

# Business Insight Example:  
# "New pricing increases satisfaction by 0.7 points"
# "Simple implementation, quick business decision"
```

---

## **4. X-Learner (Cross-Learning Approach)**

### **Core Concept**
X-learner combines T-learner flexibility with improved efficiency by using cross-fitting and propensity score weighting.

### **Detailed Algorithm**

#### **Step 1: Initial T-Learner Approach**
```python
# Same as T-learner: fit separate models
model_treated = RandomForestRegressor(random_state=42)
model_control = RandomForestRegressor(random_state=42)

model_treated.fit(treated_data[features], treated_data['outcome'])
model_control.fit(control_data[features], control_data['outcome'])
```

#### **Step 2: Impute Individual Treatment Effects**
```python
# For treated individuals: actual outcome - predicted control outcome
treated_control_pred = model_control.predict(treated_data[features])
tau_treated = treated_data['outcome'].values - treated_control_pred

# For control individuals: predicted treated outcome - actual outcome
control_treated_pred = model_treated.predict(control_data[features])  
tau_control = control_treated_pred - control_data['outcome'].values

print(f"Imputed {len(tau_treated)} treatment effects for treated group")
print(f"Imputed {len(tau_control)} treatment effects for control group")
```

#### **Step 3: Model the Treatment Effects**
```python
# Fit models to predict treatment effects
tau_model_treated = RandomForestRegressor(random_state=42)
tau_model_control = RandomForestRegressor(random_state=42)

tau_model_treated.fit(treated_data[features], tau_treated)
tau_model_control.fit(control_data[features], tau_control)
```

#### **Step 4: Propensity Score Estimation**
```python
from sklearn.linear_model import LogisticRegression

# Estimate propensity scores (probability of treatment)
propensity_model = LogisticRegression(random_state=42)
propensity_model.fit(data[features], data['treatment'])

propensity_scores = propensity_model.predict_proba(data[features])[:, 1]
print(f"Propensity scores range: {propensity_scores.min():.3f} to {propensity_scores.max():.3f}")
```

#### **Step 5: Weighted Treatment Effect Prediction**
```python
# Predict treatment effects from both models
tau_t_pred = tau_model_treated.predict(data[features])
tau_c_pred = tau_model_control.predict(data[features])

# Weight predictions by propensity scores
# High propensity (likely to be treated) → use control model's effect estimate
# Low propensity (likely to be control) → use treated model's effect estimate
ite = propensity_scores * tau_c_pred + (1 - propensity_scores) * tau_t_pred

ate = ite.mean()
print(f"X-learner Average Treatment Effect: {ate:.3f}")
```

### **X-Learner Advantages**

#### **1. Handles Imbalanced Treatments**
```python
# Example: 20% treated, 80% control
treatment_prop = data['treatment'].mean()
print(f"Treatment proportion: {treatment_prop:.2%}")

if treatment_prop < 0.3 or treatment_prop > 0.7:
    print("Imbalanced treatment detected - X-learner advantage!")
    print("X-learner uses cross-fitting to leverage both groups effectively")
```

#### **2. Leverages Cross-Information**
```python
# X-learner uses:
# - Treated group to estimate effects for control group
# - Control group to estimate effects for treated group
# This cross-pollination often improves estimates
print("Cross-information utilization:")
print(f"Control model helps estimate effects for {len(treated_data)} treated individuals")
print(f"Treated model helps estimate effects for {len(control_data)} control individuals")
```

#### **3. Propensity Score Adjustment**
```python
# Propensity weighting helps with confounding
# Individuals with high propensity to be treated but weren't treated are informative
# Individuals with low propensity but were treated are informative

high_prop_control = (propensity_scores > 0.7) & (data['treatment'] == 0)
low_prop_treated = (propensity_scores < 0.3) & (data['treatment'] == 1)

print(f"High-propensity control individuals: {high_prop_control.sum()}")
print(f"Low-propensity treated individuals: {low_prop_treated.sum()}")
print("These individuals provide valuable counterfactual information")
```

#### **4. Often Superior Performance**
```python
# Empirical studies show X-learner often outperforms T and S learners
# Especially in realistic scenarios with:
# - Imbalanced treatments
# - Selection bias
# - Moderate sample sizes
```

### **X-Learner Limitations**

#### **1. Complexity**
```python
# X-learner requires:
# - 2 outcome models (like T-learner)
# - 2 treatment effect models  
# - 1 propensity score model
# Total: 5 models vs. 1 (S-learner) or 2 (T-learner)

print("Model complexity comparison:")
print("S-learner: 1 model")
print("T-learner: 2 models") 
print("X-learner: 5 models")
```

#### **2. Propensity Score Dependence**
```python
# X-learner performance depends on propensity score quality
# Poor propensity scores can hurt performance
from sklearn.metrics import roc_auc_score

# Evaluate propensity model
prop_auc = roc_auc_score(data['treatment'], propensity_scores)
print(f"Propensity model AUC: {prop_auc:.3f}")

if prop_auc < 0.6:
    print("WARNING: Poor propensity score model!")
    print("X-learner may not perform well")
```

#### **3. Hyperparameter Tuning Complexity**
```python
# Need to tune hyperparameters for multiple models
# Grid search becomes computationally expensive
# Model selection more complex

hyperparams_to_tune = {
    'outcome_models': ['n_estimators', 'max_depth', 'min_samples_split'],
    'effect_models': ['n_estimators', 'max_depth', 'min_samples_split'],  
    'propensity_model': ['C', 'penalty', 'solver']
}
print("Hyperparameter tuning complexity increases significantly")
```

### **Real-World X-Learner Applications**

#### **1. Observational Healthcare Studies**
```python
# Business Context: Surgery effectiveness study
# Treatment: New surgical technique vs. standard technique
# Outcome: Recovery time
# Challenge: Treatment assignment not random (surgeon choice)

surgery_data = pd.read_csv('surgery_outcomes.csv')
# Features: patient_age, severity, comorbidities, surgeon_experience
# Non-random treatment assignment creates selection bias
# X-learner's propensity weighting helps address this

# Business Insight Example:
# "New technique reduces recovery time by 2.1 days on average"
# "Effect strongest for moderate severity cases (3.2 days reduction)"
# "Propensity adjustment crucial due to surgeon selection patterns"
```

#### **2. Digital Marketing with Imbalanced Treatment**
```python
# Business Context: Premium feature adoption
# Treatment: Offered premium upgrade (20% of users)
# Outcome: Monthly engagement score
# Challenge: Selective targeting creates imbalance

engagement_data = pd.read_csv('premium_feature_data.csv')
# 80% control, 20% treatment
# Selection bias: premium offered to high-value users
# X-learner handles imbalance and selection bias well

# Business Insight Example:
# "Premium feature increases engagement by 23% on average"
# "Cross-fitting reveals effect varies by user type"
# "Propensity adjustment prevents overestimation of effect"
```

#### **3. High-Stakes Business Decisions**
```python
# Business Context: Supply chain optimization
# Treatment: New inventory management system
# Outcome: Cost reduction
# Stakes: Multi-million dollar investment decision

supply_chain_data = pd.read_csv('inventory_system_test.csv')
# High accuracy requirement justifies X-learner complexity
# Multiple validation approaches needed
# Conservative estimate preferred for business case

# Business Insight Example:
# "New system reduces costs by $2.3M annually"
# "95% confidence interval: $1.8M to $2.8M savings"
# "X-learner provides most robust estimate among meta-learners"
```

---

## **5. Comparison and Selection Guide**

### **Performance Comparison Matrix**

| Scenario | Best Choice | Reasoning |
|----------|-------------|-----------|
| **Balanced treatment (40-60%)** | T-Learner | Sufficient data for both models |
| **Imbalanced treatment (<30% or >70%)** | X-Learner | Handles imbalance well |
| **Small sample size (<500)** | S-Learner | Makes best use of limited data |
| **Homogeneous population** | S-Learner | Treatment effect likely consistent |
| **Heterogeneous population** | T-Learner | Captures group differences |
| **High-stakes decision** | X-Learner | Often most accurate |
| **Quick analysis needed** | S-Learner | Simplest implementation |
| **Selection bias present** | X-Learner | Propensity adjustment helps |

### **Implementation Complexity**

```python
# Code complexity comparison
def s_learner_complexity():
    return "Low: 1 model, simple implementation"

def t_learner_complexity():  
    return "Medium: 2 models, data splitting required"

def x_learner_complexity():
    return "High: 5 models, propensity scores, cross-fitting"
```

### **Data Requirements**

```python
def data_requirements():
    return {
        'S-Learner': {
            'minimum_samples': 200,
            'treatment_balance': 'Any',
            'feature_quality': 'Medium'
        },
        'T-Learner': {
            'minimum_samples': 500,
            'treatment_balance': 'Balanced preferred',
            'feature_quality': 'Medium'
        },
        'X-Learner': {
            'minimum_samples': 800,
            'treatment_balance': 'Handles imbalance well',  
            'feature_quality': 'High (for propensity scores)'
        }
    }
```

---

## **6. Why T/S/X Learners Don't Apply to ZENON Project**

### **Fundamental Incompatibility Issues**

#### **1. Outcome Type Mismatch**
```python
# T/S/X learners expect:
outcome_format = "Binary (0/1) or continuous (real numbers)"

# ZENON project has:
survival_outcome = {
    'type': 'time-to-event',
    'components': ['time_to_termination', 'event_indicator'],
    'censoring': 'right_censored',
    'competing_risks': ['voluntary', 'involuntary', 'retirement']
}

print("INCOMPATIBLE: Meta-learners can't handle survival outcomes")
```

#### **2. Time Dimension Missing**
```python
# T/S/X learners answer:
meta_learner_question = "What's the treatment effect?" (single number)

# ZENON needs to answer:
ZENON_questions = [
    "What's the treatment effect at 6 months?",
    "What's the treatment effect at 12 months?", 
    "What's the treatment effect at 18 months?",
    "When does the treatment effect peak?",
    "How does the effect change over time?"
]

print("INCOMPATIBLE: Meta-learners ignore time dimension")
```

#### **3. Censoring Not Handled**
```python
# ZENON reality:
total_employees = 1000000
still_employed = 650000  # Right-censored observations
terminated = 350000      # Observed events

# T/S/X learners would:
# Option 1: Drop censored employees (lose 65% of data!)
# Option 2: Treat censored as "no event" (biased!)
# Option 3: Use time-to-censoring (wrong outcome!)

print("INCOMPATIBLE: Meta-learners waste censored data")
```

#### **4. No Competing Risks**
```python
# ZENON termination types:
termination_types = {
    'voluntary': 0.45,      # Target outcome
    'involuntary': 0.30,    # Competing risk
    'retirement': 0.15,     # Competing risk  
    'transfer': 0.10        # Competing risk
}

# T/S/X learners see:
binary_outcome = "Left (1) or Stayed (0)"
# Loses crucial information about WHY someone left

print("INCOMPATIBLE: Meta-learners ignore competing risks")
```

### **Business Logic Incompatibility**

#### **1. Wrong Business Question**
```python
# Meta-learner output for salary increase:
meta_output = "Treatment effect = 0.15"
# HR Manager asks: "What does 0.15 mean? Should I give John a raise?"

# ZENON G-computation output:
ZENON_output = {
    'employee': 'John Smith',
    'current_12m_retention': 0.65,
    'with_salary_increase': 0.78,
    'improvement': 0.13,
    'cost': 8500,
    'replacement_cost': 45000,
    'roi': 5.3
}
# HR Manager understands: "Yes, give John a raise - good ROI"

print("Meta-learners don't provide actionable business insights")
```

#### **2. Missing Time Horizons**
```python
# ZENON business need:
retention_targets = {
    '6_month': 0.85,   # Onboarding success
    '12_month': 0.75,  # Annual retention target
    '24_month': 0.60   # Long-term retention
}

# Meta-learners can't provide time-specific predictions
print("Meta-learners can't support time-based retention planning")
```

#### **3. No Survival Probabilities**
```python
# HR needs risk categories based on survival probabilities:
risk_categories = {
    'high_risk': 'P(12-month turnover) > 0.6',
    'medium_risk': '0.3 < P(12-month turnover) <= 0.6', 
    'low_risk': 'P(12-month turnover) <= 0.3'
}

# Meta-learners don't produce survival probabilities
print("Meta-learners incompatible with risk-based HR workflows")
```

### **Technical Implementation Issues**

#### **1. Data Format Incompatibility**
```python
# Meta-learner expected format:
meta_format = pd.DataFrame({
    'employee_id': [1, 2, 3],
    'age': [25, 35, 45],
    'salary': [60000, 75000, 90000],
    'treatment': [0, 1, 0],
    'outcome': [0, 1, 0]  # Binary: stayed/left
})

# ZENON survival format:
ZENON_format = pd.DataFrame({
    'employee_id': [1, 1, 1, 2, 2],
    'start_time': [0, 6, 12, 0, 6],
    'stop_time': [6, 12, 18, 6, 10],
    'age': [25, 25, 26, 35, 35],
    'salary': [60000, 63000, 63000, 75000, 80000],
    'treatment': [0, 1, 1, 0, 1],
    'event': [0, 0, 1, 0, 1]  # Time-to-event
})

print("Data formats fundamentally incompatible")
```

#### **2. Feature Engineering Differences**
```python
# Meta-learner features:
meta_features = ['age', 'education', 'salary', 'treatment']
# Static snapshot at one time point

# ZENON survival features:
survival_features = [
    'current_age',           # Changes over time
    'tenure_months',         # Increases over time  
    'salary_growth_rate',    # Time-varying
    'recent_promotions',     # Recent history matters
    'manager_changes',       # Sequence of changes
    'performance_trend'      # Trajectory over time
]

print("Feature engineering approaches incompatible")
```

#### **3. Model Validation Differences**
```python
# Meta-learner validation:
meta_metrics = ['AUC', 'accuracy', 'RMSE', 'R-squared']

# Survival model validation:
survival_metrics = [
    'C-index',              # Concordance for ranking
    'Brier_score',          # Time-specific prediction accuracy
    'Integrated_Brier',     # Overall prediction quality
    'Calibration_plots'     # Predicted vs observed at each time
]

print("Performance evaluation frameworks incompatible")
```

### **The Correct Path Forward**

Instead of T/S/X learners, ZENON project uses:

```python
# Correct approach: G-computation for survival analysis
survival_g_computation = {
    'step_1': 'Fit Cox/AFT models with time-varying covariates',
    'step_2': 'Create counterfactual scenarios for interventions',
    'step_3': 'Predict survival functions under each scenario',
    'step_4': 'Calculate treatment effects on survival probabilities',
    'step_5': 'Generate personalized intervention recommendations'
}

# This approach provides:
business_value = [
    'Time-specific retention probabilities',
    'Multiple intervention comparisons',
    'ROI calculations for each intervention',
    'Risk category assignments',
    'Actionable HR recommendations'
]

print("G-computation provides what ZENON actually needs")
```

---

## **7. Summary and Recommendations**

### **For Understanding Meta-Learners:**
- **T-Learner**: Good for balanced treatments, heterogeneous effects
- **S-Learner**: Good for small samples, homogeneous effects  
- **X-Learner**: Good for imbalanced treatments, highest accuracy

### **For ZENON Project:**
- **Abandon meta-learner approach** - fundamentally incompatible
- **Use G-computation with survival models** - designed for time-to-event data
- **Focus on Cox PH and XGBoost AFT** - correct modeling framework
- **Generate survival probabilities** - what HR practitioners need

### **Key Takeaway:**
Meta-learners are powerful tools for standard causal inference, but they're the wrong tool for survival analysis problems. The ZENON project requires specialized survival analysis methods that can handle time-to-event outcomes, censoring, and time-varying covariates.

The confusion arose from mixing different causal inference paradigms. Understanding both approaches helps clarify why the chosen G-computation framework is correct for ZENON's business needs.