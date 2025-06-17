# Causal Inference for ADP Employee Turnover Project
## Complete Technical Guide for Survival Analysis Context

---

## **1. Introduction: Causal Inference vs. Prediction**

### **Why Causal Inference Matters for ADP**

**Prediction Problem**: "Who will leave in the next 12 months?" (Correlational)
**Causal Problem**: "What interventions will reduce John's turnover risk?" (Actionable)

ADP's business need is **causal**: HR practitioners want to know **what actions to take**, not just who's at risk. This requires causal inference, not just predictive modeling.

### **Fundamental Causal Question**
> "If we give John a 10% salary increase, how much will his 12-month retention probability improve?"

This question requires estimating **counterfactual outcomes** - what would happen under different intervention scenarios.

---

## **2. G-Computation: The Correct Approach for ADP**

### **What is G-Computation?**

G-computation is a causal inference method that estimates treatment effects by:
1. **Modeling the outcome** conditional on treatment and confounders
2. **Creating counterfactual scenarios** (everyone treated vs. everyone untreated)  
3. **Averaging over the confounder distribution** to get population effects
4. **Computing individual-level effects** for personalized recommendations

### **G-Computation Formula**
```
ATE = E[Y¹] - E[Y⁰] = Σₗ E[Y|A=1,L=l] × P(L=l) - Σₗ E[Y|A=0,L=l] × P(L=l)

Where:
- Y¹, Y⁰ = potential outcomes under treatment/control
- A = treatment assignment (1=treated, 0=control)
- L = confounders
- ATE = Average Treatment Effect
```

### **G-Computation for Survival Analysis**

**Standard G-Computation**: Binary/continuous outcomes at single time point
**Survival G-Computation**: Time-to-event outcomes with censoring and time-varying covariates

**Survival-Specific Formula**:
```
ATE(t) = E[S¹(t)] - E[S⁰(t)]

Where:
- S¹(t) = survival function under treatment at time t
- S⁰(t) = survival function under control at time t
- ATE(t) = treatment effect on survival probability at time t
```

### **Step-by-Step G-Computation for ADP**

#### **Step 1: Fit Survival Model with Confounders**
```python
from lifelines import CoxPHFitter
import pandas as pd

# Fit Cox model with treatment + confounders
cox_model = CoxPHFitter()
cox_model.fit(
    df=employee_data,
    duration_col='tenure_months',
    event_col='voluntary_termination',
    formula='salary_increase + age + tenure + performance_rating + ' +
            'job_level + industry + manager_quality + overtime_hours'
)
```

#### **Step 2: Create Counterfactual Datasets**
```python
# Scenario 1: Everyone gets 10% salary increase
df_treated = employee_data.copy()
df_treated['salary_increase'] = 1

# Scenario 2: No one gets salary increase  
df_control = employee_data.copy()
df_control['salary_increase'] = 0
```

#### **Step 3: Predict Survival Under Each Scenario**
```python
# Predict survival functions for each scenario
survival_treated = cox_model.predict_survival_function(df_treated)
survival_control = cox_model.predict_survival_function(df_control)
```

#### **Step 4: Calculate Treatment Effects**
```python
# Average Treatment Effect (ATE) at 12 months
ate_12m = survival_treated.loc[12].mean() - survival_control.loc[12].mean()
print(f"Population ATE: {ate_12m:.3f}")
# Output: "10% salary increase improves 12-month retention by 8.5 percentage points"

# Individual Treatment Effect (ITE) for each employee
ite_12m = survival_treated.loc[12] - survival_control.loc[12]
# Output: Employee-specific retention improvements
```

#### **Step 5: Business Translation**
```python
# Convert to business-friendly output
for employee_id in high_risk_employees:
    current_risk = 1 - survival_control.loc[12, employee_id]
    treated_risk = 1 - survival_treated.loc[12, employee_id]
    risk_reduction = current_risk - treated_risk
    
    print(f"Employee {employee_id}:")
    print(f"  Current 12-month turnover risk: {current_risk:.1%}")
    print(f"  Risk with salary increase: {treated_risk:.1%}")
    print(f"  Risk reduction: {risk_reduction:.1%}")
```

### **Multiple Intervention Analysis**
```python
interventions = {
    'salary_5pct': {'salary_increase_pct': 5},
    'salary_10pct': {'salary_increase_pct': 10},
    'promotion': {'promoted': 1},
    'overtime_reduction': {'overtime_hours': 'reduce_by_20pct'},
    'manager_change': {'good_manager': 1}
}

intervention_effects = {}
for intervention_name, intervention_values in interventions.items():
    # Create counterfactual dataset
    df_intervention = employee_data.copy()
    for var, value in intervention_values.items():
        df_intervention[var] = value
    
    # Predict and calculate effects
    survival_intervention = cox_model.predict_survival_function(df_intervention)
    ate_intervention = survival_intervention.loc[12].mean() - survival_control.loc[12].mean()
    intervention_effects[intervention_name] = ate_intervention

# Rank interventions by effectiveness
ranked_interventions = sorted(intervention_effects.items(), 
                            key=lambda x: x[1], reverse=True)
```

### **G-Computation Advantages for ADP**

1. **Handles Survival Data**: Works with time-to-event outcomes and censoring
2. **Multiple Time Horizons**: Can estimate effects at 6, 12, 18 months simultaneously
3. **Individual-Level Effects**: Provides personalized intervention recommendations
4. **Multiple Interventions**: Can compare different intervention strategies
5. **Business Interpretability**: Direct translation to retention probabilities and ROI

---

## **3. T-Learner, S-Learner, X-Learner: What They Are and Why They Don't Apply**

### **Overview of Meta-Learners**

T/S/X learners are **meta-learning approaches** for causal inference that use machine learning models to estimate treatment effects. They're designed for **standard causal inference problems** with binary or continuous outcomes at a single time point.

### **T-Learner (Two-Model Approach)**

#### **What is T-Learner?**
T-learner fits **separate models** for treated and control groups, then estimates treatment effects by taking the difference in predictions.

#### **Algorithm**
```python
# Step 1: Split data by treatment status
treated_data = data[data['treatment'] == 1]
control_data = data[data['treatment'] == 0]

# Step 2: Fit separate models
model_treated = RandomForestRegressor()
model_control = RandomForestRegressor()

model_treated.fit(treated_data[features], treated_data['outcome'])
model_control.fit(control_data[features], control_data['outcome'])

# Step 3: Predict for all individuals under both scenarios
y1_pred = model_treated.predict(data[features])  # Everyone as treated
y0_pred = model_control.predict(data[features])  # Everyone as control

# Step 4: Calculate treatment effects
ate = (y1_pred - y0_pred).mean()  # Average Treatment Effect
ite = y1_pred - y0_pred           # Individual Treatment Effects
```

#### **T-Learner Advantages**
- **Flexibility**: Each model can capture different relationships in treated/control groups
- **No interference**: Treatment variable doesn't compete with other features
- **Interpretability**: Clear separation of treated and control group patterns

#### **T-Learner Limitations**
- **Data splitting**: Reduces sample size for each model
- **Imbalanced treatments**: Poor performance when treatment groups are very unequal
- **Extrapolation risk**: Control model predicts for treated individuals (and vice versa)

#### **Real-World T-Learner Use Cases**
1. **Marketing Campaign Effectiveness**: Email vs. no email impact on purchases
2. **Medical Treatment Evaluation**: Drug vs. placebo effect on recovery time
3. **Educational Interventions**: Tutoring vs. standard curriculum on test scores
4. **Pricing Strategies**: Discount vs. regular price impact on sales

### **S-Learner (Single-Model Approach)**

#### **What is S-Learner?**
S-learner fits **one model** including treatment as a feature, then estimates effects by comparing predictions with treatment on/off.

#### **Algorithm**
```python
# Step 1: Fit single model with treatment as feature
model = XGBRegressor()
features_with_treatment = features + ['treatment']
model.fit(data[features_with_treatment], data['outcome'])

# Step 2: Predict under both treatment scenarios
data_treated = data.copy()
data_control = data.copy()
data_treated['treatment'] = 1
data_control['treatment'] = 0

y1_pred = model.predict(data_treated[features_with_treatment])
y0_pred = model.predict(data_control[features_with_treatment])

# Step 3: Calculate treatment effects
ate = (y1_pred - y0_pred).mean()
ite = y1_pred - y0_pred
```

#### **S-Learner Advantages**
- **Full data utilization**: Uses all data to fit single model
- **Shared patterns**: Leverages commonalities between treated/control groups
- **Simple implementation**: Just add treatment as a feature
- **Stable with small samples**: Better than T-learner when data is limited

#### **S-Learner Limitations**
- **Treatment dilution**: Model might ignore treatment if other features are stronger
- **Homogeneous effects**: Assumes treatment effect is similar across individuals
- **Feature competition**: Treatment competes with confounders for model attention

#### **Real-World S-Learner Use Cases**
1. **Small-scale A/B tests**: Limited data, simple treatment effects
2. **Homogeneous populations**: When treatment effects are similar across groups
3. **Weak treatment signals**: When treatment effect is small relative to noise
4. **Resource-constrained projects**: When simplicity is prioritized

### **X-Learner (Cross-Learning Approach)**

#### **What is X-Learner?**
X-learner combines benefits of T and S learners by using cross-fitting and propensity score weighting.

#### **Algorithm**
```python
# Step 1: Fit models like T-learner
model_treated = RandomForestRegressor()
model_control = RandomForestRegressor()

treated_data = data[data['treatment'] == 1]
control_data = data[data['treatment'] == 0]

model_treated.fit(treated_data[features], treated_data['outcome'])
model_control.fit(control_data[features], control_data['outcome'])

# Step 2: Compute imputed treatment effects
# For treated individuals: actual outcome - predicted control outcome
tau_treated = treated_data['outcome'] - model_control.predict(treated_data[features])

# For control individuals: predicted treated outcome - actual outcome  
tau_control = model_treated.predict(control_data[features]) - control_data['outcome']

# Step 3: Model the treatment effects
tau_model_treated = RandomForestRegressor()
tau_model_control = RandomForestRegressor()

tau_model_treated.fit(treated_data[features], tau_treated)
tau_model_control.fit(control_data[features], tau_control)

# Step 4: Predict treatment effects with propensity weighting
propensity_model = LogisticRegression()
propensity_model.fit(data[features], data['treatment'])
propensity_scores = propensity_model.predict_proba(data[features])[:, 1]

tau_t_pred = tau_model_treated.predict(data[features])
tau_c_pred = tau_model_control.predict(data[features])

# Weight by propensity scores
ite = propensity_scores * tau_c_pred + (1 - propensity_scores) * tau_t_pred
ate = ite.mean()
```

#### **X-Learner Advantages**
- **Handles imbalanced treatments**: Better than T-learner for unequal group sizes
- **Leverages cross-information**: Uses both groups to estimate effects for each individual
- **Propensity weighting**: Accounts for selection bias in treatment assignment
- **Often superior performance**: Combines strengths of T and S learners

#### **X-Learner Limitations**
- **Complexity**: More complex to implement and tune
- **Propensity dependence**: Performance depends on propensity score model quality
- **Computational cost**: Requires fitting multiple models
- **Harder to interpret**: Less transparent than T or S learners

#### **Real-World X-Learner Use Cases**
1. **Observational studies**: When treatment assignment is not random
2. **Imbalanced treatments**: E.g., 80% control, 20% treatment
3. **High-stakes decisions**: When accuracy is more important than simplicity
4. **Healthcare research**: Complex treatment assignment mechanisms

### **Comparison Matrix**

| Aspect | T-Learner | S-Learner | X-Learner |
|--------|-----------|-----------|-----------|
| **Models Required** | 2 | 1 | 4+ |
| **Data Efficiency** | Low | High | Medium |
| **Imbalanced Treatment** | Poor | Good | Excellent |
| **Implementation** | Simple | Very Simple | Complex |
| **Performance** | Good | Variable | Often Best |
| **Interpretability** | High | High | Medium |

---

## **4. Why T/S/X Learners Don't Apply to ADP Project**

### **Fundamental Incompatibility**

#### **1. Outcome Type Mismatch**
- **T/S/X Learners**: Designed for binary or continuous outcomes
- **ADP Problem**: Time-to-event outcome (survival time)
- **Issue**: Can't handle censoring, competing risks, or time-varying effects

#### **2. Time Dimension Ignored**
- **T/S/X Learners**: Single time point analysis
- **ADP Need**: Multiple time horizons (6-month, 12-month predictions)
- **Issue**: Don't capture when treatment effects occur or how they change over time

#### **3. Censoring Not Handled**
- **T/S/X Learners**: Assume all outcomes are observed
- **ADP Reality**: Many employees haven't left yet (right-censored)
- **Issue**: Would require dropping censored employees, losing valuable information

#### **4. No Competing Risks**
- **T/S/X Learners**: Single outcome type
- **ADP Context**: Voluntary termination vs. involuntary vs. retirement vs. transfer
- **Issue**: Can't distinguish between different types of "leaving"

### **Specific ADP Business Incompatibilities**

#### **Wrong Business Question**
```python
# What T/S/X learners answer:
"If John gets a salary increase, will he leave or stay?" (Binary)

# What ADP needs to know:
"If John gets a salary increase, how much longer will he stay?" (Time)
"What's his retention probability at 6, 12, 18 months?" (Multiple horizons)
```

#### **Missing Time-Varying Covariates**
```python
# T/S/X learner approach (WRONG):
features = ['initial_salary', 'hire_date_performance', 'starting_manager']

# ADP reality (CORRECT):
# Employee characteristics change over time:
# - Salary increases throughout tenure
# - Performance ratings change
# - Managers change
# - Job responsibilities evolve
```

#### **No Survival Probability Outputs**
```python
# T/S/X learner output:
"John's treatment effect = 0.15" (What does this mean for HR?)

# ADP business need:
"John's 12-month retention probability increases from 65% to 78% with salary increase"
"ROI: $8,500 cost, $45,000 replacement saving → 5.3x return"
```

### **Technical Implementation Issues**

#### **Data Format Incompatibility**
```python
# T/S/X learner expected format:
employee_id | age | salary | treatment | outcome
1          | 35  | 75000  | 1         | 0 (stayed)
2          | 42  | 82000  | 0         | 1 (left)

# ADP survival data format:
employee_id | start | stop | age | salary | treatment | event
1          | 0     | 6    | 35  | 75000  | 0         | 0
1          | 6     | 12   | 35  | 80000  | 1         | 0  
1          | 12    | 18   | 36  | 80000  | 1         | 1
```

#### **Model Validation Differences**
```python
# T/S/X learner metrics:
- AUC for binary classification
- RMSE for continuous outcomes
- R² for explained variance

# Survival analysis metrics:
- C-index (concordance)
- Brier score for time-specific predictions
- Integrated Brier score across time horizons
```

---

## **5. Correct Implementation Path for ADP**

### **Week 2: Foundation (Current Priority)**
1. **Implement G-computation with Cox models**
2. **Test on salary increase intervention first**
3. **Validate basic ATE calculations**
4. **Create simple ITE examples**

### **Week 3: Advanced Features**
1. **Multiple intervention comparison**
2. **Time-varying covariate integration**
3. **Competing risks analysis**
4. **Individual-level recommendations**

### **Week 4: Business Integration**
1. **Risk category mapping**
2. **ROI calculations**
3. **Model cards for interventions**
4. **Production deployment**

### **Key Deliverables**
```python
# Business-ready output format
intervention_recommendations = {
    'employee_id': 'EMP_12345',
    'current_12m_risk': 0.45,
    'interventions': [
        {
            'type': 'salary_increase_10pct',
            'new_risk': 0.32,
            'risk_reduction': 0.13,
            'cost': 8500,
            'roi': 5.3,
            'confidence_interval': (0.08, 0.18)
        },
        {
            'type': 'promotion',
            'new_risk': 0.28,
            'risk_reduction': 0.17,
            'cost': 12000,
            'roi': 6.8,
            'confidence_interval': (0.12, 0.22)
        }
    ]
}
```

---

## **6. Summary: The Path Forward**

### **For Radhika: Focus Areas**
1. **Abandon T/S/X learner research** - not applicable to survival analysis
2. **Study G-computation for survival** - this is the correct methodology
3. **Understand Cox proportional hazards** - foundation for causal survival analysis
4. **Practice with counterfactual datasets** - core skill for G-computation

### **For the Team: Technical Priorities**
1. **Implement survival G-computation pipeline**
2. **Validate causal assumptions (SUTVA, exchangeability, positivity)**
3. **Create multiple intervention scenarios**
4. **Build business translation layer**

### **Success Metrics**
- **Technical**: Proper G-computation implementation with survival models
- **Statistical**: Valid causal assumptions and robust effect estimates  
- **Business**: Actionable intervention recommendations with ROI analysis

The ADP project requires **causal inference for survival analysis**, not standard treatment effect estimation. G-computation provides the right framework for generating actionable, time-aware intervention recommendations that ADP's HR practitioners need.