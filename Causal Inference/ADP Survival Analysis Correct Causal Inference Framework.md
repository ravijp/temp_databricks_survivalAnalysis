# ADP Survival Analysis: Correct Causal Inference Framework

## **The Confusion: T/S/X Learners vs. G-Computation**

### **What Radhika Described (WRONG for this project):**
- **T/S/X Learners**: Meta-learners for binary/continuous outcomes
- **Standard Treatment Effect**: Single time point, binary treatment
- **DoWhy Example**: Lalonde dataset (employment program evaluation)

### **What ADP Project Actually Uses (CORRECT):**
- **G-Computation for Survival**: Causal inference with time-to-event data
- **Survival Treatment Effects**: How interventions affect time-to-turnover
- **Multiple Time Horizons**: 6-month, 12-month survival impacts

---

## **Correct ADP Implementation: G-Computation for Survival**

### **Step 1: Define Causal Framework**
```
Target: How does salary increase affect employee survival (time-to-turnover)?
Treatment: 10% salary increase (binary: yes/no)
Outcome: Time to voluntary termination (survival time)
Confounders: Age, tenure, performance, industry, role level, etc.
```

### **Step 2: G-Computation Algorithm for Survival**

**Phase 1: Fit Survival Model with Confounders**
```python
# Fit Cox model with treatment + confounders
cox_model = CoxPHFitter()
cox_model.fit(df, duration_col='tenure_months', 
              event_col='voluntary_term',
              formula='salary_increase + age + tenure + performance + industry')
```

**Phase 2: Create Counterfactual Scenarios**
```python
# Everyone gets salary increase (Treatment = 1)
df_treated = df.copy()
df_treated['salary_increase'] = 1

# No one gets salary increase (Treatment = 0)  
df_control = df.copy()
df_control['salary_increase'] = 0
```

**Phase 3: Predict Under Each Scenario**
```python
# Survival predictions under treatment
survival_treated = cox_model.predict_survival_function(df_treated)

# Survival predictions under control
survival_control = cox_model.predict_survival_function(df_control)
```

**Phase 4: Calculate Treatment Effects**
```python
# Average Treatment Effect (ATE) at 12 months
ate_12m = survival_treated.loc[12].mean() - survival_control.loc[12].mean()
# Interpretation: "Salary increase improves 12-month retention by X percentage points"

# Individual Treatment Effect (ITE) for each employee
ite_12m = survival_treated.loc[12] - survival_control.loc[12]
# Interpretation: "For John, salary increase improves 12-month retention by Y%"
```

---

## **Why This Approach Fits ADP Business Needs**

### **Business Question Alignment**
✅ **ADP Question**: "What's the retention impact of giving John a 10% raise?"
✅ **G-Computation Answer**: "John's 12-month retention probability increases from 65% to 78%"

❌ **T-Learner Question**: "What's John's counterfactual outcome under treatment?"
❌ **Wrong for ADP**: Doesn't handle time-to-event or competing risks

### **Intervention Types for ADP**
1. **Salary Increases**: 5%, 10%, 15% scenarios
2. **Promotion Timing**: Immediate vs. 6-month delay
3. **Overtime Reduction**: 10%, 20%, 30% reduction
4. **Manager Quality**: Good vs. poor manager assignment
5. **Remote Work**: Full remote vs. hybrid vs. office

### **Output Format for HR Users**
```
Employee: John Smith
Current Risk: 45% (12-month turnover probability)

Intervention Recommendations:
1. 10% Salary Increase → Risk drops to 32% (13 percentage point improvement)
2. Promotion to Senior → Risk drops to 28% (17 percentage point improvement)  
3. Overtime Reduction → Risk drops to 38% (7 percentage point improvement)

ROI Analysis:
- Salary increase: $8,500 cost, $45,000 replacement saving → 5.3x ROI
```

---

## **Implementation Steps for G-Computation**

### **Week 2: Basic G-Computation**
1. **Fit baseline survival models** (Cox PH + XGBoost AFT)
2. **Define intervention scenarios** (salary, promotion, overtime)
3. **Create counterfactual datasets** for each intervention
4. **Calculate ATE for population-level insights**

### **Week 3: Individual Treatment Effects**
1. **Calculate ITE for each employee** under each intervention
2. **Rank interventions by effectiveness** and cost
3. **Generate personalized recommendations** with confidence intervals
4. **Validate causal assumptions** (SUTVA, exchangeability, positivity)

### **Week 4: Business Integration**
1. **Convert to business-friendly outputs** (risk scores, ROI calculations)
2. **Create model cards** for each intervention type
3. **Implement uncertainty quantification** for all estimates
4. **Deploy in model registry** with monitoring

---

## **Critical Causal Assumptions for ADP**

### **SUTVA (Stable Unit Treatment Value Assumption)**
✅ **No Interference**: One employee's salary increase doesn't affect another's turnover
✅ **Consistency**: Well-defined interventions (clear salary increase amounts)

### **Exchangeability (No Unobserved Confounders)**
⚠️ **Strong Assumption**: Requires comprehensive confounder adjustment
✅ **Mitigation**: Rich feature set (performance, tenure, industry, role, etc.)

### **Positivity**
✅ **Sufficient Overlap**: Employees across all confounder patterns can receive treatment
✅ **ADP Context**: Salary increases possible across all employee types

---

## **Key Deliverables for ADP**

### **Technical Outputs**
1. **ATE Estimates**: Population-level intervention impacts
2. **ITE Predictions**: Individual-level treatment recommendations  
3. **Confidence Intervals**: Uncertainty around all estimates
4. **Model Validation**: Assumption testing and robustness checks

### **Business Outputs**
1. **Risk Scores**: 0-100% turnover probability with interventions
2. **ROI Calculations**: Cost-benefit analysis for each intervention
3. **Personalized Recommendations**: Top 3 interventions per employee
4. **What-If Scenarios**: Policy simulation for HR planning

---

## **Bottom Line**

**Radhika's T/S/X learner approach is for standard causal inference problems.**
**ADP project uses G-computation within survival analysis framework.**
**The confusion comes from mixing different causal inference paradigms.**

**Correct approach: G-computation → Survival models → Treatment effects on time-to-turnover**