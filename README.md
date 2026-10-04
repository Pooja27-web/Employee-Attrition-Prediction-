# Employee Attrition Prediction Using Random Forest 🌳

A machine learning classification project that predicts whether an employee is likely to **leave or stay in an organization** based on factors such as salary, job satisfaction, working hours, experience, promotions, and distance from home.

The project uses a **Random Forest Classifier** built with Python and Scikit-learn.

---

## 📌 Project Overview

Employee attrition is an important business problem because unexpected employee turnover can affect productivity, recruitment costs, team stability, and organizational performance.

This project demonstrates how machine learning can be used to analyze employee-related factors and predict employee attrition.

The model takes employee characteristics as input and predicts one of two outcomes:

- `0` → Employee is likely to stay
- `1` → Employee is likely to leave

The project covers the complete basic machine learning workflow:

**Dataset → Data Exploration → Data Validation → Feature Selection → Train/Test Split → Model Training → Prediction → Evaluation → Feature Importance → New Employee Prediction**

---

## 🎯 Problem Statement

An organization wants to predict whether an employee is likely to leave the company based on factors such as:

- Salary
- Job Satisfaction
- Working Hours
- Experience
- Promotions
- Distance From Home

The objective is to develop a **Random Forest Classifier** capable of predicting employee attrition.

---

## 🎯 Objectives

The main objectives of this project are:

1. Create and understand an employee attrition dataset.
2. Explore the structure of the dataset.
3. Check for missing values.
4. Separate input features and the target variable.
5. Split the data into training and testing sets.
6. Build a Random Forest classification model.
7. Train the model using employee data.
8. Predict employee attrition.
9. Evaluate the model using classification metrics.
10. Analyze feature importance.
11. Use the trained model to predict attrition for a new employee.

---

## 🧠 Machine Learning Approach

### Algorithm Used

**Random Forest Classifier**

Random Forest is an ensemble machine learning algorithm that combines the predictions of multiple decision trees.

Instead of relying on a single decision tree, the model uses multiple trees and combines their predictions to produce the final classification.

In this project:

```text
Number of Trees = 100
Random State = 42
```

The Random Forest model is implemented using:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

---

## 📊 Dataset

The project uses a small employee dataset created specifically for learning and demonstration purposes.

The dataset contains **20 employee records** and the following variables:

| Feature | Description |
|---|---|
| `Salary` | Employee salary |
| `Job_Satisfaction` | Job satisfaction rating |
| `Working_Hours` | Number of working hours |
| `Experience` | Years of experience |
| `Promotions` | Number of promotions received |
| `Distance_From_Home` | Distance between home and workplace |
| `Leave_Organization` | Target variable indicating whether the employee leaves |

### Target Variable

`Leave_Organization`

```text
0 → Likely to Stay
1 → Likely to Leave
```

---

## 🔍 Input Features

The model uses six input features:

```text
Salary
Job_Satisfaction
Working_Hours
Experience
Promotions
Distance_From_Home
```

The target variable is:

```text
Leave_Organization
```

---

## 🛠️ Technologies & Libraries

The project was developed using Python and the following libraries:

- **Python**
- **Pandas** — Data manipulation and analysis
- **NumPy** — Numerical operations
- **Matplotlib** — Data visualization
- **Scikit-learn** — Machine learning and model evaluation
- **Jupyter Notebook / Spyder** — Development environment

### Installation

Install the required libraries using:

```bash
pip install pandas numpy matplotlib scikit-learn
```

---

## 🔄 Project Workflow

```text
                    ┌─────────────────────┐
                    │ Employee Dataset    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Understanding   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Missing Value Check  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature / Target     │
                    │ Separation           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Train/Test Split     │
                    │ 80% / 20%            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Random Forest        │
                    │ Classifier           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Model Training       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Predictions          │
                    └──────────┬──────────┘
                               │
                  ┌────────────┴────────────┐
                  ▼                         ▼
        ┌─────────────────┐       ┌──────────────────┐
        │ Model Evaluation│       │ Feature Importance│
        └─────────────────┘       └──────────────────┘
```

---

## 🧹 Data Preparation

The dataset is first converted into a Pandas DataFrame.

```python
df = pd.DataFrame(data)
```

Basic dataset inspection is performed using:

```python
df.head()
df.shape
df.info()
df.describe()
```

Missing values are checked using:

```python
df.isnull().sum()
```

The demonstration dataset contains no missing values.

---

## ✂️ Feature & Target Separation

The independent variables are stored in `X`:

```python
X = df[
    [
        'Salary',
        'Job_Satisfaction',
        'Working_Hours',
        'Experience',
        'Promotions',
        'Distance_From_Home'
    ]
]
```

The target variable is stored in `y`:

```python
y = df['Leave_Organization']
```

---

## 🔀 Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split

- **80%** → Training data
- **20%** → Testing data

`random_state=42` is used to make the split reproducible.

---

## 🌳 Model Development

A Random Forest Classifier is created using:

```python
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

### Model Parameters

| Parameter | Value | Purpose |
|---|---:|---|
| `n_estimators` | 100 | Number of decision trees |
| `random_state` | 42 | Reproducibility |

---

## 🏋️ Model Training

The model is trained using the training dataset:

```python
model.fit(X_train, y_train)
```

During training, the Random Forest learns patterns between employee characteristics and the target outcome:

```text
Employee Characteristics
          ↓
   Machine Learning
          ↓
     Stay / Leave
```

---

## 🔮 Making Predictions

Predictions are generated using:

```python
y_pred = model.predict(X_test)
```

The predicted values can then be compared with the actual test values.

---

## 📈 Model Evaluation

The model is evaluated using several classification metrics.

### 1. Accuracy

```python
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy * 100, "%")
```

Accuracy represents the proportion of correctly classified observations.

> **Important:** Because this project uses a very small, artificially constructed dataset, a high accuracy score should not be interpreted as evidence that the model would perform similarly on real-world employee data.

---

### 2. Confusion Matrix

```python
cm = confusion_matrix(y_test, y_pred)

print(cm)
```

The confusion matrix helps identify correct and incorrect classifications.

For binary classification:

| | Predicted Stay | Predicted Leave |
|---|---:|---:|
| **Actual Stay** | True Negative | False Positive |
| **Actual Leave** | False Negative | True Positive |

Where:

- **TN** → Correctly predicted Stay
- **TP** → Correctly predicted Leave
- **FP** → Predicted Leave when the employee actually stays
- **FN** → Predicted Stay when the employee actually leaves

---

### 3. Classification Report

```python
print(classification_report(y_test, y_pred))
```

The classification report provides:

- Precision
- Recall
- F1-score
- Support

For employee attrition analysis, recall for the `Leave` class can be particularly useful because failing to identify an employee who is actually likely to leave may be important for retention analysis.

---

## 📊 Feature Importance

One advantage of Random Forest is that it provides feature importance values.

```python
importance = model.feature_importances_

for feature, value in zip(X.columns, importance):
    print(feature, ":", value)
```

The project also visualizes feature importance using Matplotlib:

```python
plt.figure(figsize=(8, 5))

plt.bar(
    X.columns,
    model.feature_importances_
)

plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45)

plt.show()
```

This visualization helps understand which input variables the trained Random Forest relied on more heavily.

### ⚠️ Important Note

Feature importance indicates how the fitted Random Forest uses the available features. It **does not automatically mean that a feature causes employee attrition**.

---

## 👤 Predicting a New Employee

The trained model can also be used to make predictions for a new employee.

Example employee:

| Feature | Value |
|---|---:|
| Salary | ₹30,000 |
| Job Satisfaction | 2 |
| Working Hours | 50 |
| Experience | 3 years |
| Promotions | 0 |
| Distance From Home | 18 km |

The input can be represented as:

```python
new_employee = [[30000, 2, 50, 3, 0, 18]]
```

Prediction:

```python
prediction = model.predict(new_employee)
```

The model output can be converted into a meaningful result:

```python
if prediction[0] == 1:
    print("Employee is likely to leave the organization.")
else:
    print("Employee is likely to stay in the organization.")
```

---

## 📁 Suggested Repository Structure

```text
Employee-Attrition-Prediction/
│
├── README.md
│
├── employee_attrition_prediction.ipynb
│
├── employee_attrition_prediction.py
│
├── data/
│   └── employee_attrition.csv
│
├── visualizations/
│   └── feature_importance.png
│
└── requirements.txt
```

If you are only uploading your notebook, you can keep the repository simpler:

```text
Employee-Attrition-Prediction/
│
├── README.md
└── employee_attrition_prediction.ipynb
```

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Employee-Attrition-Prediction.git
```

### 2. Navigate to the project directory

```bash
cd Employee-Attrition-Prediction
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or:

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 4. Run the notebook

Open:

```text
employee_attrition_prediction.ipynb
```

using Jupyter Notebook, JupyterLab, or Google Colab.

---

## 📦 Requirements

Create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
scikit-learn
```

---

## 💡 Key Learnings

Through this project, I worked with the complete basic machine learning pipeline, including:

- Understanding a classification problem
- Creating and handling a dataset using Pandas
- Separating features and target variables
- Splitting data into training and testing sets
- Building a Random Forest Classifier
- Training a machine learning model
- Generating predictions
- Evaluating classification performance
- Understanding confusion matrices
- Interpreting precision, recall, and F1-score
- Analyzing feature importance
- Making predictions for new observations
- Understanding the limitations of small datasets

---

## 🚀 Future Improvements

This project can be extended into a more realistic employee analytics solution by:

- Using a larger real-world employee attrition dataset
- Performing exploratory data analysis
- Handling categorical variables
- Applying appropriate preprocessing
- Addressing class imbalance
- Comparing Random Forest with other classification algorithms
- Performing hyperparameter tuning
- Using cross-validation
- Adding ROC-AUC and other evaluation metrics
- Building an interactive dashboard using Power BI
- Creating a Streamlit web application
- Adding explainable AI techniques such as SHAP
- Deploying the trained model as an API

---

## ⚠️ Limitations

This project is primarily an educational demonstration.

The dataset is **small and artificially constructed**, so the model's performance should not be treated as representative of real-world employee attrition prediction.

In a production scenario, a much larger and more representative dataset would be required along with appropriate data preprocessing, validation, fairness checks, and model monitoring.

---

## 📌 Project Takeaway

This project demonstrates how a **Random Forest classification model** can be used to predict employee attrition from structured employee-related features.

The project also highlights an important machine learning principle:

> **A model's performance is only meaningful when evaluated on appropriate, representative data.**

The goal is not only to generate a prediction, but also to understand the data, evaluate the model, interpret its behavior, and recognize its limitations.

---

## 👩‍💻 Author

**Poojashree H**

BCA — Data Science  
Alliance University, Bengaluru

---

## ⭐ If You Found This Project Useful

Feel free to explore the repository, experiment with the model, and improve it using a larger real-world employee attrition dataset.

**Machine Learning → Experiment → Evaluate → Improve 🚀**
