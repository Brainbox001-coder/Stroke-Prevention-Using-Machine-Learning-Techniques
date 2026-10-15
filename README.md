# Stroke Prevention Using Machine Learning Techniques

### A Machine Learning Approach to Stroke Risk Prediction

This project explores the use of machine learning to predict stroke occurrence based on demographic, medical, and lifestyle-related factors. Developed by **Group 13** as part of the TechCrush AI/Machine Learning capstone project, it demonstrates an end-to-end machine learning workflow, from data preprocessing and exploratory data analysis to model evaluation and web application deployment.

The project compares Logistic Regression and Random Forest classifiers and deploys the selected Logistic Regression pipeline through an interactive Streamlit application.

> **Disclaimer:** This project is for educational and demonstration purposes only. It is not a medical diagnostic tool and should not be used to make clinical decisions.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Workflow](#project-workflow)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Machine Learning Models](#machine-learning-models)
- [Model Evaluation](#model-evaluation)
- [Why Logistic Regression Was Selected](#why-logistic-regression-was-selected)
- [Streamlit Application](#streamlit-application)
- [Key Challenges and Lessons Learned](#key-challenges-and-lessons-learned)
- [Limitations and Future Improvements](#limitations-and-future-improvements)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Project Overview

Stroke is a major public health concern, and identifying individuals who may be at increased risk is an important area of healthcare research. Machine learning offers techniques for examining patterns in health-related data and developing predictive models.

The goal of this project is to investigate how demographic, medical, and lifestyle information can be used to classify whether a record indicates stroke occurrence.

The project follows a structured data science workflow involving data cleaning, exploratory analysis, feature preprocessing, model development, and performance evaluation. The final selected model is integrated into a Streamlit web application to demonstrate how a trained machine learning pipeline can be used interactively.

## Objectives

The main objectives of this project are to:

1. Collect and examine a publicly available stroke prediction dataset.
2. Clean the data and handle missing values.
3. Perform exploratory data analysis (EDA) to investigate patterns and relationships among features.
4. Prepare numerical and categorical features for machine learning.
5. Train and compare Logistic Regression and Random Forest classifiers.
6. Evaluate model performance using appropriate classification metrics.
7. Select a model based on its ability to identify stroke cases.
8. Save the trained model pipeline and integrate it into a Streamlit application.

## Dataset

The project uses the **Stroke Prediction Dataset** published by Fedesoriano on Kaggle.

**Dataset source:** [Stroke Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

The dataset contains demographic, health, and lifestyle information collected from 5,110 individuals and includes 12 variables. 

### Features used

| Feature | Description |
|---|---|
| `gender` | Gender of the individual |
| `age` | Age of the individual |
| `hypertension` | Indicates whether the individual has hypertension |
| `heart_disease` | Indicates whether the individual has heart disease |
| `ever_married` | Marital status |
| `work_type` | Type of employment |
| `Residence_type` | Urban or rural residence |
| `avg_glucose_level` | Average glucose level |
| `bmi` | Body mass index |
| `smoking_status` | Smoking status |
| `stroke` | Target variable: stroke occurrence (0 or 1) |

The `id` field was removed because it was an identifier rather than a meaningful predictive feature.

The dataset is imbalanced, with approximately 95% of records belonging to the non-stroke class and 5% to the stroke class. This imbalance is an important consideration when evaluating model performance.

## Technologies Used

The project uses the following tools and libraries:

- **Python** — core programming language
- **Pandas** — data manipulation and analysis
- **NumPy** — numerical operations
- **Matplotlib and Seaborn** — data visualization
- **Scikit-learn** — preprocessing, model training, and evaluation
- **Joblib** — saving and loading the trained model pipeline
- **Jupyter Notebook / Google Colab** — development and experimentation
- **Streamlit** — interactive web application
- **Git and GitHub** — version control and project hosting

## Project Workflow

The project was completed through the following stages:

1. **Data collection:** Obtained the stroke prediction dataset from Kaggle.
2. **Data cleaning:** Inspected the dataset, removed the identifier column, and handled missing BMI values.
3. **Exploratory data analysis:** Examined target distribution, numerical variables, categorical features, outliers, and relationships in the data.
4. **Data preprocessing:** Separated features from the target and prepared numerical and categorical variables.
5. **Model development:** Trained Logistic Regression and Random Forest classifiers.
6. **Model evaluation:** Compared accuracy, precision, recall, F1-score, confusion matrices, and ROC-AUC.
7. **Model selection:** Selected Logistic Regression because it performed better at identifying stroke cases.
8. **Deployment:** Saved the trained pipeline using Joblib and built an interactive Streamlit application.

## Exploratory Data Analysis

Exploratory Data Analysis was used to understand the dataset's structure, feature distributions, and potential relationships with stroke occurrence.

The analysis examined:

- The distribution of stroke and non-stroke cases.
- The distributions of numerical features such as age, average glucose level, and BMI.
- Categorical variables such as gender, work type, marital status, residence type, and smoking status.
- Potential outliers in numerical variables.
- Relationships and correlations among selected features.

The analysis highlighted the importance of considering age, glucose level, hypertension, and heart disease when investigating patterns associated with stroke occurrence. These observed associations do not establish causation.

## Data Preprocessing

Data preprocessing was performed to prepare the dataset for reliable model training.

The main steps included:

- Removing the `id` column.
- Handling missing values in the BMI feature using median imputation.
- Separating the input features from the target variable.
- Splitting the data into training and testing sets using an 80:20 ratio with stratification.
- Applying median imputation and standard scaling to numerical features.
- Applying one-hot encoding to categorical features.
- Integrating preprocessing and classification into machine learning pipelines.

Using a pipeline helps ensure that the same fitted preprocessing transformations are applied when the model receives new input data.

## Machine Learning Models

Two classification algorithms were developed and compared.

### 1. Logistic Regression

Logistic Regression was used as an interpretable baseline classifier for the binary prediction task.

The model was configured with:

- `max_iter=1000`
- `class_weight="balanced"`

Balanced class weights were used to place greater emphasis on the minority stroke class during training.

### 2. Random Forest

Random Forest was selected to investigate whether an ensemble of decision trees could capture more complex relationships in the data.

The model was configured with:

- `n_estimators=300`
- `random_state=42`
- `class_weight="balanced"`

Both models were evaluated on the held-out test set.

## Model Evaluation

Because the dataset contains substantially more non-stroke cases than stroke cases, accuracy alone is insufficient for assessing the models. Recall, precision, F1-score, and ROC-AUC were also considered.

### Performance comparison

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 74.46% | 94.81% |
| Precision | 13.49% | 0.00% |
| Recall | 78.00% | 0.00% |
| F1-score | 0.23 | 0.00 |
| ROC-AUC | 0.843 | 0.770 |

### Interpretation

**Logistic Regression** achieved a recall of 78%, identifying 39 of the 50 actual stroke cases in the test set. However, its precision was low, indicating that it also classified many non-stroke cases as positive.

**Random Forest** achieved a higher accuracy of 94.81%, but it failed to identify any stroke cases in the test set. Its high accuracy was therefore misleading for the project's primary objective.

The results demonstrate why model evaluation for imbalanced healthcare data should consider the ability to detect the minority class rather than relying on accuracy alone.

## Why Logistic Regression Was Selected

Logistic Regression was selected as the preferred model because the project's primary objective was to identify stroke cases rather than maximize overall accuracy. It achieved a recall of 78% and a ROC-AUC of 0.843, outperforming Random Forest on both metrics. Although it produced some false-positive predictions, it identified most of the actual stroke cases in the test set, making it more suitable for the project's focus on stroke detection.

However, this choice reflects the project's modelling objectives and does not establish that the model is clinically reliable or suitable for real-world medical use. Further validation would be required before any clinical application.

## Streamlit Application

The selected Logistic Regression pipeline was saved using Joblib and integrated into a Streamlit web application.

The application provides an interactive interface where users can enter demographic, medical, and lifestyle information and receive a model-generated classification.

### Application features

- Interactive input fields for the model's features
- Loading of the saved machine learning pipeline
- Prediction using the fitted preprocessing and classification steps
- A simple interface for demonstrating the model's functionality

**Live application:** [https://stroke-prediction-demo.streamlit.app/]

## Key Challenges and Lessons Learned

### Challenges

- **Class imbalance:** Stroke cases represented only a small proportion of the dataset.
- **Missing values:** BMI contained missing entries that required imputation.
- **Misleading accuracy:** Random Forest achieved high accuracy while failing to detect stroke cases.
- **Model evaluation:** Selecting appropriate metrics was essential to assess the project's actual objective.
- **Deployment:** The trained preprocessing and classification pipeline needed to be saved and integrated into an accessible application.

### Lessons learned

- Data cleaning and preprocessing are essential parts of a machine learning workflow.
- Pipelines help maintain consistent preprocessing between training and prediction.
- Recall, precision, F1-score, and ROC-AUC provide important information when working with imbalanced classification datasets.
- A model's suitability depends on the problem objective, not simply its overall accuracy.
- Deployment demonstrates how a trained model can be made accessible through a user-friendly interface.

## Limitations and Future Improvements

This project is an educational demonstration and has several limitations:

- The model was evaluated using a single train-test split.
- The dataset is imbalanced, and the selected model produces a substantial number of false positives.
- The reported results do not establish performance on independent populations or in clinical settings.
- Further validation, calibration, and threshold analysis would be necessary before any practical healthcare application.

Potential future improvements include:

- Evaluating additional models and imbalance-handling techniques.
- Using cross-validation and independent external validation datasets.
- Investigating classification thresholds to better understand the precision-recall trade-off.
- Improving prediction explanations and communicating uncertainty.
- Conducting fairness, reliability, and clinical suitability assessments.

## Acknowledgements

We acknowledge our tutors, Tutor Micheal and Tutor Lamzy, and the TechCrush community for their guidance, support, and contributions throughout the project.

We also acknowledge Fedesoriano for making the Stroke Prediction Dataset available through Kaggle.

## References

1. Fedesoriano. (2021). *Stroke Prediction Dataset*. Kaggle.  
   https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

2. Scikit-learn Developers. *Scikit-learn Documentation*.  
   https://scikit-learn.org/stable/

3. Streamlit. *Streamlit Documentation*.  
   https://docs.streamlit.io/

4. Joblib. *Joblib Documentation*.  
   https://joblib.readthedocs.io/

---

**Built by Group 13 | TechCrush AI/Machine Learning Capstone Project | October 2026**
