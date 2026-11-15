# Medical Insurance Charges Prediction


## 1. Project Overview

- This project focuses on predicting individual medical insurance charges using Machine Learning techniques.
- The objective is to explore how factors such as age, gender, BMI, number of children, smoking habits, and geographical region influence medical insurance costs.
- The project demonstrates an end-to-end Machine Learning workflow, starting from raw data and data preparation through exploratory data analysis, model development, evaluation, and deployment using Streamlit.

## 2. Business Problem

Insurance companies need to estimate potential medical expenses to support pricing decisions, risk assessment, and business planning.
This project addresses the following question:
- Can we estimate an individual's medical insurance charges based on their demographic and health-related information?
- The prediction model uses historical insurance data to estimate insurance charges for new input records.

## 3. Dataset Description

The dataset contains individual medical insurance records with demographic, lifestyle, and health-related information. These variables are used to analyze the factors influencing medical insurance costs and to develop a regression model for predicting insurance charges.

### Dataset Features

| Feature | Description | Data Type |
|---|---|---|
| Age | Age of the individual in years | Numerical |
| Gender | Gender of the individual | Categorical |
| BMI | Body Mass Index of the individual | Numerical |
| Number of Children | Number of dependent children | Numerical |
| Smoker | Smoking status (Yes/No) | Categorical |
| Region | Geographical residential region | Categorical |
| Charges | Individual medical insurance cost | Numerical (Target) |

### Target Variable

- **Charges** is the dependent variable that the Machine Learning model aims to predict.

### Dataset Files

- **Raw Dataset:** Contains the original data before data cleaning and transformation.
- **Processed Dataset:** Contains the cleaned and prepared data used for exploratory analysis and model development.

### The repository contains:

- **Raw dataset:** Original data before transformation.
- **Processed dataset:** Data prepared for subsequent analysis and model development.

## 4. Project Workflow

### Phase 1: Data Wrangling

- Load and inspect the raw dataset.
- Identify missing values and inconsistent entries.
- Handle missing values.
- Review and correct data types.
- Prepare the cleaned dataset for analysis.

### Phase 2: Exploratory Data Analysis (EDA)

- Perform descriptive statistical analysis.
- Explore the distribution of insurance charges.
- Analyze relationships between BMI, age, smoking habits, and insurance charges.
- Examine correlations between numerical variables.
- Use visualizations to identify patterns and trends.

### Phase 3: Machine Learning

The project explores multiple regression approaches:

-  Simple Linear Regression
-  Multiple Linear Regression
-  Polynomial Regression
-  Ridge Regression
-  Hyperparameter tuning using GridSearchCV

Model evaluation and refinement are performed to assess predictive performance and compare model results.

### Phase 4: Model Deployment

A Streamlit application provides an interactive interface where users can enter individual details and receive estimated medical insurance charges.

## 5. Technologies and Tools

- Python: Data processing and Machine Learning
- Pandas and NumPy: Data manipulation
- Matplotlib and Seaborn: Data visualization
- Scikit-learn: Regression models, preprocessing, and model evaluation
- Jupyter Notebook: Data analysis and experimentation
- Streamlit: Interactive application development
- Joblib: Saving and loading trained model artifacts
- Git and GitHub: Version control and project documentation


## 6. Repository Structure

The project is organized into the following directories and files:

```text
insurance-charges-prediction/
│
├── data/
│   ├── raw/
│   │   └── medical_insurance_dataset.csv
│   │
│   └── processed/
│       └── updated_medical_insurance_dataset.csv
│
├── notebooks/
│   ├── Data_Wraggling.ipynb
│   ├── Exploratory_Data_Analysis.ipynb
│   └── model_code.ipynb
│
├── app.py
├── insurance_ridge_model.pkl
├── feature_columns.pkl
├── README.md
├── requirements.txt
└── .gitignore
```

## 7. Generative AI Contribution

Generative AI played a crucial supporting role throughout the project development process.

It was used to:

- Understand Machine Learning concepts and regression techniques.
- Assist with Python code development and debugging.
- Explain data preprocessing and model evaluation.
- Interpret analytical outputs and visualizations.
- Improve code readability and project documentation.
- Support the development and refinement of the Streamlit application.
- The Machine Learning models were trained using Scikit-learn. 
- Generative AI supported the learning, development, and documentation process rather than generating the model's predictions.

## 8. Learning Outcomes

- Through this project, I gained practical experience in:
- End-to-end data preparation and analysis.
- Exploratory Data Analysis and data visualization.
- Regression-based Machine Learning.
- Model evaluation and hyperparameter optimization.
- Saving and deploying trained Machine Learning models.
- Building interactive applications using Streamlit.
- Version control and project management using GitHub.

## 9. Running the Application

- Clone the repository:

git clone <https://github.com/saunokdutta12-cyber/insurance-charges-prediction.git>

- Navigate to the project directory:

```cd insurance-charges-prediction```

- Install the required libraries:

```pip install -r requirements.txt```

- Run the Streamlit application:

```streamlit run app.py```

### Note: The trained model file and feature-column file must be available in the project root directory. The application also requires the model's training feature configuration to match its input preprocessing.

## 10. Project Objective

This project represents a practical application of Data Analytics and Machine Learning concepts, demonstrating how raw data can be transformed into meaningful insights and an interactive predictive application.
