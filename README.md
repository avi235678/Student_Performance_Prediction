# Student Performance Prediction

## Project Overview

This project predicts a student's final G3 score using Machine Learning.

The project uses the Student Performance Dataset and includes data preprocessing,
exploratory data analysis, model training, evaluation and prediction.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning

The project uses Linear Regression and Random Forest Regression models.

The models are evaluated using:

- Mean Absolute Error (MAE)
- R² Score

## Model Results

Linear Regression:

- MAE: 0.7651
- R² Score: 0.8487

Random Forest:

- MAE: 0.7651
- R² Score: 0.8429

Linear Regression performed slightly better on this dataset.

## Streamlit Application

A Streamlit web application is included for student performance prediction.

## Project Structure

Student_Performance_Prediction/

- data/
  - student-por.csv
- main.py
- app.py
- student_performance_model.pkl
- requirements.txt
- README.md

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt