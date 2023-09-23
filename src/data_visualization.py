import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('../data/diabetes_prediction_dataset.csv')

# Data Analysis
print(data.describe())

# Plotting Gender Distribution

data['gender'].value_counts().plot(kind='bar')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Distribution of Gender')
plt.show()

# Plotting Age Distribution
data.age.plot(kind='density')
plt.title('Distribution of Age')
plt.show()

# Plotting Smoking History Distribution
data['smoking_history'].value_counts().plot(kind='bar')
plt.xlabel('Smoking History')
plt.ylabel('Count')
plt.title('Distribution of Smoking History')
plt.show()

# Plotting Heart Disease Distribution
data['heart_disease'].value_counts().plot(kind='bar')
plt.xlabel('Heart Disease')
plt.ylabel('Count')
plt.title('Distribution of Heart Disease')
plt.show()

# Plotting Hypertension Distribution
data['hypertension'].value_counts().plot(kind='bar')
plt.xlabel('Hypertension')
plt.ylabel('Count')
plt.title('Distribution of Hypertension')
plt.show()

# Plotting BMI Distribution
data.bmi.plot(kind='density')
plt.title('Distribution of BMI')
plt.show()

# Plotting HbA1c Distribution
data.HbA1c_level.plot(kind='density')
plt.title('Distribution of HbA1c')
plt.show()

# Plotting Blood Glucose Distribution
data.blood_glucose_level.plot(kind='density')
plt.title('Distribution of Glucose Level')
plt.show()
