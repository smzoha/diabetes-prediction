import os
import pandas as pd

data = pd.read_csv('../data/diabetes_prediction_dataset.csv')

# Checking for null values
print('Null Value Count:', data.isnull().sum(), sep='\n')
print('================================================')

# Checking for duplicate records
print('Duplicate Rows Count:', data.duplicated().sum())

# Dropping duplicate rows
data = data.drop_duplicates()
print('Duplicate Rows Count (after drop):', data.duplicated().sum())
print('================================================')

# Transform 'gender' and 'smoking_history' data to lower for further encoding
data['gender'] = data['gender'].str.lower()
data['smoking_history'] = data['smoking_history'].str.lower()

# Checking count of rows grouped by 'gender'
print('Gender Distribution (pre-conversion):', data['gender'].value_counts(), sep='\n')
print('================================================')

# Convert 'gender' to numeric values (Male: 0, Female: 1, Other: 2)
data.loc[data['gender'] == 'male', 'gender'] = 0
data.loc[data['gender'] == 'female', 'gender'] = 1
data.loc[data['gender'] == 'other', 'gender'] = 2

# Checking count of rows grouped by 'gender' -- post conversion
print('Gender Distribution (post-conversion):', data['gender'].value_counts(), sep='\n')
print('================================================')

# Checking count of rows grouped by 'smoking_history'
print('Smoking History Distribution (pre-conversion):', data['smoking_history'].value_counts(), sep='\n')
print('================================================')

# Convert 'smoking_history' to numeric values (current: 0, ever: 1, former: 2, never: 3, no info: 4, not current: 5)
data.loc[data['smoking_history'] == 'current', 'smoking_history'] = 0
data.loc[data['smoking_history'] == 'ever', 'smoking_history'] = 1
data.loc[data['smoking_history'] == 'former', 'smoking_history'] = 2
data.loc[data['smoking_history'] == 'never', 'smoking_history'] = 3
data.loc[data['smoking_history'] == 'no info', 'smoking_history'] = 4
data.loc[data['smoking_history'] == 'not current', 'smoking_history'] = 5

# Checking count of rows grouped by 'smoking_history' -- post conversion
print('Smoking History Distribution (post-conversion):', data['smoking_history'].value_counts(), sep='\n')
print('================================================')

print('Writing data to new file')
if not os.path.exists('../clean_data'):
    os.mkdir('../clean_data')

data.to_csv('../clean_data/clean_data.csv', index=False)
