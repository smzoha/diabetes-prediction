import pandas as pd
from mrmr import mrmr_classif
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data = pd.read_csv('../clean_data/clean_data.csv')

# Put data to separate variables for features and target
x = data[data.columns[:-1]]
y = data[data.columns[-1]]

# Run mRMR (algorithm that determines the relevance of a feature against the target value)
selected_features = mrmr_classif(X=x, y=y, K=8)
print('Selected Features:', selected_features, sep='\n')

# Keep the features deemed important by mRMR
x = x[['blood_glucose_level', 'HbA1c_level', 'age', 'hypertension', 'bmi']]

# Sampling data to avoid imbalance between classes
sampler = RandomUnderSampler(sampling_strategy='majority')
x, y = sampler.fit_resample(x, y)

# Split train-test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Scaling training and test datasets
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# LOGISTIC REGRESSION
# Training model
print('========== LOGISTIC REGRESSION ==============')
model = LogisticRegression(max_iter=17000)
model.fit(x_train, y_train)

# Calculate cross-validation score
score = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=100)
print('Cross-Validation Score:', score)
print('Mean Cross-Validation Score:', score.mean())

# Plotting Accuracy Score over Iterations
plt.plot(score)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Cross-Validation (Accuracy)')
plt.show()

y_pred = model.predict(x_test)
report = metrics.classification_report(y_true=y_test, y_pred=y_pred)
print('Metrics Report for Prediction', report)

error = mean_squared_error(y_pred, y_test)
print('MSE Error:', error)

print('================================================')

# RANDOM FOREST
# Training model
print('============ RANDOM FOREST ================')
model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

# Calculate cross-validation score
score = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=100)
print('Cross-Validation Score:', score)
print('Mean Cross-Validation Score:', score.mean())

# Plotting Accuracy Score over Iterations
plt.plot(score)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Cross-Validation (Accuracy)')
plt.show()

# Predict with test data and evaluate scores
y_pred = model.predict(x_test)
report = metrics.classification_report(y_true=y_test, y_pred=y_pred)
print('Metrics Report for Prediction', report)

error = mean_squared_error(y_pred, y_test)
print('MSE Error:', error)

print('================================================')

# SVM
# Training model
print('============ SUPPORT VECTOR MACHINE ==============')
model = SVC(kernel='poly')
model.fit(x_train, y_train)

# Calculate cross-validation score
score = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=100)
print('Cross-Validation Score:', score)
print('Mean Cross-Validation Score:', score.mean())

# Plotting Accuracy Score over Iterations
plt.plot(score)
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.title('Cross-Validation (Accuracy)')
plt.show()

# Predict with test data and evaluate scores
y_pred = model.predict(x_test)
report = metrics.classification_report(y_true=y_test, y_pred=y_pred)
print('Metrics Report for Prediction', report)

error = mean_squared_error(y_pred, y_test)
print('MSE Error:', error)

print('================================================')