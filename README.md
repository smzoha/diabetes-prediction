# Diabetes Prediction using Machine Learning Models

The project was done as a requirement of a Machine Learning coursework, for the semester of Summer 2023, in BRAC University.

The project involves comparison between three commonly used Machine Learning models, in the form of Logistic Regression,
Random Forest and Support Vector Machine (SVM).

The project also uses Maximum Relevance-Minimum Redundancy (mRMR) for features selection. More information about mRMR can be
found [here](https://github.com/smazzanti/mrmr).

The dataset used in this project was obtained from Kaggle and contains demographic data, as well as medical data, that
allows for detection of the disease in an individual.

The dataset can be found here: [Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset).

The dataset contains 8 features and 1 target variable, with the latter being a boolean value, that allows for binary classification
of the disease from the data.

The project can be divided into three parts - data clean-up, visualization, and finally, training and prediction.

The three Python scripts that reflect the aforementioned parts are:
* **data_cleanup.py:** The script that involves cleaning up the data, dropping empty and duplicate rows, along with converting
data for some of the features to make them more easily understandable to the models - e.g. convert text categories to numeric values.
* **data_visualization.py:** A simple script that includes various graphing functions that allows for visualization of the features
and the target variable.
* **train_predict.py:** The script involves training the three models with the dataset and test how well they can predict upon training
on a slice of the dataset that is kept aside as testing data.

#### Prerequisites:
* **Python 3**
  * Python 3.11 has been used for this project and is defined as a dependency in the Pipfile 
  * In order to use a different version with Pipenv, please change the version under the `[requires]` section of the Pipfile
* **Pip**
    * Required for downloading the packages required to run the project
    * Instruction on how to install Pip can be found [here](https://pip.pypa.io/en/stable/cli/pip_install/)
* **(Optional) PipEnv**
  * Allows creating the environment for running the code for the project
  * Instruction on how to install Pipenv can be found [here](https://pipenv.pypa.io/en/latest/) [Requires Pip]

#### Instructions:
1. Install the dependencies using Pipenv (or Pip)
   1. In order to manually install the dependencies using Pip, the following command can be run:
   
        `pip install numpy pandas matplotlib scikit-learn imblearn mrmr_selection`
2. Run the files in the order mentioned above

#### Libraries Used:
* Numpy
* Pandas
* Matplotlib
* Sci-Kit Learn
* Imbalanced Learn
* [mRMR](https://github.com/smazzanti/mrmr)
