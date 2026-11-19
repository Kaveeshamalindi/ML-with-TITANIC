# Titanic Survival Prediction – Machine Learning Project

## 1. About This Project

This project uses machine learning to predict whether a passenger on the Titanic survived the disaster.

The project is developed using Python in a Jupyter Notebook (`titanic.ipynb`). It uses passenger information such as age, gender, ticket class, fare, and family details to train a machine learning model.

The model learns patterns from passengers whose survival outcomes are already known. After training, it can use those patterns to predict whether a new passenger might have survived.

**Important:** A prediction is an estimate made by the model. It is not a guarantee of what would have happened to a real person.

## 2. Project Goal

The main goal is to build a **Binary Classification Model**.

Binary classification means that the model predicts one of two possible outcomes:

| Value | Meaning |
|---|---|
| `0` | Passenger did not survive |
| `1` | Passenger survived |

This project uses **Logistic Regression**, a machine learning algorithm commonly used to predict outcomes that belong to two categories.

## 3. Dataset

The project uses the Titanic dataset from Kaggle.

- **Dataset:** [Titanic Dataset – Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)
- **CSV file:** `Titanic-Dataset.csv`
- **Number of records:** 891 passengers
- **Original columns:** 12

Each row represents one passenger, and each column contains information about that passenger.

### 3.1 Dataset Column Descriptions

| Column | Description |
|---|---|
| `PassengerId` | A unique identification number assigned to each passenger. |
| `Survived` | Indicates whether the passenger survived. `0` = No, `1` = Yes. This is the target column. |
| `Pclass` | Passenger ticket class: `1` = First Class, `2` = Second Class, `3` = Third Class. |
| `Name` | Full name of the passenger. |
| `Sex` | Passenger's recorded sex: male or female. |
| `Age` | Age of the passenger in years. |
| `SibSp` | Number of siblings and spouses travelling with the passenger. |
| `Parch` | Number of parents and children travelling with the passenger. |
| `Ticket` | Passenger's ticket number. |
| `Fare` | Amount paid for the ticket. |
| `Cabin` | Cabin number assigned to the passenger, if available. |
| `Embarked` | Port where the passenger boarded: `C` = Cherbourg, `Q` = Queenstown, `S` = Southampton. |

## 4. Tools and Technologies

The following Python libraries are used in this project:

| Library | Purpose |
|---|---|
| `pandas` | Loading, managing, and analysing tabular data. |
| `numpy` | Performing numerical operations. |
| `seaborn` | Creating statistical visualizations. |
| `matplotlib` | Creating graphs and charts. |
| `kagglehub` | Downloading the dataset from Kaggle. |
| `scikit-learn` | Data splitting, feature scaling, model training, and evaluation. |
| `joblib` | Saving and loading the trained model and scaler. |
| `Jupyter Notebook` | Writing and executing Python code interactively. |

## 5. Project Workflow

The project follows a complete machine learning workflow, starting with data collection and ending with a prediction.

### Step 1 – Import Required Libraries

First, the required Python libraries are imported.

Libraries provide ready-made functions that make it easier to perform tasks such as data analysis, visualization, and machine learning.

For example:
- Pandas is used to work with datasets.
- Matplotlib and Seaborn are used to create charts.
- Scikit-learn is used to train the machine learning model.

### Step 2 – Load the Dataset

The Titanic dataset is downloaded from Kaggle using `kagglehub`.

The CSV file is loaded into a Pandas DataFrame called `df`.

A **DataFrame** is similar to an Excel spreadsheet because it contains rows and columns.

### Step 3 – Explore the Dataset

Before building a machine learning model, it is important to understand the dataset.

The notebook uses the following commands:

| Command | Purpose |
|---|---|
| `df.head()` | Displays the first five rows. |
| `df.shape` | Displays the number of rows and columns. |
| `df.columns` | Displays all column names. |
| `df.info()` | Displays column data types and non-missing value counts. |
| `df.describe()` | Displays statistical information about numerical columns. |
| `df.isnull().sum()` | Counts missing values in each column. |

This step helps us understand the structure of the dataset and identify missing information.

### Step 4 – Handle Missing Values

Some passenger records contain missing information.

Machine learning models generally require missing values to be handled before training.

The notebook performs the following operations:

**Age**

Missing age values are replaced with the median age.

The median is the middle value when all available ages are arranged in ascending order.

**Embarked**

Missing values in the `Embarked` column are replaced with the most frequently occurring value, known as the mode.

**Cabin**

The `Cabin` column is removed because it contains many missing values.

### Step 5 – Convert Categorical Data into Numerical Data

Machine learning algorithms work with numerical values. However, some dataset columns contain text categories.

For example:

- `Sex` contains male and female.
- `Embarked` contains C, Q, and S.

The notebook uses `pd.get_dummies()` to convert these categories into numerical columns.

The `drop_first=True` option removes one category from each original column to avoid redundant information.

The resulting columns are:

| New Column | Description |
|---|---|
| `Sex_male` | `1` = Male, `0` = Female. |
| `Embarked_Q` | `1` = Queenstown, `0` = Not Queenstown. |
| `Embarked_S` | `1` = Southampton, `0` = Not Southampton. |

When both `Embarked_Q` and `Embarked_S` are `0`, the passenger belongs to the remaining category, Cherbourg.

### Step 6 – Feature Engineering

**Feature Engineering** means creating new useful columns using existing dataset information.

The notebook creates two additional features.

**1. FamilySize**

This feature calculates the total number of family members travelling together.

Formula:

`FamilySize = SibSp + Parch + 1`

The number `1` represents the passenger themselves.

Example:

If a passenger has:
- 1 sibling or spouse
- 1 parent or child

Then:

`FamilySize = 1 + 1 + 1 = 3`

The total family size is 3.

**2. IsAlone**

This feature identifies whether a passenger travelled alone.

| Value | Meaning |
|---|---|
| `1` | Passenger travelled alone. |
| `0` | Passenger travelled with at least one family member. |

These additional features provide the model with more information about each passenger.

### Step 7 – Exploratory Data Analysis (EDA)

EDA stands for **Exploratory Data Analysis**.

It means analysing and visualizing data to understand patterns and relationships before training a machine learning model.

The notebook uses Matplotlib and Seaborn to create visualizations.

The following analyses are performed:

**1. Survival Count**

Compares the number of passengers who survived with the number who did not survive.

**2. Survival by Gender**

Compares survival outcomes between male and female passengers.

**3. Survival by Passenger Class**

Examines survival patterns among first-class, second-class, and third-class passengers.

**4. Age Distribution**

Displays the distribution of passenger ages.

**5. Family Size and Survival**

Examines the relationship between family size and survival outcomes.

**6. Correlation Heatmap**

Displays relationships between numerical features using correlation values.

EDA helps us understand the dataset and identify useful patterns before building the model.

### Step 8 – Remove Unnecessary Columns

Some columns are not used as model features.

The notebook removes:

- `Name`
- `Ticket`
- `PassengerId`

The `Cabin` column was already removed during data cleaning.

These columns are excluded from the final model input.

### Step 9 – Separate Features and Target

The dataset is divided into two parts.

**X – Features**

Contains the passenger information used by the model to make predictions.

Examples:
- Age
- Sex
- Passenger class
- Fare
- Family size

**y – Target**

Contains the actual survival information from the `Survived` column.

In simple terms:

- `X` contains the information used to make a prediction.
- `y` contains the correct answer that the model needs to learn.

### Step 10 – Split the Dataset

The dataset is divided into training data and testing data using `train_test_split()`.

The notebook uses an 80:20 split.

| Dataset | Percentage | Purpose |
|---|---:|---|
| Training data | 80% | Used to train the machine learning model. |
| Testing data | 20% | Used to evaluate the trained model. |

The notebook also uses `random_state=42` to make the data split reproducible.

### Step 11 – Feature Scaling

The notebook uses `StandardScaler` to standardize the feature values.

Different features have different numerical ranges.

For example:
- Age may range from a few years to many years.
- Fare may contain values much larger than age.
- Family size usually contains smaller numbers.

StandardScaler transforms numerical features to make their scales more comparable.

The notebook uses:

- `fit_transform()` on the training data.
- `transform()` on the testing data.

The scaler learns the required scaling information from the training data and applies the same transformation to the testing data.

### Step 12 – Build and Train the Model

The project uses **Logistic Regression**.

Logistic Regression is a supervised machine learning algorithm commonly used for binary classification problems.

In this project, it learns patterns between passenger information and survival outcomes.

The model is created using:

```python
log_reg = LogisticRegression(max_iter=1000)
```

The model is trained using:

```python
log_reg.fit(X_train, y_train)
```

During training, the model learns from the passenger records and their known survival outcomes.

### Step 13 – Make Predictions

After training, the model is used to predict the survival outcomes of the testing data.

```python
y_pred = log_reg.predict(X_test)
```

The predicted results are compared with the actual survival outcomes to evaluate model performance.

### Step 14 – Evaluate Model Performance

The notebook uses accuracy, a confusion matrix, and a classification report to evaluate the model.

**1. Accuracy**

Accuracy measures the percentage of predictions that the model classified correctly.

The notebook achieves approximately:

**79.89% Accuracy**

This means that approximately 80 out of every 100 test predictions were correct.

**2. Confusion Matrix**

The notebook produces the following confusion matrix:

```text
[[89 16]
 [20 54]]
```

The rows represent actual classes, while the columns represent predicted classes.

| | Predicted: Did Not Survive | Predicted: Survived |
|---|---:|---:|
| **Actual: Did Not Survive** | 89 | 16 |
| **Actual: Survived** | 20 | 54 |

Explanation:

- 89 passengers were correctly predicted as not surviving.
- 54 passengers were correctly predicted as surviving.
- 16 passengers were predicted to survive but actually did not survive.
- 20 passengers were predicted not to survive but actually survived.

**3. Classification Report**

The classification report provides additional performance measurements.

| Metric | Explanation |
|---|---|
| Precision | Of the passengers predicted to belong to a class, how many actually belonged to that class? |
| Recall | Of the passengers who actually belonged to a class, how many did the model correctly identify? |
| F1-score | A combined measurement of precision and recall. |
| Support | Number of actual test records belonging to that class. |

Approximate results from the notebook:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| `0` – Did Not Survive | 0.82 | 0.85 | 0.83 | 105 |
| `1` – Survived | 0.77 | 0.73 | 0.75 | 74 |

These results describe the model's performance on the notebook's test dataset.

### Step 15 – Save the Trained Model

After training, the notebook saves the model and scaler using `joblib`.

Two files are created:

| File | Purpose |
|---|---|
| `titanic_model.pkl` | Stores the trained Logistic Regression model. |
| `scaler.pkl` | Stores the fitted StandardScaler. |

Saving these files allows us to reuse the trained model later without training it again.

### Step 16 – Predict for a New Passenger

Finally, the notebook demonstrates how to make a prediction for a new passenger.

The example passenger has the following information:

| Feature | Value |
|---|---:|
| `Pclass` | 3 |
| `Age` | 25 |
| `SibSp` | 0 |
| `Parch` | 0 |
| `Fare` | 7.25 |
| `Sex_male` | 1 |
| `Embarked_Q` | 0 |
| `Embarked_S` | 1 |
| `FamilySize` | 1 |
| `IsAlone` | 1 |

The passenger's information is first transformed using the saved scaler.

Then the transformed data is passed to the trained model.

The model returns:

```text
[0]
```

**Prediction: The passenger is predicted not to survive.**

Remember that this is only a prediction made by the machine learning model.

## 6. Model Performance Summary

| Evaluation Metric | Result |
|---|---:|
| Algorithm | Logistic Regression |
| Training data | 80% |
| Testing data | 20% |
| Accuracy | 79.89% |
| Maximum iterations | 1000 |

The model achieves approximately 79.89% accuracy on the testing data.

## 7. Project Folder Structure

The project folder can be organized as follows:

```text
Titanic-Survival-Prediction/
│
├── titanic.ipynb
│
├── README.md
│
├── titanic_model.pkl
│
└── scaler.pkl
```

**File descriptions:**

- `titanic.ipynb` – Main notebook containing data analysis, preprocessing, model training, evaluation, and prediction.
- `README.md` – Documentation explaining the project.
- `titanic_model.pkl` – Saved trained machine learning model.
- `scaler.pkl` – Saved feature scaler.

The `.pkl` files are generated when the notebook's model-saving cells are executed.

## 8. Installation and Requirements

Make sure Python is installed on your computer.

Install the required libraries using:

```bash
pip install pandas numpy seaborn matplotlib kagglehub scikit-learn joblib jupyter
```

### Required Libraries

- pandas
- numpy
- seaborn
- matplotlib
- kagglehub
- scikit-learn
- joblib
- Jupyter Notebook

## 9. How to Run the Project

Follow these steps to run the project on your computer.

**Step 1:** Download or clone the project repository.

**Step 2:** Open a terminal or command prompt inside the project folder.

**Step 3:** Install the required Python libraries.

```bash
pip install pandas numpy seaborn matplotlib kagglehub scikit-learn joblib jupyter
```

**Step 4:** Start Jupyter Notebook.

```bash
jupyter notebook
```

**Step 5:** Open the `titanic.ipynb` file.

**Step 6:** Run the notebook cells from top to bottom.

The notebook downloads the dataset from Kaggle using `kagglehub`. If Kaggle requests authentication or dataset access, configure your Kaggle credentials and follow the dataset page's access requirements.

## 10. Learning Outcomes

Through this project, I gained practical experience in:

- Loading datasets using Pandas.
- Understanding dataset rows and columns.
- Identifying and handling missing values.
- Cleaning and preparing data.
- Converting categorical values into numerical values.
- Creating new features using feature engineering.
- Performing Exploratory Data Analysis.
- Creating visualizations using Matplotlib and Seaborn.
- Splitting datasets into training and testing sets.
- Performing feature scaling.
- Building a Logistic Regression model.
- Making predictions using machine learning.
- Evaluating model performance.
- Saving and loading machine learning models using Joblib.

## 11. Future Improvements

This project can be improved further by:

- Using a Scikit-learn Pipeline for preprocessing and model training.
- Ensuring all preprocessing steps are fitted using training data only to avoid data leakage.
- Comparing Logistic Regression with other classification algorithms.
- Applying cross-validation for more reliable model evaluation.
- Performing hyperparameter tuning to improve model performance.
- Developing a simple web application where users can enter passenger information and receive a prediction.

## 12. Dataset Source and Attribution

The dataset used in this project is available on Kaggle:

[Titanic Dataset – Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset)

Please check the dataset page for its license and terms of use before redistributing the dataset.

---

**Project Type:** Machine Learning – Binary Classification  
**Programming Language:** Python  
**Algorithm:** Logistic Regression  
**Target Column:** `Survived`  
**Development Environment:** Jupyter Notebook
