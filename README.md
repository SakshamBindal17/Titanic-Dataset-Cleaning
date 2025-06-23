# Titanic Dataset: Data Cleaning & Preprocessing

This project covers the step-by-step cleaning and preprocessing of the Titanic dataset for machine learning tasks. The goal is to prepare the data for further analysis and modeling by handling missing values, encoding categorical features, scaling numerical data, and saving the cleaned dataset.

---

## **Step 1: Import and Explore the Dataset**

We start by loading the dataset and examining its structure.

import pandas as pd

df = pd.read_csv('Titanic-Dataset.csv')
print(df.head())
print(df.info())


- **Purpose:** Understand the data types, column names, and check for missing values.

---

## **Step 2: Handle Missing Values**

### **2.1 Check Missing Values**
print(df.isnull().sum())


**Findings:**
- `Age` and `Embarked` have missing values.
- `Cabin` has a large number of missing values.

### **2.2 Fill Missing Values**

- **Age:** Fill missing values with the median age.
    ```
    df['Age'] = df['Age'].fillna(df['Age'].median())
    ```
- **Embarked:** Fill missing values with the most frequent value (mode).
    ```
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode())
    ```
- **Cabin:** Drop the column due to too many missing values.
    ```
    df = df.drop('Cabin', axis=1)
    ```

---

## **Step 3: Encode Categorical Variables**

### **3.1 Encode 'Sex' Column**

Convert 'male' to 0 and 'female' to 1.

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})


### **3.2 One-Hot Encode 'Embarked' Column**

Use one-hot encoding for the 'Embarked' column, dropping the first category (`C`) to avoid redundancy.

df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


#### **Embarked Encoding Table**

| Embarked_Q | Embarked_S | Original Embarked |
|------------|------------|-------------------|
| 0          | 0          | C                 |
| 0          | 1          | S                 |
| 1          | 0          | Q                 |

---

## **Step 4: Standardize Numerical Features**

Standardize `Age` and `Fare` columns to have mean 0 and standard deviation 1.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


---

## **Step 5: Visualize Outliers (Optional)**

Boxplots help visualize outliers in `Age` and `Fare`.

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')

plt.subplot(1, 2, 2)
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()


- Outliers are visible as dots outside the whiskers.
- For this project, outliers are **kept** in the dataset.

---

## **Step 6: Save the Cleaned Dataset**

Save the cleaned DataFrame to a CSV file for future use.

df.to_csv('Titanic-Dataset-Cleaned.csv', index=False)
---

## **Summary of Preprocessing Steps**

- Loaded and explored the dataset.
- Filled missing values in `Age` and `Embarked`, dropped `Cabin`.
- Encoded categorical columns (`Sex`, `Embarked`).
- Standardized numerical features (`Age`, `Fare`).
- Visualized and decided to keep outliers.
- Saved the cleaned dataset for modeling.

---

*Prepared by Saksham Bindal*

