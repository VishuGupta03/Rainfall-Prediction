import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('Rainfall.csv')

# Data exploration
print(df.shape)
print(df.info())
print(df.describe().T)
print(df.isnull().sum())
print(df.columns)

# Data preprocessing
df.rename(str.strip, axis='columns', inplace=True)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        val = df[col].mean()
        df[col] = df[col].fillna(val)
print(df.isnull().sum().sum())

# Plot rainfall distribution
plt.pie(df['rainfall'].value_counts().values, labels = df['rainfall'].value_counts().index, autopct='%1.1f%%')
plt.show()

# Group by rainfall
print(df.groupby('rainfall').mean())

# Select features
features = list(df.select_dtypes(include = np.number).columns)
features.remove('day')
print(features)

# Plot feature distributions
plt.subplots(figsize=(15,8))
for i, col in enumerate(features):
    plt.subplot(3,4, i + 1)
    sb.distplot(df[col])
plt.tight_layout()
plt.show()

# Plot feature boxplots
plt.subplots(figsize=(15,8))
for i, col in enumerate(features):
    plt.subplot(3,4, i + 1)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

# Encode target variable
df.replace({'yes':1, 'no':0}, inplace=True)

# Plot correlation heatmap
plt.figure(figsize=(10,10))
sb.heatmap(df.corr() > 0.8, annot=True, cbar=False)
plt.show()

# Drop features
df.drop(['maxtemp', 'mintemp'], axis=1, inplace=True)

# Split data
features = df.drop(['day', 'rainfall'], axis=1)
target = df.rainfall
X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, stratify=target, random_state=2)

# Oversample minority class
ros = RandomOverSampler(sampling_strategy='minority', random_state=22)
X, Y = ros.fit_resample(X_train, Y_train)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# Define models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

# Train and evaluate models
for i in range(3):
    models[i].fit(X, Y)
    print(f'{models[i]} : ')
    train_preds = models[i].predict_proba(X) 
    print('Training Accuracy : ', metrics.roc_auc_score(Y, train_preds[:,1]))
    val_preds = models[i].predict_proba(X_val) 
    print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, val_preds[:,1]))
    print()
    # Plot confusion matrix
    cm = metrics.confusion_matrix(Y_val, models[i].predict(X_val))
    plt.figure(figsize=(10,8))
    sb.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

# Print classification report
print(metrics.classification_report(Y_val, models[2].predict(X_val)))