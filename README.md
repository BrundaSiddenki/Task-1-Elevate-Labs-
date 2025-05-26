# Task-1-Elevate-Labs-
Import Libraries:
Common Python libraries for data manipulation, visualization, machine learning, and model persistence (e.g., pandas, numpy, seaborn, matplotlib, sklearn, and joblib) are imported.

Load Dataset:
A CSV dataset of the Titanic is loaded into a pandas DataFrame from the specified file path, and basic exploratory information is displayed:

The data types of columns.
Number of missing values in each column.
Handle Missing Values:

Missing values in the Age column are replaced with the median using SimpleImputer.
Missing values in the Embarked column are replaced with the most frequent value.
The Cabin column, which contains excessive missing values, is dropped.
Convert Categorical Features:

Categorical columns (Sex and Embarked) are converted to numerical values using label encoding with LabelEncoder.
Normalize Numerical Features:

Numerical columns (Age, Fare, SibSp, and Parch) are scaled to have zero mean and unit variance using StandardScaler.
Visualize and Remove Outliers:

Boxplots are created for each numerical column to visually identify outliers.
The interquartile range (IQR) method is applied to remove rows containing outliers for the numerical columns.
Plot Correlation Matrix:

A heatmap is generated to display correlations between variables in the dataset.
Class Balance Visualization:

The distribution of the target variable (Survived) is displayed using a bar chart.
Train-Test Split:

The dataset is split into features (X) and target (y).
The data is divided into training and testing sets (80% training, 20% testing) using train_test_split.
Cross-Validation:

A RandomForestClassifier is initialized and evaluated using 5-fold cross-validation (cross_val_score) to estimate robust model performance.
Hyperparameter Tuning (Grid Search):

A grid search (GridSearchCV) is performed to find the best combination of hyperparameters for the RandomForestClassifier. Hyperparameter options include:
Number of estimators (n_estimators),
Maximum depth of trees (max_depth),
Minimum samples required to split a node (min_samples_split).
Model Training and Evaluation:

The best model from the grid search is trained on the training data and evaluated on the test data.
The accuracy_score and classification_report provide metrics like precision, recall, and F1-score.
A confusion matrix is plotted to evaluate classification performance.
Feature Importance Visualization:

The feature importances (contributions of each feature to the prediction) are extracted from the trained classifier and visualized using a bar chart.
Save Model:

The final trained RandomForestClassifier model is saved to a file (titanic_rf_model.pkl) using joblib for future use.
Output Cleaned Dataset:

The final cleaned dataset is printed and displayed.

What the code does overall:
This code preprocesses the Titanic dataset, performs feature engineering, visualizes data patterns, trains and tunes a RandomForestClassifier model to predict survival, evaluates the model, determines feature importance, and saves the trained model for deployment or later use. It demonstrates a complete supervised machine learning pipeline.
