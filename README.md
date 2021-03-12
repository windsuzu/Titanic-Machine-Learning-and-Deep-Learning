# AI in Titanic

[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic) is a competition for Kaggle beginners. You have to use the data (e.g., age, sex, etc.) from the passengers to predict whether they survived or not.

I use `Scikit-learn` as my machine learning framework and `Pytorch` as my deep learning framework in this competition. Then I divide the **ML** and **DL** into two parts: 

The **ML** part shows you how to clean the data, build the model in both `single model` and `ensemble` ways, and do some extra works like **feature engineering, feature selection, model selection**, etc.

The **DL** part shows you how to build a neural network using `Pytorch`, reduce the boilerplate code using `Pytorch-Lightning`, and tune the hyperparameters using `Ray Tune`.

---

- Machine Learning:
  - `pandas`
    - loading data
    - cleaning data
    - correcting data
    - feature engineering
  - `sweetviz` for EDA (Exploratory Data Analysis)
  - `lazypredict` provides an overview of the performance of multiple models
  - `sklearn`
    - single classifier
    - feature selection
    - model selection (hyperparameter tuning)
    - ensembling

> I also post on Kaggle: 
> - [ML Approach (Sklearn + Pandas + Sweetviz + LazyPredict + Feature Engineering + Feature Selection + Model Selection + Model Ensembling)](https://www.kaggle.com/windsuzu/sklearn-eda-lazypredict-feat-engineering-ensemble#ML-Approach-(Sklearn-+-Pandas-+-Sweetviz-+-LazyPredict-+-Feature-Engineering-+-Feature-Selection-+-Model-Selection-+-Model-Ensembling))

---

- Deep Learning
  - `Pytorch`
    - also using pandas for data preprocessing
    - Pytorch Dataset, DataLoader (collate_fn)
    - building network and training
    - save and load checkpoints
    - predicting
  - `Pytorch-Lightning`
    - pl.LightningDataModule
    - pl.LightningModule
    - pl.Trainer
    - save and load checkpoints
    - logging with Tensorboard
  - `Ray Tune` with `Pytorch-Lightning`
    - hyperparameter search space
    - schedulers for early stopping
    - tuning report callbacks
    - tunning
    - save and load checkpoints
    - logging with Tensorboard

> I also post on Kaggle:
> - [Beyond 77% Pytorch + Lightning + Ray Tune](https://www.kaggle.com/windsuzu/beyond-77-pytorch-lightning-ray-tune)

## Table of Contents

* [AI in Titanic](#ai-in-titanic)
  * [Table of Contents](#table-of-contents)
  * [Machine Learning Core Code](#machine-learning-core-code)
    * [Data Preprocessing](#data-preprocessing)
      * [EDA using sweetviz](#eda-using-sweetviz)
      * [Data Cleaning, Correcting, Completing](#data-cleaning-correcting-completing)
      * [Feature Engineering](#feature-engineering)
    * [Benchmarking and Finding Useful Models](#benchmarking-and-finding-useful-models)
    * [Single Models](#single-models)
      * [Feature Selection](#feature-selection)
      * [Fine-Tuning (Model Selection)](#fine-tuning-model-selection)
      * [Training](#training)
      * [Predicting](#predicting)
    * [Ensembling](#ensembling)
      * [Fine-Tuning Multiple Models](#fine-tuning-multiple-models)
      * [Voting Ensembling](#voting-ensembling)
        * [Training](#training-1)
        * [Predicting](#predicting-1)
  * [Deep Learning Core Code](#deep-learning-core-code)
    * [](#)

## Machine Learning Core Code

> ❤🧡💛💚💙💜🤎🖤
> - **FULL CODE AND EXPLANATION IS HERE**: [machine-learning.ipynb](machine-learning.ipynb)

---

### Data Preprocessing

#### EDA using sweetviz

``` py
import sweetviz as sv
report = sv.compare([train_data, "Train"], [test_data, "Test"], target_feat="Survived", pairwise_analysis="on")
report.show_notebook()
```

#### Data Cleaning, Correcting, Completing

``` py
test_ids = test_data["PassengerId"]

dataset["Age"].fillna(dataset["Age"].median(), inplace=True)
dataset["Embarked"].fillna(dataset["Embarked"].mode()[0], inplace=True)
dataset["Fare"].fillna(dataset["Fare"].median(), inplace = True)

drop_column = ['PassengerId', 'Cabin', 'Ticket']
dataset.drop(drop_column, axis=1, inplace=True)
```

#### Feature Engineering

``` py
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['IsAlone'] = 1  #initialize to 1 = is alone
dataset['IsAlone'].loc[dataset['FamilySize'] > 1] = 0  # now update to no if family size is greatethan 
# Get "Mr", "Miss", "Mrs", and many titles from the name column.

dataset['Title'] = dataset["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
title_counts = dataset['Title'].value_counts() < 10
dataset["Title"] = dataset["Title"].apply(lambda x: "Misc" if title_counts.loc[x] == True else x

# Divide the `Fare` into 4 intervals with similar quantities.
dataset['FareBin'] = pd.qcut(dataset['Fare'], 4, labels=False

# Divide the `Age` into 4 discrete intervals according to its value.
dataset['AgeBin'] = pd.cut(dataset['Age'], [0, 25, 50, 75, 100], labels=False

dataset.drop(columns=["Name", "Age", "Fare"], inplace=True)

train_data = pd.get_dummies(train_data)
test_data = pd.get_dummies(test_data)
```

---

### Benchmarking and Finding Useful Models

``` py
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

models, predictions = clf.fit(X_train, X_test, y_train, y_test)
```

---

### Single Models

``` py
from sklearn.linear_model import RidgeClassifier
```

#### Feature Selection

``` py
from sklearn.feature_selection import RFECV

one_clf = RidgeClassifier()
selector = RFECV(one_clf)
selector.fit(X, y)

new_features = train_data.columns.values[selector.get_support()]
new_features
```

#### Fine-Tuning (Model Selection)

``` py
from sklearn.model_selection import GridSearchCV

tuning_params = {
    "alpha": [0.2, 0.4, 0.6, 0.8, 1],
    "normalize": [False, True],
    "tol": [1e-2, 1e-3, 1e-4, 1e-5],
    "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    }

search = GridSearchCV(one_clf, tuning_params, "accuracy")
search.fit(X[new_features], y)
search.best_params_
```

#### Training

``` py
one_clf = RidgeClassifier(**search.best_params_).fit(X[new_features], y)
```

#### Predicting

``` py
predictions = one_clf.predict(test_data[new_features])
output = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})
output.to_csv(output_dir / "single_model_submission.csv", index=False)
```

---

### Ensembling

``` py
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
```

#### Fine-Tuning Multiple Models

``` py
estimators=[
        ("xgb", XGBClassifier()),
        ("lr", LogisticRegression()),
        ("rc", RidgeClassifier()),
        ("rccv", RidgeClassifierCV()),
        ("lda", LinearDiscriminantAnalysis()),
        ("cccv", CalibratedClassifierCV()),
        ("svc", SVC()),
        ("lsvc", LinearSVC()),
        ("nc", NearestCentroid()),
        ("ada", AdaBoostClassifier())
    ]

multiple_params = [
    {
        "learning_rate": grid_lr,
        "max_depth": grid_max_depth,
        "n_estimators": grid_n_estimator,
    },
    {
        "C": grid_C,
        "penalty": ["l1", "l2", "elasticnet"],
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
    },
    {
        "alpha": grid_ratio,
        "normalize": grid_bool,
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
    },
    {
        "fit_intercept": grid_bool,
        "normalize": grid_bool,
    },
    {
        "solver": ["svd", "lsqr", "eigen"],
    },
    {
        "method": ["sigmoid", "isotonic"],
    },
    {
        "C": grid_C,
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": grid_max_depth,
        "gamma": ["scale", "auto"],
    },
    {
        "penalty": ["l1", "l2"],
        "loss": ["hinge", "squared_hinge"],
        "C": grid_C,
        "fit_intercept": grid_bool,
    },
    {},
    {
        "n_estimators": grid_n_estimator,
        "learning_rate": grid_lr,
        "algorithm": ["SAMME", "SAMME.R"],
    },
]

for (algo, clf), params in zip(estimators, multiple_params):
    search = GridSearchCV(clf, params)
    search.fit(X[new_features], y)

    print(algo, ":", search.best_params_)
    clf.set_params(**search.best_params_)
```

#### Voting Ensembling 

##### Training

``` py
vote_clf = VotingClassifier(
    estimators=estimators,
    voting='hard'
)

vote_clf.fit(X[new_features], y)
```

##### Predicting

``` py
predictions = vote_clf.predict(test_data[new_features])
output = pd.DataFrame({'PassengerId': test_ids, 'Survived': predictions})
output.to_csv(output_dir / 'ensemble_submission.csv', index=False)
```
---

## Deep Learning Core Code

> ❤🧡💛💚💙💜🤎🖤
> - **FULL CODE AND EXPLANATION IS HERE**: [Deep Learning](deep-learning.ipynb)

### 

