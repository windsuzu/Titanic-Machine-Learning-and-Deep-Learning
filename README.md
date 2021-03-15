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
    * [Dataset](#dataset)
    * [Pytorch](#pytorch)
      * [Custom Dataset](#custom-dataset)
      * [DataLoader with custom collate_fn](#dataloader-with-custom-collate_fn)
      * [Network](#network)
      * [Train](#train)
      * [Restore Checkpoint and Predict](#restore-checkpoint-and-predict)
    * [Pytorch Lightning](#pytorch-lightning)
      * [DataModule](#datamodule)
      * [LightningModule](#lightningmodule)
      * [Fit with Trainer](#fit-with-trainer)
      * [Test with Trainer](#test-with-trainer)
    * [Ray Tune](#ray-tune)
      * [Config (Search Space)](#config-search-space)
      * [Scheduler](#scheduler)
      * [Report Callback](#report-callback)
      * [Trainable function](#trainable-function)
      * [Tune](#tune)
      * [Restore Weights from Checkpoint and Predict](#restore-weights-from-checkpoint-and-predict)

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

### Dataset

|     | Survived | Pclass | Sex | SibSp | Parch | Embarked | FamilySize | IsAlone | Title | FareBin | AgeBin |
| --- | -------- | ------ | --- | ----- | ----- | -------- | ---------- | ------- | ----- | ------- | ------ |
| 0   | 0        | 3      | 1   | 1     | 0     | 2        | 2          | 0       | 3     | 0       | 0      |
| 1   | 1        | 1      | 0   | 1     | 0     | 0        | 2          | 0       | 4     | 3       | 1      |
| 2   | 1        | 3      | 0   | 0     | 0     | 2        | 1          | 1       | 2     | 1       | 1      |
| 3   | 1        | 1      | 0   | 1     | 0     | 2        | 2          | 0       | 4     | 3       | 1      |
| 4   | 0        | 3      | 1   | 0     | 0     | 2        | 1          | 1       | 3     | 1       | 1      |
| ... | ...      | ...    | ... | ...   | ...   | ...      | ...        | ...     | ...   | ...     | ...    |
| 886 | 0        | 2      | 1   | 0     | 0     | 2        | 1          | 1       | 1     | 1       | 1      |
| 887 | 1        | 1      | 0   | 0     | 0     | 2        | 1          | 1       | 2     | 2       | 0      |
| 888 | 0        | 3      | 0   | 1     | 2     | 2        | 4          | 0       | 2     | 2       | 1      |
| 889 | 1        | 1      | 1   | 0     | 0     | 0        | 1          | 1       | 3     | 2       | 1      |
| 890 | 0        | 3      | 1   | 0     | 0     | 1        | 1          | 1       | 3     | 0       | 1      |

``` py
y = train_data.pop("Survived")
X = train_data


X_train = X.values
y_train = y.values

X_test = test_data.values
```

### Pytorch

#### Custom Dataset

``` py
# Same as torch.utils.data.TensorDataset
class MyTensorDataset(Dataset):
    def __init__(self, X, y=None):
        if y is None:
            self.data = X
        else:
            self.data = list(zip(X, y))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


train_dataset = MyTensorDataset(X_train, y_train)
test_dataset = MyTensorDataset(X_test)
```

#### DataLoader with custom collate_fn

``` py
train_size = int(len(train_dataset) * 0.8)
val_size = len(train_dataset) - train_size
train_set, val_set = random_split(train_dataset, [train_size, val_size])


def contrived_fn(batch_data):
    """
    Simulate the behavior of the default collate_fn.
    The return values must be tensor type.
    """
    tensor_X = []
    tensor_y = []
    for x, y in batch_data:
        tensor_X.append(x)
        tensor_y.append(y)
    
    return torch.FloatTensor(tensor_X), torch.LongTensor(tensor_y)


train_loader = DataLoader(train_set, batch_size, True, num_workers=0, collate_fn=contrived_fn)
val_loader = DataLoader(val_set, batch_size, num_workers=0, collate_fn=contrived_fn)
test_loader = DataLoader(test_dataset, batch_size, num_workers=0)
```

#### Network

``` py
class Net(nn.Module):
    def __init__(self, feature_size, target_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, target_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = Net(feature_size=feature_size, target_size=target_size)
model
```

#### Train

``` py
def run_step(model, opt, dev, criterion, batch_X, batch_y, training=True):
    batch_X = batch_X.to(dev)
    batch_y = batch_y.to(dev)

    batch_pred = model(batch_X)
    loss = criterion(batch_pred, batch_y)
    acc = (batch_pred.max(1)[1] == batch_y).sum().item()

    if training:
        opt.zero_grad()
        loss.backward()
        opt.step()

    return batch_pred, loss, acc


def run_epoch(model, opt, dev, criterion, data_loader, training=True):
    if training:
        model.train()
    else:
        model.eval()
    
    epoch_loss = 0
    epoch_acc = 0

    for batch_X, batch_y in tqdm(data_loader):
        _, step_loss, step_acc = run_step(model, opt, dev, criterion, batch_X, batch_y, training)
        epoch_loss += step_loss
        epoch_acc += step_acc

    epoch_loss = (epoch_loss / len(data_loader)).item()
    epoch_acc = (epoch_acc / len(data_loader.dataset)) * 100
    return epoch_loss, epoch_acc
```

``` py
opt = torch.optim.SGD(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epoch = 50
model.to(dev)

train_loss = []
train_acc = []
eval_loss = []
eval_acc = []

min_loss = np.inf

for i in range(epoch):
    epoch_train_loss, epoch_train_acc = run_epoch(model, opt, dev, criterion, train_loader, training=True)
    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)

    with torch.no_grad():
        epoch_eval_loss, epoch_eval_acc = run_epoch(model, opt, dev, criterion, val_loader, training=False)
        eval_loss.append(epoch_eval_loss)
        eval_acc.append(epoch_eval_acc)

    if epoch_eval_loss < min_loss:
        min_loss = epoch_eval_loss
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

    print(f"Epoch {i+1}: \ntrain=loss: {epoch_train_loss}, acc: {epoch_train_acc}\nvalidation=loss: {epoch_eval_loss}, acc: {epoch_eval_acc}")
```

#### Restore Checkpoint and Predict

``` py
model.load_state_dict(torch.load(ckpt_dir / "model.pt"))
model.eval()

result = []

with torch.no_grad():
    for X_test_batch in test_loader:
        X_test_batch = X_test_batch.to(dev)
        pred = model(X_test_batch.float())
        pred = pred.max(1)[1]
        result.extend(pred.tolist())

submission = pd.DataFrame({'PassengerId': test_ids, 'Survived': result})
submission.to_csv(submission_dir / "torch_submission.csv", index=False)
```

### Pytorch Lightning

#### DataModule

``` py
from torch.utils.data import TensorDataset

class TitanicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir


    def prepare_data(self):
        """
        Download datasets here. Not to assign variables here.
        ie: `prepare_data` is called from a single GPU. Do not use it to assign state (self.x = y).
        """


    def setup(self, stage=None):
        if stage == "fit":
            full_dataset = pd.read_csv(self.data_dir / "train.csv")
            full_dataset = self._data_preprocess(full_dataset)

            y = full_dataset.pop("Survived")
            X = full_dataset
            full_dataset = TensorDataset(torch.Tensor(X.values), torch.Tensor(y.values).long())

            train_size = int(len(full_dataset) * 0.8)
            val_size = len(full_dataset) - train_size
            self.train_set, self.val_set = random_split(full_dataset, [train_size, val_size])

        if stage == "test":
            test_dataset = pd.read_csv(self.data_dir / "test.csv")
            self.test_ids = test_dataset["PassengerId"]

            test_dataset = self._data_preprocess(test_dataset)
            self.test_set = TensorDataset(torch.Tensor(test_dataset.values))
            

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=0, pin_memory=True)


    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=0, pin_memory=True)


    def _data_preprocess(self, pd_dataset):
        pd_dataset["Age"].fillna(pd_dataset["Age"].median(), inplace=True)
        pd_dataset["Embarked"].fillna(pd_dataset["Embarked"].mode()[0], inplace=True)
        pd_dataset["Fare"].fillna(pd_dataset["Fare"].median(), inplace = True)
        # Data Cleaning
        drop_column = ['PassengerId', 'Cabin', 'Ticket']
        pd_dataset.drop(drop_column, axis=1, inplace=True)
        # Data Creating (Feature Engineering)
        pd_dataset['FamilySize'] = pd_dataset['SibSp'] + pd_dataset['Parch'] + 1
        pd_dataset['IsAlone'] = 1  #initialize to 1 = is alone
        pd_dataset['IsAlone'].loc[pd_dataset['FamilySize'] > 1] = 0  # now update to no if family size is greater than 1
        pd_dataset['Title'] = pd_dataset["Name"].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
        title_counts = pd_dataset['Title'].value_counts() < 10
        pd_dataset["Title"] = pd_dataset["Title"].apply(lambda x: "Misc" if title_counts.loc[x] == True else x)
        ## Divide the `Fare` into 4 intervals with similar quantities.
        pd_dataset['FareBin'] = pd.qcut(pd_dataset['Fare'], 4, labels=False)
        ## Divide the `Age` into 4 discrete intervals according to its value.
        pd_dataset['AgeBin'] = pd.cut(pd_dataset['Age'], [0, 25, 50, 75, 100], labels=False)
        ## Drop these columns since we have these features in the discrete version.
        pd_dataset.drop(columns=["Name", "Age", "Fare"], inplace=True)

        label_encoder = LabelEncoder()
        pd_dataset["Sex"] = label_encoder.fit_transform(pd_dataset["Sex"])
        pd_dataset["Title"] = label_encoder.fit_transform(pd_dataset["Title"])
        pd_dataset["Embarked"] = label_encoder.fit_transform(pd_dataset["Embarked"])

        return pd_dataset
```

``` py
dm = TitanicDataModule(data_dir, 64)

# Test whether the data module works by setting it manually.
dm.setup("fit")
first_batch, *_ = dm.train_dataloader()
print(first_batch[0].shape)

dm.setup("test")
first_batch, *_ = dm.test_dataloader()
print(first_batch[0].shape)
```

#### LightningModule

``` py
class TitanicModel(pl.LightningModule):
    def __init__(self, feature_size, hidden_size, target_size, dropout, lr):
        # super(TitanicModel, self).__init__()
        super().__init__()
        self.fc1 = nn.Linear(feature_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, target_size)
        self.dropout = nn.Dropout(dropout)
        self.lr = lr
        self.save_hyperparameters()


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = F.cross_entropy(pred, y)
        acc = self.accuracy(pred, y)
        
        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger.
        # detail: https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#log
        self.log_dict({"train_loss": loss, "train_acc": acc}, on_step=False, on_epoch=True, prog_bar=True)

        # must return loss for continued training (ie: grad and step)
        return {"loss": loss, "pred": pred}


    def training_epoch_end(self, output):
        """
        If you need to do something with all the outputs of each training_step, override training_epoch_end yourself.
        """


    def validation_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = F.cross_entropy(pred, y)
        acc = self.accuracy(pred, y)
        
        self.log_dict({"val_loss": loss, "val_acc": acc}, on_step=False, on_epoch=True, prog_bar=True)

        # return when you want to do something at validation_epoch_end()
        # return pred
    

    def validation_epoch_end(self, output):
        """
        If you need to do something with all the outputs of each validation_step, override validation_epoch_end.
        """        


    def test_step(self, batch, batch_idx):
        X = batch[0]
        pred = self(X)
        return pred

    
    def test_epoch_end(self, output):
        pred = torch.cat([batch.max(1)[1] for batch in output])
        self.test_result = pred.detach()

    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


    def accuracy(self, batch_pred, batch_y):
        correct = (batch_pred.max(1)[1] == batch_y).sum().detach()
        accuracy = correct / len(batch_y)
        return torch.tensor(accuracy)


model = TitanicModel(feature_size=10, 
                     hidden_size=512, 
                     target_size=2,
                     dropout=0.3, 
                     lr=0.1)
```

#### Fit with Trainer

``` py
# define model checkpoint
checkpoint = pl.callbacks.ModelCheckpoint(dirpath=ckpt_dir,  # path for saving checkpoints
                                          filename="{epoch}-{val_loss:.3f}",
                                          monitor="val_loss",
                                          mode="min")


trainer = pl.Trainer(fast_dev_run=False,
                     max_epochs=100,
                     default_root_dir=log_dir,  # path for saving logs
                     weights_save_path=ckpt_dir,  # path for saving checkpoints
                     gpus=1,
                     callbacks=[checkpoint])

trainer.fit(model, dm)
```

#### Test with Trainer

``` py
model.load_from_checkpoint(checkpoint.best_model_path)

trainer.test(model, datamodule=dm)
submission = pd.DataFrame({'PassengerId': dm.test_ids, 'Survived': model.test_result.cpu()})
submission.to_csv(submission_dir / "torch_lightning_submission.csv", index=False)
```

### Ray Tune

#### Config (Search Space)

``` py
config = {
    "batch_size": tune.choice([64, 128, 256]),
    "hidden_size": tune.grid_search([128, 256, 512]),
    "dropout": tune.uniform(0.1, 0.3),
    "lr": tune.loguniform(0.01, 0.1),
}

from torch.utils.data import TensorDataset

class TitanicDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, config):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = config["batch_size"]

    ...
    ...


class TitanicModel(pl.LightningModule):
    def __init__(self, feature_size, target_size, config):
        # super(TitanicModel, self).__init__()
        super().__init__()
        self.fc1 = nn.Linear(feature_size, config["hidden_size"])
        self.fc2 = nn.Linear(config["hidden_size"], target_size)
        self.dropout = nn.Dropout(config["dropout"])
        self.lr = config["lr"]
        self.save_hyperparameters()

    ...
    ...
```

#### Scheduler

``` py
# Example of ASHA Scheduler
scheduler_asha = ASHAScheduler(
    max_t=100,
    grace_period=1,
    reduction_factor=2,
)
```

#### Report Callback

``` py
tune_report_callback = TuneReportCheckpointCallback(
    metrics={
    "val_loss": "val_loss",
    "val_acc": "val_acc",
    },
    filename="ray_ckpt",
    on="validation_end",
)
```

#### Trainable function

``` py
def run_with_tune(config, data_dir=None, feature_size=10, target_size=2, epochs=50, gpus=0, trained_weights=None):
    model = TitanicModel(feature_size, target_size, config)
    dm = TitanicDataModule(data_dir, config)

    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=gpus,
        fast_dev_run=False,
        progress_bar_refresh_rate=0,
        weights_summary=None,
        default_root_dir=log_dir / "ray_logs",  # path for saving logs
        callbacks=[
            tune_report_callback,
        ],
    )

    if not trained_weights:
        trainer.fit(model, dm)

    else:
        model.load_state_dict(trained_weights)
        trainer.test(model, datamodule=dm)
        submission = pd.DataFrame({'PassengerId': dm.test_ids, 'Survived': model.test_result})
        submission.to_csv(submission_dir / "ray_tune_submission.csv", index=False)
```

#### Tune

``` py
reporter = CLIReporter(
    parameter_columns=["batch_size", "hidden_size", "lr"],
    metric_columns=["val_loss", "val_acc", "training_iteration"]
)

result = tune.run(
    tune.with_parameters(
        run_with_tune,
        data_dir=data_dir.absolute(),
        feature_size=10,
        target_size=2,
        epochs=100,
        gpus=1,
        ),
    resources_per_trial={
        "cpu": 1,
        "gpu": 1,
    },
    local_dir=ckpt_dir / "ray_ckpt",  # path for saving checkpoints
    metric="val_loss",
    mode="min",
    config=config,
    num_samples=16,
    scheduler=scheduler_asha,
    progress_reporter=reporter,
    name="tune_titanic_asha",
)
```

#### Restore Weights from Checkpoint and Predict

``` py
trained_weights = torch.load(Path(result.best_checkpoint) / "ray_ckpt")["state_dict"]

run_with_tune(result.best_config, 
              data_dir,
              trained_weights=trained_weights,
              )
```