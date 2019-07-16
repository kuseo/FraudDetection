#%% Import modules
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold
import warnings
import os
warnings.simplefilter("ignore")


#%% Read datas
test_identity = pd.read_csv('./data/test_identity.csv', index_col="TransactionID")
test_transaction = pd.read_csv('./data/test_transaction.csv', index_col="TransactionID")
train_identity = pd.read_csv('./data/train_identity.csv', index_col="TransactionID")
train_transaction = pd.read_csv('./data/train_transaction.csv', index_col="TransactionID")
ss = pd.read_csv('./data/sample_submission.csv', index_col="TransactionID")
# sanity check
assert test_identity.shape==(141907, 40)
assert test_transaction.shape==(506691, 392)
assert train_identity.shape==(144233, 40)
assert train_transaction.shape==(590540, 393)
assert ss.shape==(506691, 1)

#%% Merge identity with transaction
X_test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
# sanity check
assert train.shape==(590540, 433)
assert X_test.shape==(506691, 432)

#%% Create train x y pair
X_train = train.drop("isFraud", axis=1) # deep copy and drop
Y_train = train["isFraud"].copy(deep=True)

#%% fill n/a
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)
assert X_train.shape==(590540, 432)
assert X_test.shape==(506691, 432)

del train, test_identity, test_transaction, train_identity, train_transaction

#%% Create catboost data pool
# catboost automatically transform categorical features into encoded format
CAT_FEATURES = list(X_train.select_dtypes("object").columns)

train_dataset = Pool(data=X_train, label=Y_train, cat_features=CAT_FEATURES)
test_dataset = Pool(data=X_test, cat_features=CAT_FEATURES)

#%% train model
model = CatBoostClassifier(iterations=1000, task_type="GPU")
model.fit(train_dataset, verbose=True)

#%% save model and predict
if not os.path.exists("./result/"):
    os.makedirs("./result/")
model.save_model("./result/model_catboost.json", format="json")
test_pred = model.predict_proba(test_dataset, verbose=True)[:,1]
ss["isFraud"] = test_pred

ss.to_csv("./result/submit_catboost.csv")

#%% feature importance
import matplotlib.pylab as plt
fi = pd.DataFrame(index=model.feature_names_)
fi['importance'] = model.feature_importances_
fi.loc[fi['importance'] > 0.1].sort_values('importance').plot(kind='barh', figsize=(15, 25), title='Feature Importance')
plt.savefig("./result/feature_importance.png")
plt.show()