# %% read dataframe from part1
import pandas as pd

df = pd.read_pickle("data.pkl")


# %% select some yes/no columns and convert into booleans
rfs = [col for col in df.columns if col.startswith("rf_")]
css = [col for col in df.columns if col.startswith("cs_")]
armed = [
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
    "othrweap",
]

x = df[rfs + css + armed]
x
# %%
x = x == "YES"
x

# %% create label for the dataset, then remove the weapons columns
y = (
    x["contrabn"]
    | x["pistol"]
    | x["riflshot"]
    | x["asltweap"]
    | x["knifcuti"]
    | x["machgun"]
    | x["othrweap"]
)
x = x.drop(columns=armed)

# %% grab some number and category feats
num_cols = ["age", "height", "weight"]
cat_cols = ["race", "city", "build"]

x[num_cols] = df[num_cols]
x[cat_cols] = df[cat_cols]


# %% split into training / testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)


# %% massage the categorical columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

enc = OneHotEncoder(handle_unknown="ignore")
ct = ColumnTransformer(
    [
        ("ohe", enc, cat_cols),
    ],
    remainder="passthrough",
)

x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)


# %%
def evaluate(clf, y_train, pred_train, y_test, pred_test):
    print(clf.__class__.__name__)
    print("..Training Result:")
    print(f"....acc: {accuracy_score(y_train, pred_train)}")
    print(f"....precision: {precision_score(y_train, pred_train)}")
    print(f"....recall: {recall_score(y_train, pred_train)}")
    print(f"....f1: {f1_score(y_train, pred_train)}")
    print("..Testing Result:")
    print(f"....acc: {accuracy_score(y_test, pred_test)}")
    print(f"....precision: {precision_score(y_test, pred_test)}")
    print(f"....recall: {recall_score(y_test, pred_test)}")
    print(f"....f1: {f1_score(y_test, pred_test)}")


# %% apply decision tree classifier and check results
# (it would be nice to make all the print statements into a reusable function)
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

evaluate(clf, y_train, pred_train, y_test, pred_test)


# %% plot the decision tree and look for important features
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(18, 12))
plot_tree(clf, 
    filled=True, 
    max_depth=3, 
    feature_names=ct.get_feature_names(), 
    fontsize=10
)


# %% apply logistic regression classifier and check results
from sklearn.linear_model import LogisticRegression


clf = LogisticRegression()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

evaluate(clf, y_train, pred_train, y_test, pred_test)


# %% look for important features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
sns.barplot(y=ct.get_feature_names(), x=clf.coef_[0])


# %% apply naive bayes classifier and check results
from sklearn.naive_bayes import MultinomialNB


clf = MultinomialNB()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)

evaluate(clf, y_train, pred_train, y_test, pred_test)


# %% look for important features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
sns.barplot(y=ct.get_feature_names(), x=clf.feature_log_prob_[0])


# %%
