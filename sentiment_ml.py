import pandas as pd
import numpy as np
import re
import os
csv_file = os.path.join(christosskylogiannis_amazonmobile_path, 'Amazon_Unlocked_Mobile.csv')
df = pd.read_csv(csv_file)

# Sample the data to speed up computation
# Comment out this line to match with lecture
#df = df.sample(frac=0.1, random_state=10)

df.head()

from sklearn.linear_model import LogisticRegression

# Drop missing values
df.dropna(inplace=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)  # Remove symbols/numbers
    text = re.sub(r'\s+', ' ', text)         # Remove extra spaces
    return text

df['Reviews'] = df['Reviews'].apply(clean_text)

# Remove any 'neutral' ratings equal to 3
df = df[df['Rating'] != 3]

# Encode 4s and 5s as 1 (rated positively)
# Encode 1s and 2s as 0 (rated poorly)
df['Positively Rated'] = np.where(df['Rating'] > 3, 1, 0)
df.head(10)

# Most ratings are positive
df['Positively Rated'].mean()

from sklearn.model_selection import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Reviews'],
                                                    df['Positively Rated'],
                                                    random_state=0)

print('X_train first entry:\n\n', X_train.iloc[0])
print('\n\nX_train shape: ', X_train.shape)

"""# CountVectorizer"""

from sklearn.feature_extraction.text import CountVectorizer

# Fit the CountVectorizer to the training data
vect = CountVectorizer(
    stop_words='english',
    min_df=3,
    ngram_range=(1,2)
).fit(X_train)

vect

vect.get_feature_names_out()[::1000]

len(vect.get_feature_names_out())

# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)

X_train_vectorized

X_train_vectorized[0]

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='logloss',
    n_jobs=-1
)

model.fit(X_train_vectorized, y_train)

from sklearn.metrics import roc_auc_score

# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

pred_proba = model.predict_proba(vect.transform(X_test))[:,1]
roc_auc_score(y_test, pred_proba)

print("=== 1-gram model ===")
vect1 = CountVectorizer(
    min_df=5,
    stop_words='english',
    ngram_range=(1,1)
).fit(X_train)

X1_train = vect1.transform(X_train)
X1_test = vect1.transform(X_test)

model1 = LogisticRegression(max_iter=3000)
model1.fit(X1_train, y_train)

auc_1gram = roc_auc_score(y_test, model1.predict(X1_test))
print("AUC (1-gram):", auc_1gram)


print("\n=== 1-2 gram model ===")
vect2 = CountVectorizer(
    min_df=5,
    stop_words='english',
    ngram_range=(1,2)
).fit(X_train)

X2_train = vect2.transform(X_train)
X2_test = vect2.transform(X_test)

model2 = LogisticRegression(max_iter=3000)
model2.fit(X2_train, y_train)

auc_2gram = roc_auc_score(y_test, model2.predict(X2_test))
print("AUC (1-2 gram):", auc_2gram)

"""Η χρήση 2-grams βελτιώνει σημαντικά την απόδοση του μοντέλου (AUC από 0.912 σε 0.947). Αυτό συμβαίνει επειδή οι bigrams αποτυπώνουν σημαντικές φράσεις όπως “not working”, “works great”, “very disappointed”, οι οποίες εκφράζουν ξεκάθαρο συναίσθημα και βοηθούν το μοντέλο να διαχωρίσει θετικές από αρνητικές αξιολογήσεις."""

# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names_out()
)



# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1]
# so the list returned is in order of largest to smallest
importances = model.feature_importances_
sorted_idx = np.argsort(importances)

print("Least important features:\n", feature_names[sorted_idx[:10]])
print("Most important features:\n", feature_names[sorted_idx[-10:]])

"""# Tfidf"""

from sklearn.feature_extraction.text import TfidfVectorizer

# Fit the TfidfVectorizer to the training data specifiying a minimum document frequency of 5
vect = TfidfVectorizer(
    min_df=5,
    stop_words='english',
    ngram_range=(1,2),
    sublinear_tf=True
).fit(X_train)

vect.get_feature_names_out()

X_train_vectorized = vect.transform(X_train)

model = LogisticRegression(max_iter=2000)

model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names_out())


sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()

print('Smallest tfidf:\n{}\n'.format(feature_names[sorted_tfidf_index[:10]]))
print('Largest tfidf: \n{}'.format(feature_names[sorted_tfidf_index[:-11:-1]]))

sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

import matplotlib.pyplot as plt
import seaborn as sns

# 1.feature names
feature_names = np.array(vect.get_feature_names_out())

# 2.2-grams
mask_2grams = np.array([len(f.split()) == 2 for f in feature_names])
feature_names_2g = feature_names[mask_2grams]
coefs_2g = model.coef_[0][mask_2grams]

# 3. Top 20 positive & negative 2-grams
top_pos_idx = np.argsort(coefs_2g)[-20:]
top_neg_idx = np.argsort(coefs_2g)[:20]

top_features = np.concatenate([top_neg_idx, top_pos_idx])
selected_features = feature_names_2g[top_features]

# 4.TF-IDF
X_small = X_train_vectorized[:, mask_2grams][:, top_features].toarray()

# 5.reviews που έχουν σημασία
rows_with_values = np.where(X_small.sum(axis=1) > 0)[0]

# 6.μέχρι 40 από αυτά
rows_to_plot = rows_with_values[:40]

# 7.heatmap
plt.figure(figsize=(14,12))
sns.heatmap(X_small[rows_to_plot], cmap="viridis",
            yticklabels=False,
            xticklabels=selected_features)

plt.title("Heatmap – TF-IDF weights for important 2-grams")
plt.xticks(rotation=90)
plt.show()

# These reviews are treated the same by our current model
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))

"""# n-grams"""

# Fit the CountVectorizer to the training data specifiying a minimum
# document frequency of 5 and extracting 1-grams and 2-grams
vect = CountVectorizer(min_df=5, ngram_range=(1,2)).fit(X_train)

X_train_vectorized = vect.transform(X_train)

len(vect.get_feature_names_out())

model = LogisticRegression(max_iter=3000)

model.fit(X_train_vectorized, y_train)

predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

feature_names = np.array(vect.get_feature_names_out())


sorted_coef_index = model.coef_[0].argsort()

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

# These reviews are now correctly identified
print(model.predict(vect.transform(['not an issue, phone is working',
                                    'an issue, phone is not working'])))
