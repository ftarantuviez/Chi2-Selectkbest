import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2, SelectKBest
from PIL import Image

import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title='Feature Selection with Statistical Methods', page_icon="./f.png")
st.title('Feature Selection with Statistical Methods')
st.subheader('By [Francisco Tarantuviez](https://www.linkedin.com/in/francisco-tarantuviez-54a2881ab/) -- [Other Projects](https://franciscot.dev/portfolio)')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.write('---')
st.write("""
The idea of this app is compare the results of prediction between a model which the data was previously transformed via some statistical methods against the results of the same model but using the original data.

The project consist in predict if the patient has cancer or if does not have cancer.The data that we are gonna use is provided by Scikit-learn in their package called: "load_breast_cancer". \n
The specific statistical techiniques that we will use are:
* **Chi2**
* **SelectKBest**

(both are provided by scikit-learn package)
""")

st.write(""" 
## Loading Data

As previously was said, we are gonna use the dataset given by scikit-learn which give us 30 features to fit the model, and another column which represents the binary target: 1 if the patient has cancer, 0 if has not.

```
bc_data = load_breast_cancer()

# separate into features and target
bc_features = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
bc_classes = pd.DataFrame(bc_data.target, columns=['IsMalignant'])
bc_X = np.array(bc_features)
bc_y = np.array(bc_classes).T[0]
```

""")
bc_data = load_breast_cancer()
bc_features = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
bc_classes = pd.DataFrame(bc_data.target, columns=['IsMalignant'])
bc_X = np.array(bc_features)
bc_y = np.array(bc_classes).T[0]

st.write("#### Features")
st.dataframe(bc_features)
st.write("There are {} rows and {} columns".format(bc_features.shape[0], bc_features.shape[1]))

st.write(""" 
---
## Why feature selection?

* Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
* Improves Accuracy: Less misleading data means modeling accuracy improves.
* Reduces Training Time: Less data means that algorithms train faster

If you wanna know more about this topic, I highly recommend you visit the resources I dropped at the end of the page.

### **Chi2**

Pearson's chi-squared test is used to determine whether there is a statistically significant difference between the expected frequencies and the observed frequencies in one or more categories of a contingency table.

This technique only works when the test statistic is chi-squared distributed under the null hypothesis. Mathematically is represented as follow:
""")
st.latex("""
x^{2} = \sum_{}\\frac{(O_{i} - E_{i})}{E_{i}}
""")
st.write("""
&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; $x^{2}$ = chi squared

&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; $O_{i}$ = observed value

&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; $E_{i}$ = expected value
""")

st.write("""
### **Select K Best**

Select features according to the k highest scores. It basically select the best K-features given by the method you pass to it. In this case we use Chi2 to determine the significance level of each feature in the dataset. Then we set an ar bitrary K which return that number of features in relation with Chi2 answer.

We import the libraries:
```
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import chi2, SelectKBest
```
And we intialize the SelectKBest instance passing to it the following parameters: 

* score_func: function which return an array of p-values (chi2 in our project)
* k: the number of features it has to return (15 in this example, you can change from the sidebar)

```
skb = SelectKBest(score_func=chi2, k=15)
skb.fit(bc_X, bc_y)
```


And in the below table, we can see the importance of each of the features. We see that 'worst area' impacts more than twice times the second features (mean area)
""")
st.sidebar.header("Number of features")
k = st.sidebar.slider("K", 1, 30, value=15)

skb = SelectKBest(score_func=chi2, k=k)
skb.fit(bc_X, bc_y)
feature_scores = [(item, score) for item, score in zip(bc_data.feature_names, skb.scores_)]
st.dataframe(pd.DataFrame(feature_scores).rename({0: "Feature", 1: "Score"}, axis=1).sort_values(by="Score", ascending=False).reset_index(drop=True))

st.write(""" 
From all these features we only selected the first 15 features to use:
""")
st.code("""
select_features_kbest = skb.get_support()
feature_names_kbest = bc_data.feature_names[select_features_kbest]
feature_subset_df = bc_features[feature_names_kbest]
bc_SX = np.array(feature_subset_df)
print(bc_SX.shape)
print(feature_names_kbest)

(569, 15)
['mean radius' 'mean texture' 'mean perimeter' 'mean area'
 'mean concavity' 'radius error' 'perimeter error' 'area error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst compactness' 'worst concavity' 'worst concave points']
""")

select_features_kbest = skb.get_support()
feature_names_kbest = bc_data.feature_names[select_features_kbest]
feature_subset_df = bc_features[feature_names_kbest]
bc_SX = np.array(feature_subset_df)

st.write(""" 
Finally, using cross validation technique, we are gonna train two differents models: one will use all the features in the original dataset, and the other just gonna use the features selected by chi2 score.
""")
st.code(""" 
# build logistic regression model
lr = LogisticRegression(max_iter=100000)

# evaluating accuracy for model built on full featureset
full_feat_acc = cross_val_score(lr, bc_X, bc_y, scoring='accuracy', cv=5)
# evaluating accuracy for model built on selected featureset
sel_feat_acc = cross_val_score(lr, bc_SX, bc_y, scoring='accuracy', cv=5)
""")

lr = LogisticRegression(max_iter=100000)
# evaluating accuracy for model built on full featureset
full_feat_acc = pd.Series(cross_val_score(lr, bc_X, bc_y, scoring='accuracy', cv=5))
# evaluating accuracy for model built on selected featureset
sel_feat_acc = pd.Series(cross_val_score(lr, bc_SX, bc_y, scoring='accuracy', cv=5))

df_acc = pd.concat([full_feat_acc, sel_feat_acc], axis=1)
st.line_chart(df_acc)

st.write(""" 
### **Accuracy**
""")
st.dataframe(pd.DataFrame([np.average(full_feat_acc), np.average(sel_feat_acc)]).T.rename({0: "30 features", 1: "{} features".format(k)}, axis=1))

st.write(""" 
## Conclusion

We can conclude that using techniques of feature selection we can get very good metrics of our models without using all the features. This allow us to avoid overfitting and also is less processing for the computer.

### Feature selection resourcees:
* [Feature Selection For Machine Learning in Python](https://machinelearningmastery.com/feature-selection-machine-learning-python/)
* [sklearn.feature_selection.chi2](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html)
* [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
* [sklearn.feature_selection.SelectKBest](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html)

And on Twitter, [svpino](https://twitter.com/svpino) can guide you better than anyone to understand how to start with Machine Learning and AI.
""")

# This app repository

st.write("""
## App repository

[Github](https://github.com/ftarantuviez/chi2_selectkbest)
""")
# / This app repository