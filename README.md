# Predicting-Blood-Donations
Use of machine learning package sci-kit learn to predict if students are likely to donate glood given the chance.
<br />
<br />
This is important because it can give blood drive centers more efficient planning when they are scheduling a mobile blood drive (in which they travel to different universities to ask for blood donations from students). 

### Data detailing individual students and their blood donation status was used to complete the following:
* Identify target/ training variable (student did not donate blood  = 0, student did donate blood  = 0)
  * Compute incidence of these target variables
* Use train_test_split() to split blood tranfusion data in training and testing sets 
```
X_train, X_test, y_train, y_test = train_test_split(
    transfusion.drop(columns='target'),
    transfusion.target,
    test_size=0.25,
    random_state=42,
    stratify=transfusion['target']
)
```
* Use TPOT package to optimize machine learning model (sci-kit learn pipeline)
```
tpot = TPOTClassifier(
    generations=5,
    population_size=20,
    verbosity=2,
    scoring='roc_auc',
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
```

* Correct for high variance using log normalization
```
# Import numpy
import numpy as np

# Copy X_train and X_test into X_train_normed and X_test_normed
X_train_normed, X_test_normed = X_train.copy(), X_test.copy()

# Specify which column to normalize
col_to_normalize = 'Monetary (c.c. blood)'


# Log normalization
for df_ in [X_train_normed, X_test_normed]:
    # Add log normalized column
    df_['monetary_log'] = np.log(df_[col_to_normalize])
    # Drop the original column
    df_.drop(columns=col_to_normalize, inplace=True)
```
