import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

### try and load the model back
loaded_model = pickle.load(open('model_dev1/models/xgboost.sav', 'rb'))
### load scaler
loaded_scaler = pickle.load(open('model_dev1/models/scaler.sav', 'rb'))

## Creating a new dataframe with the same column names and values
df_test = pd.DataFrame(columns=['occur_date', 'boro', 'precinct', 'jurisdiction_code',
       'statistical_murder_flag', 'perp_age_group', 'perp_sex', 'perp_race', 'vic_age_group', 'vic_sex', 'vic_race'])


## occur_date = 5 (Friday)
## boro = 1 (Brooklyn)
## precinct = 112 (precinct 112)
## jurisdiction_code = 2.0
## statistical_murder_flag = 1 (True)
## perp_age_group = 4 (25-44)
## perp_sex = 1 (Female)
## perp_race = 6 (White)
## vic_age_group = 2 (25-44)
## vic_sex = 0 (Female))
## vic_race = 5 (White))

df_test.loc[0] = [5,1,112,2.0,1,4,1,6,2,0,5]
df_test_scaled = loaded_scaler.transform(df_test)

# Predict on the test set
y_test_pred = loaded_model.predict(df_test_scaled)
# print value of prediction
print(y_test_pred[0])