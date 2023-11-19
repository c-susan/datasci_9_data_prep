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
