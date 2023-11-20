import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## Loading raw data (pickle)
df = pd.read_pickle('model_dev2/data/raw/diesel_prices.pkl')

## Getting column names
df.columns

## Cleaning column names 
## Making them all lower case and removing white spaces
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## Getting data types
df.dtypes

## Dropping these columns because too missing values
to_drop = [
    'batavia_average_($/gal)',
    'dutchess_average_($/gal)',
    'elmira_average_($/gal)',
    'glens_falls_average_($/gal)',
    'ithaca_average_($/gal)',
    'kingston_average_($/gal)',
    'watertown_average_($/gal)',
    'white_plains_average_($/gal)']
df.drop(to_drop, axis=1, inplace=True, errors='ignore')

#######################

## Changing date column from object data type to datetime data type
df['date'] = pd.to_datetime(df['date'])
# Changing the values to show only the month and year: yyyy-mm for the ordinal encoding
df['date'] = df['date'].dt.to_period('M')

# Performing ordinal encoding on date column.
enc = OrdinalEncoder()
enc.fit(df[['date']])
df['date'] = enc.transform(df[['date']])

# Creating a dataframe with mapping
df_mapping_date = pd.DataFrame(enc.categories_[0], columns=['date'])
df_mapping_date['date_ordinal'] = df_mapping_date.index
df_mapping_date.head(5)

# Saving mapping to csv
df_mapping_date.to_csv('model_dev2/data/processed/mapping_date.csv', index=False)

############


## Saving processed dataset to a csv file to test the model
df.to_csv('model_dev2/data/processed/processed_diesel_prices.csv', index=False)

