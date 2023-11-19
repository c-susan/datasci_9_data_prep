import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

## Loading raw data (pickle)
df = pd.read_pickle('model_dev1/data/raw/shooting_data.pkl')

## Getting column names
df.columns

## Cleaning column names 
## Making them all lower case and removing white spaces
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

## Getting data types
df.dtypes 

## drop columns
to_drop = [
    'incident_key',
    'occur_time',
    'x_coord_cd',
    'y_coord_cd',
    'latitude',
    'longitude',
    'lon_lat',
    'loc_of_occur_desc',
    'loc_classfctn_desc',
    'location_desc']
df.drop(to_drop, axis=1, inplace=True, errors='ignore')

#######################

## Changing occur_date column from mm/dd/yyyy format to day of the week 
df['occur_date'] = pd.to_datetime(df['occur_date'])
df['occur_date'] = df['occur_date'].dt.day_name()

# Performing ordinal encoding on occur_date column. 1-Monday, 2-Tuesday....7-Sunday. 
day_to_number = {'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7}
df['occur_date'] = df['occur_date'].replace(day_to_number).astype(int)

# Creating dataframe with mapping
df_mapping_date = pd.DataFrame({'occur_date': list(day_to_number.keys()), 'occur_date_num': list(day_to_number.values())})
df_mapping_date

# Saving mapping to csv
df_mapping_date.to_csv('model_dev1/data/processed/mapping_date.csv', index=False)

############

## Encoding the boro column
enc = OrdinalEncoder()
enc.fit(df[['boro']])
df['boro'] = enc.transform(df[['boro']])

# Creating a dataframe with mapping
df_mapping_boro = pd.DataFrame(enc.categories_[0], columns=['boro'])
df_mapping_boro['boro_ordinal'] = df_mapping_boro.index
df_mapping_boro.head(5)
# save mapping to csv
df_mapping_boro.to_csv('model_dev1/data/processed/mapping_boro.csv', index=False)

############

## Encoding the statistical_murder_flag column
enc = OrdinalEncoder()
enc.fit(df[['statistical_murder_flag']])
df['statistical_murder_flag'] = enc.transform(df[['statistical_murder_flag']])

# Creating a dataframe with mapping
df_mapping_statistical_murder_flag = pd.DataFrame(enc.categories_[0], columns=['statistical_murder_flag'])
df_mapping_statistical_murder_flag['statistical_murder_flag_ordinal'] = df_mapping_statistical_murder_flag.index
df_mapping_statistical_murder_flag.head(5)
# save mapping to csv
df_mapping_statistical_murder_flag.to_csv('model_dev1/data/processed/mapping_statistical_murder_flag.csv', index=False)

############

## Encoding the perp_age_group column
enc = OrdinalEncoder()
enc.fit(df[['perp_age_group']])
df['perp_age_group'] = enc.transform(df[['perp_age_group']])

# Creating a dataframe with mapping
df_mapping_perp_age_group = pd.DataFrame(enc.categories_[0], columns=['perp_age_group'])
df_mapping_perp_age_group['perp_age_group_ordinal'] = df_mapping_perp_age_group.index
df_mapping_perp_age_group.head(5)
# save mapping to csv
df_mapping_perp_age_group.to_csv('model_dev1/data/processed/mapping_perp_age_group.csv', index=False)

############

## Encoding the perp_sex column
enc = OrdinalEncoder()
enc.fit(df[['perp_sex']])
df['perp_sex'] = enc.transform(df[['perp_sex']])

# Creating a dataframe with mapping
df_mapping_perp_sex = pd.DataFrame(enc.categories_[0], columns=['perp_sex'])
df_mapping_perp_sex['perp_sex_ordinal'] = df_mapping_perp_sex.index
df_mapping_perp_sex.head(5)
# save mapping to csv
df_mapping_perp_sex.to_csv('model_dev1/data/processed/mapping_perp_sex.csv', index=False)

############

## Encoding the perp_race column
enc = OrdinalEncoder()
enc.fit(df[['perp_race']])
df['perp_race'] = enc.transform(df[['perp_race']])

# Creating a dataframe with mapping
df_mapping_perp_race = pd.DataFrame(enc.categories_[0], columns=['perp_race'])
df_mapping_perp_race['perp_race_ordinal'] = df_mapping_perp_race.index
df_mapping_perp_race.head(5)
# save mapping to csv
df_mapping_perp_race.to_csv('model_dev1/data/processed/mapping_perp_race.csv', index=False)

############

## Encoding the vic_age_group column
enc = OrdinalEncoder()
enc.fit(df[['vic_age_group']])
df['vic_age_group'] = enc.transform(df[['vic_age_group']])

# Creating a dataframe with mapping
df_mapping_vic_age_group = pd.DataFrame(enc.categories_[0], columns=['vic_age_group'])
df_mapping_vic_age_group['vic_age_group_ordinal'] = df_mapping_vic_age_group.index
df_mapping_vic_age_group.head(5)
# save mapping to csv
df_mapping_vic_age_group.to_csv('model_dev1/data/processed/mapping_vic_age_group.csv', index=False)

############
## Encoding the vic_sex column
enc = OrdinalEncoder()
enc.fit(df[['vic_sex']])
df['vic_sex'] = enc.transform(df[['vic_sex']])

# Creating a dataframe with mapping
df_mapping_vic_sex = pd.DataFrame(enc.categories_[0], columns=['vic_sex'])
df_mapping_vic_sex['vic_sex_ordinal'] = df_mapping_vic_sex.index
df_mapping_vic_sex.head(5)
# save mapping to csv
df_mapping_vic_sex.to_csv('model_dev1/data/processed/mapping_vic_sex.csv', index=False)

############

## Encoding the vic_race column
enc = OrdinalEncoder()
enc.fit(df[['vic_race']])
df['vic_race'] = enc.transform(df[['vic_race']])

# Creating a dataframe with mapping
df_mapping_vic_race = pd.DataFrame(enc.categories_[0], columns=['vic_race'])
df_mapping_vic_race['vic_race_ordinal'] = df_mapping_vic_race.index
df_mapping_vic_race.head(5)
# save mapping to csv
df_mapping_vic_race.to_csv('model_dev1/data/processed/mapping_vic_race.csv', index=False)


#######################

## Saving processed dataset to a csv file to test the model
df.to_csv('model_dev1/data/processed/processed_shooting_data.csv', index=False)