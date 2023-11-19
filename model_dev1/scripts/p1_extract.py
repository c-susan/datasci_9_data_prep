import pandas as pd 

## Loading/Extracting Data

# Landing Page: https://data.cityofnewyork.us/Public-Safety/NYPD-Shooting-Incident-Data-Historic-/833y-fsy8
# Data Download Link: 
datalink = 'https://data.cityofnewyork.us/api/views/833y-fsy8/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.shape
df.sample(5)

df.columns


df.LOCATION_DESC.isna().sum()


## Saving data as a csv to model_dev1/data/raw folder
df.to_csv('model_dev1/data/raw/shooting_data.csv', index=False)

## Saving as pickle to model_dev1/data/raw folder folder
df.to_pickle('model_dev1/data/raw/shooting_data.pkl')
