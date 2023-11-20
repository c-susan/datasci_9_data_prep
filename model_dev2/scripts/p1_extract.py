import pandas as pd 

## Loading/Extracting Data

# Landing Page: https://data.ny.gov/Energy-Environment/Diesel-Retail-Price-Weekly-Average-by-Region-Begin/dtfv-pchi
# Data Download Link: 
datalink = 'https://data.ny.gov/api/views/dtfv-pchi/rows.csv?accessType=DOWNLOAD'

df = pd.read_csv(datalink)
df.shape
df.sample(5)

df.columns


## Saving data as a csv to model_dev1/data/raw folder
df.to_csv('model_dev2/data/raw/diesel_prices.csv', index=False)

## Saving as pickle to model_dev1/data/raw folder folder
df.to_pickle('model_dev2/data/raw/diesel_prices.pkl')
