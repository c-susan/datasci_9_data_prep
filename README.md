# datasci_9_data_prep
HHA507 / Data Science / Assignment 9 / ML Data Prep 

Selecting datasets suitable for a machine learning experiment, with an emphasis on data cleaning, encoding, and transformation steps necessary to prepare the data.

## This repo contains the following: 
+ **datasets** folder: contains the datasets used in this assignment
+ **model_dev1** folder: contains the extraction, transformation, and computing of dataset 1  <a href=https://github.com/c-susan/datasci_9_data_prep/blob/main/datasets/NYPD_Shooting_Incident_Data__Historic_.csv> NYPD_Shooting_Incident_Data__Historic_.csv </a>
+ **model_dev2** folder: contains the extraction, transformation, and computing of dataset 2
<a href=https://github.com/c-susan/datasci_9_data_prep/blob/main/datasets/Diesel_Retail_Price_Weekly_Average_by_Region__Beginning_2007.csv>Diesel_Retail_Price_Weekly_Average_by_Region__Beginning_2007.csv</a>

# Documentation
## Dataset 1
<a href=https://github.com/c-susan/datasci_9_data_prep/blob/main/datasets/NYPD_Shooting_Incident_Data__Historic_.csv>NYPD_Shooting_Incident_Data__Historic_.csv</a>
+ This dataset contains a detailed overview of every shooting incident that took place in New York City from 2006 to the end of the previous calendar year. Each record contains information about the incident, including details about the event, location, time of occurrence, and demographic information of both suspects and victims. As of November 19, 2023, the dataset was last updated on April 27, 2023. Dataset taken from: https://catalog.data.gov/dataset/nypd-shooting-incident-data-historic.
+ The intended machine learning task for this dataset is classification, with column 'vic_sex' (victim sex) being the target variable (X).

**Steps to Cleaning and Transforming Data**
1. The column names were first cleaned to replace spaces with underscores(_) and changed to lowercase.
2. The following columns were dropped with the following script:
```
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
```

3. The date column was changed to a datetime datatype and then changed to their corresponding day names (Monday....Sunday). The days were then coded to an integer value where 1 = Monday....7 = Sunday, ending with the date column changing to an integer datatype. A data dictionary of the column is then mapped to a csv file located in the <a href=https://github.com/c-susan/datasci_9_data_prep/tree/main/model_dev1/data/processed>/model_dev1/data/processed</a> folder.
4. The other columns, boro, statistical_murder_flag, perp_age_group, perp_sex, perp_race, vic_age_group, vic_sex, and vic_race were also ordinally encoded, changed to an integer datatype, and their data dictionary saved in the <a href=https://github.com/c-susan/datasci_9_data_prep/tree/main/model_dev1/data/processed>/model_dev1/data/processed</a> folder.
5. A copy of the cleaned and processed data was saved into the <a href=https://github.com/c-susan/datasci_9_data_prep/tree/main/model_dev1/data/processed>/model_dev1/data/processed</a> folder. 

## Dataset 2
<a href=https://github.com/c-susan/datasci_9_data_prep/blob/main/datasets/Diesel_Retail_Price_Weekly_Average_by_Region__Beginning_2007.csv>Diesel_Retail_Price_Weekly_Average_by_Region__Beginning_2007.csv</a>
+ This dataset contains information of the weekly average retail diesel prices in U.S. dollars per gallon for New York State and eight New York metropolitan regions. Data starts from October 2007 to the present., with some regions starting in 2017. Dataset taken from: https://catalog.data.gov/dataset/diesel-retail-price-weekly-average-by-region-beginning-2007. 
+ The intended machine learning task for this dataset is regression, with column 'new_york_state_average_($/gal)' being the target variable(X).

**Steps to Cleaning and Transforming Data**
1. The column names were first cleaned to replace spaces with underscores(_) and changed to lowercase.
2. The following columns were dropped with the following script because they contained too missing values:
```
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
```

3. The date column was changed to a datetime datatype and then changed to yyyy-mm format. The date column was then ordinally encoded, with the data dictionary saved in the <a href=https://github.com/c-susan/datasci_9_data_prep/tree/main/model_dev2/data/processed>/model_dev2/data/processed</a> folder.
4.  A copy of the cleaned and processed data was saved into the <a href=https://github.com/c-susan/datasci_9_data_prep/tree/main/model_dev2/data/processed>/model_dev2/data/processed</a> folder.


## Dataset Splitting
In the `p3_compute.py` filw of each `model_dev` folder, script was created  to split each dataset into three parts:
+ Training data (`train_x`, `train_y`)
+ Validation data (`val_x`, `val_y`)
+ Testing data (`test_x`, `test_y`) 
