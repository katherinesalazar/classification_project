# imports for acquire
import pandas as pd
import numpy as np
import os
from env import host, user, password

###################### Acquire telco_churn Data from SQL Database ######################

# SQL database query to join the customers, contract_types, payment_types and internet_service_types tables from the telco_churn SQL database
query = '''
SELECT * 
FROM customers
JOIN contract_types USING (contract_type_id)
JOIN payment_types USING (payment_type_id)
JOIN internet_service_types USING (internet_service_type_id);
'''

data_base_name = "telco_churn"

# Connection for the env file to the telco_churn SQL database
def get_sql_connection(host=host, user=user, password=password):
    '''
    Function used to read SQL query from SQL database
    ''' 
    global query
    global data_base_name

    url = f'mysql+pymysql://{user}:{password}@{host}/{data_base_name}'
    df = pd.read_sql(query, url)
    return df

###################### Acquire telco_churn.csv Data ######################

# SQL database data from SQL query to csv file
def get_telco_data():
    '''
    This function reads in telco_churn data from Codeup database, writes data to
    a csv file.
    '''
    global data_base_name
    global query
    url = f'mysql+pymysql://{user}:{password}@{host}/{data_base_name}'
    plain_data = pd.read_sql(query, url)
    plain_data.to_csv("telco_churn.csv")
    
###################### Ensure no duplicate telco_churn.csv Data ######################

# Duplicate file check
def duplicate_csv_file_check(file_name):
    '''
    Checks if there is a csv file with the matching name in the directory. If there is not 
    it will create a new csv using the env file in the directory. 
    '''
    if os.path.exists(file_name) == False:
        get_telco_data()

###################### Read telco_churn.csv Data while dropping unnamed columns ######################

# Reads the telco to CSV file and drops the unnamed column in the dataframe
def read_telco_data():
    ''' 
    Used to read the telco csv file, and also drops the unnamed columns within the dataframe
    '''
    df = pd.read_csv("telco_churn.csv")
    df = df.drop(columns="Unnamed: 0")
    return df