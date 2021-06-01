# imports for prepare:
import acquire
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.feature_selection as feat_select
import warnings
warnings.filterwarnings('ignore')


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import scipy.stats as stats


# dataframe from the acquire file
df = acquire.read_telco_data()


###################################### Test Train Split ######################################
############# DATA IS SPLIT INTO TRAIN, VALIDATE, TEST WITH REASONABLE PROPORTIONS ###########
##################################### RANDOM STATE IS SET ###################################
################ Data is split prior to exploration of variable relationships ###############

# Train Test Split function
def train_validate_test_split(df, is_churn, seed=789):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[is_churn])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[is_churn])
    return train, validate, test

def test_train_split(df, stratify_value = 'churn'):
    '''
    This function takes in the telco_churn data data acquired by aquire.py,
    performs a split, stratifies by churn.
    Returns train, validate, and test dataframe
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=789,
                                        stratify = df[stratify_value])
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=789,
                                   stratify= train_validate[stratify_value])
    return train, validate, test

# X_train function
def X_train_function(X_cols, y_col, train, validate, test):
    '''
    X_cols = list of column names you want as your features: online_security, online_backup, device_protection
    y_col = string that is the name of your target colums
    train = the name of your train dataframe
    validate = the name of your validate dataframe
    test = the name of your test dataframe
    outputs X_train and y_train, X_validate and y_validate, and X_test and y_test
    '''
    
    # X is the data frame of the features, y is a series of the target
    X_train, y_train = train[X_cols], train[y_col]
    X_validate, y_validate = validate[X_cols], validate[y_col]
    X_test, y_test = test[X_cols], test[y_col]
    
    return X_train, y_train, X_validate, y_validate, X_test, y_test

###################################### Prep Telco ######################################
################## HANDLING MISSING VALUES, DATA TYPES, ETC. ###########################

def prep_telco(df):
    """
    This functions takes in the telco dataframe from acquire and cleaned dataset

    """
##### missing values, outliers, data integrity issues are discovered and documented #####
# There were missing values within the 'total charges' column in the telco_churn dataset, thus replaced with 0 and updated to float #
    df.total_charges = df.total_charges.str.replace(' ', '0').astype(float)
    train, validate, test = test_train_split(df)

    
    return train, validate, test


def prep_telco_model(df):
    '''
    These functions take in the telco churn dataframe and returns the cleaned dataset for model
    
    '''
    
    #df.replace(to_replace = " ", value = np.nan, inplace = True)
    
    # column to drop with unique identifiers:
    columns_to_drop =  ['customer_id']
    
    # dropping column with unique identifiers:
    df = df.drop(columns=columns_to_drop)
    
    # get list of columns that need to be encoded as they are 'object' types: 
    cols = [col for col in list(df.columns) if df[col].dtype == 'object']

    #turn all text (object) columns to numbers using LabelEncoder()
    #label_encoder = LabelEncoder()
    #for col in cols:
     #   df[col] = label_encoder.fit_transform(df[col])

    train, validate, test = test_train_split(df)


###################################### Prep Telco Add Columns ######################################
################## HANDLING MISSING VALUES, DATA TYPES, ETC. ###########################

# Creating tenure year column
def create_tenure_year(df):
    '''
    Function used to create a new column and outputs years of tenure
    '''
    df["tenure_years"] = df.tenure/12
    df.tenure_years = df.tenure_years.astype(int)
    return df

# Creating is_churn column
def is_churn(df):
    '''
    Function used to create a new column for churn
    '''
    df['is_churn'] = (df.churn == "Yes")
    return df

# Encoding the columns
def encode_all(df):
    '''encodes all Yes values to 1, No values to 0, and 2 for n/a results of no internet service
    or no phone service,  Female to 1 and Male to 0 then turns encode columns into integers'''
    
    df = df.replace({"Yes": 1,
                          "No": 0,
                           "No internet service": 2,
                           "No phone service": 2
    })
    df = df.replace({"Female": 1,
                           "Male": 0 
    })
    for c in df.columns:
        if c == 'monthly_charges' or c== 'total_charges':
            df[c] = df[c]
        elif df[c].any() == 1:
            df[c] = df[c].astype(int)

    return df

# Creating security features column
def security_features(df):
    '''
    Function used to create a new column for: 
    - online_security
    '''
    df['security_features'] = (df.online_security == 'Yes')
    return df

# Creating backup features column
def backup_features(df):
    '''
    Function used to create a new column for online_backup
    '''
    df['backup_features'] = (df.online_backup == "Yes")
    return df

# Creating device protection features column 
def device_protection_features(df):
    '''
    Function used to create a new column for device_protection
    '''
    df['device_protection_features'] = (df.device_protection == "Yes")
    return df

# Encoding the features columns to numbers
def encode_feature_columns(df):
    '''
    Main function used to encode new columns, this helps combine columns to help reduced the number of features.
    '''
    df["security_features"] = df.apply(lambda row: security_features(row), axis = 1)

    df["backup_features"] = df.apply(lambda row: backup_features(row), axis = 1)

    df["device_protection_features"] = df.apply(lambda row: device_protection_features(row), axis = 1)
    
    #new code
    
    df["is_churn"] = df["is_churn"].astype(int)
    df["security_features"] = df["security_features"].astype(int)
    df["backup_features"] = df["backup_features"].astype(int)
    df["device_protection_features"] = df["device_protection_features"].astype(int)


    return df

####### INDIVIDUAL VARIABLE DISTRIBUTIONS ARE PLOTTED ###########

# not object columns
#num_cols = df.columns[[df[col].dtype != 'O' for col in df.columns]]

# object columns
#obj_cols = df.columns[[df[col].dtype == 'O' for col in df.columns]]

# individual variable distribution CHURN is plotted in barchart
#sns.displot(df, x="churn")


def prep_full_telco(df):
    '''
    Does the same thing as prep_telco_model except no split. For creating CSV
    '''

        
    #df.replace(to_replace = " ", value = np.nan, inplace = True)
    
    # column to drop with unique identifiers:
    #columns_to_drop =  ['customer_id']
    
    # dropping column with unique identifiers:
    #df = df.drop(columns=columns_to_drop)
    
    # get list of columns that need to be encoded as they are 'object' types: 
    #cols = [col for col in list(df.columns) if df[col].dtype == 'object']

    #turn all text (object) columns to numbers using LabelEncoder()
    #label_encoder = LabelEncoder()
    #for col in cols:
     #   df[col] = label_encoder.fit_transform(df[col])

    return df