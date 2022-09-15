import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.datasets import load_diabetes, load_iris, make_moons, load_wine, fetch_california_housing

## Preparing Adult dataset
def PrepareAdult(dataset_path, dataset_name):
    
    raw_data = np.genfromtxt('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                             delimiter=', ', dtype=str, invalid_raise=False)

    #  column names from "https://archive.ics.uci.edu/ml/datasets/Adult"
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'educational-num', 'marital-status', 'occupation',
                    'relationship', 'race', 'gender', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
                    'income']

    adult_data = pd.DataFrame(raw_data, columns=column_names)

    # For more details on how the below transformations are made, please refer to https://rpubs.com/H_Zhu/235617
    adult_data = adult_data.astype({"age": np.int64, "educational-num": np.int64, "hours-per-week": np.int64})

    adult_data = adult_data.replace({'workclass': {'Without-pay': 'Other/Unknown', 'Never-worked': 'Other/Unknown'}})
    adult_data = adult_data.replace({'workclass': {'Federal-gov': 'Government', 'State-gov': 'Government',
                                     'Local-gov': 'Government'}})
    adult_data = adult_data.replace({'workclass': {'Self-emp-not-inc': 'Self-Employed', 'Self-emp-inc': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'Never-worked': 'Self-Employed', 'Without-pay': 'Self-Employed'}})
    adult_data = adult_data.replace({'workclass': {'?': 'Other/Unknown'}})

    adult_data = adult_data.replace(
        {
            'occupation': {
                'Adm-clerical': 'White-Collar', 'Craft-repair': 'Blue-Collar',
                'Exec-managerial': 'White-Collar', 'Farming-fishing': 'Blue-Collar',
                'Handlers-cleaners': 'Blue-Collar',
                'Machine-op-inspct': 'Blue-Collar', 'Other-service': 'Service',
                'Priv-house-serv': 'Service',
                'Prof-specialty': 'Professional', 'Protective-serv': 'Service',
                'Tech-support': 'Service',
                'Transport-moving': 'Blue-Collar', 'Unknown': 'Other/Unknown',
                'Armed-Forces': 'Other/Unknown', '?': 'Other/Unknown'
            }
        }
    )

    adult_data = adult_data.replace({'marital-status': {'Married-civ-spouse': 'Married', 'Married-AF-spouse': 'Married',
                                                        'Married-spouse-absent': 'Married', 'Never-married': 'Single'}})

    adult_data = adult_data.replace({'race': {'Black': 'Other', 'Asian-Pac-Islander': 'Other',
                                              'Amer-Indian-Eskimo': 'Other'}})

    adult_data = adult_data[['age', 'workclass', 'education', 'marital-status', 'occupation',
                             'race', 'gender', 'hours-per-week', 'income']]

    adult_data = adult_data.replace({'income': {'<=50K': 0, '>50K': 1}})

    adult_data = adult_data.replace({'education': {'Assoc-voc': 'Assoc', 'Assoc-acdm': 'Assoc',
                                                   '11th': 'School', '10th': 'School', '7th-8th': 'School',
                                                   '9th': 'School', '12th': 'School', '5th-6th': 'School',
                                                   '1st-4th': 'School', 'Preschool': 'School'}})

    df = adult_data.rename(columns={'income': 'class', 'marital-status': 'marital_status', 'hours-per-week': 'hours_per_week'})

    ## Recognizing inputs
    class_name = 'class'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['age', 'hours_per_week']
    discrete_features = ['workclass', 'education', 'marital_status', 'occupation',
                         'race', 'gender']

    continuous_availability = True
    discrete_availability = True

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler =StandardScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([scaled_data, ord_encoded_data], axis=1)
    df_X_ohe = pd.concat([scaled_data, ohe_encoded_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = [df_X_org.iloc[:, continuous_indices].shape[1], df_X_org.shape[1]]

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = [scaled_data.shape[1], df_X_ord.shape[1]]

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = [scaled_data.shape[1], df_X_ohe.shape[1]]

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }

    return dataset

## Preparing Student dataset
def PrepareStudent(dataset_path, dataset_name):

    ## Reading data from a csv file
    df = pd.read_csv(dataset_path+dataset_name, delimiter=';')
    df.loc[df['G3'] > 10, 'grade'] = 'high'
    df.loc[df['G3'] <= 10, 'grade'] = 'low'
    
     ## Handling missing values
    df = df.dropna().reset_index(drop=True)
    df=df.drop(['school', 'G3', 'address', 'famsize', 'nursery', 'activities', 'guardian', 'reason', 'Mjob', 'Fjob', 'Medu', 'traveltime', 'failures', 'Fedu','internet','health'], axis=1)
    
    df.replace(to_replace = 'A', value = 'apart', inplace = True)
    df.replace(to_replace = 'T', value = 'together', inplace = True)
    cols = ['studytime', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc']
    df[cols] = df[cols].replace([1, 2, 3, 4, 5], ['very low', 'low', 'medium', 'high', 'very high'])
    
    ## Recognizing inputs
    class_name = 'grade'
    df_X_org = df.loc[:, df.columns!=class_name]
    df_y = df.loc[:, class_name]

    continuous_features = ['age','G1', 'G2', 'absences']
    discrete_features = ['sex', 'Pstatus', 'studytime', 'schoolsup', 'famsup', 'paid', 'higher',  'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc']

    continuous_availability = True
    discrete_availability = True 

    df_X_org = pd.concat([df_X_org[continuous_features], df_X_org[discrete_features]], axis=1)

    continuous_indices = [df_X_org.columns.get_loc(f) for f in continuous_features]
    discrete_indices = [df_X_org.columns.get_loc(f) for f in discrete_features]

    feature_values = []
    for c in continuous_features:
        feature_values.append({c:[min(df_X_org[c]),max(df_X_org[c])]})
    for d in discrete_features:
        feature_values.append({d: set(df_X_org[d].unique())})

    ## Extracting the precision of continuous features
    types = df_X_org[continuous_features].dtypes
    continuous_precision = []
    for c in continuous_features:
        if types[c] == float:
            len_dec = []
            for val in df_X_org[c]:
                len_dec.append(len(str(val).split('.')[1]))
            len_dec = max(set(len_dec), key=len_dec.count)
            continuous_precision.append(len_dec)
        else:
            continuous_precision.append(0)

    precision = pd.Series(continuous_precision, index=continuous_features)
    df_X_org = df_X_org.round(precision)

    ## Scaling continuous features
    num_feature_scaler =StandardScaler()
    scaled_data = num_feature_scaler.fit_transform(df_X_org.iloc[:, continuous_indices].to_numpy())
    scaled_data = pd.DataFrame(data=scaled_data, columns=continuous_features)

    ## Encoding discrete features
    # Ordinal feature transformation
    ord_feature_encoder = OrdinalEncoder()
    ord_encoded_data = ord_feature_encoder.fit_transform(df_X_org.iloc[:, discrete_indices].to_numpy())
    ord_encoded_data = pd.DataFrame(data=ord_encoded_data, columns=discrete_features)

    # One-hot feature transformation
    ohe_feature_encoder = OneHotEncoder(sparse=False)
    ohe_encoded_data = ohe_feature_encoder.fit_transform(ord_encoded_data.to_numpy())
    ohe_encoded_data = pd.DataFrame(data=ohe_encoded_data)

    # Creating ordinal and one-hot data frames
    df_X_ord = pd.concat([scaled_data, ord_encoded_data], axis=1)
    df_X_ohe = pd.concat([scaled_data, ohe_encoded_data], axis=1)

    ## Encoding labels
    df_y_le = df_y.copy(deep=True)
    label_encoder = {}
    le = LabelEncoder()
    df_y_le = le.fit_transform(df_y_le)
    label_encoder[class_name] = le

    ## Extracting raw data and labels
    X_org = df_X_org.values
    X_ord = df_X_ord.values
    X_ohe = df_X_ohe.values
    y = df_y_le

    ## Indexing labels
    labels = {i: label for i, label in enumerate(list(label_encoder[class_name].classes_))}

    ## Indexing features
    feature_names = list(df_X_org.columns)
    feature_indices = {i: feature for i, feature in enumerate(feature_names)}
    feature_ranges = {feature_names[i]: [min(X_ord[:, i]), max(X_ord[:, i])] for i in range(X_ord.shape[1])}
    feature_width = np.max(X_ord, axis=0) - np.min(X_ord, axis=0)

    n_cat_discrete = ord_encoded_data.nunique().to_list()

    len_continuous_org = [0, df_X_org.iloc[:, continuous_indices].shape[1]]
    len_discrete_org = [df_X_org.iloc[:, continuous_indices].shape[1], df_X_org.shape[1]]

    len_continuous_ord = [0, scaled_data.shape[1]]
    len_discrete_ord = [scaled_data.shape[1], df_X_ord.shape[1]]

    len_continuous_ohe = [0, scaled_data.shape[1]]
    len_discrete_ohe = [scaled_data.shape[1], df_X_ohe.shape[1]]

    ## Returning dataset information
    dataset = {
        'name': dataset_name.replace('.csv', ''),
        'df': df,
        'df_y': df_y,
        'df_X_org': df_X_org,
        'df_X_ord': df_X_ord,
        'df_X_ohe': df_X_ohe,
        'df_y_le': df_y_le,
        'class_name': class_name,
        'label_encoder': label_encoder,
        'labels': labels,
        'ord_feature_encoder': ord_feature_encoder,
        'ohe_feature_encoder': ohe_feature_encoder,
        'num_feature_scaler': num_feature_scaler,
        'feature_names': feature_names,
        'feature_values': feature_values,
        'feature_indices': feature_indices,
        'feature_ranges': feature_ranges,
        'feature_width': feature_width,
        'continuous_availability': continuous_availability,
        'discrete_availability': discrete_availability,
        'discrete_features': discrete_features,
        'discrete_indices': discrete_indices,
        'continuous_features': continuous_features,
        'continuous_indices': continuous_indices,
        'continuous_precision': continuous_precision,
        'n_cat_discrete': n_cat_discrete,
        'len_discrete_ord': len_discrete_ord,
        'len_continuous_ord': len_continuous_ord,
        'len_discrete_ohe': len_discrete_ohe,
        'len_continuous_ohe': len_continuous_ohe,
        'len_discrete_org': len_discrete_org,
        'len_continuous_org': len_continuous_org,
        'X_org': X_org,
        'X_ord': X_ord,
        'X_ohe': X_ohe,
        'y': y
    }
    
    
    return dataset