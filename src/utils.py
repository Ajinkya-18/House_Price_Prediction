def assert_path(file_path:str):
    import os
    from pathlib import Path

    corrected_file_path = file_path.replace('\\', '/')
    cwd = os.getcwd()
    full_file_path = os.path.join(cwd, corrected_file_path)

    if os.path.exists(full_file_path):
        return full_file_path
    
    else:
        raise ValueError('Invalid file path or file extension. Check file path and accepted file extensions are- ".csv" and ".joblib"')

#-----------------------------------------------------------------------------------------------------------------------

def load_data(data_path:str):
    asserted_data_path = assert_path(data_path)

    if asserted_data_path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(asserted_data_path)

        return df

    else:
        raise ValueError('data file extension is invalid. Pass a .csv file')
    
#--------------------------------------------------------------------------------------------------------------------

def split_data(df, target:str='median_house_value'):
    from sklearn.model_selection import train_test_split

    X, Y = df.drop([target], axis=1), df[target]
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)

    return x_train, x_val, y_train, y_val

#--------------------------------------------------------------------------------------------------------------------

def preprocess_data(df, target:str='median_house_value'):
    try:
        import pandas as pd

        df.dropna(inplace=True)
        df = pd.get_dummies(df, columns=['ocean_proximity'], dtype=int)

        df['rooms_per_house'] = df['total_rooms'] / df['households']
        df['bedrooms_per_house'] = df['total_bedrooms'] / df['households']

        df.drop(['total_rooms', 'total_bedrooms', 'population', 'households'], axis=1, inplace=True)

        df.drop(df[df['bedrooms_per_house'] > 3].index, axis=0, inplace=True)
        df.drop(df[df['rooms_per_house'] > 12].index, axis=0, inplace=True)
        df.drop(df[df['median_income'] > 12].index, axis=0, inplace=True)

        x_train, x_val, y_train, y_test = split_data(df)

        rfecv = load_model('models\RFECV_fitted.joblib')
        scaler = load_model('models\RobustScaler_fitted.joblib')

        x_train_new = rfecv.transform(x_train)
        x_val_new = rfecv.transform(x_val)

        x_train_new_scaled = scaler.transform(x_train_new)
        x_val_new_scaled = scaler.transform(x_val_new)


        return x_train_new_scaled, x_val_new_scaled, y_train, y_test

    except Exception as e:
        raise e
    
#------------------------------------------------------------------------------------------------------------------

def load_model(model_path:str):
    asserted_model_path = assert_path(model_path)

    try:
        from joblib import load
        with open(asserted_model_path, 'rb') as f:
            model = load(f)

            return model
        
    except Exception as e:
        raise e
    
#------------------------------------------------------------------------------------------------------------------

def save_model(model_instance, save_dir:str='models/model_instance.joblib'):
    try:
        from joblib import dump

        with open(save_dir, 'wb') as f:
            print('saving model..')
            dump(model_instance, f)
            print('Saved model successfully.')

    except Exception as e:
        raise e
    
#-----------------------------------------------------------------------------------------------------------------------

def train_model(model, x_train, y_train):
    try:
        print('started training...')
        model.fit(x_train, y_train)
        print('Completed model training. Returning trained model')

        return model
    
    except Exception as e:
        raise e
    
#------------------------------------------------------------------------------------------------------------------

def test_model(model, x_test, y_test):
    try:
        y_preds = model.predict(x_test)

        from sklearn.metrics import r2_score
        score = r2_score(y_test, y_preds)

        return score
    
    except Exception as e:
        raise e
    
#-------------------------------------------------------------------------------------------------------------

def get_inference_data():
    from random import choice, uniform
    import pandas as pd

    inf_data = {'longitude':[round(uniform(-124.00, -114.50), 2)], 
                'latitude': [round(uniform(32.00, 42.00), 2)], 
                'housing_median_age': [float(choice(range(1, 61)))], 
                'median_income': [round(uniform(0.3, 13.00), 2)], 
                'ocean_proximity_INLAND': [float(choice(range(0, 2)))], 
                'rooms_per_house': [float(choice(range(3, 16)))], 
                'bedrooms_per_house': [float(choice(range(1, 6)))]
                }
    
    inf_df = pd.DataFrame.from_dict(inf_data)

    scaler = load_model('models/RobustScaler_fitted.joblib')
    inf_df_scaled = scaler.transform(inf_df)

    return inf_df_scaled

#------------------------------------------------------------------------------------------------------------------------
    

