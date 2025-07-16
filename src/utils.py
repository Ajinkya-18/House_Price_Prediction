def assert_path(file_path:str):
    import os
    from pathlib import Path

    corrected_file_path = file_path.replace('\\', '/')
    cwd = os.getcwd()
    full_file_path = os.path.join(cwd, corrected_file_path)

    if os.path.exists(full_file_path) and (full_file_path.endswith('.csv') or full_file_path.endswith('.joblib')):
        return full_file_path
    
    else:
        raise ValueError('Invalid file path or ')

#-----------------------------------------------------------------------------------------------------------------------

def load_data(data_path:str):
    pass

#--------------------------------------------------------------------------------------------------------------------

def preprocess_data(df, target:str, mode:str='train'):
    pass

