import json
import logging
import pandas as pd
import os
import dill

path = os.environ.get('PROJECT_PATH', '.')

def predict():
    path_models = os.path.join(path, "data", "models")
    path_test = os.path.join(path, "data", "test")
    path_preds = os.path.join(path, "data", "predictions")
    
    list_of_models = [os.path.join(path_models, one_file) for one_file in os.listdir(path_models) if (one_file[-4:] == ".pkl")]
    if len(list_of_models) == 0:
        return(0)
    
    full_model_file_name = max(list_of_models, key=os.path.getctime)
    model_version = full_model_file_name.split("cars_pipe_")[1].split(".pkl")[0]
    
    with open(full_model_file_name, 'rb') as hfile:
        curr_model = dill.load(hfile)

    df_save_preds = pd.DataFrame(columns=["ModelName", "DataFile", "Prediction"])
    for test_filename in os.listdir(path_test):
        cur_test_file = os.path.join(path_test, test_filename)

        with open(cur_test_file) as hfile:
            test_data = json.load(hfile)
        
        df = pd.DataFrame.from_dict([test_data])
        y = curr_model.predict(df)
        
        df_save_preds = df_save_preds.append(
            [
                {
                    "ModelName": "cars_pipe_" + model_version,
                    "DataFile": test_filename,
                    "Prediction": y[0]
                }
            ]
        )
        
        csv_file = os.path.join(path_preds, "preds_" + model_version + ".csv")
    
    df_save_preds.to_csv(csv_file, index=False)

    return(0)

if __name__ == '__main__':
    predict()