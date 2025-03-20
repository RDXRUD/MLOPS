import os
import logging
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
import yaml

log_dir='logs'
logger=logging.getLogger("model_training")
logger.setLevel("DEBUG")

log_file_path=os.path.join(log_dir,"model_training.log")
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

console_handler=logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter=logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path:str)->dict:
    try:
        with open(params_path,"r") as file:
            params=yaml.safe_load(file)
        logger.debug("Parameters retrived from %s",params)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s",e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s",e)
        raise

def load_data(file_path:str)->pd.DataFrame:
    try:
        df=pd.read_csv(file_path)
        logger.debug("Data loaded from %s ",file_path)
        return df
    
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e)
        raise
        
    except FileNotFoundError as e:
        logger.error("File not found %s",e)
        raise
        
    except Exception as e:
        logger.error("Unexpected error occur while loading the data: %s",e)
        raise
    
def train_model(X_train:np.ndarray,y_train:np.ndarray,params:dict)->RandomForestClassifier:
    try:
        if X_train.shape[0]!=y_train.shape[0]:
            raise ValueError("The number of sample in x_train and y_train must be same")
        
        logger.debug("Initializing RandomForest model with parameters: %s",params)
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        
        logger.debug("Model training started with %d samples",X_train.shape[0])
        clf.fit(X_train,y_train)
        logger.debug("Model Training Completed")
        
        return clf
    
    except ValueError as e:
        logger.error("Value error during model training: %s",e)
        raise
        
    except Exception as e:
        logger.error("Error during model training: %s",e)
        raise
    
    
def save_model(model,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
            
        logger.debug("Model saved to %s",file_path)
    
    except FileNotFoundError as e:
        logger.error("File not found: %s",e)
        raise
        
    except Exception as e:
        logger.error("Error occured while saving the model: %s",e)
        raise
    
def main():
    try:
        params=load_params(params_path="params.yaml")["model_building"]
        # params={'n_estimators':25,'random_state':2}
        train_data=load_data('./data/processed/train_tfidf.csv')
        X_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values
        
        clf=train_model(X_train,y_train,params)
        
        model_save_path='model/model.pkl'
        save_model(clf,model_save_path)
        
    except Exception as e:
        logger.error("Failed to complete model building process: %s",e)
        print(f"Error: {e}")
        
if __name__=="__main__":
    main()
        
    
    
        
        