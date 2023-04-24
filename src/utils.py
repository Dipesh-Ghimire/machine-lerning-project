# It will have all common functions that we use during entire project
import os
import sys 
import numpy as np 
import pandas as pd 
import dill
from src.exception import CustomException


def  save_object(file_path,obj):
   # Caller: data_transformation
   try:
      dir_path = os.path.dirname(file_path)
      os.makedirs(dir_path, exist_ok=True )
      with open(file_path,"wb") as file_obj:
         dill.dump(obj,file_obj)
   
   except Exception as e:
      raise CustomException(e,sys)