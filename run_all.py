
import sys
import os

sys.path.insert(0,"data_preprocessing/")
import main_preprocessing

sys.path.insert(1,"machine_learning/")
import main_ml

if __name__ == "__main__":
  
  print("All starting")

  #os.system('python data_preprocessing/main_preprocessing.py')
  os.system('python machine_learning/main_ml.py')


  print("Over ALL Done")