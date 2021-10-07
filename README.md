Instructions on how to run the code

Step 1: Download Conda

	1. Go to https://www.anaconda.com /products/individual#windows and download the Anaconda Installer.
	2. Run the installer. 
	3. Choose the default options and click "install". 
	4. Click "Finish" to end, and verify the installation by opening Anaconda Navigator or Anaconda Prompt


Step 2: Install environment.yaml
	
	1. Open the Anaconda command prompt 
	2. Type the following: conda env create -f environment.yml -p <location of the environement.yml>

Step 3: Generate Results
	
	1. Activate the copied virtual environment
	2. Go to the repository directory of the main folder "candle_clinical_track 2021"
	3. Generate preprocessed data, predictions and LOOCV evaluation by typing the following: python run_all.py


Step 4: View results here

	1. Go to the folder labeled output
	2. The summary of the results are in the evaluation folder with the filename "LOOCV evaluation.csv"
	3. The predictions are in the folder predictions 



Replicating the results 
	The result can be replaced using the generated predictions in the prediction folders

	1. Go to the repository directory of the main folder "candle_clinical_track 2021"
	2. Replicate the LOOCV evaluation by typing the following: python machine_learning\evaluation.py
	3. Go to the folder labeled output
	4. The replicated summary of the results are in the evaluation folder with the filename "Evaluation Result.csv"



Notes: 	
 - The labels in the "Data.csv" are in the Baseline column. 1 represents abnormal and 0 represents normal liver status
 - All the original predictions were run using Windows 10 Home 