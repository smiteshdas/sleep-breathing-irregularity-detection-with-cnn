# Sleep Breathing Irregularities Detection using a 1D-CNN model [AI for Health - SRIP 2026]

This project deals with training and evaluation of a 1D CNN model for detecting breathing irregularities that occur during sleep. 
eg: *Hypopnea, Obstructive Sleep Apnea* etc.

Run the `train_test_model.py` script to execute the model training and evaluation process.

![Cover Signal Graph](https://raw.githubusercontent.com/smiteshdas/sleep-breathing-irregularity-detection-with-cnn/refs/heads/main/cover.jpg)

## Usage
Run all the commands in sequence after `cd` to the root directory of the project to generate visualisation pdf's , dataset and then train and evaluate the model.
### Understanding the Data and Visualization

Run this command for each participants data directories.
```
python scripts/vis.py -name "Data/AP01" 
```
Similarly for all participants `python scripts/vis.py -name "Data/AP02"` , `python scripts/vis.py -name "Data/AP03"` etc.

This saves the plotted signal graphs (Nasal Airflow, Thoracic Movement, SpO₂) as PDF's in the *Visualizations* folder.

### Signal Preprocessing and Dataset Creation
```
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```
This command creates an dataset ready for 1D CNN training and testing by filering the signal noise and takes out 30 second windows with 50% overlapping . It takes input directory *Data* and output Directory as *Dataset*.

### Basic Modeling and Evaluation
```
python scripts/train_test_model.py
```
This command runs the *train_test_model.py* which does the training and testing on the 1D CNN algorithm. The model is evaluated in a Leave-One-Participant-Out Cross-Validation manner.

## Requirements
- Python
- Matplotlib
- Pandas
- NumPy
- Scikit Learn
- PyTorch
- SciPy


**NOTE:** ChatGPT was used as a supportive tool while completing this project.
