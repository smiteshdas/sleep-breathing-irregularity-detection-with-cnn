# Breathig Irregularities Detection with 1D CNN model [AI for Health - SRIP 2026]



Understanding the Data and Visualization
```
python scripts/vis.py -name "Data/AP01" 
```
Similarly for all participants `python scripts/vis.py -name "Data/AP02"` , `python scripts/vis.py -name "Data/AP03"` etc.

Signal Preprocessing and Dataset Creation
```
python scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

Basic Modeling and Evaluation
```
python scripts/train_test_model.py
```
