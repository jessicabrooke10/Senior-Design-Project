# LSTM Model

Intstructions to replicate:

Take original data file (in csv format), and run it through new_editor1.py, then through new_editor2.py, then through half_hour_agg.py. This outputs by_half_hour.csv

To evaluate the model different datasets, run them through new_editor1.py, then through new_editor2.py, and then input_time_agg.py, which allows for the user to manually pick a time to aggregate by. This outputs LSTM_by_time_period.csv

Final_LSTM_SD_v5.ipynb is the main file for this part of the project. It now contains the data aggregators and editors within itself, so it is the only file needed other than the initial data file(s).
