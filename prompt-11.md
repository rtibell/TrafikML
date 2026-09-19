# Premiss
You are a sesoned senior full stack developer with 30 years of experiances of programming and developing IT-systems mainly with Java and Spring boot. 
You are about to develop a macine learning model that can predict congestions on roads in the Stockholm region. The source data comes from the TrafikenNu site that is measuring trafic speed on thousends of segments. 
The collecting service is in place and is currently collecting data for the macine learning application. The data will be deliverd in Json format. 
Recurrent sequence models per segment (LSTM/GRU or seq2seq)
Treat each road segment's minute-by-minute speed and flow-status as its own multivariate time series, and train an LSTM/GRU (or an encoder-decoder seq2seq for multi-step-ahead forecasts) either per segment or as one model with the segment ID as an embedding feature. 

# Task
- Generate code for a application that can create, evaluate and run a model with properties as described above. 
- Partition data into training set (90%) and test set (10%), shuffle the data with a random algorithm using seed 4711.
- Evaluate the model and suggest improvments to the code and machine learning parameters. 
- Generate documentation for how the solution is designed in markdown file MachineLearning_MK/Solution.md.
- Generate documentation on how to run the solution in markdown file MachineLearning_MK/Operations.md.

# Instruction
- Use Python for training of the model.
- Use frameworks PyTorch Geometric or DGL (Deep Graph Library) on top of PyTorch.
- Use a GTX 1080 to accelerate creation and evaluation of model.
- Data for the model is available in Json file "raw-trafic-data.json" and contains a array of data in format described in "SegmentSpeed json datastructure".
- This version of the code should be placed in the subdirectory MachineLearning_MK-1.

# SegmentSpeed json datastructure
	 {
	    "sectionId": 32371,
	    "measureTime": "2026-09-13T15:04:00",
	    "status": "freeflow",
	    "statusEnum": 0,
	    "speed": 54,
	    "dayNr": 6,
	    "daysUntilHoliday": 48,
	    "holidayType": 0,
	    "minutesSincDaybreak": 544,
	    "monthOfYear": 9,
	    "holidayNum": 10
	  }