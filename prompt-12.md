# Premiss
You are a sesoned senior full stack developer with 30 years of experience of programming and developing machine learning solutions based on Python and IT-systems, mainly applications based on Java and Spring boot. 
You are about to develop a machine learning model that can predict congestions on roads in the Stockholm region. The source data comes from the TrafikenNu site that is measuring traffic speed on thousands of segments. 
The collecting service is in place and is currently collecting data for the machine learning application. The data will be delivered in Json format. 
Recurrent sequence models per segment (LSTM/GRU or seq2seq)
I would like you to create a machine learning application that can create a neural network based model, evaluate it, run tuning iterations and finally test its performance. Base the model on a CNN that can predict the speed from the SegmentSpeed data.


# Task
- Generate code for a application that can create, improve, evaluate and run a model with properties as described above. 
- Partition data into training set (90%) and test set (10%), shuffle the data with a random algorithm using seed 4711.
- Evaluate the model and suggest improvements to the code and machine learning parameters. 
- Generate documentation for how the solution is designed in markdown file MachineLearning_MK/Solution.md.
- Generate documentation on how to run the solution in markdown file MachineLearning_MK/Operations.md.

# Instruction
- Use Python for training of the model.
- Use appropriate frameworks, for example PyTorch.
- Use a GTX 1080 to accelerate creation and evaluation of model.
- Data for the model is available in Json file "raw-trafic-data.json" and contains a array of data in format described in "SegmentSpeed json data structure".
- This version of the code should be placed in the subdirectory MachineLearning_MK-2.
- The features in the input data structure SegmentSpeed is: statusEnum, dayNr, daysUntilHoliday, holidayType, minutesSincDaybreak, monthOfYear, holidayNum. 
- The target variable in the input data structure SegmentSpeed is speed.
- Normalize input data.
- Try out if target variable should be transformed with log or not.

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