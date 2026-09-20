# Premiss
You are a sesoned senior full stack developer with 30 years of experience of programming and developing machine learning solutions based on Python and IT-systems, mainly applications based on Java and Spring boot. 
You are about to develop a machine learning model that can predict congestions on roads in the Stockholm region. The source data comes from the TrafikenNu site that is measuring traffic speed on thousands of segments. 
The collecting service is in place and is currently collecting data for the machine learning application. The data will be delivered in Json format. 
Recurrent sequence models per segment (LSTM/GRU or seq2seq)
I would like you to create a machine learning application that can create a neural network based model, evaluate it, run tuning iterations and finally test its performance. Base the model on a CNN that can predict the speed from the SegmentSpeed data.


# Task
- Update the Python code to work with the revised SegmentSpeed json datastructure shown below. The code resides in directory MachineLearning_MK-4.
- Generate updated documentation for how the solution is designed in markdown file MachineLearning_MK-4/Solution.md.
- Generate updated documentation on how to run the solution in markdown file MachineLearning_MK-4/Operations.md.

# Instruction
- Use Python for training of the model.
- Use appropriate frameworks, for example PyTorch.
- Use a GTX 1080 or built in Mac graphical card to accelerate creation and evaluation of model.
- Data for the model is available in Json file "raw-trafic-data.json" and contains a array of data in format described in "SegmentSpeed json data structure".
- This version of the code should be placed in the subdirectory MachineLearning_MK-4.
- The features in the input data structure SegmentSpeed is: holiday_type_regular_day, holiday_type_eve, holiday_type_holiday, freeflow_status, heavy_status, congested_status, imposible_status, holiday_is_nyar, holiday_is_jul, holiday_is_forsta_maj, holiday_is_nationaldagen, holiday_is_pask, holiday_is_kristihimmelsfard, holiday_is_pingst, holiday_is_midsommar, holiday_is_allahelgona, holiday_is_trettondag, dayNr, daysUntilHoliday, minutesSincDaybreak, monthOfYear. 
- The target variable in the input data structure SegmentSpeed is speed.
- Normalize input data.
- Command line argument to select if target variable should be transformed with log or not.
- Command line argument to set the number of layers in the MLP
- Command line argument switch for determin type of activation function:
	Sigmoid function
	Hyperbolic tangent function
	Softsign function
	ReLU
	Leaky ReLU

# Features to implement
- Use a multilayer perceptron to create the predictive model
- Utilize gradient descent and backpropagation when updating the MLP weights
- Implement the following machine learning paradigms:
	Gradient descent
	MiniBatch
	Early stopping
	Adjustable learning rate and momentum
	Random initialization of weights


# Revised SegmentSpeed json datastructure
	 {
	    "sectionId": 32371,
	    "measureTime": "2026-09-13T15:04:00",
	    "status": "freeflow",
	    "statusEnum": 0,
	    "speed": 54,
	    "dayNr": 6,
	    "daysUntilHoliday": 48,
	    "holiday_type_regular_day": 1,
	    "holiday_type_eve": 0,
	    "holiday_type_holiday": 0,
	    "freeflow_status: 1,
	    "heavy_status": 0,
	    "congested_status": 0,
	    "imposible_status": 0,
	    "minutesSincDaybreak": 544,
	    "monthOfYear": 9,
	    "holiday_is_nyar": 0,
	    "holiday_is_jul": 0,
	    "holiday_is_forsta_maj": 0,
	    "holiday_is_nationaldagen": 0,
	    "holiday_is_pask": 0,
	    "holiday_is_kristihimmelsfard": 0,
	    "holiday_is_pingst": 0,
	    "holiday_is_midsommar": 0,
	    "holiday_is_allahelgona": 0,
	    "holiday_is_trettondag": 0
	  }


