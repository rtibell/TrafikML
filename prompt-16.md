# Premiss
You are a sesoned senior full stack developer with 30 years of experience of programming and developing machine learning solutions based on Python and IT-systems, mainly applications based on Java and Spring boot. 
You are about to develop a machine learning model that can predict congestions on roads in the Stockholm region. The source data comes from the TrafikenNu site that is measuring traffic speed on thousands of segments. 
The collecting service is in place and is currently collecting data for the machine learning application. The data will be delivered in Json format. 
Recurrent sequence models per segment (LSTM/GRU or seq2seq)
I would like you to create a machine learning application that can create a neural network based model, evaluate it, run tuning iterations and finally test its performance. Base the model on a CNN that can predict the speed from the SegmentSpeed data.


# Task
- Update the Python code use the status fields; freeflow_status, heavy_status, congested_status, imposible_status as target variable. The code resides in directory MachineLearning_MK-5.
- Generate updated documentation for how the solution is designed in markdown file MachineLearning_MK-5/Solution.md.
- Generate updated documentation on how to run the solution in markdown file MachineLearning_MK-5/Operations.md.

# Passus
- note that the target variables freeflow_status, heavy_status, congested_status, imposible_status  can only have the value 0 or 1. So it might be a good ide to us e a soft max based solution. But it's up to you to make the correcdt desision.


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


