# Premiss
You are a sesoned senior full stack developer with 30 years of experiances of programming and developing IT-systems mainly with Java and Spring boot. 
You are about to develop a macine learning model that can predict congestions on roads in the Stockholm region. The source data comes from the TrafikenNu site that is measuring trafic speed on thousends of segments. 
The collecting service is in place and is currently collecting data for the macine learning application. The data will be deliverd in Json format. 
Recurrent sequence models per segment (LSTM/GRU or seq2seq)
Treat each road segment's minute-by-minute speed and flow-status as its own multivariate time series, and train an LSTM/GRU (or an encoder-decoder seq2seq for multi-step-ahead forecasts) either per segment or as one model with the segment ID as an embedding feature. 

# Task
To increase the performance of the neural network I need some changes to the Java Spring Boot application that alters the format of the SegmentSpeed json datastructure and the REST endpoint that extracts the information.

# Instructions
- The SegmentSpeed json structor is in class MachineLearningSpeedOfSectionView

# Changes to the SegmentSpeed data structure from the original format to look like the revised format.
- Change the holidayType field from a integer to be represented as three different integer fields like below
	- if holidayType value is 0 set field holiday_type_regular_day = 1
	- if holidayType value is 1 set field holiday_type_eve = 1
	- if holidayType value is 2 set field holiday_type_holiday = 1
- Change the statusEnum field from string with 4 different values to be represented as four different integer fields like below	
	- if statusEnum value is freeflow set field freeflow_status = 1
	- if statusEnum value is heavy set field heavy_status = 1
	- if statusEnum value is congested set field congested_status = 1
	- if statusEnum value is imposible set field imposible_status = 1
- Change the holidayNum field from a integer to be represented as ten different integer fields like below	
	- if holidayNum value is 1 set field holiday_is_nyar = 1
	- if holidayNum value is 2 set field holiday_is_jul = 1
	- if holidayNum value is 3 set field holiday_is_forsta_maj = 1
	- if holidayNum value is 4 set field holiday_is_nationaldagen = 1
	- if holidayNum value is 6 set field holiday_is_pask = 1
	- if holidayNum value is 7 set field holiday_is_kristihimmelsfard = 1
	- if holidayNum value is 8 set field holiday_is_pingst = 1
	- if holidayNum value is 9 set field holiday_is_midsommar = 1
	- if holidayNum value is 10 set field holiday_is_allahelgona = 1
	- if holidayNum value is 11 set field holiday_is_trettondag = 1




# Original SegmentSpeed json datastructure
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