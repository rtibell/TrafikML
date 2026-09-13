# Task
Please add a additional field to the MachineLearningSpeedOfSection json structure produced by the service SpeedOfSectionMLData that tracks the holiday number. 

# Instructions
- add the integer field holidayNum to the HolidayResult class in the utility class SwedishHolidays.java that keeps track of the holiday as a predefined integer.
- update the method nextHoliday in the utility class SwedishHolidays.java to calculate the integer represntation of the holida and add it to holidayNum in the HolidayResult.
- add the holidayNum field returned by the method nextHoliday in class SwedishHolidays to the output formated as MachineLearningSpeedOfSection of service SpeedOfSectionMLData.


## MachineLearningSpeedOfSection Json format
{
	"sectionId": 3333,
	"measureTime": "2026-09-13T13:44:00",
	"status": "freeflow",
	"speed": 55,
	"statusEnum": 1,
	"dayNr": 0,
	"daysUntilHoliday": 5,
	"holidayNr": 1
	"minutesSincDaybreak: 533243, 
	"monthOfYear": 9,
	"holidayNum": 2
}