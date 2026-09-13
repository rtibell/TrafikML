# Task
I would like you to add an additional REST endpoint and a corresponding service that can return json formated information targeted to be used in a machine learning senario. 

# Instructions
- Creata a service named SpeedOfSectionMLData that will create speed of section data for a given section formated for usage by a machine learning model. The service should create Json formated data conforming to the format given in section MachineLearningSpeedOfSection and following the rules in Mappings below.
- Create a REST endpoint mapped to /api/v1/sections/{id}/ml-speed-data. The endpoint shoud handle GET requests and forwarad them to the SpeedOfSectionMLData service.
- Create a documentation describing how to utilize the REST endpoints, there address and arguments. Give examples of how to access them using CURL. Create the documentation in a markdown file REST-opperations.md


# Mappings
## statusEnum
	freeflow = 0
	heavy = 1
	congested = 2
	impossible = 3
## dayNr 
	map week days to integer number:
		monday = 0
		tuesday = 1
		wednsday = 2
		thursday = 3
		friday = 4
		saturday = 5
		sunday = 6
## daysUntilHoliday
	use the java file SwedishHolidays.java to calculate the daysUntilHoliday field


## holidayNr
	map type of holiday to a number:
	regular day = 0
	eve = 1
	holiday = 2
## minutesSincDaybreak
	number of minutes since 06:00 in the morning. Use measureTime as input value.
## monthOfYear
	month number starting with Januari = 1 and december = 12


# MachineLearningSpeedOfSection Json format
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
	"monthOfYear": 9
}