# Premiss
You are a sesoned senior full stack developer with 30 years of experiances of programming and developing IT-systems mainly with Java and Spring boot. You are about to development a REST backend service with a SQL based database.
Follow the instructions given below and conform to god programming routines and architectural norms and patterns. For each service, utility och function there should excist a complet set of tests. Keep all configuration and dynamic information in application.yml configuration files. Note that ther are tree different environments: develoopment, test and production.
If you hava any questions don't hasitate to ask me any relevant question. 

Non functional requirements:
* Backend must use Java 25+
* Backend must use Spring Boot 4+, Spring Web, and expose REST endpoints returning JSON
* Use the build tool Gradle for Groovy to build the backend service
* Use Gradle V9 or later
* Use a Postgress geo-location enabled database.

# Task
Your goal is to develop a backend service that can be used to gather information on the traffic flow in the Stockholm region and store it in a Postgress SQL database. The information will in the next stage be used to drive a Machine Larning service that can determin when traffic congestions a likely to occure, whare and about what time. 

# Instruction
The solution should contain three different services eadch fetching datata from one of the spcified REST endpoints. Each service is opperatin within it's own time period and is doing one request each time intervall. Each service is making a REST request to it's target URL and retrievs the respons. The respons is processd and a verification is done against the database to terming if the information is new or updated. If the information is new, the field "recordCreated" is set to current time and it is insterted into the database. For updated information the excisting record is updated with the new informatioin. The field "recordUpdated" is updated with the current time. 

In the subchapters below the individual services are described.

## SpeedOfSection service
Generate a Spring Boot Service that:
- Request information from the given URL periodically by a scheduled service
- Convert the Json data into POJOs
- Verify if the POJO contains new or updated information using the id and measureTime as a composit key to determin if the record excists or not
- For POJOs containing new data insets it into the database
- For POJOS containing excisting data update the database with information from the POJO.
- The field "id" is used as a foreign key to the DefinitionOfSection
- The field "id" is converted to a Long

## DefinitionOfSection service
Generate a Spring Boot Service that:
- Request information from the given URL periodically by a scheduled service
- Convert the Json data into POJOs
- Verify if the POJO contains new or updated information using the id, modifiedTime and geometryModifiedTime 
- For POJOs containing new data insets it into the database
- For POJOS containing excisting data update the database with information from the POJO.
- The field "id" is used as the primary key
- The field "id" is converted to a Long


## TrafficMessages service
Generate a Spring Boot Service that:
- Request information from the given URL periodically by a scheduled service
- Convert the Json data into POJOs
- Insert the POJO into the database
- The field "id" is used as the primary key
- The field "id" is converted to a Long
- Format a information message containing the traffic information. Store the template in the application.yml configuration file
- Store the information message to the file system with the id as file name and ".dat" as prefix
- Send the informatioin message to Slack using a Slack webhook. The slack channel name is stored in the application.yml configuration file

Pleace generate the source code, information structures, database tables and files required for the backend service.



# Rules




# REST services used to obtain information regarding the trafic in stockholm
Below is three different REST endpoints presented. The table below gives the name, URL and short description.
For each endpint the json data format is given. 
| Name                | URL                                                                           | Intervall                            | Information returned                                              |
|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|
| SpeedOfSection      | https://trafiken.nu/api/traveltime?region=vst                                 | every minute between 07:00 and 19:00 | Speed and status of traffic on section.                           |
| DefinitionOfSection | https://trafiken.nu/api/traveltime/action/getsections?region=vst              | onece a day at 06:00                 | Definition of section geographical shape defined by polygon data. |
| TrafficMessages     | https://trafiken.nu/api/trafficmessages?region=vst&trafficType=V%C3%A4gtrafik | every 10 minutes                     | Traffic messages sent when extrordenary events occures.           |

## SpeedOfSection information format
```json
[{
    "id": "44469",
    "status": "freeflow",
    "speed": 48,
    "measureTime": "2026-09-12T17:19:00"
}]

## DefinitionOfSection information format
```json
[{
    "type": "Feature",
    "id": "45553",
    "geometry": {
      "type": "LineString",
      "coordinates": [
        [
          17.910571,
          58.972645
        ],
        [
          17.911648,
          58.974388
        ],
        [
          17.91179,
          58.974601
        ],
        [
          17.912065,
          58.974961
        ],
        [
          17.912582,
          58.975545
        ],
        [
          17.913287,
          58.976272
        ],
        [
          17.913494,
          58.97644
        ],
        [
          17.913982,
          58.976745
        ],
        [
          17.914089,
          58.976846
        ],
        [
          17.914212,
          58.977026
        ]
      ]
    },
    "properties": {
      "name": "Avfart Norrut trafikplats Ösmo",
      "region": "VST",
      "countyNo": 1,
      "modifiedTime": "2026-05-21T08:30:52",
      "geometryModifiedTime": "2026-03-11T03:29:52",
      "activated": true,
      "averageFunctionalRoadClass": 3,
      "importedTime": "2026-05-21T10:31:12.6484827"
    }
}]

## TrafficMessages information format
```json
[{
    "id": 11851210,
    "region": "VST",
    "title": "Väg 950 vid Granby båda riktningarna",
    "message": "<span>Akut ledningsarbete.</span><br/><br/><b>Tillfälliga begränsningar:</b><br/>Ett körfält avstängt",
    "provider": "Trafik Stockholm",
    "messageType": "Vägarbete",
    "messageCode": "Vägarbete",
    "trafficType": "Vägtrafik",
    "wgs84Position": {
      "type": "Point",
      "coordinates": [
        18.09670398124536,
        59.57501383535391
      ]
    },
    "sweRef99Extent": {
      "type": "LineString",
      "coordinates": [
        [
          675188.3998505928,
          6605920.663208873
        ],
        [
          675319.3698504879,
          6606228.813208379
        ],
        [
          675304.9798505168,
          6606940.903207208
        ],
        [
          675184.8398506269,
          6607291.853206627
        ],
        [
          675009.479850783,
          6607611.403206097
        ],
        [
          674893.8398508943,
          6608173.683205169
        ],
        [
          674961.6498508446,
          6608564.3132045325
        ],
        [
          674974.2998508346,
          6608570.853204522
        ],
        [
          674967.6598508407,
          6608584.613204499
        ],
        [
          675064.3298507637,
          6608814.333204132
        ],
        [
          675097.1098507366,
          6608878.463204029
        ],
        [
          675233.8498506227,
          6608992.803203846
        ],
        [
          675310.4898505617,
          6609158.013203579
        ],
        [
          675431.8298504723,
          6609760.653202596
        ],
        [
          675429.6098504759,
          6609852.823202445
        ],
        [
          675381.7998505204,
          6610002.483202198
        ],
        [
          675208.6798506747,
          6610290.923201717
        ],
        [
          675214.0998506695,
          6610309.103201689
        ]
      ]
    },
    "affectedDirection": "Both",
    "startTime": "2026-09-14T07:00:00",
    "endTime": "2026-10-12T16:00:00",
    "scheduledOccurrences": [],
    "versionTime": "2026-09-11T13:17:26",
    "iconId": 162,
    "severity": 2,
    "isFuture": true,
    "roadClosed": false,
    "detailsPath": "https://trafiken.nu/stockholm/trafikinformation/11851210/vag-950-vid-granby-bada-riktningarna/",
    "sortIndex": 4347,
    "severityText": "Liten påverkan",
    "workState": 0
}]

