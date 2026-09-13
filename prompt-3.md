# Modifications
- Database has a column named "measure_time" but the corresponding attribut "measureTime" is missing in the Java code. 
- Verify that the desing of the database and the entities is in sync.
- Verify that the class DefinitionOfSectionRawDto is compatible with the structure returned from the endpoint https://trafiken.nu/api/traveltime/action/getsections?region=vst