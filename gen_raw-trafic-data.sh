  
rm -f  MachineLearning_MK-1/raw-trafic-data.json
curl http://localhost:8080/api/v1/sections/32371/ml-speed-data\?size\=200\&page\=0 | jq '.'  | sed -e 's/]/,/' > MachineLearning_MK-1/raw-trafic-data.json
curl http://localhost:8080/api/v1/sections/32371/ml-speed-data\?size\=200\&page\=1 | jq '.'  | sed -e 's/]/,/' | sed -e 's/\[//' >> MachineLearning_MK-1/raw-trafic-data.json
curl http://localhost:8080/api/v1/sections/32371/ml-speed-data\?size\=200\&page\=2 | jq '.'  | sed -e 's/]/,/' | sed -e 's/\[//' >> MachineLearning_MK-1/raw-trafic-data.json
curl http://localhost:8080/api/v1/sections/32371/ml-speed-data\?size\=200\&page\=3 | jq '.'  | sed -e 's/]/,/' | sed -e 's/\[//' >> MachineLearning_MK-1/raw-trafic-data.json
