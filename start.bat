docker rm container fatsapiproject-rest-1
docker rmi fastapi
docker build -t fastapi .
pause
docker-compose up