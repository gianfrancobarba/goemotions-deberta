# start.ps1
$env:COMPOSE_BAKE = "true"
docker compose up --build
