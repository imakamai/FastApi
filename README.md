# FastApi
Ova skripta (start.abt) služi za automatsko pokretanje FastAPI aplikacije koristeći Docker i docker-compose. U nastavku je detaljan opis komandi koje se izvršavaju i kako da koristite ovu skriptu.

Sadržaj skripte
bash
Copy
Edit
docker rm container fatsapiproject-rest-1
docker rmi fastapi
docker build -t fastapi .
pause
docker-compose up


Objašnjenje komandi
docker rm container fatsapiproject-rest-1

Briše postojeći kontejner po imenu fatsapiproject-rest-1 (ako postoji).

Napomena: Ovo će raditi samo ako kontejner postoji i nije pokrenut. Ako jeste, potrebno je prethodno zaustaviti kontejner (docker stop fatsapiproject-rest-1).

docker rmi fastapi

Briše prethodno napravljenu Docker sliku sa imenom fastapi kako bi se ponovo izgradila sveža slika.

docker build -t fastapi .

Pravi novu Docker sliku iz Dockerfile-a u trenutnom direktorijumu i imenuje je fastapi.

pause

Pauzira izvršavanje skripte dok korisnik ne pritisne taster. Ovo omogućava korisniku da vidi rezultate prethodnih komandi pre nastavka.

docker-compose up

Pokreće aplikaciju koristeći docker-compose.yml fajl. Svi servisi definisani u ovom fajlu biće pokrenuti.


Kako koristiti
Uverite se da su Dockerfile i docker-compose.yml fajlovi pravilno podešeni u glavnom direktorijumu projekta.

Pokrenite skriptu dvoklikom na start.abt fajl (ili je otvorite u terminalu).

Nakon pauze, pritisnite bilo koji taster da nastavite i pokrenete aplikaciju.


Napomene
Proverite da li je Docker pokrenut pre pokretanja skripte.

Ako dobijete grešku da kontejner ne može da se obriše jer je aktivan, dodajte docker stop fatsapiproject-rest-1 pre prve komande.

Ime kontejnera (fatsapiproject-rest-1) i ime slike (fastapi) moraju se poklapati sa onima koje koristite u docker-compose.yml.
