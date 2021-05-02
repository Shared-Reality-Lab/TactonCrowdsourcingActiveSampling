# TactonCrowdsourcingActiveSampling
Code for the TactonCrowdsourcingActiveSampling project, accepted at the World Haptics Conference (WHC) 2021.

### Features

* Android application for generating tactons on smartphone and rate them (open with Android Studio)
* Active sampling scheme for selecting tactons to be rated
* Containerized PostgreSQL database and web-app for handling server-side requests
* Flask Application for the webapp
* Slave & Master database replication for scheduled backups and data security

### To spin up Docker containers

Install Docker, and docker-compose, then run:
```sudo docker-compose up --build -d```
