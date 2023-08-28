# Debian - Python - Mod_Wsgi

This web application is exposing api for prediction running a python app under `mod_wsgi` on apache in a docker container

This repository is for a production ready environment, it does not run a flask local server but an apache server in docker.

This docker image will be built in stages
It will contain the application written using the Flask framework

I will be using a serialezed statistical model (a pickle file)

The command to run the `Dockerfile` is:

`docker run -d -p 80:80 --name <name> vamax-as-a-service`

Alternatively, you can use docker-compose with:

`docker-compose up -d`

 * Download the repo
 * build the image: `docker build -t vamax-as-a-service .`

The swagger API page can be accessed via the following url:

http://localhost/apidocs/

#### The docker file runs through the following steps:  

 - get debian bullseye-slim image.  
 - install the requirements for python and flask on debian  
 - copy over the `requirements.txt` file and run `pip install` on it  
 - This is copied separately so that the dependencies are cached and dont need to run everytime the image is rebuilt  
 - copy over the application config file for apache  
 - copy over the `.wsgi` file. This is the entrypoint for our application, the `run.py` file, and the application directory  
 - enable the new apache config file and headers   
 - dissable the default apache config file  
 - expose port 80  
 - point the container to the application directory  
 - the run command. 
