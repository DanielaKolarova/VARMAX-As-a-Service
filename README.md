## Deploying Statistical Models using Flask and Python
This is a project to elaborate how to deploy a statistical model from the VARMAX family using a combination of Flask API deployed using a production ready docker (docker-compose) file.


### Project Structure
This project has two major parts :
1. src/models - This folder contains code fot the statistical model, data loading, model optimization, testing, serialization
2. web-app - This folder contains the Flask APIs, web project, serialized model and Docker/Docker-Compose setup files

In addition to the main folders there are some helper folders:

1. notebooks - This folder contains jypiter notebooks with various model experiments
2. diagrams - This folder contains documentation diagrams
3. data - This folder is dedicated to data storage that might be needed for models training and testing

### Running the project
1. Ensure that you are in the project home directory. Create the model by running below command from command prompt -
```
python varmax_model.py
```
This would create a serialized version of our model into a file model.pkl located in web-app/savedmodels

2. Run app.py using below command to start Flask API
```
python run.py

or using docker-compose within web-app run

 - docker-compose up -d
 - 

3. Navigate to URL http://127.0.0.1:80 (or) http://localhost:80 You should be able to view the homepage.

4. To access MinIO web console navigate to http://localhost:9101/ (or) http://127.0.0.1:9101/

