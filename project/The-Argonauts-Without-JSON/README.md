# Spotify Recommender with fastAPI Integration

A recommendation algorithm that provides recommendations for Spotify music and artist.
The algorithm we plan on using is a simple unsupervised KNN algorithm that will predict the
nearest neighbors of each song and therefore provide similar songs as recommendations
when prompted.

the playbook located at [a relative link](sample_model/explore_spotify_data.ipynb) /sample_model/explore_spotify_data.ipynb contains all info for model again we use a clean version but that to show our poc


### Team: Argnauts without JSON 

# tools used
- KNN 
- FastAPI
- uvicorn for deploying endpoints

# MODEL KNN 
This template contains code and pipeline definition for a machine learning project demonstrating how to automate the end to end ML/AI project. The build pipelines include DevOps tasks for data sanity test, unit test, model training on different compute targets, model version management, model evaluation/model selection, model deployment as realtime web service, staged deployment to QA/prod, integration testing and functional testing.

## Project structure 
.
├── LICENSE
├── MANIFEST
├── README.md
├── docs
│   ├── DOCS.md
│   ├── authorize.png
│   ├── sample_payload.json     # Sample Payload to test ML model to be added for front end
│   └── sample_payload.png
├── fastapi_skeleton            # Skeleton Module
│   ├── __init__.py.             
│   ├── api                     # API related code
│   │   ├── __init__.py
│   │   └── routes              # All routes provided by API
│   │       ├── __init__.py
│   │       ├── heartbeat.py    # Route to check if is server is up
│   │       ├── playlist.py   # Route to playlist using ML model( to be finalized)
│   │       └── router.py       # Main router to  serves the routers
│   ├── core
│   │   ├── __init__.py
│   │   ├── config.py           # Server configuration helper
│   │   ├── event_handlers.py   # Handle server start/stop
│   │   ├── messages.py         # Shared messages/resources
│   │   └── security.py         # Common security helpers
│   ├── main.py                 # App entrypoint
│   ├── models
│   │   ├── __init__.py
│   │   ├── heartbeat.py        # Data model for heartbeat response
│   │   ├── payload.py          # Data model for ML model payload
│   │   └── playlist.py         # Data model for playlist raccomander result
│   └── services
│       ├── __init__.py      
│       └── models.py           # Services to provide ML model
├── requirements.txt            # Project requirements
├── sample_model                # Sample ML model folder
│   ├── Models                  # our bin
│   └── Utils                   # funtions to improve performance      
│       └── scaler.py           # data manuplation and scaling
│       └── split_data.py       # split the data        
│   ├── interference.py         
│   └── train.py                # traim our model         
│   ├── poc.py                  # our initial prove of concept to be removed 
│   └── train.py                # traim our model       
├── setup.py                    # Python setup script for dist
├── tests                       # Test folder
│   ├── __init__.py
│   ├── conftest.py             # Test configuration / bootstrap
│   ├── test_api                # API tests
│   │   ├── __init__.py 
│   │   ├── test_api_auth.py    # API authentication test cases
│   │   ├── test_heartbeat.py   # API Heartbeat test cases
│   │   └── test_playlist.py  # API ML model raccomander test cases(future)
│   └── test_service
│       ├── __init__.py       
│       └── test_models.py      # ML model test cases ( next stage)
└── tox.ini                     # Tox configuration


## FastApi
- we have a token saved inside .env to authenticate the app
- ![Alt text](env.png "Optional title")
- end points will look like following once hooked
- ![Alt text](structure.png "Optional title")



### Next steps
- Finalize the end point integration
- Dockerize the backend
- Create a vue front end and dockerize it
- Create a docker-compose so we have elastic front-backend elasticity
- use NoSQL db instead of flat csv ( for that we created a data adapter for easier code modifications)
- Finalize unit tests and connect to CICD pipeline

# Contributing
we are welcoming contributors to finish the project


