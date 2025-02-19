![](https://i.imgur.com/wBGNsPw.jpeg)
# Meta Comprehensive RAG Benchmark (CRAG) Mock API for KDD Cup 2024: A Quick Start Guide

Welcome to the official repository of the Meta Comprehensive RAG Benchmark (CRAG) Mock API.

## Prerequisites

Before diving into the setup and usage of the CRAG Mock API, ensure you have the following prerequisites installed and set up on your system:
- Git (for cloning the repository)
- Docker (optional, for containerized execution)
- Python 3.10

## Installation Guide

### Cloning the Repository

First, clone the repository to your local machine using Git.

```
git clone git@gitlab.aicrowd.com:aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024/crag-mock-api.git
```

### Setting Up Your Environment

Navigate to the repository directory and install the necessary dependencies:

```
cd crag-mock-api
pip install -r requirements.txt
```

## Running the API Server

To launch the API server on your local machine, use the following Uvicorn command. This starts a fast, asynchronous server to handle API requests.

```
uvicorn server:app --reload
```

Access the API documentation and test the endpoints at `http://127.0.0.1:8000/docs`.

For custom server configurations, specify the host and port as follows:

```
uvicorn server:app --reload --host [HOST] --port [PORT]
```

### Using Docker for Simplified Deployment

Running the API server with Docker is an excellent option for consistent and isolated environments. To start the server in a Docker container:

```
export IMAGE_NAME="docker.io/aicrowd/kdd-cup-24-crag-mock-api:v0"
docker run -d -p=8000:8000 $IMAGE_NAME
```

The Docker container version simplifies setup, especially across different operating systems and configurations.

## System Requirements

- **Supported OS**: Linux, Windows, macOS
- **Python Version**: 3.10
- See `requirements.txt` for a complete list of Python package dependencies.

## Python API Wrapper

For Python developers, the [apiwrapper/pycragapi.py](apiwrapper/pycragapi.py) provides a convenient way to interact with the API. An example usage is demonstrated in [apiwrapper/example_call.ipynb](apiwrapper/example_call.ipynb), showcasing how to efficiently integrate the API into your development workflow.

## Docker Releases and Versioning

- Current Release: `v1`
    - **Release Date**: Tue May 14 16:42:11 UTC 2024 
    - **Docker Images**:
        - `docker.io/aicrowd/kdd-cup-24-crag-mock-api:v1`
        - `docker.io/aicrowd/kdd-cup-24-crag-mock-api:v1-613ccf7`

### Release History
- Release: `v1`
    - **Release Date**: Tue May 14 16:42:11 UTC 2024 
    - **Docker Images**:
        - `docker.io/aicrowd/kdd-cup-24-crag-mock-api:v1`
        - `docker.io/aicrowd/kdd-cup-24-crag-mock-api:v1-613ccf7`

- Release: `v0`
    - **Release Date**: Sat Mar 30 04:05:14 UTC 2024
    - **Docker Images**:
        - `docker.io/aicrowd/kdd-cup-24-crag-mock-api:v0`
        - `docker.io/aicrowd/kdd-cup-24-crag-mock-api:v0-4fb3254`

The versioning of Docker images allows you to choose between the latest updates or specific stable versions for your testing environment.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This license permits sharing and adapting the work, provided it's not used for commercial purposes and appropriate credit is given. For a quick overview, visit [Creative Commons License](https://creativecommons.org/licenses/by-nc/4.0/).


### Contributions

We welcome contributions to the CRAG Mock API! If you have suggestions or improvements, please feel free to send across a pull request.
