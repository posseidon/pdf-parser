# Python Container Application

This project is a Python application designed to run in a container for scalability. It includes all necessary files to build and run the application using Docker.

## Project Structure

```
python-container-app
├── src
│   └── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

## Getting Started

To get started with this project, follow the instructions below.

### Prerequisites

Make sure you have the following installed on your machine:

- Docker
- Python (for local development)

### Installation

1. Clone the repository:

   ```
   git clone <repository-url>
   cd python-container-app
   ```

2. Install the required Python packages:

   ```
   pip install -r requirements.txt
   ```

### Building the Docker Image

To build the Docker image for the application, run the following command in the project root directory:

```
docker build -t python-container-app .
```

### Running the Application

Once the image is built, you can run the application using:

```
docker run -p 5000:5000 python-container-app
```

### Accessing the Application

After running the container, you can access the application at `http://localhost:5000`.

### Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request.

### License

This project is licensed under the MIT License. See the LICENSE file for details.