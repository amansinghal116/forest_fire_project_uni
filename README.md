# Forest Fire Burn Area Predictor

This is a containerized FastAPI application that predicts the burn area of forest fires based on meteorological and geographical inputs. 
The application uses a pre-trained Keras model and provides an interactive web interface for making predictions.

## Technologies Used:
FastAPI: Backend framework for building APIs.
TensorFlow/Keras: Framework for the pre-trained model.
Jinja2: Template engine for rendering HTML pages.
Docker: For containerized deployment.

## Project Structure:
forest_fire_project/
│
├── app/
│   ├── main.py               # FastAPI Main application 
│   ├── templates/            # HTML templates for the app
│   │   ├── index.html        # Form for user input
│   │   └── result.html       # Displays prediction results
│   ├── static/               # Static files (CSS, images, etc.)
│   └── forest_fire_model.keras  # Pre-trained Keras model
├── Dockerfile                # Docker configuration
├── requirements.txt          # List of Python dependencies
└── README.md                 # Documentation

## Local Setup Without Docker
Step1: Navigate to the forest_fire_project directory
Step2: Install dependencies: pip install -r requirements.txt
Step3: cd app
Step4: Run the application: uvicorn main:app --host 0.0.0.0 --port 8000


## Setup Using Docker
Step1: Navigate to the forest_fire_project directory
Step2: Build the Docker image: docker build -t forest_fire_predict .
Step3: Run the Docker container: docker run --name forest_fire -p 8000:8000 forest_fire_predict
Step4: Access the application: Open your browser and navigate to http://0.0.0.0:8000 

## Usage
Launch the application in your browser at http://0.0.0.0:8000
Fill in the required fields:
X, Y: Spatial coordinates.
Month, Day: Temporal information.
FFMC, DMC, DC, ISI: Meteorological indexes.
Temp, RH, Wind, Rain: Weather conditions.
Click Submit to get the predicted burn area.

## Custom Loss Function:
The model uses a custom mean squared error (MSE) function during training and evaluation
def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)
The custom function is registered and passed while loading the model
model = load_model(model_path, custom_objects={'mse': mse})

## Troubleshooting
1.Model File Not Found: Ensure forest_fire_model.keras is present in the app/ directory.
2.Custom Function Error: If encountering issues with mse, verify that it is correctly defined and passed to load_model.
3.Docker Port Conflict: If port 8000 is in use, modify the -p flag during container run:
docker run --name forest_fire -p 8080:8000 forest_fire_predict
4.Dependencies Issue: If you face any dependency errors, ensure all libraries in requirements.txt are installed correctly.
