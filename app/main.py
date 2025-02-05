from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import os

# Categorical Mappings for both month and days
month_mapping = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 
                 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
day_mapping = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5, 'sat': 6, 'sun': 7}

# Defining the custom mse function
def mse(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)

# Creating the FastAPI application
app = FastAPI(title="Forest Fire Burn Area Predictor")

# Determining the base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Conditionally mount static files if directory exists
static_dir = os.path.join(BASE_DIR, 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Setting up the Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, 'templates'))

# Loading the pre-trained model
model_path = os.path.join(BASE_DIR, 'forest_fire_model.keras')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Addding custom objects when loading the model
model = load_model(model_path, custom_objects={'mse': mse})

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request):
    # Getting the form data
    form_data = await request.form()
    try:
        # Preparing the input data with categorical mapping
        input_dict = {
            'X': [int(form_data['X'])],
            'Y': [int(form_data['Y'])],
            'month': [month_mapping[form_data['month']]],  # Mapping to numerical value
            'day': [day_mapping[form_data['day']]],        # Mapping to numerical value
            'FFMC': [float(form_data['FFMC'])],
            'DMC': [float(form_data['DMC'])],
            'DC': [float(form_data['DC'])],
            'ISI': [float(form_data['ISI'])],
            'temp': [float(form_data['temp'])],
            'RH': [int(form_data['RH'])],
            'wind': [float(form_data['wind'])],
            'rain': [float(form_data['rain'])]
        }
    except KeyError as e:
        return {"error": f"Missing or invalid form field: {e}"}

    # Converting the input_dict to DataFrame for prediction
    input_df = pd.DataFrame(input_dict)

    # Storing the original categorical values for display from html
    original_month = form_data['month']
    original_day = form_data['day']

    # Making the prediction using our already saved ML model forest_fire_model.keras
    prediction = model.predict(input_df)
    prediction = np.expm1(prediction)  # Reverse log transformation
    prediction = np.clip(prediction, 0, None)  # Replace negative values with 0

    # Converting the prediction to a scalar value and rounding it
    prediction_value = float(np.round(prediction[0], 2))

    # Rendering the result template with both mapped and original values for displaying the output to result.html
    return templates.TemplateResponse(
        "result.html", 
        {
            "request": request, 
            "prediction": prediction_value,
            "input_data": input_dict,
            "original_month": original_month,
            "original_day": original_day
        }
    )
