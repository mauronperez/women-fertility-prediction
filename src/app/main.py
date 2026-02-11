
"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="Women Fertility Prediction API",
    description="ML API for predicting childless women",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class WomenData(BaseModel):
    """
    Customer data schema for churn prediction.
    
    This schema defines the exact 18 features required for churn prediction.
    All features match the original dataset structure for consistency.
    """
    age: str             
    married: str               # "Yes" or "No"
    housing_payment: str     
    
    housing_income: str         
    job: str         # "Yes", "No"
    
    level_studies: str
    income: str 
    number_living: str 
    house_income :str

    


# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: WomenData):
    """
    Main prediction endpoint for customer churn prediction.
    
    This endpoint:
    1. Receives validated customer data via Pydantic model
    2. Calls the inference pipeline to transform features and predict
    3. Returns churn prediction in JSON format
    
    Expected Response:
    - {"prediction": "Likely to churn"} or {"prediction": "Not likely to churn"}
    - {"error": "error_message"} if prediction fails
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        # Return error details for debugging (consider logging in production)
        return {"error": str(e)}


# =================================================== # 


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    age,married,housing_payment,level_studies,income,number_living,job,house_income
):
    """
    Gradio interface function that processes form inputs and returns prediction.
    
    This function:
    1. Takes individual form inputs from Gradio UI
    2. Constructs the data dictionary matching the API schema
    3. Calls the same inference pipeline used by the API
    4. Returns user-friendly prediction string
    
    """
    # Construct data dictionary matching WomenData schema
    data = {
        "Age": age,
        "Married": married,
        "Renting/Income in %": housing_payment,
        "Level of Studies": level_studies,
        "Income": income,
        "Residence's number of people living": number_living,
        "Job": job,
        "House Income": house_income
    }
    
    # Call same inference pipeline as API endpoint
    result = predict(data)
    return str(result)  # Return as string for Gradio display

# === GRADIO UI CONFIGURATION ===
# Build comprehensive Gradio interface with all customer features
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Demographics section
        gr.Dropdown(["Older than 30", "Less than 30"], label="Age", value="Older than 30"),
        gr.Dropdown(["Yes", "No"], label="Married", value="Yes"),
        gr.Dropdown(["Yes", "No"], label="Job", value="No"),
        gr.Dropdown(["Low", "Middle", "High"], label="Income", value="No"),
        gr.Dropdown(["Low", "Middle", "High"], label="House Income", value="No"),
        gr.Dropdown(["Less than 20%", "Between 20 and 60%", "More than 60%"], label="Renting/Income in %", value="No"),
        gr.Dropdown(["No studies", "Graduate", "Post-Graduate"], label="Level of Studies", value="No"),
        gr.Dropdown(["Less than 4", "More than 4 or 4"], label="Residence's number of people living", value="No"),
    ],
    
    outputs=gr.Textbox(label="Prediction Result", lines=2),
    title="👶 Women's Fertility Prediction",
    description="""
    **Predict whether a woman has children using machine learning**
    
    Fill in the demographic and socioeconomic details below to get a prediction. The model uses 
    XGBoost trained on historical data to predict the likelihood of having children.
    
    💡 **Tip**: Factors like age, marital status, employment, and income level are strong 
    indicators in the prediction model.
    """,
    examples=[
        # High likelihood of having children
        ["Older than 30", "Yes", "Yes", "High", "High", "Between 20 and 60%", "Post-Graduate", "More than 4 or 4"],
        
        # Low likelihood of having children
        ["Less than 30", "No", "No", "Low", "Low", "Less than 20%", "No studies", "Less than 4"],
        
        # Medium likelihood of having children
        ["Older than 30", "Yes", "No", "Middle", "Middle", "More than 60%", "Graduate", "Less than 4"],
        
        # Young married professional
        ["Less than 30", "Yes", "Yes", "High", "High", "More than 60%", "Post-Graduate", "Less than 4"],
    ],
    
    )

# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)
