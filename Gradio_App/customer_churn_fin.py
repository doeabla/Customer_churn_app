import gradio as gr
import pickle
import sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from gradio.components import Label



# copy all necessary features for encoding and scaling in the same sequence from your python codes
expected_inputs = ["TotalCharges","MonthlyCharges","Tenure","Gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
#group the numeric and categorical features exactly as was done in the python codes
numerical = ["TotalCharges","MonthlyCharges","Tenure"]
categoricals = ["Gender","SeniorCitizen","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]


# Load your pipeline, Scaler, and encoder
def load_pipeline(file_path):
    with open(file_path, "rb") as file:
        pipeline = pickle.load(file)
    return pipeline

# Load the pipeline (assuming it contains preprocessing steps)
pipeline = load_pipeline(r"C:\Users\USER\Azubi LP2\Gradio_App\pipeline2.pkl")

# Load your RandomForestClassifier model separately
with open(r'C:\Users\USER\Azubi LP2\Gradio_App\random_forest_classifier.pkl', "rb") as file:
    randf_classifier = pickle.load(file)




def predict_churn(*args, model=randf_classifier, pipeline=pipeline):
    try:
        # Create a DataFrame with input data
        input_data = pd.DataFrame([args], columns=expected_inputs)

        # Preprocess the input data using the loaded pipeline
        preprocessed_data = pipeline.transform(input_data)

        # Make predictions using the RandomForestClassifier model
        model_output = model.predict_proba(preprocessed_data)[:, 1]  # Use predict_proba to get probabilities

        # Convert model output to a single probability value
        probability_of_churn = model_output[0]

        # You can adjust the threshold as needed
        threshold = 0.5

        # Compare the probability with the threshold to make a binary prediction
        if probability_of_churn > threshold:
            return "Churn"
        else:
            return "No Churn"
    except Exception as e:
        # Handle exceptions gracefully
        return f"Error: {str(e)}"
    


    
# Define expected_inputs, numerical, and categoricals here
TotalCharges= gr.Number(label="Total Charges ($)")
MonthlyCharges = gr.Number(label="Monthly Charges ($)")
Tenure= gr.Number(label="Number of months being on the network",minimum=1)
Gender = gr.Radio(label="Gender ", choices=["Male", "Female"])
SeniorCitizen = gr.Radio(label="Senior or non Senior citizen", choices=["Yes", "No"])
Partner = gr.Radio(label="Partner", choices=["Yes", "No"])
Dependents = gr.Radio(label="Dependants", choices=["Yes", "No"])
PhoneService = gr.Radio(label="Phone services", choices=["Yes", "No"])
MultipleLines = gr.Radio(label="Multiple Lines", choices=["Yes", "No"])
InternetService = gr.Radio(label="Internet Service", choices=["DSL", "Fiber optic", "No"])
OnlineSecurity = gr.Radio(label="Online Security", choices=["Yes", "No", "No internet service"])

OnlineBackup = gr.Radio(label="Requested for Online backup", choices=["Yes", "No", "No internet service"])
DeviceProtection = gr.Radio(label="Device Protection", choices=["Yes", "No", "No internet service"])
TechSupport = gr.Radio(label="Tech Support", choices=["Yes", "No", "No internet service"])
StreamingTV = gr.Radio(label="Streaming TV", choices=["Yes", "No", "No internet service"])
StreamingMovies = gr.Radio(label="Streaming Movies", choices=["Yes", "No", "No internet service"])
Contract = gr.Radio(label="Contract Term", choices=["Month-to-month", "One year", "Two year"])
PaperlessBilling = gr.Radio(label=" Paperless Billing", choices=["Yes", "No"])
PaymentMethod = gr.Radio(label="Payment Method", choices=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])


# Define Gradio interface components
inputs = [TotalCharges, MonthlyCharges, Tenure, Gender, SeniorCitizen, Partner, Dependents, PhoneService,
          MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
          StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod]





# Define the image path and designer text
image_path = "C:/Users/USER/Azubi LP2/Gradio_App/customer churn.png"

designers_names = "Designed by: Doe Edinam, Enoch Taylor-Nketiah & Kofi Asare Bamfo"





# Define Gradio interface with the image and names included in the description
interface = gr.Interface(
    fn=predict_churn,
    inputs=inputs,
    outputs=gr.outputs.Label(),
    title="<b>Customer Attrition Prediction App</b>",
    description=f"<div style='text-align: center;'><img src='{image_path}' style='width: 50%;'><br><br>{designers_names}</div><br><b>Enter customer information to predict churn.</b>",
    live=True,
    css=".gradio-container {background-color: lightblue;}",
)



# Launch the Gradio app
interface.launch(inbrowser=True)



