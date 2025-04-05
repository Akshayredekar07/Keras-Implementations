import pandas as pd
import numpy as np
import gradio as gr
import pickle
import os
from tensorflow.keras.models import load_model

print("Loading model and preprocessor...")

# Check if files exist
if not os.path.exists('churn_model.h5') or not os.path.exists('churn_preprocessor.pkl'):
    print("Error: Model files not found. Please run train_model.py first.")
    exit(1)

# Load the model
model = load_model('churn_model.h5')

# Load preprocessing objects
with open('churn_preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

scaler = preprocessor['scaler']
geo_encoder = preprocessor['geo_encoder']
gender_encoder = preprocessor['gender_encoder']
geo_mapping = preprocessor['geo_mapping']
gender_mapping = preprocessor['gender_mapping']
feature_names = preprocessor['feature_names']

# Function to make predictions
def predict_churn(credit_score, geography, gender, age, tenure, balance, 
                  num_of_products, has_credit_card, is_active_member, estimated_salary):
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [int(has_credit_card)],
        'IsActiveMember': [int(is_active_member)],
        'EstimatedSalary': [estimated_salary]
    })
    
    # Encode categorical variables
    input_data['Geography'] = input_data['Geography'].map(geo_mapping)
    input_data['Gender'] = input_data['Gender'].map(gender_mapping)
    
    # Ensure column order matches what the model expects
    input_data = input_data[feature_names]
    
    # Scale the input data
    scaled_data = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_data)
    
    # Get the probability of churn (class 1)
    churn_probability = prediction[0][1] * 100
    
    # Create result message
    if churn_probability > 50:
        result = f"⚠️ HIGH RISK OF CHURN: This customer has a {churn_probability:.2f}% probability of leaving the bank."
    else:
        result = f"✅ LOW RISK OF CHURN: This customer has only a {churn_probability:.2f}% probability of leaving the bank."
    
    # Create detailed explanation
    explanation = f"""
### Customer Profile Analysis:
- **Credit Score**: {credit_score} {'(Good)' if credit_score > 700 else '(Average)' if credit_score > 600 else '(Poor)'}
- **Location**: {geography}
- **Gender**: {gender}
- **Age**: {age} years {'(Higher churn risk)' if age > 45 else ''}
- **Tenure**: {tenure} years with the bank
- **Balance**: ${balance:,.2f}
- **Products**: {num_of_products} {'(Multiple products may indicate higher loyalty)' if num_of_products > 1 else ''}
- **Credit Card**: {'Yes' if has_credit_card else 'No'}
- **Active Member**: {'Yes' if is_active_member else 'No'} {'(Inactive members have higher churn risk)' if not is_active_member else ''}
- **Estimated Salary**: ${estimated_salary:,.2f}

### Churn Probability: {churn_probability:.2f}%
"""
    
    return result, explanation, churn_probability

# Create Gradio interface
print("Setting up Gradio interface...")
with gr.Blocks(title="Bank Customer Churn Predictor") as app:
    gr.Markdown("# 🏦 Bank Customer Churn Prediction")
    gr.Markdown("Enter customer details to predict the likelihood of churn")
    
    with gr.Row():
        with gr.Column():
            credit_score = gr.Slider(label="Credit Score", minimum=300, maximum=900, value=650, step=1)
            geography = gr.Dropdown(label="Country", choices=list(geo_mapping.keys()), value=list(geo_mapping.keys())[0])
            gender = gr.Radio(label="Gender", choices=list(gender_mapping.keys()), value=list(gender_mapping.keys())[0])
            age = gr.Slider(label="Age", minimum=18, maximum=100, value=35, step=1)
            tenure = gr.Slider(label="Tenure (Years with Bank)", minimum=0, maximum=10, value=5, step=1)
        
        with gr.Column():
            balance = gr.Number(label="Account Balance ($)", value=75000)
            num_of_products = gr.Slider(label="Number of Products", minimum=1, maximum=4, value=1, step=1)
            has_credit_card = gr.Checkbox(label="Has Credit Card", value=True)
            is_active_member = gr.Checkbox(label="Is Active Member", value=True)
            estimated_salary = gr.Number(label="Estimated Salary ($)", value=50000)
    
    with gr.Row():
        predict_btn = gr.Button("Predict Churn Probability", variant="primary")
    
    with gr.Row():
        output_result = gr.Textbox(label="Prediction Result")
    
    with gr.Row():
        output_explanation = gr.Markdown(label="Customer Analysis")
    
    with gr.Row():
        output_gauge = gr.Number(label="Churn Probability (%)")
    
    predict_btn.click(
        fn=predict_churn,
        inputs=[
            credit_score, geography, gender, age, tenure, 
            balance, num_of_products, has_credit_card, 
            is_active_member, estimated_salary
        ],
        outputs=[output_result, output_explanation, output_gauge]
    )

# Launch the app
if __name__ == "__main__":
    print("Launching Gradio interface. Press Ctrl+C to exit.")
    app.launch()