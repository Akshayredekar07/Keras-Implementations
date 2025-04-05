import numpy as np
import tensorflow as tf
import gradio as gr
import joblib

def predict_admission(
    gre_score, 
    toefl_score, 
    university_rating, 
    sop, 
    lor, 
    cgpa, 
    research
):
    """
    Predict admission probability using the trained model
    """
    # Create input array
    input_data = np.array([[
        gre_score, 
        toefl_score, 
        university_rating, 
        sop, 
        lor, 
        cgpa, 
        research
    ]])
    
    try:
        # Load the scaler
        scaler = joblib.load('admission_scaler.pkl')
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Load the model
        model = tf.keras.models.load_model('admission_model.h5')
        
        # Make prediction
        prediction = model.predict(input_scaled)[0][0]
        
        # Convert to percentage
        admission_prob = round(prediction * 100, 2)
        
        # Return interpretation
        if admission_prob >= 70:
            result = "High chance of admission"
        elif admission_prob >= 40:
            result = "Moderate chance of admission"
        else:
            result = "Low chance of admission"
        
        return f"Admission Probability: {admission_prob}%\n{result}"
    
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Define Gradio interface components
inputs = [
    gr.Slider(290, 340, step=1, label="GRE Score", value=310),
    gr.Slider(92, 120, step=1, label="TOEFL Score", value=100),
    gr.Slider(1, 5, step=1, label="University Rating", value=3),
    gr.Slider(1.0, 5.0, step=0.5, label="SOP Rating", value=3.5),
    gr.Slider(1.0, 5.0, step=0.5, label="LOR Rating", value=3.5),
    gr.Slider(6.8, 9.92, step=0.01, label="CGPA", value=8.0),
    gr.Radio([0, 1], label="Research Experience (0=No, 1=Yes)", value=1),
]

outputs = gr.Textbox(label="Admission Prediction")

title = "Graduate Admission Predictor"
description = "Predict your chance of admission to graduate school based on your academic profile"

# Create and launch the interface
app = gr.Interface(
    fn=predict_admission,
    inputs=inputs,
    outputs=outputs,
    title=title,
    description=description,
    examples=[
        [320, 110, 4, 4.5, 4.5, 9.0, 1],  # High chance example
        [300, 100, 3, 3.0, 3.5, 8.0, 0],  # Moderate chance example
        [290, 92, 1, 2.0, 2.5, 7.0, 0]    # Low chance example
    ]
)

if __name__ == "__main__":
    app.launch()


    