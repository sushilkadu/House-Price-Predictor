import gradio as gr
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# Load the trained models
linear_model = joblib.load('models/linear_model.pkl')
poly_model = joblib.load('models/poly_model.pkl')


def predict_sale_price(gr_liv_area, year_built, overall_qual, model_type):
    features = np.array([[gr_liv_area, year_built, overall_qual]])
    model = linear_model if model_type == "Linear" else poly_model
    # If using the polynomial model, expand input features the same way
    # they were expanded during training (degree=2, default include_bias=True).
    if model_type != "Linear":
        poly = PolynomialFeatures(degree=2)
        features = poly.fit_transform(features)

    price = model.predict(features)[0]
    return f"🏠 Predicted Price: ${price:,.0f}"

with gr.Blocks() as demo:
    gr.Markdown("# House Price Prediction")
    with gr.Row():
        gr_liv_area = gr.Number(label="Above Grade Living Area (sq ft)", value=1500)
        year_built = gr.Number(label="Year Built", value=2000)
        overall_qual = gr.Number(label="Overall Quality (1-10)", value=5)
    
    model_type = gr.Radio(choices=["Linear", "Polynomial"], label="Model Type", value="Linear")
    predict_button = gr.Button("Predict Price")
    output = gr.Textbox(label="Prediction Output")
            

    predict_button.click(fn=predict_sale_price,
                         inputs=[gr_liv_area, year_built, overall_qual, model_type],
                         outputs=output)
if __name__ == "__main__":
    demo.launch()