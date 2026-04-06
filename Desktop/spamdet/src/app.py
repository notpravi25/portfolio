import gradio as gr
from llm_explain import analyse_message  # your function that predicts spam/tone

# Gradio function wrapper
def predict_interface(message, tone):
    # analyse_message should return the prediction, confidence, explanation
    prediction, confidence, explanation = analyse_message(message, tone)
    return f"Prediction: {prediction}\nConfidence: {confidence}\nExplanation: {explanation}"

# Gradio UI
demo = gr.Interface(
    fn=predict_interface,
    inputs=[
        gr.Textbox(label="Enter your message here"),
        gr.Dropdown(["Formal", "Casual", "Neutral"], label="Choose Tone")
    ],
    outputs=gr.Textbox(label="Result"),
    title="Spam Detector + LLM Explainer",
    description="Enter a message to see spam prediction and explanation."
)

demo.launch()