import gradio as gr
from utils import model_predict

def predict_spam(text):
    prediction = model_predict(text)
    if prediction == 1:
        return "🚫 SPAM!"
    else:
        return "✅ NOT spam"

# Create Gradio interface
demo = gr.Interface(
    fn=predict_spam,
    inputs=gr.Textbox(label="Enter email text", lines=4, placeholder="Type or paste email text here..."),
    outputs=gr.Textbox(label="Prediction"),
    title="Eduardo Hernandez | Email Spam Detector with TFID and SVM",
    description="Enter an email text to check if it's spam or not.",
    examples=[
        ["Work From Home and Earn $5,000 a Week! We are looking for individuals to work from home. No experience needed. Just send us your resume and personal details to get started"],
        ["Invoice #12345 - Payment Due. Please find attached your invoice for the recent purchase. Payment is due within 7 days"],
        ["Buy now! 90% OFF luxury watches. Authentic Rolex, Omega, TAG. Limited stock!"],
        ["Dear Mr. Thompson, Thank you for your recent purchase. Your order #12345 has been shipped."]
    ]
)

if __name__ == "__main__":
    demo.launch() 