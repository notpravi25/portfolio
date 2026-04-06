"""
llm_explain.py — Spam Detector with LLM Explanation (Groq + Gradio)
Fixed: client.generate() → client.chat.completions.create()
"""

import os
import pickle
import re
import random
import gradio as gr
from groq import Groq

# ── API Key ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")   # set in your terminal before running:
                                            # Windows: set GROQ_API_KEY=your_key
                                            # Mac/Linux: export GROQ_API_KEY=your_key
client = Groq(api_key=GROQ_API_KEY)

# ── Stopwords & Lemmatizer (same as Step 2) ───────────────────────────────────
STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers',
    'herself','it','its','itself','they','them','their','theirs','themselves',
    'what','which','who','whom','this','that','these','those','am','is','are',
    'was','were','be','been','being','have','has','had','having','do','does',
    'did','doing','a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between','into',
    'through','during','before','after','above','below','to','from','up','down',
    'in','out','on','off','over','under','again','further','then','once','here',
    'there','when','where','why','how','all','both','each','few','more','most',
    'other','some','such','no','nor','not','only','own','same','so','than',
    'too','very','s','t','can','will','just','don','should','now','d','ll',
    'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn',
    'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
    'won','wouldn'
}

def simple_lemmatize(word):
    rules = [
        ('ings',''),('ing',''),('edly',''),('ness',''),('ment',''),
        ('tion',''),('ions',''),('ies','y'),('ied','y'),('eed','ee'),
        ('ed',''),('ers','er'),('es','e'),('ly',''),
    ]
    for suffix, replacement in rules:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)] + replacement
    return word

def preprocess(text):
    text   = text.lower()
    text   = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [simple_lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) >= 2]
    return ' '.join(tokens)

# ── ML Classifier ─────────────────────────────────────────────────────────────
# Option A: Load your real trained model (recommended)
try:
    ml_model  = pickle.load(open('models/spam_model.pkl', 'rb'))
    tfidf_vec = pickle.load(open('models/tfidf_vectorizer.pkl', 'rb'))
    USE_REAL_MODEL = True
    print("Loaded real ML model.")
except FileNotFoundError:
    USE_REAL_MODEL = False
    print("Model files not found — using dummy classifier.")

def run_ml_classifier(message):
    if USE_REAL_MODEL:
        cleaned    = preprocess(message)
        vectorized = tfidf_vec.transform([cleaned])
        prediction = ml_model.predict(vectorized)[0]
        proba      = ml_model.predict_proba(vectorized)[0]
        classes    = list(ml_model.classes_)
        spam_prob  = proba[classes.index('spam')]
        ham_prob   = proba[classes.index('ham')]
    else:
        # Fallback dummy classifier
        spam_prob = random.uniform(0, 1)
        ham_prob  = 1 - spam_prob
        prediction = "spam" if spam_prob > ham_prob else "ham"

    return prediction.upper(), spam_prob, ham_prob

# ── LLM Explanation via Groq ──────────────────────────────────────────────────
def get_explanation(message, prediction, confidence, tone="simple"):
    tone_instructions = {
        "simple":    "Use clear plain English. No jargon. Be concise. Suitable for any user.",
        "technical": "Use NLP/ML terminology. Mention token patterns and keyword weights. Be precise.",
        "friendly":  "Use a warm conversational tone. Speak like a helpful friend. Use simple analogies.",
    }
    instruction = tone_instructions.get(tone, tone_instructions["simple"])

    prompt = f"""You are an AI assistant explaining spam detection results.
A machine learning model classified an SMS message.

Message: "{message}"
Prediction: {prediction}
Confidence: {confidence}

{instruction}

Explain clearly why this message is classified as {prediction}.
Focus on:
- Keywords present (suspicious or reassuring)
- Tone (urgent, promotional vs casual, personal)
- Patterns typical of {('spam' if prediction == 'SPAM' else 'legitimate')} messages

Keep it to 2-4 sentences. Do NOT re-classify — only explain the ML decision.
Start directly. No preamble like "Certainly!" or "Sure!"."""

    # ── FIXED: correct Groq SDK call ──────────────────────────────────────────
    response = client.chat.completions.create(
       # FIXED — current active model
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

# ── Main Analysis Function ────────────────────────────────────────────────────
def analyse_message(message, tone):
    if not message.strip():
        return "—", "—", "—", "—", "Please enter a message."

    prediction, spam_prob, ham_prob = run_ml_classifier(message)
    confidence = f"{max(spam_prob, ham_prob) * 100:.1f}%"

    explanation = get_explanation(message, prediction, confidence, tone)

    warning = ""
    if prediction == "SPAM":
        warning = "\n⚠ WARNING: Do not click links or share personal info in this message.\n\n"

    return (
        prediction,
        confidence,
        f"{spam_prob * 100:.1f}%",
        f"{ham_prob  * 100:.1f}%",
        warning + explanation
    )

# ── Gradio Interface ──────────────────────────────────────────────────────────
with gr.Blocks() as demo:
    gr.Markdown("## Spam Detector with LLM Explanation")
    gr.Markdown("Enter a message below. The ML model classifies it, and an LLM explains why.")

    with gr.Row():
        message_input = gr.Textbox(
            label="Enter your message",
            placeholder="Type or paste a message here...",
            lines=3
        )
        tone_input = gr.Radio(
            choices=["simple", "technical", "friendly"],
            label="Explanation tone",
            value="simple"
        )

    submit_btn = gr.Button("Analyse")

    with gr.Row():
        pred_output      = gr.Textbox(label="Prediction")
        conf_output      = gr.Textbox(label="Confidence")
        spam_prob_output = gr.Textbox(label="Spam probability")
        ham_prob_output  = gr.Textbox(label="Ham probability")

    explanation_output = gr.Textbox(label="LLM Explanation", lines=6)

    submit_btn.click(
        analyse_message,
        inputs=[message_input, tone_input],
        outputs=[pred_output, conf_output, spam_prob_output, ham_prob_output, explanation_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)