import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer from Hugging Face
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Labels used by the model
labels = ['Negative', 'Neutral', 'Positive']

def classify_sentiment(text):
    # Preprocess text
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True)
    
    # Run model
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Get probabilities
    scores = F.softmax(output.logits, dim=1)
    top_class = torch.argmax(scores).item()
    sentiment = labels[top_class]
    confidence = scores[0][top_class].item()

    return sentiment, confidence

# Test it with a headline
headline = input("Enter a news headline: ")
sentiment, confidence = classify_sentiment(headline)
print(f"Sentiment: {sentiment} ({confidence:.2f} confidence)")
