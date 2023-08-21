import torch
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import BertTokenizer, BertForSequenceClassification

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny")

# Load the state dictionary of the trained model

model.load_state_dict(torch.load("sentiment_analysis_model.pth", map_location=torch.device('cpu')))

#If you are using a GPU device, then uncomment the following line and comment out the previous
#model.load_state_dict(torch.load("sentiment_analysis_model.pth"))


# Set the model to evaluation mode
model.eval()

# Create an inference function
def inference(text):
    """
    Predict the sentiment of a movie review text.

    Args:
        text (str): The movie review text.

    Returns:
        str: The predicted sentiment (positive or negative).
    """

    # Preprocess the text
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk_stopwords = set(stopwords.words('english'))
    def preprocess_text(text):
        text = text.lower()
        text = text.replace(r"[^a-zA-Z0-9]", " ")
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in nltk_stopwords]
        return tokens
    preprocessed_text = preprocess_text(text)

    # Convert the text to a tensor
    inputs = tokenizer(" ".join(preprocessed_text), padding=True, truncation=True, max_length=128, return_tensors="pt")

    # Predict the sentiment
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_sentiment = outputs.logits.argmax(dim=1)

    return "Positive" if predicted_sentiment.item() == 1 else "Negative"

# Test the inference function
test_review = "It was a waste of my time and money"
sentiment = inference(test_review)
print(f"The sentiment of the movie review is: {sentiment}")
