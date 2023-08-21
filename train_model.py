import torch
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

# Read the data
df = pd.read_csv('imdb.csv')
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
nltk_stopwords = set(stopwords.words('english'))

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df["review"], df["sentiment"], test_size=0.25
)

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = text.replace(r"[^a-zA-Z0-9]", " ")
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in nltk_stopwords]
    return tokens

x_train_tokenized = [preprocess_text(text) for text in x_train]
x_test_tokenized = [preprocess_text(text) for text in x_test]

# Create a vocabulary and convert words to indices
word_to_index = {}
for tokens in x_train_tokenized:
    for word in tokens:
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

x_train_padded = [torch.tensor([word_to_index.get(word, 0) for word in tokens]) for tokens in x_train_tokenized]
x_test_padded = [torch.tensor([word_to_index.get(word, 0) for word in tokens]) for tokens in x_test_tokenized]

# Convert lists of tensors to padded sequences
x_train_padded = pad_sequence(x_train_padded, batch_first=True)
x_test_padded = pad_sequence(x_test_padded, batch_first=True)


# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=2)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Convert text tokens to input format for BERT
x_train_encoded = tokenizer(x_train.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
x_test_encoded = tokenizer(x_test.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')

# Build data loaders
train_dataset = data.TensorDataset(x_train_encoded['input_ids'], x_train_encoded['attention_mask'], torch.tensor(y_train.values, dtype=torch.long))
train_loader = data.DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataset = data.TensorDataset(x_test_encoded['input_ids'], x_test_encoded['attention_mask'], torch.tensor(y_test.values, dtype=torch.long))
val_loader = data.DataLoader(val_dataset, batch_size=16, shuffle=False)

optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*10)

train_losses = []
val_losses = []


# Training loop
for epoch in range(25):
    model.train()
    train_loss_sum = 0.0

    for batch_x, attention_mask, batch_y in train_loader:
        batch_x, attention_mask, batch_y = batch_x.to(device), attention_mask.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=batch_x, attention_mask=attention_mask).logits  # Use .logits for classification
        loss = nn.CrossEntropyLoss()(outputs, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss_sum += loss.item()

    train_loss = train_loss_sum / len(train_loader)
    train_losses.append(train_loss)

    # Validation loop
    model.eval()
    val_loss_sum = 0.0

    with torch.no_grad():
        for batch_x, attention_mask, batch_y in val_loader:
            batch_x, attention_mask, batch_y = batch_x.to(device), attention_mask.to(device), batch_y.to(device)
            outputs = model(input_ids=batch_x, attention_mask=attention_mask).logits  # Use .logits for classification
            val_loss = nn.CrossEntropyLoss()(outputs, batch_y)
            val_loss_sum += val_loss.item()

        val_loss = val_loss_sum / len(val_loader)
        val_losses.append(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Plot the training and validation loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


torch.save(model.state_dict(), "sentiment_analysis_model.pth")
