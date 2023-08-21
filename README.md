# Sentiment-Analysis-using-Bert-Tiny
Train a deep learning model to perform sentiment analysis on text data using the IMDb movie reviews dataset.

## Instructions to run:
For training:

``` python train_model.py ```

For inference, first change the test_review variable as per your requirement,then:

``` python inference_sentiment_analysis.py ```

## Dataset Link:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

## Data Preprocessing:
The dataset was processed to map sentiment labels ("positive" and "negative") to numerical values (1 and 0).
NLTK library was used for preprocessing.

## Model Architecture:
I used the BERT model for sequence classification, specifically the "bert-tiny" variant, which is a smaller version of the original BERT model. 

## Training and Validation:
I used Google Colab for training as it offers free GPU.
I used the AdamW optimizer with a learning rate of 2e-5 and a linear learning rate scheduler with warm-up steps.
Training and validation losses were recorded for each epoch.

## Challenges and Solutions:
Due to limited resources, I used the "bert-tiny", but it still provided competitive results.
