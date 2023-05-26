import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer
import nltk
nltk.download('punkt')
nltk.download('slovene')

from nltk.tokenize import sent_tokenize

from sentiment.custom_model import CustomModel


class CustomSentimentAnalysis():

    def __init__(self, model_location):
        self.bert_tokenizer = BertTokenizer.from_pretrained("distilbert-base-cased")
        self.model = CustomModel(vocabulary_size=len(self.bert_tokenizer.vocab),
                        embedding_dimensions=150,
                        cnn_filters=100,
                        dnn_units=256,
                        model_output_classes=2,
                        dropout_rate=0.2)
    
        self.model.load_weights("models/custom")

    def get_token_ids(self, texts):
        return self.bert_tokenizer.batch_encode_plus(texts, 
                                                add_special_tokens=True, 
                                                max_length = 128, 
                                                pad_to_max_length = True)["input_ids"]

    def predict(self, text: str):
        tokens = self.get_token_ids([text])
        p = self.model.predict(tokens)[0]

        return p[0]

    def samples_creation(self, text, all_positions):
        data = {}

        for character in all_positions:
            positions = all_positions[character]

            cumulative_string = ""

            for position in positions:
                while(position[0]-1 > 0 and text[position[0]-1] != " "):
                    position[0] -= 1
                while(position[1] < len(text) and text[position[1]] != " "):
                    position[1] += 1
                text_before = text[:position[0]]
                text_after = text[position[1]:]

                if(len(cumulative_string) != 0):
                    cumulative_string += " $ "

                if(0 < len(text_before)):
                    sentences_before = sent_tokenize(text_before, language='slovene')
                    cumulative_string += sentences_before[-1]
                    cumulative_string += " "

                chosen_word = text[position[0]:position[1]]
                cumulative_string += chosen_word
                
                if(0 < len(text_after)):
                    sentences_after = sent_tokenize(text_after, language='slovene')
                    cumulative_string += sentences_after[0]

            data[character] = cumulative_string
        
        return data

    def character_sentiment(self, text: str, characters):
        data = self.samples_creation(text, characters)
        results = {}

        for character in characters:
            prediction = self.predict(data[character])
            results[character] = prediction

        return results
