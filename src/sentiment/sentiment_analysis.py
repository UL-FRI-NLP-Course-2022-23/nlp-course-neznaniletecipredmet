import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from transformers import BertTokenizer

from src.sentiment.custom_model import CustomModel


class CustomSentimentAnalysis():

    def __init__(self, model_location):
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.model = CustomModel(vocabulary_size=len(self.bert_tokenizer.vocab),
                        embedding_dimensions=150,
                        cnn_filters=100,
                        dnn_units=256,
                        model_output_classes=2,
                        dropout_rate=0.2)

    @staticmethod
    def get_token_ids(texts):
        return self.bert_tokenizer.batch_encode_plus(texts, 
                                                add_special_tokens=True, 
                                                max_length = 128, 
                                                pad_to_max_length = True)["input_ids"]

    def predict(self, text: str):
        tokens = CustomSentimentAnalysis.get_token_ids([text])
        p = custom_model.predict(tokens)[0]

        return p[0]

    def samples_creation(text, all_positions):
        data = {}

        for character in all_positions:
            positions = all_positions[character]
            sentiment = sentiments[character]

            cumulative_string = ""

            for position in positions:
                while(position[0]-1 > 0 and text[position[0]-1] != " "):
                    position[0] -= 1
                while(position[1] < len(text) and text[position[1]] != " "):
                    position[1] += 1
                text_before = text[:position[0]]
                text_after = text[position[1]:]

                words_before = text_before.split()
                words_before.reverse()
                words_after = text_after.split()

                if(num_words_after <= len(words_after)):
                    words_after = words_after[:num_words_after]
                
                if(num_words_before <= len(words_before)):
                    words_before = words_before[:num_words_before]
                    words_before.reverse()

                string_before = " ".join(words_before)
                string_after = " ".join(words_after)
                chosen_word = text[position[0]:position[1]]

                cumulative_string += " $ "

                cumulative_string += string_before
                cumulative_string += " "
                cumulative_string += chosen_word
                cumulative_string += " "
                cumulative_string += string_after

            data[character] = cumulative_string
        
        return data

    def character_sentiment(self, text: str, characters):
        data = self.samples_creation(text, characters)
        results = {}

        for character in characters:
            prediction = self.predict(characters[character])
            results[character] = prediction

        return results



class SentimentAnalysis():

    def __init__(self, model_location):
        self.bert_model = TFBertForSequenceClassification.from_pretrained(model_location)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    @staticmethod
    def convert_to_input(reviews):
        input_ids,attention_masks,token_type_ids=[],[],[]
        
        for x in tqdm(reviews,position=0, leave=True):
            inputs = bert_tokenizer.encode_plus(x,add_special_tokens=True, max_length=max_length)
            
            i, t = inputs["input_ids"], inputs["token_type_ids"]
            m = [1] * len(i)

            padding_length = max_length - len(i)

            i = i + ([pad_token] * padding_length)
            m = m + ([0] * padding_length)
            t = t + ([pad_token_segment_id] * padding_length)
            
            input_ids.append(i)
            attention_masks.append(m)
            token_type_ids.append(t)
        
        return [np.asarray(input_ids), 
                    np.asarray(attention_masks), 
                    np.asarray(token_type_ids)]

    @staticmethod
    def example_to_features(input_ids,attention_masks,token_type_ids):
        return {"input_ids": input_ids,
                "attention_mask": attention_masks,
                "token_type_ids": token_type_ids}

    def predict(self, text: str):
        sample = SentimentAnalysis([text])
        test_ds=tf.data.Dataset.from_tensor_slices((sample[0],sample[1],sample[2])).map(example_to_features).batch(12)
        
        results = self.bert_model.predict(test_ds)

        results_predicted = np.argmax(results.logits, axis=1)

        return results_predicted[0]