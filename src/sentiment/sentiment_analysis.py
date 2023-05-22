import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class CustomSentimentAnalysis():

    def __init__(self, model_location):
        self.model = tf.keras.models.load_model(model_location)
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

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