import csv
import classla
from src.name_entity_recognition import find_all_entities
from src.sentiment.sentiment_analysis import CustomSentimentAnalysis
from src.characters_features import characteristics, weights_of_links, character_importance, link_classification
from src.data_processing import remove_new_lines
import json
import os

nlp = classla.Pipeline('sl')
analyser = CustomSentimentAnalysis("./models/custom")

def analyse_story(title, text):
    text = remove_new_lines(text)

    nlp_results = nlp(text)
    all_entities = find_all_entities(text, "src/resources/characters.txt", nlp_results)

    character_adjectives = characteristics(all_entities, list(nlp_results.iter_tokens()))
    character_protagonist = character_importance(all_entities, list(nlp_results.iter_tokens()))
    relationship_weights = weights_of_links(all_entities, list(nlp_results.iter_tokens()))
    relationship_class = link_classification(all_entities, list(nlp_results.iter_tokens()))

    character_sentiment = analyser.character_sentiment(text, all_entities)


    # Save the result to a CSV file
    with open("visualisation/data/characters/" + title + ".csv", "w", newline="", encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)
        for character in all_entities:
            row = [character, character_sentiment[character], character_protagonist[character]] + character_adjectives[character]
            writer.writerow(row)

    with open("visualisation/data/relationships/" + title + ".csv", "w", newline="", encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)
        for pair in relationship_class:
            pair_w = pair
            if pair not in relationship_weights:
                pair_w = (pair[1], pair[0])
                if pair_w not in relationship_weights:
                    relationship_weights[pair_w] = ""
            writer.writerow([pair[0], pair[1], relationship_class[pair], relationship_weights[pair_w]])


stories_path = "data/farytales/stories/"
for story in os.listdir(stories_path)[77:]:
    print(story)
    with open(stories_path + story, "r", encoding='utf8') as file:
        text = file.read()
        analyse_story(story[:-4], text)
