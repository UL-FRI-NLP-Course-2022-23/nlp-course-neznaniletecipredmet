import csv
import classla
from src.name_entity_recognition import find_all_entities
from src.sentiment.sentiment_analysis import CustomSentimentAnalysis
from src.characters_features import characteristics, weights_of_links, character_importance, link_classification
from src.data_processing import remove_new_lines
# from coreference import coreference
import json


def analyse_story(text):
    text = remove_new_lines(text)

    nlp = classla.Pipeline('sl')
    nlp_results = nlp(text)
    all_entities = find_all_entities(text, "src/resources/characters.txt", nlp_results)
    # occurrences = coreference(remove_new_lines(text), all_entities, trust=0.6)

    # with open("static/data/entities.csv", "w", encoding='UTF8') as f:
    #     json.dump(all_entities, f)

    character_adjectives = characteristics(all_entities, list(nlp_results.iter_tokens()))
    character_protagonist = character_importance(all_entities, list(nlp_results.iter_tokens()))
    relationship_weights = weights_of_links(all_entities, list(nlp_results.iter_tokens()))
    relationship_class = link_classification(all_entities, list(nlp_results.iter_tokens()))

    analyser = CustomSentimentAnalysis("./models/custom")
    character_sentiment = analyser.character_sentiment(text, all_entities)


    # Save the result to a CSV file
    with open("src/static/data/characters.csv", "w", newline="", encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)
        for character in all_entities:
            row = [character, character_sentiment[character], character_protagonist[character]] + character_adjectives[character]
            writer.writerow(row)

    with open("src/static/data/relationships.csv", "w", newline="", encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)
        for pair in relationship_class:
            pair_rotated = (pair[1], pair[0]) if pair not in relationship_weights else pair
            writer.writerow([pair[0], pair[1], relationship_class[pair], relationship_weights[pair_rotated]])

