import json
import csv
import classla
from name_entity_recognition import find_all_entities
from sentiment.sentiment_analysis import CustomSentimentAnalysis
from characters_features import characteristics, weights_of_links, character_importance, link_classification
# from coreference import coreference
# from data_processing import remove_new_lines


def analyse_story(text):
    nlp = classla.Pipeline('sl')
    nlp_results = nlp(text)
    all_entities = find_all_entities(text, "./src/resources/characters.txt", nlp_results)
    # occurrences = coreference(remove_new_lines(text), all_entities, trust=0.6)

    characteristics = characteristics(all_entities, list(nlp_results.iter_tokens()))
    weights_of_links = weights_of_links(all_entities, list(nlp_results.iter_tokens()))
    character_importance = character_importance(all_entities, list(nlp_results.iter_tokens()))
    link_classification = link_classification(all_entities, list(nlp_results.iter_tokens()))

    analyser = CustomSentimentAnalysis("./models/custom")
    sentiment = analyser.character_sentiment(text, all_entities)


    # Save the result to a CSV file
    with open("static/data/characters.csv", "w", newline="", encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)
        for character in all_entities:
            row = [character, sentiment[character], character_importance[character]] + characteristics
            writer.writerow(row)

    with open("static/data/relationships.csv", "w", newline="", encoding='UTF8') as csvfile:
        writer = csv.writer(csvfile)
        for pair in link_classification:
            writer.writerow([pair[0], pair[1], link_classification[pair], weights_of_links[pair]])

