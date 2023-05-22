import json
import csv
import os
import nltk
nltk.download('punkt')
nltk.download('slovene')

from nltk.tokenize import sent_tokenize

def remove_new_lines(txt: str) -> str:
    # Replace new lines
    txt = txt.replace("\n", " ")
    txt = txt.replace("\"", " ")

    txt = " ".join(txt.split())

    return txt

def make_dataset_surrounding_words(num_words_before: int, num_words_after: str, stories_dir: str,
                                   positions_dir: str, sentiment_dir: str, result_path: str
                                   separate: bool) -> None:
    dataset = []

    for filename in os.listdir(stories_dir):
        print(filename, flush=True)

        f = os.path.join(stories_dir, filename)
        with open(f, encoding='utf-8') as f_story:
            text = f_story.read()
            text = remove_new_lines(text)

        filename = filename.replace(".txt", ".json")
        f = os.path.join(positions_dir, filename)
        with open(f, encoding='utf-8') as json_file:
            all_positions = json.load(json_file)

        f = os.path.join(sentiment_dir, filename)
        with open(f, encoding='utf-8') as json_file:
            sentiments = json.load(json_file)

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

                if(len(cumulative_string) != 0):
                    cumulative_string += " $ "

                cumulative_string += string_before
                cumulative_string += " "
                cumulative_string += chosen_word
                cumulative_string += " "
                cumulative_string += string_after
            
                if(separate):
                    dataset.append([cumulative_string, sentiment])
                    cumulative_string = ""

            if(not separate):
                dataset.append([cumulative_string, sentiment])

    with open(result_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(dataset)

def make_dataset_sentences(stories_dir: str, positions_dir: str, sentiment_dir: str,
                           result_path: str, separate: bool) -> None:
    dataset = []

    for filename in os.listdir(stories_dir):
        print(filename, flush=True)

        f = os.path.join(stories_dir, filename)
        with open(f, encoding='utf-8') as f_story:
            text = f_story.read()
            text = remove_new_lines(text)

        filename = filename.replace(".txt", ".json")
        f = os.path.join(positions_dir, filename)
        with open(f, encoding='utf-8') as json_file:
            all_positions = json.load(json_file)

        f = os.path.join(sentiment_dir, filename)
        with open(f, encoding='utf-8') as json_file:
            sentiments = json.load(json_file)

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

                if(separate):
                    dataset.append([cumulative_string, sentiment])
                    cumulative_string = ""

            if(not separate):
                dataset.append([cumulative_string, sentiment])

    with open(result_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["review", "sentiment"])
        writer.writerows(dataset)


def remove_all_new_lines():
    stories_dir = "../data/farytales/stories"
    for filename in os.listdir(stories_dir):
        print(filename, flush=True)

        f = os.path.join(stories_dir, filename)
        with open(f, "r", encoding='utf-8') as f_story:
            text = f_story.read()
            text = remove_new_lines(text)

        with open(f, "w", encoding='utf-8') as f_story:
            f_story.write(text)

