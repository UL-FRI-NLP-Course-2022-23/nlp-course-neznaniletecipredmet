import os
import json
import classla
from src.name_entity_recognition import find_all_entities
from src.data_processing import remove_new_lines

directory = "./data/farytales/stories"
result_dir = "./data/farytales/ner_output2"
nlp = classla.Pipeline('sl') 

for filename in os.listdir(directory):
    print(filename, flush=True)
    f = os.path.join(directory, filename)
    with open(f) as f_story:
        text = f_story.read()
        text = remove_new_lines(text)

    s = find_all_entities(text, "./src/resources/characters.txt", nlp)
    
    result_file = os.path.join(result_dir, filename.replace("txt", "json"))

    with open(result_file, 'w', encoding='utf-8') as fp:
        json.dump(s, fp,ensure_ascii=False)