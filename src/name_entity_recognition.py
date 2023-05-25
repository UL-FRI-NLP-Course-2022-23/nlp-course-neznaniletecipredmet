from typing import Dict, List, Union
import Levenshtein as ls
import classla

def find_character_occurrence_levenshtein(character: str, nlp_results, threshold: float = 0.9) -> List:
    results = []
    
    # Iterate over all words
    for token in nlp_results.iter_tokens():
        # Get lemma and test if lemma is by levenshtein close enough to character
        lemma = token.words[0].lemma.lower()
        if(ls.ratio(lemma, character) > threshold and token.words[0].upos in ["NOUN", "PROPN"] and token.ner not in ["B-PER", "I-PER"]):
            results.append(token)
    
    return results

def find_all_characters(characters: List[str], nlp_results, threshold: float = 0.9) -> Dict:
    all_results = {}

    for character in characters:
        results = find_character_occurrence_levenshtein(character, nlp_results, threshold)

        if(len(results) > 0):
            all_results[character] = [[res.start_char, res.end_char] for res in results]

    return all_results

def find_entities_from_list(characters_file: str, nlp_results, threshold: float = 0.9) -> Dict:
    with open(characters_file) as f:
        characters = f.readlines()
    
    characters = [character.strip("\n") for character in characters]
    
    return find_all_characters(characters, nlp_results, threshold)

def find_entities_of_ner_old(nlp_results, threshold: float = 0.9) -> Dict:
    results = []
    unique = []
    last_id = -1
    word = ""
    to_append = []
    for token in nlp_results.iter_tokens():
        if(token.ner == "B-PER" or token.ner == "I-PER"):
            if(last_id == token.id[0]-1):
                word += token.text
                to_append.append(token)
            else:
                to_append = [token]

                word = ""
                for w in token.words:
                    word += w.lemma

                already = None
                for u in unique:
                    if(ls.ratio(word, u) > threshold):
                        already = u
                        break
                if(already is None):
                    unique.append(word)
                    already = word
                
                for token_to_append in to_append:
                    results.append((already, token_to_append))

            last_id = token.id[0]
    
    return results, unique

def find_entities_of_ner(nlp_results, threshold: float = 0.95) -> Dict:
    results = {}

    for token in nlp_results.iter_tokens():
        if(token.ner == "B-PER" or token.ner == "I-PER"):
            lemma = token.words[0].lemma.lower()

            already = None
            for r in results:
                if(ls.ratio(lemma, r) > threshold):
                    already = r
                    break
            if(already is None):
                results[lemma] = [[token.start_char, token.end_char]]
            else:
                results[lemma].append([token.start_char, token.end_char])
            
    return results

def find_all_entities_old(text: str, character_file: str, nlp, threshold: float = 0.95) -> Dict:
    # Compute classla results
    nlp_results = nlp(text)

    results = find_entities_from_list(character_file, nlp_results, threshold)
    results_ner = find_entities_of_ner(nlp_results, threshold)

    for res_ner in results_ner:
        if(res_ner in results):
            results[res_ner] += results_ner[res_ner]
        else:
            results[res_ner] = results_ner[res_ner]

    return results


def find_all_entities(text: str, character_file: str, nlp_results, threshold: float = 0.95) -> Dict:
    results = find_entities_from_list(character_file, nlp_results, threshold)
    results_ner = find_entities_of_ner(nlp_results, threshold)

    for res_ner in results_ner:
        if(res_ner in results):
            results[res_ner] += results_ner[res_ner]
        else:
            results[res_ner] = results_ner[res_ner]

    c = 0
    for r in results:
        c += len(results[r])

    limit = int(c*0.1)

    to_remove = []
    for r in results:
        if(limit > len(results[r])):
            to_remove.append(r)

    for to_r in to_remove:
        del results[to_r]

    return results