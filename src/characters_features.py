import os
import json
import classla


def characteristics(coref_dict, doc_tokens, occurrences=3, neighborhood=3, deviation=0):
    characteristics = {}
    for key, value in coref_dict.items():
        neighbors = []
        first_three = value[:occurrences]  # prvih N pojavitev samostalnika
        for i, j in first_three:
            lst = [(ix, x) for ix, x in enumerate(doc_tokens) if x.start_char >= i - deviation and x.end_char <= j + deviation]
            for ids, s in lst:
                neighbors.extend(
                    [doc_tokens[x] for x in range(ids - neighborhood, ids + neighborhood + 1) if x >= 0 and x != ids])
        # adjectives
        characteristics[key] = [x.words[0].lemma for x in neighbors if x.words[0].upos == 'ADJ']

    return characteristics


def weights_of_links(coref_dict, doc_tokens):
    pairs = {}
    connected = []
    for token in doc_tokens:
        character = [key for key, value in coref_dict.items() for i, j in value if token.start_char >= i and token.end_char <= j]
        if len(character) != 0:
            connected.append(character[0])
        if token.words[0].upos == 'PUNCT' and token.text != ",":
            if len(connected) > 1:
                connected.sort()
                p = [(a, b) for idx, a in enumerate(connected) for b in connected[idx + 1:] if a != b]
                for i in p:
                    pairs[i] = pairs[i] + 1 if i in pairs else 1
            connected = []
    # normalized
    S = sum(pairs.values())
    return {key: value / S for key, value in pairs.items()}


def character_importance(coref_dict, doc_tokens):
    importance = {}
    weights = weights_of_links(coref_dict, doc_tokens)
    S = sum({key: len(value) for key, value in coref_dict.items()}.values())
    for key, value in coref_dict.items():
        w = sum([v for (k1, k2), v in weights.items() if k1 == key or k2 == key])
        importance[key] = (len(value) / S) + w
    # normalized
    S2 = sum(importance.values())
    return {key: value / S2 for key, value in importance.items()}


def link_classification(coref_dict, doc_tokens):
    links = {}
    connected, verbs = set(), []
    for token in doc_tokens:
        character = [key for key, value in coref_dict.items() for i, j in value if
                     token.start_char >= i and token.end_char <= j]
        if len(character) != 0:
            connected.add(character[0])
        if token.words[0].upos == 'VERB':
            verbs.append(token.words[0].lemma)
        if token.words[0].upos == 'PUNCT' and token.text != ",":
            if len(connected) > 1:
                links[tuple(connected)] = verbs
            connected, verbs = set(), []

    afinn = {}
    with open("../src/Slovene_sentiment_lexicon_JOB.txt", encoding="utf8") as file:
        lines = [line for line in file]

    for l in lines:
        afinn[l.split()[0]] = l.split()[3]

    affin={}
    for l in links:
        summ = 0
        length = 0
        for i in links[l]:
            if i in afinn:
                length +=1
                summ += float(afinn[i])
        affin[l] = summ/length

    return affin


def features():
    nlp = classla.Pipeline('sl')
    stories = "../data/farytales/stories"
    corefpath = "../data/farytales/coreference"

    for filename in os.listdir(stories):
        f = os.path.join(stories, filename)
        with open(f, encoding="utf8") as file:
            text = file.read()
        doc = nlp(text)
        name = os.path.splitext(filename)[0]
        f1 = os.path.join(corefpath, name)
        with open(f1 + '.json', encoding="utf-8") as json_file:
            coreference = json.load(json_file)

        print(name)
        print(characteristics(coreference, list(doc.iter_tokens())))
        print(weights_of_links(coreference, list(doc.iter_tokens())))
        print(character_importance(coreference, list(doc.iter_tokens())))
        print(link_classification(coreference, list(doc.iter_tokens())))
        print()

features()