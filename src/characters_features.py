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
    connected, verbs = [], []
    for token in doc_tokens:
        character = [key for key, value in coref_dict.items() for i, j in value if
                     token.start_char >= i and token.end_char <= j]
        if len(character) != 0:
            connected.add(character[0])
        if token.words[0].upos == 'VERB':
            verbs.append(token.text)
        if token.words[0].upos == 'PUNCT' and token.text != ",":
            if len(connected) > 1:
                links[tuple(connected)] = verbs
            connected, verbs = set(), []

    return links