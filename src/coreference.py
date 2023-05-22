import os
import classla

from helpers_coref import ContextualControllerBERT
from helpers_coref import Document, Token, Mention
from data_processing import remove_new_lines
import json


def classla_output_to_coref_input(classla_output):
    # Transforms CLASSLA's output into a form that can be fed into coref model.
    output_tokens = {}
    output_sentences = []
    output_mentions = {}
    output_clusters = []

    str_document = classla_output.text
    start_char = 0
    MENTION_MSD = {"N", "P"}  # "N"-noun, "V"-verb, "R"-adverb, "P"-pronoun

    current_mention_id = 1
    token_index_in_document = 0
    for sentence_index, input_sentence in enumerate(classla_output.sentences):
        output_sentence = []
        mention_tokens = []
        for token_index_in_sentence, input_token in enumerate(input_sentence.tokens):
            input_word = input_token.words[0]
            output_token = Token(str(sentence_index) + "-" + str(token_index_in_sentence),
                                 input_word.text,
                                 input_word.lemma,
                                 input_word.xpos,
                                 sentence_index,
                                 token_index_in_sentence,
                                 token_index_in_document)

            # FIXME: This is a possibly inefficient way of finding start_char of a word. Stanza has this functionality
            #  implemented, Classla unfortunately does not, so we resort to a hack
            new_start_char = str_document.find(input_word.text, start_char)
            output_token.start_char = new_start_char
            if new_start_char != -1:
                start_char = new_start_char

            if len(mention_tokens) > 0 and mention_tokens[0].msd[0] != output_token.msd[0]:
                output_mentions[current_mention_id] = Mention(current_mention_id, mention_tokens)
                output_clusters.append([current_mention_id])
                mention_tokens = []
                current_mention_id += 1

            # Simplistic mention detection: consider nouns, verbs, adverbs and pronouns as mentions
            if output_token.msd[0] in MENTION_MSD:
                mention_tokens.append(output_token)

            output_tokens[output_token.token_id] = output_token
            output_sentence.append(output_token.token_id)
            token_index_in_document += 1

        # Handle possible leftover mention tokens at end of sentence
        if len(mention_tokens) > 0:
            output_mentions[current_mention_id] = Mention(current_mention_id, mention_tokens)
            output_clusters.append([current_mention_id])
            mention_tokens = []
            current_mention_id += 1

        output_sentences.append(output_sentence)

    return Document(1, output_tokens, output_sentences, output_mentions, output_clusters)


def init_classla():
    os.environ["CLASSLA_RESOURCES_DIR"] = "C:/Users/Jana/classla_resources"
    CLASSLA_RESOURCES_DIR = os.getenv("CLASSLA_RESOURCES_DIR", None)
    if CLASSLA_RESOURCES_DIR is None:
        raise Exception(
            "CLASSLA resources path not specified. Set environment variable CLASSLA_RESOURCES_DIR as path to the dir where CLASSLA resources should be stored.")

    processors = 'tokenize,pos,lemma,ner'

    # Docker image already contains these resources
    # classla.download('sl', processors=processors)

    return classla.Pipeline('sl', processors=processors)


def init_coref():
    os.environ["COREF_MODEL_PATH"] = "./contextual_model_bert"
    COREF_MODEL_PATH = os.getenv("COREF_MODEL_PATH", None)
    if COREF_MODEL_PATH is None:
        raise Exception(
            "Coref model path not specified. Set environment variable COREF_MODEL_PATH as path to the model to load.")

    instance = ContextualControllerBERT.from_pretrained(COREF_MODEL_PATH)
    # instance.eval_mode()
    return instance


classla_model = init_classla()
coref_model = init_coref()


def coref_mentions(input, threshold, return_singletons=False):
    classla_output = classla_model(input)
    coref_input = classla_output_to_coref_input(classla_output)
    coref_output = coref_model.evaluate_single(coref_input)
    coreferences = []
    coreferenced_mentions = set()

    for id2, id1s in coref_output["predictions"].items():
        if id2 is not None:
            for id1 in id1s:
                mention_score = coref_output["scores"][id1]

                if threshold is not None and mention_score < threshold:
                    continue

                coreferenced_mentions.add(id1)
                coreferenced_mentions.add(id2)

                coreferences.append({
                    "id1": int(id1),
                    "id2": int(id2),
                    "score": mention_score
                })

    mentions = []
    for mention in coref_input.mentions.values():
        [sentence_id, token_id] = [int(idx) for idx in mention.tokens[0].token_id.split("-")]
        mention_score = coref_output["scores"][mention.mention_id]

        if return_singletons is False and mention.mention_id not in coreferenced_mentions:
            continue

        mention_raw_text = " ".join([t.raw_text for t in mention.tokens])
        mentions.append(
            {
                "id": mention.mention_id,
                "start_idx": mention.tokens[0].start_char,
                "length": len(mention_raw_text),
                "ner_type": classla_output.sentences[sentence_id].tokens[token_id].ner.replace("B-", "").replace("I-", ""),
                "msd": mention.tokens[0].msd,
                "text": mention_raw_text
            }
        )

    return {
        "mentions": mentions,
        "coreferences": sorted(coreferences, key=lambda x: x["id1"])
    }


def coreference(text, filename, trust=0.5, window_size=1100, offset=550, deviation=2):
    with open("../data/farytales/ner_output2/" + filename+'.json', encoding="utf-8") as json_file:
        extraction = json.load(json_file)

    start_offset = 0
    while start_offset < len(text):
        snip = text[start_offset: min(start_offset + window_size, len(text)) - 1]
        cf = coref_mentions(snip, trust)
        coref = cf['coreferences']
        mentions = cf['mentions']

        for c in coref:
            inst1, inst2 = c['id1'], c['id2']
            ment1, ment2 = list(filter(lambda m: m['id'] == inst1, mentions))[0], list(filter(lambda m: m['id'] == inst2, mentions))[0]

            key1, key2 = -1, -1
            for key, value in extraction.items():
                for start, end in value:
                    if ment1['start_idx']+start_offset in range(start - deviation, start + deviation) and ment1['start_idx']+ment1["length"]+start_offset in range(end - deviation, end + deviation):
                        key1 = key
                    if ment2['start_idx']+start_offset in range(start - deviation, start + deviation) and ment2['start_idx']+ment2["length"]+start_offset in range(end - deviation, end + deviation):
                        key2 = key

                if key1 != -1 and key2 != -1:
                    break

            if key1 != key2:
                if key1 != -1 and key2 != -1:
                    if ment1['ner_type'] == 'PER' and ment2['ner_type'] != 'PER':
                        extraction[key1].extend(extraction[key2])
                        del extraction[key2]
                    elif ment1['ner_type'] != 'PER' and ment2['ner_type'] == 'PER':
                        extraction[key2].extend(extraction[key1])
                        del extraction[key1]
                    elif ment1['ner_type'] != 'PER' and ment2['ner_type'] != 'PER':
                        extraction[key1].extend(extraction[key2])
                        del extraction[key2]

                elif key1 != -1:
                    extraction[key1].append([ment2['start_idx']+start_offset, ment2['start_idx']+ment2['length']+start_offset])
                elif key2 != -1:
                    extraction[key2].append([ment1['start_idx']+start_offset, ment1['start_idx']+ment1['length']+start_offset])

        start_offset += offset

    return extraction


def run_all():
    directory = "../data/farytales/stories"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        with open(f, encoding="utf8") as file:
            text = file.read()
        name = os.path.splitext(filename)[0]

        result = coreference(remove_new_lines(text), name, trust=0.6)
        with open("../data/farytales/coreference/"+name+".json", "w", encoding="utf-8") as outfile:
            json.dump(result, outfile, ensure_ascii=False)


run_all()
