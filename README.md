# Natural language processing course 2022/23: `Character analysis in Slovene short stories`

Team members:
 * `GAL PETKOVŠEK`, `63170020`, `gp1914@student.uni-lj.si`
 * `MARIJA MAROLT`, `63170017`, `mm7522@student.uni-lj.si`
 * `JANA ŠTREMFELJ`, `63170284`, `js7437@student.uni-lj.si`
 
Group public acronym/name: `KARABIN`
 > This value will be used for publishing marks/scores. It will be known only to you and not you colleagues.

## Environment setup
To install all required dependencies for this project run: \
```pip install -r requirements.txt```

To prepare the environment for the NER process you need to execute the following command in python (this downloads the NER, POS and lemma models from classla that are required for this process): \
```python -c "import classla; classla.download('sl')"```

TODO: Add env setup for coreference and possibly visualization and evaluation

## Component usage
This section describes the general pipeline and structure of our repo. Before this make shur that the environment is set up.

### Name entity recognition
To use the ner pipeline for stories we can follow the example in `calculate_ner.py`. We first need to initialize the classla model (the easiest way to do that is just to construct a pipeline that includes processor tokenizer, ner, pos and lemma). After that we need to have a txt file of all predefined characters to search for (can be located in `src/resources/characters.txt`). We then input all those values into find_all_entities function (located in `src/name_entity_recognition.py`). The output of the function is a dictionary which has characters for keys and a list of tuples that represent occurrences of those entities in the text as values.
### NER evaluation

### Coreference resolution

### Visualization


# TODOs:
* Marija extend the character list
* Marija add visualization to repo
* Marija add results to repo
* Jana add coreference module to repo
* Implement the relationship extraction
* Implement the relationship classification
* Implement character classification
* Implement the protagonist classification
* Implement character feature extraction