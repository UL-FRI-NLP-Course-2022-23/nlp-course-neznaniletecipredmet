from data_processing import make_dataset_surrounding_words, remove_all_new_lines, make_dataset_sentences

#remove_all_new_lines()

# make_dataset_surrounding_words(1, 1, "../data/farytales/stories_to_extract", "../data/farytales/location_gt",
#                                "../data/farytales/character_sentiment",
#                                "../data/farytales/sentiment_dataset/dataset.csv")

make_dataset_sentences("../data/farytales/stories_to_extract", "../data/farytales/location_gt",
                               "../data/farytales/character_sentiment",
                               "../data/farytales/sentiment_dataset/dataset_separate.csv", True)