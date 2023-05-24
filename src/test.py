from data_processing import make_dataset_surrounding_words, remove_all_new_lines, make_dataset_sentences, make_dataset_only_surrounding

#remove_all_new_lines()

make_dataset_surrounding_words(5, 5, "../data/farytales/stories_to_extract", "../data/farytales/location_gt",
                               "../data/farytales/character_sentiment",
                               "../data/farytales/sentiment_dataset/dataset_5_5_separate.csv", True)

make_dataset_surrounding_words(3, 10, "../data/farytales/stories_to_extract", "../data/farytales/location_gt",
                               "../data/farytales/character_sentiment",
                               "../data/farytales/sentiment_dataset/dataset_3_10_separate.csv", True)

make_dataset_surrounding_words(10, 3, "../data/farytales/stories_to_extract", "../data/farytales/location_gt",
                               "../data/farytales/character_sentiment",
                               "../data/farytales/sentiment_dataset/dataset_10_3_separate.csv", True)

make_dataset_only_surrounding(6, 6, "../data/farytales/stories_to_extract", "../data/farytales/location_gt",
                               "../data/farytales/character_sentiment",
                               "../data/farytales/sentiment_dataset/dataset_surroundings_6_6_separate.csv", True)

make_dataset_sentences("../data/farytales/stories_to_extract", "../data/farytales/location_gt",
                               "../data/farytales/character_sentiment",
                               "../data/farytales/sentiment_dataset/dataset_sentences_separate.csv", True)
