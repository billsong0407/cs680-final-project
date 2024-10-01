**Algorithmic Analysis of Local Dialects: Develop a model that analyzes spoken language or text to identify and categorize local dialects and slang usage in small communities.**
- Approach 1: first use large language model to first identify the segments of the input audio corresponding to each word, then use methods like K-means Cluster or K nearest neighbours to classify the dialects. Finally, average out the classification of each words in an audio to classify the entire sentence.
- Approach 2: simply train a machine learning model to classify the audio.
- Dataset: can look for existing dataset. Otherwise, getting audio data from local news station can also be viable


