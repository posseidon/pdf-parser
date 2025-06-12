from typing import List, Dict
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import numpy as np

class Topic:
    def __init__(self, method: str = "lda", topic_range=range(2, 11)):
        """
        method: "lda" for Latent Dirichlet Allocation, "hdp" for NMF (as HDP alternative)
        topic_range: range of topic numbers to search for best model
        """
        self.method = method.lower()
        self.topic_range = topic_range
        self.vectorizer = CountVectorizer(stop_words='english')
        self.model = None
        self.n_topics = None

    def fit(self, documents: List[str]):
        """
        Fits the topic model to the given documents.
        This method processes the documents through vectorization and fits either a Latent 
        Dirichlet Allocation (LDA) or Non-negative Matrix Factorization (NMF) model. It uses
        grid search with cross-validation to find the optimal number of topics within the
        specified topic range.
        Parameters
        ----------
        documents : List[str]
            A list of document strings to be processed and fitted to the topic model.
        Returns
        -------
        None
            The method updates the instance variables self.model and self.n_topics in place.
        Raises
        ------
        ValueError
            If the specified method is neither 'lda' nor 'hdp'.
        Notes
        -----
        The method performs the following steps:
        1. Vectorizes the input documents
        2. Initializes either LDA or NMF model based on self.method
        3. Performs grid search over different numbers of topics
        4. Selects the best model and stores it in self.model
        5. Updates self.n_topics with the optimal number of components
        """
        X = self.vectorizer.fit_transform(documents)
        if self.method == "lda":
            model = LatentDirichletAllocation(random_state=42)
        elif self.method == "hdp":
            model = NMF(random_state=42)
        else:
            raise ValueError("Unknown method. Use 'lda' or 'hdp'.")

        param_grid = {'n_components': list(self.topic_range)}
        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X)
        self.model = grid.best_estimator_
        self.n_topics = self.model.n_components

    def get_topics(self, n_words: int = 10) -> List[List[str]]:
        """
        Extract top words for each topic from the trained topic model.

        This method returns the most representative words for each topic identified by the model,
        based on their importance scores within each topic.

        Args:
            n_words (int, optional): Number of top words to extract per topic. Defaults to 10.

        Returns:
            List[List[str]]: A list of topics, where each topic is represented by a list of its top n_words.
                             Each inner list contains strings representing the most relevant words for that topic.

        Example:
            >>> topic_model.get_topics(n_words=5)
            [['word1', 'word2', 'word3', 'word4', 'word5'],
             ['topic2_word1', 'topic2_word2', 'topic2_word3', 'topic2_word4', 'topic2_word5']]
        """
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        for topic_idx, topic in enumerate(self.model.components_):
            top_features = [feature_names[i] for i in topic.argsort()[:-n_words - 1:-1]]
            topics.append(top_features)
        return topics

    def classify(self, documents: List[str]) -> List[int]:
        """
        Classify a list of documents into their most probable topics.

        Args:
            documents (List[str]): A list of text documents to classify.

        Returns:
            List[int]: A list of topic indices, where each index represents the most probable
                topic for the corresponding document in the input list.
                The indices correspond to the topics learned during model training.

        Example:
            >>> documents = ["sports news about football", "politics election results"] 
            >>> topic_model.classify(documents)
            [0, 2]  # Where 0 might represent 'sports' topic and 2 'politics' topic
        """
        X = self.vectorizer.transform(documents)
        topic_distributions = self.model.transform(X)
        return np.argmax(topic_distributions, axis=1).tolist()