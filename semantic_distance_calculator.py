from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticDistanceCalculator:
    """
    A utility class for calculating semantic similarity between a query and a list of sentences
    using a SentenceTransformer model.
    """

    def __init__(self, model='bert-base-nli-mean-tokens'):
        """
        Initializes the SemanticDistanceCalculator with a specified SentenceTransformer model.
        """
        self.model = SentenceTransformer(model)

    def get_top_k_sentences(self, query, sentences, k):
        """
        Returns the top-k sentences most semantically similar to the query.
        """
        sentences.insert(0, query)
        sentence_embeddings = self.model.encode(sentences)
        similarity_scores = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])

        similarity_scores = list(similarity_scores[0])

        sorted_sentences = [x for _, x in sorted(zip(similarity_scores, sentences[1:]), key=lambda pair: pair[0])]

        return sorted_sentences[:k]
