from sentence_transformers import SentenceTransformer, util

# Loaded once at module import.
model = SentenceTransformer("all-MiniLM-L6-v2")


def compute_embeddings(texts: list) -> object:
    """Return a tensor of sentence embeddings for a list of strings."""
    return model.encode(texts, convert_to_tensor=True)


def recommend_articles(
    user_history_embeddings,
    article_embeddings,
    articles: list,
    top_k: int = 5,
) -> list:
    """
    Return the top-k most similar articles to a user's reading history.

    Fixes vs original:
    - The original returned a raw TopkResult tensor (indices + scores),
      leaving the caller to figure out which articles those indices mapped to.
      This function now returns the actual article objects so callers get
      something directly usable.
    - Averages similarity scores across all history embeddings (mean pooling)
      so the recommendation reflects the overall reading profile, not just
      the last article read.

    Args:
        user_history_embeddings: Tensor of embeddings for articles the user has read.
        article_embeddings:      Tensor of embeddings for candidate articles.
        articles:                List of article dicts/objects matching article_embeddings.
        top_k:                   Number of recommendations to return.

    Returns:
        List of up to top_k articles from `articles`.
    """
    if not articles:
        return []

    # Average over the user's history to get a single profile vector
    scores      = util.cos_sim(user_history_embeddings, article_embeddings)
    avg_scores  = scores.mean(dim=0)

    k           = min(top_k, len(articles))
    top_indices = avg_scores.topk(k).indices.tolist()

    return [articles[i] for i in top_indices]
