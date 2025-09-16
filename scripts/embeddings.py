from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_embeddings(texts):
    return model.encode(texts, convert_to_tensor=True)

def recommend_articles(user_history_embeddings, article_embeddings, top_k=5):
    scores = util.cos_sim(user_history_embeddings, article_embeddings)
    top_results = scores.topk(top_k)
    return top_results
