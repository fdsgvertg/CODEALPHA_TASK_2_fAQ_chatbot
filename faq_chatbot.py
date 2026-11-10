# faq_chatbot.py
# TASK 2: Chatbot for FAQs
# Uses pure Python + NumPy (no external NLP packages required)

import re
import math
from collections import Counter, defaultdict
import numpy as np

# ============================================================
# 1. Collect FAQs (example: e-commerce / online store)
# ============================================================
FAQS = [
    {
        "question": "How do I track my order?",
        "answer": "You can track your order by logging into your account and going to 'My Orders'. Click on the order and then 'Track Package'. You will also receive a tracking link via email once the order ships."
    },
    {
        "question": "What is your return policy?",
        "answer": "We accept returns within 30 days of delivery. Items must be unused and in original packaging. Start a return from your account under 'My Orders' or contact support."
    },
    {
        "question": "How long does shipping take?",
        "answer": "Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. International orders may take 10-15 business days."
    },
    {
        "question": "Do you offer free shipping?",
        "answer": "Yes, we offer free standard shipping on all orders over $50 within the continental United States."
    },
    {
        "question": "How can I change or cancel my order?",
        "answer": "You can cancel or modify an order within 1 hour of placing it from the 'My Orders' page. After that, contact support immediately and we will try to help."
    },
    {
        "question": "What payment methods do you accept?",
        "answer": "We accept Visa, Mastercard, American Express, PayPal, Apple Pay, and Google Pay."
    },
    {
        "question": "How do I reset my password?",
        "answer": "Go to the login page and click 'Forgot Password'. Enter your email address and we will send you a password reset link."
    },
    {
        "question": "Is my personal information secure?",
        "answer": "Yes. We use industry-standard SSL encryption and never store full credit card details on our servers. Your data is protected according to our Privacy Policy."
    },
    {
        "question": "Do you ship internationally?",
        "answer": "Yes, we ship to most countries. Shipping costs and delivery times vary by destination. You can see exact costs at checkout."
    },
    {
        "question": "How do I contact customer support?",
        "answer": "You can reach us via live chat on the website, email at support@example.com, or phone at 1-800-123-4567 (Mon-Fri 9am-6pm EST)."
    },
]

# ============================================================
# 2. Preprocess text
# ============================================================
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "up", "about", "into", "over", "after", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "must", "shall",
    "can", "need", "i", "you", "he", "she", "it", "we", "they", "me", "him",
    "her", "us", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "what", "which", "who", "whom", "how", "when",
    "where", "why", "if", "then", "than", "so", "just", "also", "only"
}

def preprocess(text: str) -> list:
    """Clean, tokenize, and remove stopwords."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)          # keep only alphanumeric
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# ============================================================
# 3. Build TF-IDF + Cosine Similarity
# ============================================================
def build_tfidf(documents: list):
    """
    documents: list of tokenized docs
    Returns: vocabulary, idf dict, tfidf matrix
    """
    # Document frequency
    df = defaultdict(int)
    for doc in documents:
        for term in set(doc):
            df[term] += 1

    N = len(documents)
    idf = {term: math.log((N + 1) / (freq + 1)) + 1 for term, freq in df.items()}
    vocab = sorted(idf.keys())

    # TF-IDF vectors (as dense numpy for easy cosine)
    tfidf_matrix = []
    for doc in documents:
        tf = Counter(doc)
        vec = np.zeros(len(vocab))
        for i, term in enumerate(vocab):
            if term in tf:
                tf_val = tf[term] / len(doc)
                vec[i] = tf_val * idf[term]
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        tfidf_matrix.append(vec)

    return vocab, idf, np.array(tfidf_matrix)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))   # already normalized

# Precompute FAQ vectors
faq_questions = [faq["question"] for faq in FAQS]
faq_tokens = [preprocess(q) for q in faq_questions]
vocab, idf, faq_vectors = build_tfidf(faq_tokens)

def vectorize_query(query: str) -> np.ndarray:
    tokens = preprocess(query)
    tf = Counter(tokens)
    vec = np.zeros(len(vocab))
    for i, term in enumerate(vocab):
        if term in tf:
            tf_val = tf[term] / max(len(tokens), 1)
            vec[i] = tf_val * idf.get(term, 0)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec

# ============================================================
# 4. Match + Answer
# ============================================================
def get_best_answer(user_question: str, threshold: float = 0.25):
    """
    Returns (answer, score, matched_question) or a fallback message.
    """
    q_vec = vectorize_query(user_question)
    scores = [cosine_similarity(q_vec, faq_vec) for faq_vec in faq_vectors]
    best_idx = int(np.argmax(scores))
    best_score = scores[best_idx]

    if best_score < threshold:
        return (
            "I'm not sure I understand. Could you rephrase your question? "
            "You can ask about orders, shipping, returns, payments, or support.",
            best_score,
            None
        )

    return FAQS[best_idx]["answer"], best_score, FAQS[best_idx]["question"]

# ============================================================
# 5. Simple Console Chat UI
# ============================================================
def chat():
    print("=" * 60)
    print("  FAQ Chatbot  |  Type 'quit' or 'exit' to leave")
    print("=" * 60)
    print("Try asking about: order tracking, returns, shipping, payments...\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye", "q"}:
            print("Bot: Goodbye! Have a great day.")
            break

        answer, score, matched = get_best_answer(user_input)
        print(f"Bot: {answer}")
        if matched:
            print(f"      (matched: \"{matched}\"  |  similarity: {score:.2f})")
        print()

if __name__ == "__main__":
    chat()
