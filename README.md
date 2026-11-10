# TASK 2: Chatbot for FAQs

## Features
- Pre-collected FAQs (e-commerce domain – easily replaceable)
- Text preprocessing (cleaning, tokenization, stopword removal)
- TF-IDF vectorization + Cosine similarity matching
- Interactive console chatbot
- Shows matched FAQ and similarity score

## Requirements
- Python 3.8+
- NumPy (`pip install numpy`)

## How to run
```bash
python faq_chatbot.py
```

Type questions like:
- "How do I track my order?"
- "What is the return policy?"
- "Do you ship internationally?"

Type `quit` or `exit` to leave.

## Customization
Edit the `FAQS` list at the top of the file to use your own questions and answers.
