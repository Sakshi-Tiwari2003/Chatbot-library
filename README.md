# IISER Bhopal Library Chatbot

A simple chatbot application for the IISER Bhopal Central Library website that can answer frequently asked questions about library services, facilities, and policies.

## Features

- **Interactive Chat Interface**: User-friendly chat interface using Streamlit
- **Pre-loaded Q&A Database**: Contains 37 frequently asked questions about the library
- **🔥 Sentence Embeddings**: Uses AI-powered semantic matching to understand question meaning
- **Intelligent Question Matching**: Combines embeddings with fallback string matching
- **Admin Panel**: Add new questions and answers dynamically
- **Fallback Response**: Default response for unknown questions with contact information
- **Sample Questions**: Quick access to common questions + semantic variations
- **Persistent Storage**: Q&A pairs saved in JSON format, embeddings cached for performance
- **Smart Caching**: Embedding cache automatically updates when new Q&A pairs are added

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run library_chatbot.py
```

2. Open your web browser and go to `http://localhost:8501`

3. Start asking questions about the library!

## Admin Features

- **Add New Q&A Pairs**: Use the sidebar to add new questions and answers
- **View All Q&A Pairs**: Expand the existing pairs to see all questions and answers
- **Clear Chat History**: Reset the current conversation

## Sample Questions

### Exact Matches
- What services does the library offer?
- How do I borrow a book?
- What are the library's hours?
- Can I renew my books online?
- How do I get a library card?
- Is there Wi-Fi in the library?

### Semantic Variations (Now Supported!)
- What can the library do for me?
- How to check out books?
- When is the library open?
- Can I extend my book loan?
- How to get access to the library?
- Is internet available?

## Technical Details

### Sentence Embeddings
The chatbot uses the `all-MiniLM-L6-v2` model from sentence-transformers to:
- Convert questions into 384-dimensional embeddings
- Perform semantic similarity matching using cosine similarity
- Understand synonyms and different phrasings of the same question

### Performance Optimizations
- **Embedding Caching**: Pre-computed embeddings are cached in `question_embeddings.pkl`
- **Smart Fallback**: Short queries (< 3 characters) use string matching
- **Automatic Cache Refresh**: Cache is cleared when new Q&A pairs are added

## Contact Information

For questions not covered by the chatbot, users are directed to contact:
- Email: librarian@iiserb.ac.in
- Website: https://library.iiserb.ac.in/