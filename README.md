# IISER Bhopal Library Chatbot

A simple chatbot application for the IISER Bhopal Central Library website that can answer frequently asked questions about library services, facilities, and policies.

## Features

- **Interactive Chat Interface**: User-friendly chat interface using Streamlit
- **Pre-loaded Q&A Database**: Contains 37 frequently asked questions about the library
- **Intelligent Question Matching**: Uses fuzzy string matching to find the best answer
- **Admin Panel**: Add new questions and answers dynamically
- **Fallback Response**: Default response for unknown questions with contact information
- **Sample Questions**: Quick access to common questions
- **Persistent Storage**: Q&A pairs are saved in JSON format

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

- What services does the library offer?
- How do I borrow a book?
- What are the library's hours?
- Can I renew my books online?
- How do I get a library card?
- Is there Wi-Fi in the library?

## Contact Information

For questions not covered by the chatbot, users are directed to contact:
- Email: librarian@iiserb.ac.in
- Website: https://library.iiserb.ac.in/