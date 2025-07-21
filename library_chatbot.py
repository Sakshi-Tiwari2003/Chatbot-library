import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Initialize session state
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'embeddings_model' not in st.session_state:
    st.session_state.embeddings_model = None

if 'question_embeddings' not in st.session_state:
    st.session_state.question_embeddings = None

if 'questions_list' not in st.session_state:
    st.session_state.questions_list = []

# Default Q&A pairs
default_qa_pairs = {
    "What services does the library offer?": "Central Library offer many services and it is available in Services tab in our website.\nhttps://library.iiserb.ac.in/\nPlease see the above link for library services.",
    
    "Can I suggest a book for the library to purchase?": "Yes! You can submit a purchase request form, available it on our website.\nhttps://library.iiserb.ac.in/dropdown/services/form.html\nPlease see the above link for suggesting a book for the library.",
    
    "How do I borrow a book?": "You can borrow books using your library card at the RFID kiosk, which allows you to check in and check out books.",
    
    "How long can I keep borrowed items?": "You can find the information by visiting\nhttps://library.iiserb.ac.in/dropdown/services/Borrowing_facilities.html.",
    
    "Can I renew my books online?": "Yes. You can find the information by visiting\nhttps://library.iiserb.ac.in/dropdown/services/Borrowing_facilities.html",
    
    "What happens if I return the books late?": "Users must return the books within stimulated time. If the book is not returned on time, there will be a fine of Rs. 2 per day for each book until it is returned.",
    
    "Can I access databases and journals from home?": "Yes, You can access our subscribed E-resources anytime and from anywhere by using the MyLOFT remote access platform.",
    
    "How do I get a library card?": "You have to apply for the library membership by filling out the membership form. Your institute identity card will serve as your library card.",
    
    "What should I do if I lose my library card?": "Please reach out to the library staff for assistance.",
    
    "Is there Wi-Fi in the library?": "Yes, It is available in the library.",
    
    "Can I print or scan documents?": "Yes, You can go to the photocopy section in the library.",
    
    "What are the library's hours?": "You can find the information by visiting https://library.iiserb.ac.in/dropdown/about-us/library-timings.html",
    
    "How do I reserve discussion room?": "You can easily reserve a discussion room online by visiting\nhttps://web.iiserb.ac.in/crbs/index.php.",
    
    "How do I search books in the library?": "You can find books in the library by using the Web OPAC (Online Public Access Catalog).\nPlease visit the following link: https://webopac.iiserb.ac.in/.",
    
    "Who is the Librarian of IISER Bhopal?": "Dr. Sandeep Kumar Pathak is the Librarian of Central Library IISER Bhopal.",
    
    "What are the Library Timings?": "You can find the information by visiting https://library.iiserb.ac.in/dropdown/about-us/library-timings.html",
    
    "What is the total area of Library?": "The library has three floors with a sitting capacity of 500 users.",
    
    "Do Library has washroom facility at each floor?": "Yes.",
    
    "Do Library has Wi-Fi facility?": "Yes, It is available in the library.",
    
    "Whether IISER Bhopal Library is fully air conditioned?": "Yes.",
    
    "Whether Library has Discussion Room Facility?": "Yes, You can easily reserve a discussion room online by visiting\nhttps://web.iiserb.ac.in/crbs/index.php.",
    
    "Does the Library organize an Author/ Editorial workshop?": "Yes, The Central Library, from time to time, organizes such types of workshops to help users make the best use of e-resources.",
    
    "What are the timing of Library Photocopy Facility": "The photocopying service is open from Monday to Saturday, between 9:30 AM and 5:30 PM.",
    
    "What are the timing of e-Library?": "It is the same as Library timings. You can find the information by visiting\nhttps://library.iiserb.ac.in/dropdown/about-us/library-timings.html.",
    
    "Where is the e-Library?": "The e-Library is located on the 2nd floor of the library.",
    
    "Where is the librarian sitting area located?": "The librarian sitting area is located on the ground floor of the library.",
    
    "How many books and e‑resources does the library have?": "The library houses around 16,000 books across various disciplines and provides access to over 8,000 e‑journals and multiple databases.",
    
    "Which holidays is the library closed?": "The library is closed on the following holidays:\n- Independence Day (August 15)\n- Republic Day (January 26)\n- Gandhi Jayanti (October 2)",
    
    "Who is eligible for library membership?": "Anyone who is a legitimate member of IISER Bhopal can access the library. This includes students, PhD scholars, faculty, staff, and visitors. Your membership is valid as long as you are connected to the institute.",
    
    "What e‑resources and databases are available?": "You can find the list by visiting https://library.iiserb.ac.in/dropdown/online-resources/az_list_of_subs_e_resources.html.",
    
    "What is the Institutional Digital Repository (IDR)?": "It archives and disseminates the scholarly output of IISER Bhopal. It is accessible at\nhttp://idr.iiserb.ac.in:8080/jspui/.",
    
    "Can visitors use the library?": "Yes, for on‑site reference only (no borrowing). Prior notice is appreciated.",
    
    "How is the library secured and automated?": "It uses RFID technology for self-check-in/out, theft prevention, inventory management, sorting, smart cards, people counters, and efficient operations.",
    
    "How do I request a book or journal not available in the library?": "Please email your request at: librarian@iiserb.ac.in.",
    
    "Can I get help with research or referencing?": "Yes, the library offers reference services and user training programs.",
    
    "Where can I find membership or book requisition forms?": "To get the forms in either English or Hindi, go to the \"Forms\" section on the library website.\nYou can download them from there. For more details, please visit\nhttps://library.iiserb.ac.in/dropdown/services/form.html.",
    
    "What should I do if I lose a library book?": "Report it immediately to the library staff. You may be asked to replace it or pay a fine.",
    
    "Can I eat or talk in the library?": "No. Silence must be maintained. Food and drinks are not allowed."
}

# Load Q&A pairs from file or use defaults
def load_qa_pairs():
    if os.path.exists('qa_pairs.json'):
        try:
            with open('qa_pairs.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default_qa_pairs
    return default_qa_pairs

# Save Q&A pairs to file
def save_qa_pairs(qa_pairs):
    with open('qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

# Initialize Q&A pairs
if not st.session_state.qa_pairs:
    st.session_state.qa_pairs = load_qa_pairs()

# Load or initialize the sentence transformer model
@st.cache_resource
def load_embeddings_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load embeddings from cache or compute them
def load_or_compute_embeddings(qa_pairs):
    embeddings_file = 'question_embeddings.pkl'
    
    if os.path.exists(embeddings_file):
        try:
            with open(embeddings_file, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data['questions'] == list(qa_pairs.keys()):
                    return cached_data['embeddings'], cached_data['questions']
        except:
            pass
    
    # Compute embeddings
    model = load_embeddings_model()
    questions_list = list(qa_pairs.keys())
    embeddings = model.encode(questions_list)
    
    # Cache the embeddings
    try:
        with open(embeddings_file, 'wb') as f:
            pickle.dump({
                'questions': questions_list,
                'embeddings': embeddings
            }, f)
    except:
        pass
    
    return embeddings, questions_list

# Function to find the best matching answer using sentence embeddings
def find_best_match_embeddings(user_question, qa_pairs, threshold=0.5):
    if not qa_pairs:
        return None
    
    # Load model and embeddings
    model = load_embeddings_model()
    embeddings, questions_list = load_or_compute_embeddings(qa_pairs)
    
    # Encode user question
    user_embedding = model.encode([user_question])
    
    # Compute cosine similarities
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    
    # Find best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    if best_score >= threshold:
        best_question = questions_list[best_idx]
        return qa_pairs[best_question], best_score, best_question
    
    return None, best_score, None

# Fallback function using string matching for very short queries
def find_best_match_fallback(user_question, qa_pairs, threshold=0.6):
    best_match = None
    best_score = 0
    
    user_question_lower = user_question.lower().strip()
    
    for question, answer in qa_pairs.items():
        question_lower = question.lower().strip()
        
        # Check for exact match first
        if user_question_lower == question_lower:
            return answer, 1.0, question
        
        # Check for substring matches
        if user_question_lower in question_lower or question_lower in user_question_lower:
            score = 0.8
            if score > best_score and score >= threshold:
                best_score = score
                best_match = answer
    
    return best_match, best_score, None

# Combined matching function
def find_best_match(user_question, qa_pairs, threshold=0.5):
    # For very short queries, use fallback method
    if len(user_question.strip()) < 3:
        result, score, question = find_best_match_fallback(user_question, qa_pairs, threshold)
        return result
    
    # Use embeddings for longer queries
    try:
        result, score, question = find_best_match_embeddings(user_question, qa_pairs, threshold)
        return result
    except Exception as e:
        # Fallback to string matching if embeddings fail
        result, score, question = find_best_match_fallback(user_question, qa_pairs, threshold)
        return result

# Default response for unknown questions
def get_default_response():
    return "I'm sorry, I don't have information about that specific question. Please contact the library office for assistance:\n\nEmail: librarian@iiserb.ac.in\nPhone: Contact library reception\n\nYou can also visit our website: https://library.iiserb.ac.in/"

# Streamlit app
def main():
    st.set_page_config(
        page_title="IISER Bhopal Library Chatbot",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 IISER Bhopal Library Chatbot")
    st.markdown("Welcome to the IISER Bhopal Central Library chatbot! Ask me anything about library services, facilities, and policies.")
    
    # Add info about embeddings
    with st.expander("🔬 About This Chatbot"):
        st.markdown("""
        This chatbot uses **sentence embeddings** to understand your questions better:
        - **Semantic matching**: Understands the meaning behind your questions, not just keywords
        - **Improved accuracy**: Better matches even when you phrase questions differently
        - **Faster responses**: Cached embeddings for quick retrieval
        - **Smart fallback**: Uses traditional string matching for very short queries
        """)
    
    # Show loading status for embeddings
    if st.session_state.qa_pairs and st.session_state.embeddings_model is None:
        with st.spinner("Loading AI model for better question understanding..."):
            st.session_state.embeddings_model = load_embeddings_model()
        st.success("✅ AI model loaded successfully!")
        st.rerun()
    
    # Sidebar for admin functions
    with st.sidebar:
        st.header("Admin Panel")
        
        # Add new Q&A pair
        st.subheader("Add New Q&A Pair")
        new_question = st.text_input("Question:")
        new_answer = st.text_area("Answer:", height=100)
        
        if st.button("Add Q&A Pair"):
            if new_question and new_answer:
                st.session_state.qa_pairs[new_question] = new_answer
                save_qa_pairs(st.session_state.qa_pairs)
                # Clear embedding cache when new Q&A is added
                if os.path.exists('question_embeddings.pkl'):
                    os.remove('question_embeddings.pkl')
                st.success("Q&A pair added successfully! Embeddings will be recomputed.")
                st.rerun()
            else:
                st.error("Please provide both question and answer.")
        
        # Display existing Q&A pairs
        st.subheader("Existing Q&A Pairs")
        if st.button("Show All Q&A Pairs"):
            for i, (q, a) in enumerate(st.session_state.qa_pairs.items(), 1):
                with st.expander(f"Q{i}: {q[:50]}..."):
                    st.write(f"**Question:** {q}")
                    st.write(f"**Answer:** {a}")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Chat with the Library Bot")
        
        # Chat history
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.chat_message("user").write(chat["question"])
                st.chat_message("assistant").write(chat["answer"])
        
        # User input
        user_question = st.chat_input("Type your question here...")
        
        if user_question:
            # Find the best matching answer
            answer = find_best_match(user_question, st.session_state.qa_pairs)
            
            if answer is None:
                answer = get_default_response()
            
            # Add to chat history
            st.session_state.chat_history.append({
                "question": user_question,
                "answer": answer
            })
            
            # Display the new conversation
            st.chat_message("user").write(user_question)
            st.chat_message("assistant").write(answer)
            
            st.rerun()
    
    with col2:
        st.subheader("Quick Actions")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Sample questions
        st.subheader("Sample Questions")
        sample_questions = [
            "What services does the library offer?",
            "How do I borrow a book?",
            "What are the library's hours?",
            "Can I renew my books online?",
            "How do I get a library card?",
            "Is there Wi-Fi in the library?"
        ]
        
        # Add semantic matching examples
        st.subheader("Try These Semantic Variations")
        semantic_questions = [
            "What can the library do for me?",  # Similar to "What services does the library offer?"
            "How to check out books?",  # Similar to "How do I borrow a book?"
            "When is the library open?",  # Similar to "What are the library's hours?"
            "Can I extend my book loan?",  # Similar to "Can I renew my books online?"
            "How to get access to the library?",  # Similar to "How do I get a library card?"
            "Is internet available?"  # Similar to "Is there Wi-Fi in the library?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=f"sample_{question}"):
                answer = find_best_match(question, st.session_state.qa_pairs)
                if answer is None:
                    answer = get_default_response()
                
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })
                st.rerun()
        
        for question in semantic_questions:
            if st.button(question, key=f"semantic_{question}"):
                answer = find_best_match(question, st.session_state.qa_pairs)
                if answer is None:
                    answer = get_default_response()
                
                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer
                })
                st.rerun()

if __name__ == "__main__":
    main()