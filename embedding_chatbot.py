import streamlit as st
import json
import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize session state
if 'qa_pairs' not in st.session_state:
    st.session_state.qa_pairs = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'model' not in st.session_state:
    st.session_state.model = None

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

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

# Load Q&A pairs
def load_qa_pairs():
    if os.path.exists('qa_pairs.json'):
        try:
            with open('qa_pairs.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return default_qa_pairs
    return default_qa_pairs

# Save Q&A pairs
def save_qa_pairs(qa_pairs):
    with open('qa_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

# Initialize Q&A pairs
if not st.session_state.qa_pairs:
    st.session_state.qa_pairs = load_qa_pairs()

# Load sentence transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for all questions
def generate_embeddings(questions, model):
    with st.spinner("🔄 Generating embeddings for questions..."):
        embeddings = model.encode(questions, show_progress_bar=True)
    return embeddings

# Save embeddings to cache
def save_embeddings_cache(questions, embeddings):
    cache_data = {
        'questions': questions,
        'embeddings': embeddings
    }
    try:
        with open('embeddings_cache.pkl', 'wb') as f:
            pickle.dump(cache_data, f)
    except:
        pass

# Load embeddings from cache
def load_embeddings_cache():
    try:
        with open('embeddings_cache.pkl', 'rb') as f:
            cache_data = pickle.load(f)
        return cache_data['questions'], cache_data['embeddings']
    except:
        return None, None

# Initialize embeddings
def initialize_embeddings(qa_pairs, model):
    questions_list = list(qa_pairs.keys())
    
    # Try to load from cache
    cached_questions, cached_embeddings = load_embeddings_cache()
    
    if cached_questions is not None and cached_questions == questions_list:
        st.success("✅ Loaded embeddings from cache!")
        return cached_embeddings, questions_list
    else:
        # Generate new embeddings
        st.info("🔄 First time setup - generating embeddings...")
        embeddings = generate_embeddings(questions_list, model)
        save_embeddings_cache(questions_list, embeddings)
        st.success("✅ Embeddings generated and cached!")
        return embeddings, questions_list

# Find best match using cosine similarity
def find_best_match_embedding(user_query, qa_pairs, model, embeddings, questions_list, threshold=0.7):
    if not qa_pairs or embeddings is None:
        return None, 0, None
    
    # Generate embedding for user query
    user_embedding = model.encode([user_query])
    
    # Calculate cosine similarities
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    
    # Find the best match
    best_idx = np.argmax(similarities)
    best_score = similarities[best_idx]
    
    if best_score >= threshold:
        best_question = questions_list[best_idx]
        best_answer = qa_pairs[best_question]
        return best_answer, best_score, best_question
    
    return None, best_score, None

# Default response
def get_default_response():
    return "I'm sorry, I don't have information about that specific question. Please contact the library office for assistance:\n\n📧 Email: librarian@iiserb.ac.in\n📞 Phone: Contact library reception\n🌐 Website: https://library.iiserb.ac.in/"

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="IISER Bhopal Library Chatbot - AI Powered",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 IISER Bhopal Library Chatbot - AI Powered")
    st.markdown("**Powered by Sentence Embeddings & Cosine Similarity** - Ask me anything about library services in any way you like!")
    
    # Load model if not already loaded
    if st.session_state.model is None:
        with st.spinner("🤖 Loading AI model..."):
            st.session_state.model = load_model()
        st.success("✅ AI Model loaded successfully!")
    
    # Initialize embeddings if not already done
    if st.session_state.embeddings is None:
        st.session_state.embeddings, st.session_state.questions_list = initialize_embeddings(
            st.session_state.qa_pairs, st.session_state.model
        )
    
    # Technical details
    with st.expander("🔬 How This AI Chatbot Works"):
        st.markdown("""
        **Advanced Semantic Understanding using:**
        
        1. **📊 Sentence Embeddings**: Converts questions into 384-dimensional vectors using `all-MiniLM-L6-v2`
        2. **🎯 Cosine Similarity**: Measures semantic similarity between your question and stored questions  
        3. **⚡ Caching System**: Pre-computed embeddings for instant responses
        4. **🎚️ Similarity Threshold**: 70% threshold ensures accurate matching
        
        **Examples of what I can understand:**
        - "How do I borrow a book?" ≈ "How can I borrow books?" ≈ "Book borrowing process?"
        - "What are library hours?" ≈ "When is library open?" ≈ "Library timing information?"
        - "Is WiFi available?" ≈ "Does library have internet?" ≈ "Wireless connection access?"
        """)
    
    # Sidebar
    with st.sidebar:
        st.header("🛠️ Admin Panel")
        
        # Display current stats
        st.metric("📚 Total Questions", len(st.session_state.qa_pairs))
        st.metric("🔢 Embedding Dimension", "384")
        st.metric("🎯 Similarity Threshold", "70%")
        
        # Add new Q&A
        st.subheader("➕ Add New Q&A")
        new_question = st.text_input("Question:")
        new_answer = st.text_area("Answer:", height=100)
        
        if st.button("Add Q&A Pair"):
            if new_question and new_answer:
                st.session_state.qa_pairs[new_question] = new_answer
                save_qa_pairs(st.session_state.qa_pairs)
                # Regenerate embeddings
                st.session_state.embeddings, st.session_state.questions_list = initialize_embeddings(
                    st.session_state.qa_pairs, st.session_state.model
                )
                st.success("✅ Q&A added and embeddings updated!")
                st.rerun()
            else:
                st.error("❌ Please provide both question and answer.")
        
        # Show existing Q&A
        st.subheader("📋 Existing Questions")
        if st.button("View All Q&A"):
            for i, (q, a) in enumerate(st.session_state.qa_pairs.items(), 1):
                with st.expander(f"Q{i}: {q[:30]}..."):
                    st.write(f"**Q:** {q}")
                    st.write(f"**A:** {a}")
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat Interface")
        
        # Display chat history
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.chat_message("user").write(chat["question"])
                # Show similarity score if available
                similarity_info = ""
                if "similarity" in chat:
                    similarity_info = f" *(Similarity: {chat['similarity']:.1%})*"
                st.chat_message("assistant").write(chat["answer"] + similarity_info)
        
        # User input
        user_question = st.chat_input("Ask me anything about the library...")
        
        if user_question:
            # Find best match using embeddings
            answer, similarity, matched_question = find_best_match_embedding(
                user_question, 
                st.session_state.qa_pairs, 
                st.session_state.model,
                st.session_state.embeddings,
                st.session_state.questions_list
            )
            
            if answer is None:
                answer = get_default_response()
                similarity = 0
            
            # Add to chat history
            chat_entry = {
                "question": user_question,
                "answer": answer,
                "similarity": similarity
            }
            if matched_question:
                chat_entry["matched_question"] = matched_question
            
            st.session_state.chat_history.append(chat_entry)
            st.rerun()
    
    with col2:
        st.subheader("⚡ Quick Test")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.subheader("📝 Try These Examples")
        
        # Original questions
        original_questions = [
            "What services does the library offer?",
            "How do I borrow a book?",
            "What are the library's hours?",
            "Can I renew my books online?",
            "Is there Wi-Fi in the library?"
        ]
        
        # Different variations to test semantic matching
        variations = [
            "What facilities are available at library?",  # Similar to services
            "How can I borrow books from library?",      # Similar to borrow
            "When does the library open and close?",     # Similar to hours
            "Can I extend my book loan period?",         # Similar to renew
            "Does library provide internet access?"      # Similar to Wi-Fi
        ]
        
        all_test_questions = original_questions + variations
        
        for question in all_test_questions:
            if st.button(question, key=f"test_{hash(question)}"):
                answer, similarity, matched_question = find_best_match_embedding(
                    question,
                    st.session_state.qa_pairs,
                    st.session_state.model,
                    st.session_state.embeddings,
                    st.session_state.questions_list
                )
                
                if answer is None:
                    answer = get_default_response()
                    similarity = 0
                
                chat_entry = {
                    "question": question,
                    "answer": answer,
                    "similarity": similarity
                }
                if matched_question:
                    chat_entry["matched_question"] = matched_question
                
                st.session_state.chat_history.append(chat_entry)
                st.rerun()

if __name__ == "__main__":
    main()