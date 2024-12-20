import streamlit as st
from langchain_helper import query_database

st.title("Ecommerce T_Shirts👕: Q&A")

# Initialize session state for question history if it doesn't exist
if 'question_history' not in st.session_state:
    st.session_state.question_history = []

question = st.text_input("Question: ")

if question:
    try:
        # Add question to history
        if question not in st.session_state.question_history:
            st.session_state.question_history.append(question)
        
        # Get response using the query_database function
        response = query_database(question)
        
        st.header("Answer")
        st.write(response)
        
        # Show question history
        if st.session_state.question_history:
            st.sidebar.header("Question History")
            for past_question in st.session_state.question_history:
                if st.sidebar.button(past_question):
                    response = query_database(past_question)
                    st.header("Answer")
                    st.write(response)
                    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please try refreshing the page if the error persists.")