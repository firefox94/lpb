import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Streaming Agent", page_icon="🤖")
st.title("VietHa-AllMighty 😼")

# get response
def get_response(query, chat_history):
    template = """
    You are a very handsome boy with wide knowledge of many domains.
    You are here to answer questions to help people.
    Chat history: {chat_history}
    User question: {user_question}
    """

    prompt = ChatPromptTemplate.from_template(template)
    
    llm = OllamaLLM(model='vietha')

    chain = prompt | llm | StrOutputParser()
    
    return chain.invoke({
        "chat_history": chat_history,
        "user_question": query
    })
# conversation
for message in st.session_state.chat_history:
    if isinstance(message,HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message,AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    else:
        st.warning(f"Unsupported message type {type(message)}")

# user input
user_query = st.chat_input("Challenge accepted 😏")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        ai_response = get_response(user_query,st.session_state.chat_history)
        st.markdown(ai_response)
        
    st.session_state.chat_history.append(AIMessage(content=ai_response))