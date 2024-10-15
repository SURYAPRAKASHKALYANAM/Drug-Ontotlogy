import streamlit as st
from langchain_community.graphs import Neo4jGraph
from langchain_groq import ChatGroq
from langchain.chains import GraphCypherQAChain
import time

st.title("Drug Ontology")

NEO4J_URI = st.secrets["NEO4J_URI"]
NEO4J_USERNAME = st.secrets["NEO4J_USERNAME"]
NEO4J_PASSWORD = st.secrets["NEO4J_PASSWORD"]

groq_api_key = st.secrets["GROQ_API_KEY"]


graph=Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama3-groq-70b-8192-tool-use-preview")

chain=GraphCypherQAChain.from_llm(llm=llm,graph=graph,verbose=True,allow_dangerous_requests=True)

def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

if "messages" not in st.session_state:
        st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        output = chain.invoke({"query": prompt})
        response = st.write_stream(response_generator(output["result"]))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

