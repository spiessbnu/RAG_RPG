"""
Streamlit app implementing a simple chatbot that uses OpenAI's Conversation
and Retrieval APIs.  Each user session starts a fresh conversation state
and performs a vector search against a pre-existing vector store to
retrieve relevant context before answering the user's query.  The app
leverages Streamlit's session state to persist the conversation ID and
chat history across interactions within a single session.

To use this app you will need an OpenAI API key with access to the
Conversation and Retrieval APIs and the ID of a vector store (previously
created via the OpenAI API).  For demonstration purposes the vector store
ID is read from an environment variable named ``VECTOR_STORE_ID``.

Usage:

    streamlit run streamlit_app.py

This will start a local web server where you can chat with the model.

Note: this example does not perform any explicit web browsing to fetch
additional information—it relies solely on the contents of the specified
vector store.  It also includes a safety guard in the prompt to ask the
model to decline answering if the retrieved context is insufficient.
"""

import os
import streamlit as st
from openai import OpenAI


def init_openai_client(api_key: str) -> OpenAI:
    """Instantiate an OpenAI client with the provided API key."""
    return OpenAI(api_key=api_key)


def create_conversation(client: OpenAI) -> str:
    """Create a new conversation via the OpenAI API and return its ID."""
    conv = client.conversations.create()
    return conv.id


def search_vector_store(client: OpenAI, vector_store_id: str, query: str, k: int = 5) -> str:
    """
    Search the given vector store for documents relevant to the query and
    build a single context string by concatenating the retrieved documents.

    Args:
        client: An authenticated OpenAI client.
        vector_store_id: The ID of the vector store to search.
        query: The user's question.
        k: How many top results to retrieve.

    Returns:
        A string containing the concatenated documents.
    """
    search_results = client.vector_stores.search(
        vector_store_id=vector_store_id,
        query=query,
        k=k,
    )
    # The response contains a .data attribute which is a list of matches.
    # Each match has a `.document` field with the original document text.
    context_parts = []
    for match in search_results.data:
        # Only include non-empty documents to avoid unnecessary whitespace.
        doc_text = match.document.strip()
        if doc_text:
            context_parts.append(doc_text)
    return "\n\n".join(context_parts)


def build_prompt(context: str, question: str) -> str:
    """
    Build a prompt for the responses API that clearly separates the
    retrieved context from the user's question and instructs the model
    not to answer beyond the provided context.

    Args:
        context: A string of retrieved documents from the vector store.
        question: The user's question.

    Returns:
        A formatted string suitable for passing to the responses API.
    """
    instructions = (
        "Você é um assistente que responde apenas com base no contexto fornecido."
        " Se o contexto não contiver informações relevantes para a pergunta,"
        " responda que não sabe ou que não pode ajudar. Não invente fatos."
    )
    prompt = f"Instruções: {instructions}\n\nContexto:\n{context}\n\nPergunta: {question}\n\nResposta:"
    return prompt


def main():
    st.set_page_config(page_title="Chatbot com Retrieval e Conversa", layout="centered")
    st.title("Chatbot com Retrieval e Conversation API")

    # Collect the OpenAI API key from the user if not provided via environment
    default_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input("Chave da API OpenAI", value=default_key, type="password")
    if not api_key:
        st.info("Insira sua chave de API na barra lateral para começar.")
        st.stop()

    # Collect the vector store ID from environment or the user.
    default_vs = os.environ.get("VECTOR_STORE_ID", "")
    vector_store_id = st.sidebar.text_input("ID do Vector Store", value=default_vs)
    if not vector_store_id:
        st.info("Insira o ID do vector store na barra lateral.")
        st.stop()

    # Initialise the client and conversation on first run.
    if "client" not in st.session_state:
        st.session_state.client = init_openai_client(api_key)

    if "conversation_id" not in st.session_state:
        # Start a new conversation for this session.
        st.session_state.conversation_id = create_conversation(st.session_state.client)
        st.session_state.chat_history = []

    # Display previous chat messages.
    for entry in st.session_state.chat_history:
        role = entry["role"]
        content = entry["content"]
        if role == "user":
            st.chat_message("Usuário").markdown(content)
        else:
            st.chat_message("Assistente").markdown(content)

    # User input box for the next question.
    question = st.chat_input("Faça sua pergunta aqui...")
    if question:
        # Append user message to history and display it.
        st.session_state.chat_history.append({"role": "user", "content": question})
        st.chat_message("Usuário").markdown(question)

        # Retrieve context from the vector store.
        context = search_vector_store(
            client=st.session_state.client,
            vector_store_id=vector_store_id,
            query=question,
            k=5,
        )

        # Build the prompt with safety instructions.
        prompt = build_prompt(context=context, question=question)

        # Call the responses API passing the existing conversation_id.
        try:
            response = st.session_state.client.responses.create(
                model="gpt-4o-mini",
                conversation_id=st.session_state.conversation_id,
                input=prompt,
            )
            answer = response.output_text.strip()
        except Exception as e:
            answer = f"Erro ao chamar a API: {e}"

        # Append assistant response to history and display it.
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.chat_message("Assistente").markdown(answer)


if __name__ == "__main__":
    main()