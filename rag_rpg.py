import os
import streamlit as st
from openai import OpenAI

# ----------------------------
# Config
# ----------------------------
MODEL = "gpt-4o-mini"
VECTOR_STORE_ID_DEFAULT = "vs_696e5b25f30081918c3ebf06a27cf520"

STRICT_SYSTEM = """
Você é um assistente do cenário NYCS (RPG cyberpunk/pós-apocalíptico) operando com Retrieval.
Regras obrigatórias:

1) Use APENAS informações recuperadas via ferramenta file_search (vector store do lore NYCS) e o histórico da conversa.
2) Se a resposta NÃO estiver sustentada pelo material recuperado, responda EXATAMENTE:
   "Não há informação suficiente no lore indexado para responder com segurança."
3) Não invente fatos, não especule, não complete lacunas.
4) Se a pergunta for ambígua, faça 1 pergunta de esclarecimento (máx. 1 frase) e apresente 2 interpretações possíveis (em bullets).
5) Mantenha a resposta objetiva e bem estruturada (títulos curtos e bullets quando ajudar).
""".strip()


def get_client() -> OpenAI:
    return OpenAI()


def ensure_conversation(client: OpenAI) -> str:
    """Create one conversation per Streamlit session."""
    if "conversation_id" not in st.session_state:
        conv = client.conversations.create(
            metadata={"app": "nycs_streamlit", "world": "NYCS"}
        )
        st.session_state.conversation_id = conv.id
    return st.session_state.conversation_id


def call_nycs_assistant(client: OpenAI, conversation_id: str, vector_store_id: str, user_text: str) -> str:
    resp = client.responses.create(
        model=MODEL,
        conversation=conversation_id,
        input=[
            {"role": "system", "content": STRICT_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vector_store_id],
            # opcional (dependendo da versão do SDK):
            # "max_num_results": 8,
        }],
    )
    return resp.output_text


def main():
    st.set_page_config(page_title="NYCS RAG Chat", page_icon="🗽")
    st.title("🗽 NYCS RAG Chat (OpenAI Responses + Conversation State)")

    # Sidebar config
    with st.sidebar:
        st.header("Configuração")
        vector_store_id = st.text_input("Vector Store ID", value=VECTOR_STORE_ID_DEFAULT)
        st.caption("Cada sessão do Streamlit = uma conversa nova (conversation state).")

        if st.button("🔄 Nova conversa"):
            # reset conversation + chat UI
            for key in ["conversation_id", "messages"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY não está definido no ambiente.")
        st.stop()

    client = get_client()
    conversation_id = ensure_conversation(client)

    # UI message history (local, só para exibição)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_msg = st.chat_input("Pergunte algo sobre NYCS...")
    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Consultando lore indexado..."):
                answer = call_nycs_assistant(
                    client=client,
                    conversation_id=conversation_id,
                    vector_store_id=vector_store_id,
                    user_text=user_msg,
                )
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
