
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from vector import retriever

st.set_page_config(
    page_title="Restaurant Review Assistant",
    page_icon="🍕",
    layout="centered",
)

@st.cache_resource
def build_chain():
    model = OllamaLLM(model="llama3.2")

    template = """
You are an expert assistant answering questions about a pizza restaurant.

Answer the user's question using the restaurant reviews provided below.
Do not invent information that is not supported by the reviews.
If the reviews do not contain enough information, clearly say so.

Relevant restaurant reviews:
{reviews}

User's question:
{question}

Answer:
"""

    prompt = ChatPromptTemplate.from_template(template)

    return prompt | model


chain = build_chain()


st.title("🍕 Restaurant Review Assistant")

st.write(
    "Ask questions about restaurant food, service, "
    "ratings, and customer experiences."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


question = st.chat_input(
    "Ask a question about the restaurant..."
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Searching reviews and generating an answer..."
        ):
            try:
                relevant_reviews = retriever.invoke(question)

                if not relevant_reviews:
                    answer = (
                        "I couldn't find relevant restaurant "
                        "reviews to answer your question."
                    )

                else:
                    reviews_context = "\n\n".join(
                        (
                            f"Review {index}:\n"
                            f"{document.page_content}\n"
                            f"Rating: {document.metadata.get('rating', 'Not provided')}\n"
                            f"Date: {document.metadata.get('date', 'Not provided')}"
                        )
                        for index, document in enumerate(
                            relevant_reviews,
                            start=1,
                        )
                    )

                    result = chain.invoke(
                        {
                            "reviews": reviews_context,
                            "question": question,
                        }
                    )

                    answer = str(result).strip()

                    if not answer:
                        answer = (
                            "The model returned an empty response. "
                            "Please try asking your question again."
                        )


                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except Exception as error:
                error_message = (
                    "I couldn't process your question. "
                    "Please check that Ollama is running, "
                    "the required models are available, and "
                    "the vector database can be accessed."
                )

                st.error(error_message)

                with st.expander("Technical error details"):
                    st.code(str(error))