import streamlit as st
import openai
import pdfplumber
from io import BytesIO
import os
from openai import OpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = ""

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def summarize_text(text):
    """Summarizes the text using OpenAI's latest API."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that summarizes text."},
            {"role": "user", "content": f"Summarize the following text: {text}"}
        ],
        max_tokens=300,
        temperature=0.7
    )
    # summary = response['choices'][0]['message']['content']
    summary = response.choices[0].message.content
    return summary

def ask_question_about_text(text, question):
    """Asks a question about the text using OpenAI's latest API."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on provided text."},
            {"role": "user", "content": f"Based on the following text: {text}\n\nAnswer this question: {question}"}
        ],
        max_tokens=300,
        temperature=0.7
    )
    # answer = response['choices'][0]['message']['content']
    answer = response.choices[0].message.content
    return answer


def main():
    st.title("Chat GPT AI - PDF Chatbot")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file is not None:
        text = extract_text_from_pdf(uploaded_file)
        if text:
            st.write("Text extracted from the PDF:")
            st.text_area("Extracted Text", text, height=150)
            
            # Show a summary of the text
            if st.button("Summarize Text"):
                summary = summarize_text(text)
                st.subheader("Summary:")
                st.write(summary)

            # Ask questions about the text
            question = st.text_input("Ask a question based on the text")
            if st.button("Get Answer"):
                answer = ask_question_about_text(text, question)
                st.subheader("Answer:")
                st.write(answer)
        else:
            st.error("No text could be extracted from this PDF.")

if __name__ == "__main__":
    main()