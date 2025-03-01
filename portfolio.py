import pandas as pd
import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class Portfolio:
    def __init__(self, file_path=r"C:\Users\dell\OneDrive\Desktop\Cold_email_generator\project-genai-cold-email-generator\my_portfolio.csv"):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.vectorstore = None
        self.load_portfolio()

    def get_vector_store(self, text_chunks):
        """Create a vector store from text chunks."""
        vector_store = FAISS.from_texts(text_chunks, embedding=self.embeddings)
        vector_store.save_local("faiss_index")
        return vector_store

    def load_portfolio(self):
        """Load portfolio data into the vector store."""
        if not self.vectorstore:
            documents = self.data["Techstack"].tolist()
            metadatas = [{"links": link} for link in self.data["Links"].tolist()]
            # Create the FAISS vector store from documents
            self.vectorstore = FAISS.from_texts(texts=documents, embedding=self.embeddings, metadatas=metadatas)
            # Optionally, save the vector store locally
            self.vectorstore.save_local("faiss_index")

    def query_links(self, skills):
        """Query the vector store with given skills."""
        if not self.vectorstore:
            print("Vector store is not loaded. Please load the portfolio first.")
            return []
        # Perform similarity search
        results = self.vectorstore.similarity_search(skills, k=2)
        return [result.metadata for result in results]

# Example usage:
portfolio = Portfolio()
print(portfolio.query_links("Python, Machine Learning"))


