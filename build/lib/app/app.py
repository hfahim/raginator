import os
import getpass
import sys
import openai
sys.path.insert(0, '.')
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


load_dotenv('/Users/ampeezah/code/lodgify/lodgify_test/app/.env')
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"




class RAG:

    def vectorize(self, path):
        path = "/Users/ampeezah/code/lodgify/lodgify_test/app/eBook.pdf"
        loader = PyPDFLoader(path)
        documents = loader.load_and_split()
        print(len(documents))

        # The vector store will be persisted in the current directory as .chromadb folder
        db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=".chromadb")


    def load_queries(self, path_to_questions):
        with open(path_to_questions) as f:
            self.queries = f.readlines()

    def main(self):
        """
        Execute the full Retrieval-Augmented Generation (RAG) pipeline to process a list of test questions from (test_questions.txt)
        generate answers for each, and extract relevant contexts from the original the pdf text sources (eBook.pdf).

        This method orchestrates the workflow of the RAG pipeline, starting from question processing,
        through context retrieval, answer generation, and finally compiling the results into a structured JSON format.
        The JSON object includes the original questions, their corresponding answers, and the contexts used to generate these answers.

        Note: The results will be assessed using the 'ragas' library, focusing on the following metrics:
            - Faithfulness: The degree to which the generated answers accurately reflect the information in the context.
            - Answer Relevancy: How relevant the generated answers are to the input questions.
            - Context Utilization: The accuracy and relevance of the contexts retrieved to inform the answer generation.

        Returns:
            dict: A JSON-structured dictionary containing the following keys:
                - "question": A list of the input test questions.(the 3 questions from the test_questions.txt)
                - "answer": A list of generated answers corresponding to each question.
                - "contexts": The extracted contexts from original text sources that were used to inform the generation of each answer.

        """
        # Implementation of the RAG pipeline goes here

        ## vectorize book
        path = "/Users/ampeezah/code/lodgify/lodgify_test/app/eBook.pdf"
        #RAG.vectorize(path=path)
        print(" HELLOO RAG")
        #process questions
        #self.load_queries("/Users/ampeezah/code/lodgify/lodgify_test/app/test_questions.pdf")



        #myrag.ask(myrag.queries[0])

        

