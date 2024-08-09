import os
import getpass
import sys
import openai
import chromadb
import json
import jsonpickle

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

sys.path.insert(0, '.')

load_dotenv('/Users/ampeezah/code/lodgify/lodgify_test/app/.env')
EMBEDDING_MODEL = "text-embedding-ada-002"


class RAG:
    #this funtion will vectorize the eBook in Chroadb and persist the embeddings in Chromadb index
    #here are we using load and split, it will automatically split by page. this is good enough, but we can experiment with different splitting strategies like text splitter or charachter splitter
    def vectorize(self, path) -> Chroma:
        loader = PyPDFLoader(path)
        documents = loader.load_and_split()
        print(len(documents))
        # The vector store will be persisted in the current directory as .chromadb folder
        self.db = Chroma.from_documents(documents, OpenAIEmbeddings(), persist_directory=".chromadb")
        return self.db

    #load_queries function will iterate line by line through the test_questions.txt file to get each question
    def load_queries(self, path_to_questions):
        with open(path_to_questions) as f:
            self.queries = f.readlines()

    #create a langchain agent to for QA task, it will return the reference page for the answer it provided
    # I use "stuff" as chain_type, sine its a small document and simple question which all can fit easily into memory and provide quick answers
    # I tried setting the chain_type as map_rerank ,however this method failed the faithfullness score
    # I tried setting the chain_type as map_rerank ,however this method took much longer 155 secs vs 42 secs (due to refining process)

    def create_agent_chain(self):
        model_name = "gpt-4o"
        llm = ChatOpenAI(model_name=model_name)
        self.chain = load_qa_with_sources_chain(llm, chain_type="refine")
        return self.chain


    #this function will loop through the questions and get an answer for each question
    # after its done it will append the answer and content to be used for the final output
    def get_answers(self, questions):
        answers = []
        contexts = []
        for question in questions:
            matching_docs = self.db.similarity_search(question)
            result = self.chain.run(input_documents=matching_docs, question=question)
            answers.append(result)
            context_texts = [doc.text if hasattr(doc, 'text') else str(doc) for doc in matching_docs]
            contexts.append(context_texts)
        return answers, contexts
    
    #this function will be format the out in the right format as per requirements. the output needs to be in dict which will then be fed into the ragas/pytest library.
    def format_output(self, questions, answers, contexts):
        output = {
            "question": questions,
            "answer": answers,
            "contexts": contexts
        }
        return output
    
    #this is needed to get the contexts list of strings in the right json format as per the test expectations
    def generate_output(self, questions):
        answers, contexts = self.get_answers(questions)
        formatted_output = self.format_output(questions, answers, contexts)
        return formatted_output


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
        current_directory = os.getcwd()
        path = os.path.join(current_directory, "app/eBook.pdf")
        print(path)
        db = self.vectorize(path)
        chain = self.create_agent_chain()
        #load questions
        self.load_queries("app/test_questions.txt")
        self.output = self.generate_output(self.queries)
            
        #this is needed to get the contexts list of strings in the right json format as pytest/RAGAS expectations
        json_string = jsonpickle.encode(self.output)
        self.output = jsonpickle.decode(json_string)

        return self.output

        #For debugging only
        #matching_docs = db.similarity_search(myrag.queries[0])
        #self.answer = chain.run(input_documents=matching_docs, question=myrag.queries[0])
        #return(self.answer)


#For debugging only
myrag = RAG()
print(myrag.main())

