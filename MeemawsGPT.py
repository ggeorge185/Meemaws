from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import chromadb
import os
import argparse
import time


model = os.environ.get("MODEL", "Mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

def qa():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    llm = Ollama(model=model, callbacks=callbacks)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=not args.hide_source)

    print("Welcome to Meemaw! Let's find the perfect dish for you.")
    
    while True:
        cuisine = input("What type of cuisine are you in the mood for? (e.g., Italian, Chinese, Mexican): ").strip()
        main_ingredient = input("What main ingredient would you like to include? (e.g., chicken, beef, tofu): ").strip()
        dietary_preference = input("Any dietary preferences or restrictions? (e.g., vegetarian, gluten-free): ").strip()
        
        if not cuisine or not main_ingredient:
            print("Cuisine and main ingredient are required to provide a recommendation. Please try again.")
            continue

        query = f"Recommend atleast 5 dishes with {cuisine} dish with {main_ingredient} that is {dietary_preference if dietary_preference else 'no specific dietary preference. If my query is in german,spanish,italian or french please reply with the same'}."
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()
        
        # print(f"\n\nRecommended Dish: {answer} \n")
        # if not args.hide_source:
        #     print("\nSources:")
        #     for document in docs:
        #         print("\n> " + document.metadata["source"] + ":")
        #         print(document.page_content)

        another_query = input("\nWould you like another recommendation? (yes/no): ").strip().lower()
        if another_query != 'yes':
            break
        
        andanother_query = input("\n Would you like to see the source? (yes/no) or you can type exit: ").strip().lower()
        if andanother_query == "exit":
            break
        elif andanother_query == "yes":
            # Print the relevant sources used for the answer
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
        elif andanother_query == "no":
            continue
        else:
            print("Invalid input, continuing with new recommendation.")


def parse_arguments():
    parser = argparse.ArgumentParser(description='BabushkaGPT: Ask for recipe recommendations based on your preferences.')
    parser.add_argument("--hide-source", "-S", action='store_true', help='Use this flag to disable printing of source documents used for answers.')
    parser.add_argument("--mute-stream", "-M", action='store_true', help='Use this flag to disable the streaming StdOut callback for LLMs.')
    return parser.parse_args()

if __name__ == "__main__":
    qa()
