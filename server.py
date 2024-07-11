#!/usr/bin/env python3
from flask import Flask, request, jsonify
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
import os
import time

app = Flask(__name__)

model = os.environ.get("MODEL", "Mistral")
embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME", "all-MiniLM-L6-v2")
persist_directory = os.environ.get("PERSIST_DIRECTORY", "db")
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))

embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
llm = Ollama(model=model)
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    cuisine = data.get('cuisine')
    main_ingredient = data.get('mainIngredient')
    dietary_preference = data.get('dietaryPreference', 'no specific dietary preference')

    query = f"Recommend a {cuisine} dish with {main_ingredient} that is {dietary_preference}."
    start = time.time()
    res = qa(query)
    answer, docs = res['result'], res['source_documents']
    end = time.time()

    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in docs]

    return jsonify({
        "answer": answer,
        "sources": sources
    })

if __name__ == "__main__":
    app.run(debug=True)
