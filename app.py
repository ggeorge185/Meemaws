from flask import Flask, request, jsonify, render_template
from BabushkaGPT import qa  # Import your QA function from BabushkaGPT

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    cuisine = data.get('cuisine')
    main_ingredient = data.get('mainIngredient')
    dietary_preference = data.get('dietaryPreference')

    if not cuisine or not main_ingredient:
        return jsonify({"error": "Cuisine and main ingredient are required"}), 400

    query = f"Recommend a {cuisine} dish with {main_ingredient} that is {dietary_preference if dietary_preference else 'no specific dietary preference'}."
    res = qa(query)
    answer, docs = res['result'], res.get('source_documents', [])

    sources = [{'source': doc.metadata["source"], 'page_content': doc.page_content} for doc in docs]

    return jsonify({"answer": answer, "sources": sources})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
