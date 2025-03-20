from flask import Flask, request, jsonify, render_template
from analyze import get_sentiment, compute_embeddings, classify_email
import json
app = Flask(__name__, template_folder='templates')

@app.route("/")
def home():
    print("Home page")
    return render_template('index.html')


@app.route("/api/v1/sentiment-analysis/", methods=['POST'])
def analysis():
    if request.is_json:
        data = request.get_json()
        sentiment = get_sentiment(data['text'])
        return jsonify({"message": "Data received", "data": data, "sentiment": sentiment}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/valid-embeddings/", methods=['GET'])
def valid_embeddings():
    embeddings = compute_embeddings()
    formatted_embeddings = []
    for text, vector in embeddings:
        formatted_embeddings.append({
            "text": text,
            "vector": vector.tolist() if hasattr(vector, 'tolist') else vector
        })
    embeddings = formatted_embeddings
    return jsonify({"message": "Valid embeddings fetched", "embeddings": embeddings}), 200


@app.route("/api/v1/classify/", methods=['POST'])
def classify():
    if request.is_json:
        data = request.get_json()
        text = data['text']
        classifications = classify_email(text)
        return jsonify({"message": "Email classified", "classifications": classifications}), 200
    else:
        return jsonify({"error": "Invalid Content-Type"}), 400


@app.route("/api/v1/classify-email/", methods=['GET'])
def classify_with_get():
    text = request.args.get('text')
    classifications = classify_email(text)
    return jsonify({"message": "Email classified", "classifications": classifications}), 200
    
def load_classes():
    with open('email_class.json', 'r') as file:
        data = json.load(file)
    return data['classes']
    
def update_classes(new_classes):
    with open('email_class.json', 'w') as file:
        json.dump({"classes": new_classes}, file)       

@app.route('/api/v1/add_class', methods=['POST'])
def add_class():
    # Get the class name from the request
    new_class = request.json.get('class_name')
    
    # Validate that a class name is provided
    if not new_class:
        return jsonify({'error': 'No class name provided'}), 400
    
    # Load current classes
    classes = load_classes()
    
    # Check if the class already exists
    if new_class in classes:
        return jsonify({'error': f'Class "{new_class}" already exists'}), 400
        
        # Add the new class and update the classes file
    classes.append(new_class)
    update_classes(classes)
    
    return jsonify({'message': f'Class "{new_class}" added successfully', 'classes': classes}), 200
    
    
@app.route('/api/v1/delete_class', methods=['DELETE'])
def delete_class():
    # Get the class name to delete from the request
    class_to_delete = request.json.get('class_name')
    
    if not class_to_delete:
        return jsonify({'error': 'No class name provided'}), 400

    classes = load_classes()

    # Check if the class exists
    if class_to_delete not in classes:
        return jsonify({'error': f'Class "{class_to_delete}" not found'}), 404

    # Remove the class and update the classes file
    classes.remove(class_to_delete)
    update_classes(classes)

    return jsonify({'message': f'Class "{class_to_delete}" deleted successfully', 'classes': classes}), 200




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000, debug=True)