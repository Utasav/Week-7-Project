from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
from striprtf.striprtf import rtf_to_text  # Library for reading RTF files
import numpy as np

app = Flask(__name__)

# Path to the folder with the RTF documents
doc_folder = r"C:\News documents\Week 6 NEWS Documents"

# Function to read the content of an RTF document
def read_rtf_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        rtf_content = file.read()
    return rtf_to_text(rtf_content)

# Load documents from the folder
documents = []
document_content = []

# Debugging: Check if the directory exists and print its contents
if not os.path.exists(doc_folder):
    print(f"Error: The folder '{doc_folder}' does not exist.")
    raise ValueError(f"Folder '{doc_folder}' does not exist.")
else:
    print(f"Folder '{doc_folder}' found. Listing files...")

files_in_directory = os.listdir(doc_folder)
if not files_in_directory:
    print(f"Error: The folder '{doc_folder}' is empty.")
    raise ValueError(f"Folder '{doc_folder}' is empty.")
else:
    print(f"Files found: {files_in_directory}")

# Process RTF documents and add debug print statements
for filename in files_in_directory:
    if filename.endswith(".rtf"):
        file_path = os.path.join(doc_folder, filename)
        print(f"Processing file: {file_path}")  # Debug: Show which file is being processed
        try:
            content = read_rtf_file(file_path)  # Read the content of the RTF document
            if content.strip():  # Ensure the document is not empty
                documents.append(filename)  # Add the document name
                document_content.append(content)
                print(f"Successfully read file: {filename}")  # Debug: Confirm file was read
            else:
                print(f"Warning: File '{filename}' is empty.")  # Debug: Warn if file is empty
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# If no documents were processed, raise an error
if not documents:
    raise ValueError("No valid documents to process.")

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer()
doc_vectors = vectorizer.fit_transform(document_content)

# Simulated relevance labels (manually labeled for simplicity, use actual labels in a real-world case)
# In real projects, you'd collect feedback or create a dataset of relevant vs non-relevant documents
relevance_labels = [1, 0, 1, 1]  # Assuming relevance for some documents, adjust this for your data

# Train Logistic Regression (or Naive Bayes) to predict relevance
X_train, X_test, y_train, y_test = train_test_split(doc_vectors, relevance_labels, test_size=0.2, random_state=42)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Metrics calculation functions
def calculate_precision_at_k(similarity_scores, true_labels, k=5):
    top_k_indices = similarity_scores.argsort()[-k:][::-1]
    top_k_relevant = sum([true_labels[i] for i in top_k_indices])
    precision_at_k = top_k_relevant / k
    return precision_at_k

def calculate_recall_at_k(similarity_scores, true_labels, k=5):
    top_k_indices = similarity_scores.argsort()[-k:][::-1]
    total_relevant = sum(true_labels)
    recall_at_k = sum([true_labels[i] for i in top_k_indices]) / total_relevant if total_relevant > 0 else 0
    return recall_at_k

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    
    # Check if the query is empty
    if not query.strip():
        return render_template('result.html', query=query, results=[], error="Please enter a valid search query.")
    
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity
    similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()

    # Get top 5 most similar documents
    top_indices = similarity_scores.argsort()[-5:][::-1]
    results = [
        {"title": documents[i], "relevance": round(similarity_scores[i] * 100, 2), "snippet": document_content[i][:200]}
        for i in top_indices
    ]

    # Precision and Recall at K (k=5)
    precision_at_k = calculate_precision_at_k(similarity_scores, relevance_labels, k=5)
    recall_at_k = calculate_recall_at_k(similarity_scores, relevance_labels, k=5)

    # Logistic Regression metrics
    predictions = logreg.predict(query_vector)
    accuracy = accuracy_score(relevance_labels, predictions)
    precision = precision_score(relevance_labels, predictions, zero_division=0)
    recall = recall_score(relevance_labels, predictions, zero_division=0)

    return render_template('result.html', query=query, results=results, 
                           precision_at_k=precision_at_k, recall_at_k=recall_at_k,
                           accuracy=accuracy, precision=precision, recall=recall)

if __name__ == '__main__':
    app.run(debug=True)
