import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Path to your documents
docs_path = r"C:\Users\umesh\Project Documents"
documents = []
doc_names = []

# Read PDF documents
for filename in os.listdir(docs_path):
    if filename.endswith('.pdf'):
        doc_path = os.path.join(docs_path, filename)
        with fitz.open(doc_path) as doc:
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append(text)
            doc_names.append(filename)

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Save the vectorizer and TF-IDF matrix for later use
with open(os.path.join(docs_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(docs_path, 'tfidf_matrix.pkl'), 'wb') as f:
    pickle.dump(tfidf_matrix, f)

# Save document names
with open(os.path.join(docs_path, 'doc_names.pkl'), 'wb') as f:
    pickle.dump(doc_names, f)

print("Documents preprocessed and saved!")
