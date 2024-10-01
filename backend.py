from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your_secret_key')
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:MERImarzi@1@localhost/erp_database'
db = SQLAlchemy(app)

# Document Model (Example for handling uploaded files)
class Document(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    content = db.Column(db.Text, nullable=False)

# Endpoint to upload and store documents (PDFs, Images)
@app.route('/upload', methods=['POST'])
def upload_document():
    files = request.files.getlist("files")
    documents = []
    
    for file in files:
        content = file.read().decode('utf-8') if file.filename.endswith('.pdf') else ''
        # Add document to database
        new_doc = Document(filename=file.filename, content=content)
        db.session.add(new_doc)
        db.session.commit()
        documents.append({"filename": file.filename, "content": content})
    
    return jsonify({"documents": documents}), 201

# Example API endpoint to retrieve stored documents
@app.route('/documents', methods=['GET'])
def get_documents():
    documents = Document.query.all()
    result = [{"id": doc.id, "filename": doc.filename, "content": doc.content} for doc in documents]
    return jsonify(result), 200

# Example API endpoint for advanced ERP-related processing (e.g., budgeting, expense tracking)
@app.route('/erp-data', methods=['POST'])
def erp_data():
    data = request.json
    # Perform some processing based on the data provided
    # This could be budget forecasting, expense tracking, etc.
    
    # Placeholder response for ERP processing
    return jsonify({"status": "success", "message": "ERP data processed successfully!"}), 200

if __name__ == "__main__":
    app.run(debug=True)
