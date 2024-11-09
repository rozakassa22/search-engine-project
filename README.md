# 🌐 **Search Engine Application**

This project is a simple search engine built using **Flask**, **HTML**, and **CSS**. It allows users to search for terms within a collection of `.txt` documents, displaying snippets of relevant matches and showing the number of documents that contain the query.

---

## 🚀 **Features**

- **🔍 Search Functionality**: Users can enter a query, and the application searches through all `.txt` files in the `documents` directory.
- **📊 TF-IDF Vectorization**: Uses TF-IDF to represent documents and user queries for effective searching.
- **📈 Cosine Similarity**: Calculates similarity between the query and documents to find relevant matches.
- **📜 Snippet Display**: Shows a relevant snippet from each matching document that includes the query terms.
- **📄 Document Count**: Displays the number of documents containing the query.
- **💻 Responsive & Modern Design**: Inline CSS provides a clean and visually appealing layout.

---

## 🛠 **Requirements**

Ensure you have the following Python packages installed:

```plaintext
Flask
nltk
scikit-learn
numpy
```

## ⚙️ **Installation**

 - Clone the Repository
    ```bash
    git clone <repository_url>
    ```
 - Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```
  - Run the Application
    ```bash
    python app.py
    ```  
  - Access the Application
  Open your web browser and go to http://localhost:5000.  
## 💡 **Usage**  
  - Open the application in your browser.
  - Enter a search query in the search bar on the homepage.
  - View the results page, which will display:
        - The number of documents containing the query.
        - A snippet from each matching document.