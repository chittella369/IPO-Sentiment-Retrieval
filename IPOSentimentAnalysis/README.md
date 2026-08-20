# IPO Information Retrieval & Sentiment Analysis

This project performs information retrieval, custom document ranking, and unsupervised sentiment analysis on news results gathered for Initial Public Offering (IPO) companies. It ranks news results using different search approaches and estimates investor sentiment using vector-space distance calculations against the Loughran-McDonald (LM) financial sentiment lexicon.

A domain specific LM Dictionary (Loughran-McDonald dictionary) which is a specialized sentiment word list created for finance and economics is used.
Unlike standard dictionaries like Merriam-Webster, it reclassifies common business terms (like liability or tax) that normal tools wrongly label as negative, helping analysts accurately score corporate filings and news

# Note : Do not use this model to make your investment decision. This is only to be used for knowledge purpose and shouldn't be used to trade/invest in any kind of securities what so ever.

---

## 📂 Project Structure & Notebooks

### 1. [IPO_InformationRetrieval-v0.1.ipynb](file:///Users/aditya/workspace/Dissertation/IPO_InformationRetrieval-v0.1.ipynb)
This notebook focuses on **Information Retrieval (IR)** and ranking evaluation:
* **Crawling & Data Aggregation:** Aggregates search results for a given company query from Google News and Bing News.
* **Text Preprocessing:** Cleans and normalizes title and snippet text (lowercasing, punctuation removal, word tokenization).
* **Document Ranking:**
  * **Approach 1 (Sort):** Ranks documents by sorting search results based on the search engine's original rank and source engine.
  * **Approach 2 (Term Frequency):** Ranks documents using a Term Frequency (TF) relevance scoring function matching query terms.
* **Evaluation:** Computes Precision@n (P@5, P@10, P@15, P@20, P@25, P@30) using the aggregated news list as a pseudo-ground truth. It plots a precision curve showing both approaches and prints their respective Mean Average Precision (MAP).

### 2. [IPO_ML_Model-Unsupervised-Final-v1.0.ipynb](file:///Users/aditya/workspace/Dissertation/IPO_ML_Model-Unsupervised-Final-v1.0.ipynb)
This notebook implements **Unsupervised Machine Learning** for sentiment classification:
* **Preprocessing:** Tokenizes text, filters stopwords, and writes out the processed vocabulary.
* **Lexical Sentiment Dictionary Mapping:** Integrates dictionaries (pysentiment2, LM, HIV4) to calculate basic polarity.
* **TF-IDF Sentiment Vector Distance Analysis:** Vectorizes document snippets and sentiment lexicons into a shared TF-IDF space and derives sentiment by measuring distance to positive vs. negative semantic centroids.

---

## 🛠️ Setup & Execution Instructions

Follow these steps to run the notebooks without errors:

### 1. Install Dependencies
Ensure you have Python 3.x and Jupyter installed. Install the required Python packages:
```bash
pip install pandas numpy matplotlib scikit-learn openpyxl nltk pysentiment2
```

### 2. Prepare Lexicons
Place the Loughran-McDonald sentiment list spreadsheet (`LoughranMcDonald_SentimentWordLists_2018.xlsx`) in the same directory as the notebooks.

### 3. Run the Information Retrieval Notebook
1. Open and run [IPO_InformationRetrieval-v0.1.ipynb](file:///Users/aditya/workspace/Dissertation/IPO_InformationRetrieval-v0.1.ipynb) first.
2. Enter the IPO company name when prompted (e.g., `Dhoot Transmission Ltd`, `Technocraft Ventures`, `India Pesticides`, or `Shiprocket`).
3. Running this notebook creates the processed text outputs (e.g., `<Company>_ResultantRanks_A1.txt`, `<Company>_ResultantRanks_A2.txt`, and `<Company>_RankedDocuments.csv`) inside a subdirectory named after the company.

### 4. Run the Sentiment ML Notebook
1. Open and run [IPO_ML_Model-Unsupervised-Final-v1.0.ipynb](file:///Users/aditya/workspace/Dissertation/IPO_ML_Model-Unsupervised-Final-v1.0.ipynb).
2. This notebook loads the generated text outputs from the IR step and evaluates overall investor sentiment using vector distance measures.

---

## 📐 Sentiment Derivation via Distance Measure

The unsupervised model classifies the sentiment of the document corpus using a **Vector Space Model (VSM)**:

1. **Shared Vocabulary Vectorization:**
   A single `TfidfVectorizer` is fitted on all document snippets to establish a shared $N$-dimensional vocabulary space (where $N$ is the number of unique terms). Both document snippets and the positive/negative lists from the Loughran-McDonald sentiment lexicon are mapped into this space.

2. **Centroid Representations:**
   * **Document Corpus Vector ($\mathbf{v}_{\text{doc}}$):** The average TF-IDF vector representing the average news sentiment in the corpus:
     $$\mathbf{v}_{\text{doc}} = \frac{1}{|D|} \sum_{d \in D} \mathbf{v}_d$$
   * **Positive Sentiment Vector ($\mathbf{v}_{\text{pos}}$):** The average TF-IDF vector representing positive terms:
     $$\mathbf{v}_{\text{pos}} = \frac{1}{|W_{\text{pos}}|} \sum_{w \in W_{\text{pos}}} \mathbf{v}_w$$
   * **Negative Sentiment Vector ($\mathbf{v}_{\text{neg}}$):** The average TF-IDF vector representing negative terms:
     $$\mathbf{v}_{\text{neg}} = \frac{1}{|W_{\text{neg}}|} \sum_{w \in W_{\text{neg}}} \mathbf{v}_w$$

3. **Euclidean Distance Calculation:**
   The Euclidean distances ($d_{\text{pos}}$ and $d_{\text{neg}}$) between the document corpus vector and the sentiment vectors are calculated:
   $$d_{\text{pos}} = \sqrt{\sum_{i=1}^N (v_{\text{doc},i} - v_{\text{pos},i})^2}$$
   $$d_{\text{neg}} = \sqrt{\sum_{i=1}^N (v_{\text{doc},i} - v_{\text{neg},i})^2}$$

4. **Sentiment Classification Decision:**
   * If **$d_{\text{pos}} < d_{\text{neg}}$**, the document centroid is closer to positive words than to negative ones. The sentiment is classified as **Positive** (recommend applying).
   * Otherwise, the sentiment is classified as **Negative** (recommend avoiding).
