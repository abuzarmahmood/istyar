- Visualizing the timeline of Katz Lab publication and making it easier
    to explore relationship between lab papers and extended literature

Features:
    - Network graph visualization of timeline of article from the lab
        - Edges between publications will be determined by:
            1) Manual input
            2) Automatically inferred using:
                a) Hierarchical Dirichlet Process
            3) Orphan edges will be defaulted to inferred
    - Summary statistics of each node (article)
        1) Citations (timeline)
        2) Title, authors, abstract, link
    - Suggested further reading:
        1) From within lab articles
        2) Citations from article + Articles citing current one
        2) Extended literature
    - Highlight by Author

Backend:
    - Interactive plotting done in:
        - Plotly
    - Scraping using 
        - Biopython via Entrez
    - Inference using:
        - Hierarchical Dirichlet Process (package unknown)
    - Deployment:
        - Heroku
    - Storage (for lab and extended lit scraped articles):
        - Amazon S3
    - Periodic updates:
        - Airflow

Layout:
    - Front end:
        - Easiesnt thing would be to allow people to enter a query (restricted 
            to authors) and simply receive a link to a figure.
    - Data:
        - Notes:
            - If database for text is centralized, this will allow easier
                development of more complex models (Hierachical Dirichlet Process
                or Hierarchical NMF) more easily.
    - NLP:
        - Notes:
            - Avoiding recalculating of embeddings/features with each addition 
                to the database will be challenging
            - Using TF or TFIDF by keeping count of <words per document>
                will allow updates without recalculating i.e. we can store a
                sparse array with word counts for documents. As more documents
                are added, both rows (documents) and word counts (columns) can
                be appended. If the dictionary size (total words) gets too large,
                a max number can be instantiated and words can be replaced using
                some relevance criteria (highest TFIDF?)
        - Organization:
            - Feature extraction
                - TFIDF
                - TF
                ...
                - PCA of features?
            - Clustering/Topic Modelling
                - LDA
                - NMF
                - PCA of weights?
            - Distance calculation (Either on raw features or using topics)
                - Cosine similarity
                - KMEANS
