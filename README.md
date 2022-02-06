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

