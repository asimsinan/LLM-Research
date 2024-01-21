import json
import arxiv
from datetime import datetime

keyword_mapping = {
    'cs.CL': 'Computation and Language',
    'cs.AI': 'Artificial Intelligence',
    'cs.LG': 'Machine Learning',
    'cs.RO': 'Robotics',
    'cs.CV': 'Computer Vision',
    'cs.CM': 'Computer Motion',
    'cs.MM': 'Multimedia',
    'cs.MA': 'Multiagent Systems',
    'cs.CY': 'Computers and Society',
    'cs.IR': 'Information Retrieval',
    'cs.HC': 'Human-Computer Interaction',
    'q-bio.TO': "Quantitative Biology",
    'cs.SE': 'Software Engineering',
    'cs.CR': 'Cryptography and Security',
    'cs.DL':'Digial Libraries',
    'cs.SI':'Social and Information Networks',
    'cs.NE':'Neural and Evolutionary Computing',
    'I.2.7':'Natural Language Processing',
    'q-fin.GN':'General Finance',
    'cs.DC':'Distributed, Parallel, and Cluster Computing',
    'cs.IT':'Information Theory',
    'math.IT':'Information Theory',
    'physics.ao-ph':'Atmospheric and Oceanic Physics',
    'math.DG':'Differential Geometry',
    'math-ph':'Mathematical Physics'


    # Add more mappings as needed
}
with open('papers.json', 'r') as f:
    papers = json.load(f)

client = arxiv.Client()
for document in papers:
    for paper_id, details in document.items():
        search_by_id = arxiv.Search(id_list=[paper_id])
        first_result = next(client.results(search_by_id))
        formatted_date = first_result.published.strftime("%d-%m-%Y")
        details["published_date"] = formatted_date
        date_object = datetime.strptime(formatted_date, "%d-%m-%Y")
        details["year"] = date_object.year
       # categories = [keyword_mapping[category] for category in first_result.categories]
        details["categories"] = first_result.categories
        print(first_result.title)
        with open('papers.json', 'w') as f:
            json.dump(papers, f)
