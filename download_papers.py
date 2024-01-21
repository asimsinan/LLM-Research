import arxiv,os
from datetime import datetime
import json
download_location = "LLM"
with open('Papers.json', 'r') as f:
    papers = json.load(f)

client = arxiv.Client()
for document in papers:
    for paper_id, details in document.items():
        search_by_id = arxiv.Search(id_list=[paper_id])
        paper = next(client.results(search_by_id))
        formatted_date = paper.published.strftime("%d-%m-%Y")
        date_object = datetime.strptime(formatted_date, "%d-%m-%Y")
        filename = paper.title + '.pdf'
        directory=download_location + "/" + str(date_object.year)
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_path=os.path.join(directory)
        paper.download_pdf(full_path)
        print(paper.title)




