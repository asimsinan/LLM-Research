import subprocess

import arxiv, os, re
from datetime import datetime
import json


def rename_file(filename):

    new_filename = re.sub(r'^\d+\.\d+v\d+_', '', filename)
    new_filename = new_filename[new_filename.find(' ') + 1:]

    new_filename = re.sub(r'_', ' ', new_filename)
    os.rename(filename, new_filename)


def add_paper_by_id(paper_id):
    client = arxiv.Client()
    search_by_id = arxiv.Search(id_list=[paper_id])
    paper = next(client.results(search_by_id))
    formatted_date = paper.published.strftime("%d-%m-%Y")
    date_object = datetime.strptime(formatted_date, "%d-%m-%Y")
    directory = str(date_object.year)
    if not os.path.exists(directory):
        os.makedirs(directory)
    subprocess.run(['getpaper','-d', directory, paper_id])

    with open(directory+'/000_Paper_List.json', 'r') as f:
        paper_json = json.load(f)

    paper_json[paper_id]["published_date"] = formatted_date
    paper_json[paper_id]["year"] = date_object.year
    paper_json[paper_id]["categories"]=paper.categories
    rename_file(directory+"/"+paper_json[paper_id]["download_name"])

    with open('papers.json', 'r') as f:
        papers = json.load(f)
        papers.append(paper_json)
    with open('papers.json', 'w') as f:
        json.dump(papers, f)



add_paper_by_id("2401.08577")

