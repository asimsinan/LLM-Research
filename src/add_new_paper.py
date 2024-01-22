import subprocess
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import arxiv, os, re
from datetime import datetime
import json

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
    'cs.DL': 'Digial Libraries',
    'cs.SI': 'Social and Information Networks',
    'cs.NE': 'Neural and Evolutionary Computing',
    'I.2.7': 'Natural Language Processing',
    'q-fin.GN': 'General Finance',
    'cs.DC': 'Distributed, Parallel, and Cluster Computing',
    'cs.IT': 'Information Theory',
    'math.IT': 'Information Theory',
    'physics.ao-ph': 'Atmospheric and Oceanic Physics',
    'math.DG': 'Differential Geometry',
    'math-ph': 'Mathematical Physics',
    'stat.ML': 'Machine Learning',
    'math.MG': 'Algebraic Geometry',
    'math.MP': 'Mathematical Physics',
    'eess.SP': 'Signal Processing',
    'cs.OS': 'Operating Systems'

    # Add more mappings as needed
}
def init_llm():
    checkpoint = "MBZUAI/LaMini-Flan-T5-248M"
    tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)
    summarizer = pipeline(
        task="summarization",
        model=base_model,
        tokenizer=tokenizer,
        max_length=200,
        min_length=50
    )
    return summarizer
summarizer = init_llm()
def summarize(abstract):
    output = summarizer(abstract)
    if (len(abstract) > 50):
        summary = output[0]['summary_text']
        bullet_points = summary.split(". ")
        markdown = f"### Bullet Points\n\n"
        for point in bullet_points:
            markdown = markdown + (f"    * {point}\n\n")
        return markdown

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
    subprocess.run(['getpaper', '-d', directory, paper_id])

    with open(directory + '/000_Paper_List.json', 'r') as f:
        paper_json = json.load(f)

    paper_json[paper_id]["published_date"] = formatted_date
    paper_json[paper_id]["year"] = date_object.year
    paper_json[paper_id]["categories"] = paper.categories
    rename_file(directory + "/" + paper_json[paper_id]["download_name"])

    paper = paper_json[paper_id]
    with open('./src/' + paper_id + '.md', 'w') as f:
        # Write the paper title, authors, and published date
        f.write(
            f"\n\n{1}. [{paper['title']}]({paper['abs_url']}), {','.join(paper['authors'])}, {paper['published_date']}\n")
        # Write arxiv category
        if (len(paper["categories"]) >= 1):
            categories = [keyword_mapping[category] for category in paper["categories"]]
            f.write(f"      ### Categories\n")
            f.write(f"      {', '.join(categories)}\n")
        # Write the abstract
        abstract = paper['abstract'].strip()
        f.write(f"     ### Abstract\n")
        f.write(f"     {abstract}\n")
        f.write(f"     {summarize(abstract)}\n")
        print(paper["title"])

    # with open('./src/papers.json', 'r') as f:
    #    papers = json.load(f)
    #    papers.append(paper_json)
    # with open('./src/papers.json', 'w') as f:
    #    json.dump(papers, f)


add_paper_by_id("2401.08577")
