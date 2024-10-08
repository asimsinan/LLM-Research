import subprocess
from transformers import pipeline
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import arxiv, os, re,fnmatch
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
    'cond-mat.mtrl-sci':'Material Science',
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
    'cs.OS': 'Operating Systems',
    '68T50':'Natural Language Processing',
    'cs.GT':'Computer Science and Game Theory',
    'H.3.3':'Information Search and Retrieval',
    'cs.PL':"Programming Languages",
    'econ.GN':"General Economics",
    'q-fin.EC':"Quantitative Finance",
    'q-bio.OT':"Quantitative Biology"

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
    directory = 'papers'
    filename = paper_id + '.md'
    full_path = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(full_path, 'w') as f:

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



def find_pdf_files(folder_path):
    # Initialize an empty list to store PDF file names
    pdf_file_names = []

    # Ensure the folder path ends with a slash for proper concatenation
    if not folder_path.endswith(os.sep):
        folder_path += os.sep

    # Get a list of all files in the specified folder
    all_files = os.listdir(folder_path)

    # Filter the list to include only PDF files
    pdf_files = [file for file in all_files if fnmatch.fnmatch(file, '*.pdf')]

    # Add the PDF file names to the list, excluding the '.pdf' extension
    for pdf_file in pdf_files:
        pdf_file_names.append(pdf_file.replace('.pdf', ''))

    sorted_pdf_file_names = sorted(pdf_file_names)
    return sorted_pdf_file_names
folder_path = '/Users/sinanyuksel/Desktop/LLM Papers/new papers'
new_papers = find_pdf_files(folder_path)
for paper in new_papers:
    add_paper_by_id(paper)

