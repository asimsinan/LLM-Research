import json,os
from collections import defaultdict
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
    'math-ph':'Mathematical Physics',
    'stat.ML':'Machine Learning',
    'math.MG':'Algebraic Geometry',
    'math.MP':'Mathematical Physics',
'eess.SP':'Signal Processing',
    'cs.OS':'Operating Systems'


    # Add more mappings as needed
}
with open('papers.json', 'r') as f:
    papers = json.load(f)

# Group the papers by year
papers_by_year = defaultdict(list)
for paper in papers:
    for paper_id, paper_details in paper.items():
        papers_by_year[paper_details["year"]].append(paper_details)

# Open the Markdown file
with open('Readme.md', 'w') as f:
    f.write(f"A curated list of papers on Large Language Models by year. I'll try to update the list if new papers are published. Let me know if I am missing important papers.\n\n")
    f.write(f"## TO-DO:\n\n")
    f.write(f"* Add tools, frameworks etc.tutorials\n\n"
            f"* Open source models\n\n"
            f"* Add datasets, benchmarks etc.\n\n"
            f"* Add Ms.C. and Ph.D thesis around the world\n\n"
            f"* Add university courses thought around the world\n\n")
    # Loop over the years
    sorted_papers = {}
    for year, papers in sorted(papers_by_year.items(), reverse=False):
        sorted_papers[year] = sorted(papers, key=lambda x: datetime.strptime(x['published_date'], '%d-%m-%Y'), reverse=False)
    for j, (year, papers) in enumerate(sorted_papers.items(), start=1):
        filename = f"{year}.md"  # create a filename for each year
        with open(filename, "w") as file:
            paper_count=len(papers)
            if (paper_count == 1):
                file.write(f"\n## {year} (1 paper)")
                f.write(f"- [{year}]({filename}) (1 paper)\n")
            else:
                file.write(f"\n## {year} ({paper_count} papers)")
                f.write(f"- [{year}]({filename}) ({paper_count} papers)\n")
            i=1
            for paper in papers:
                # Write the paper title, authors, and published date
                file.write(f"\n\n{i}. [{paper['title']}]({paper['abs_url']}), {','.join(paper['authors'])}, {paper['published_date']}\n")
                # Write arxiv category
                if(len(paper["categories"])>=1):
                    categories = [keyword_mapping[category] for category in paper["categories"]]
                    file.write(f"      ### Categories\n")
                    file.write(f"      {', '.join(categories)}\n")
                # Write the abstract
                abstract=paper['abstract'].strip()
                file.write(f"     ### Abstract\n")
                file.write(f"     {abstract}\n")
                i+=1

