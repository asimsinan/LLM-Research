import json
from collections import defaultdict
from datetime import datetime
from operator import itemgetter
# Load the JSON file
with open('papers.json', 'r') as f:
    papers = json.load(f)

# Group the papers by year
papers_by_year = defaultdict(list)
for paper in papers:
    papers_by_year[paper['year']].append(paper)

# Open the Markdown file
with open('Readme.md', 'w') as f:
    f.write(f"A curated list of papers on Large Language Models by year. I'll try to update the list if new papers are published. Let me know if I am missing important papers.\n")
    # Loop over the years
    sorted_papers = {}
    for year, papers in sorted(papers_by_year.items(), reverse=False):
        sorted_papers[year] = sorted(papers, key=lambda x: datetime.strptime(x['published'], '%d-%m-%Y'), reverse=False)
    for j, (year, papers) in enumerate(sorted_papers.items(), start=1):
            if (len(papers) == 1):
                f.write(f"\n## {year} ({len(papers)} paper)")
            else:
                f.write(f"\n## {year} ({len(papers)} papers)")
            i=1
            for paper in papers:
                # Write the paper title, authors, and published date
                f.write(f"\n\n{i}. [{paper['title']}]({paper['url']}), {paper['authors']}, {paper['published']}\n")
                # Write arxiv category

                if(len(paper["categories"])>=1):
                    f.write(f"      ### Categories\n")
                    f.write(f"      {', '.join(paper['categories'])}\n")
                # Write the abstract
                abstract=paper['abstract'].strip()
                f.write(f"     ### Abstract\n")
                f.write(f"     {abstract}\n")
                i+=1
