import json
from collections import defaultdict

# Load the JSON file
with open('papers.json', 'r') as f:
    papers = json.load(f)

# Group the papers by year
papers_by_year = defaultdict(list)
for paper in papers:
    year = paper['published'].split('-')[2]
    papers_by_year[year].append(paper)

# Open the Markdown file
with open('Readme.md', 'w') as f:
    # Loop over the years
    for year, papers in sorted(papers_by_year.items(), reverse=False):
        # Write the year header
        if(len(papers)==1):
            f.write(f"## {year} ({len(papers)} paper)\n\n")
        else:
            f.write(f"## {year} ({len(papers)} papers)\n\n")


        # Loop over the papers
        for i, paper in enumerate(sorted(papers, key=lambda x: x['published'], reverse=False), start=1):
            # Write the paper title, authors, and published date

            f.write(f"\n\n{i}. [{paper['title']}]({paper['url']}), {paper['authors']}, {paper['published']}\n")
            # Write arxiv category

            if(len(paper["categories"])>=1):
                f.write(f"     ### Categories\n")
                f.write(f"     {', '.join(paper['categories'])}\n")
            # Write the abstract
            abstract=paper['abstract'].strip()
            f.write(f"    ### Abstract\n    {abstract}\n")
