import os
import json
import datetime

import requests
from xml.etree import ElementTree as ET
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Define the directory where the PDF files are stored
directory = "LLM Papers"

keyword_mapping = {
   'cs.CL': 'Computation and Language',
   'cs.AI': 'Artificial Intelligence',
    'cs.LG':'Machine Learning'
   # Add more mappings as needed
}
# Initialize an empty list to store the results
papers_list = []
paper_id=1
# Loop over all files in the directory
for root, dirs, files in os.walk(directory):
    for filename in files:
        # Check if the file is a PDF
     if filename.endswith('.pdf'):
        # Remove the .pdf extension from the filename
        paper_name = filename.replace(".pdf", "")
        # Construct the search URL
        search_url = f"http://export.arxiv.org/api/query?search_query=ti:{paper_name}"

        # Send the HTTP request
        response = requests.get(search_url)

        # Parse the XML response
        root = ET.fromstring(response.content)

        # Loop over the entries in the response
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            # Get the title of the paper
            title = entry.find("{http://www.w3.org/2005/Atom}title").text

            # Calculate the Jaccard similarity between the title and the paper name
            vectorizer = CountVectorizer().fit_transform([title, paper_name])
            vectors = vectorizer.toarray()
            jaccard_sim = cosine_similarity(vectors)[0][1]

            # Check if the Jaccard similarity is above a certain threshold
            if jaccard_sim > 0.9:
                published = entry.find("{http://www.w3.org/2005/Atom}published").text
                date_obj = datetime.datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                published_ddmmyyyy = date_obj.strftime("%d-%m-%Y")
                categories = [keyword_mapping[category.attrib['term']] for category in
                              entry.findall("{http://www.w3.org/2005/Atom}category") if
                              category.attrib['term'] in keyword_mapping]

                # Create a dictionary to store the paper information
                print(paper_id,". ",title)
                paper_id+=1
                paper_dict = {
                    'title': title,
                    'url': entry.find("{http://www.w3.org/2005/Atom}id").text,
                    'authors': ', '.join([author.text for author in entry.findall(
                        "{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name")]),
                    'abstract': entry.find("{http://www.w3.org/2005/Atom}summary").text,
                    'published': published_ddmmyyyy,
                    'year':published_ddmmyyyy.split('-')[2],
                    'categories':categories

                }

                # Add the dictionary to the list of papers
                papers_list.append(paper_dict)

# Sort the papers by title
papers_list.sort(key=lambda x: x['title'], reverse=False)

# Write the papers to a JSON file
with open('papers.json', 'w') as f:
    json.dump(papers_list, f)

