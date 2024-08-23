import os,glob,re
def combine_specific_prefix_files(src_dir,prefix):
    # Use glob to find all files starting with '24' in the src directory
    file_pattern = os.path.join(src_dir, prefix+'*')
    files_to_combine = glob.glob(file_pattern, recursive=True)

    # Initialize an empty string to hold the combined content
    combined_content = ''

    # Iterate over each file found
    for file_path in files_to_combine:
        # Open the file and read its content
        with open(file_path, 'r', encoding='utf-8') as file:
            # Append the content of the current file to the combined content
            combined_content += file.read() + '\n\n'  # Add two newlines between files

    # Write the combined content to final.md
    with open('../final.md', 'w', encoding='utf-8') as final_file:
        final_file.write(combined_content)

# Example usage
#combine_specific_prefix_files('./papers','24')
