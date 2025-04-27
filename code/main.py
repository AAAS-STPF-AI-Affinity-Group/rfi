import os
import requests
import zipfile
import argparse
import shutil
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv
from datetime import datetime
import json
import jinja2
import pandas as pd
from tqdm import tqdm

load_dotenv()

def create_directory(directory_path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    return directory_path


def download_file(url, destination_path):
    """Download a file from a URL to the specified destination."""
    print(f"Downloading from: {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(destination_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    
    print(f"Download complete: {destination_path}")
    return destination_path


def extract_zip(zip_path, extract_to):
    """Extract a zip file to the specified directory."""
    print(f"Extracting zip file: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extraction complete: {extract_to}")
    return extract_to


def move_files_from_subfolder(parent_dir):
    """Move files from the first subfolder to the parent directory and delete the subfolder."""
    # Find the first subfolder in the parent directory
    subfolders = [f.path for f in os.scandir(parent_dir) if f.is_dir()]
    
    if not subfolders:
        print("No subfolders found in the extraction directory.")
        return
    
    subfolder = subfolders[0]
    print(f"Moving files from: {subfolder} to: {parent_dir}")
    
    # Move all files from subfolder to parent directory
    for item in os.listdir(subfolder):
        source = os.path.join(subfolder, item)
        destination = os.path.join(parent_dir, item)
        
        if os.path.isfile(source):
            shutil.move(source, destination)
    
    # Delete the now-empty subfolder
    shutil.rmtree(subfolder)
    print(f"Moved files and deleted subfolder: {subfolder}")


def extract_text_from_pdf(pdf_path, output_path):
    """Extract text from a PDF file using PyPDF2 and save it to a text file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            
            print(f"Text extracted: {output_path}")
            
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")


def process_pdfs(input_dir, output_dir):
    """Process all PDFs in input directory and save text to output directory."""
    create_directory(output_dir)
    
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_path = os.path.join(output_dir, os.path.splitext(pdf_file)[0] + '.txt')
        extract_text_from_pdf(pdf_path, output_path)


def download_and_extract(raw_data_dir):
    """Download and extract the zip file from the fixed URL."""
    url = "https://files.nitrd.gov/90-fr-9088/90-fr-9088-combined-responses.zip"
    zip_filename = url.split('/')[-1]
    zip_path = os.path.join(raw_data_dir, zip_filename)
    
    download_file(url, zip_path)
    extract_zip(zip_path, raw_data_dir)
    
    # Move files from subfolder to raw_data_dir and delete the subfolder
    move_files_from_subfolder(raw_data_dir)
    

class Analyzer:
    """LLM-based analyzer for extracting information from text files."""
    def __init__(self, model="gpt-4o-mini", api_base="https://api.openai.com/v1/chat/completions", api_key=None):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
    
    def get_system_prompt(self):
        return """You are an assistant that reads public comments submitted to the government.
For each text you receive, please extract the following fields:

- summary: A short paragraph summarizing the main points.
- submitter_type: The type of submitter, chosen ONLY from the following list:
  - Academia
  - Individual
  - Industry/professional/scientific association
  - Non-federal government
  - Non-profit
  - Private sector
  If unclear, guess based on context.
- agencies: A list of any U.S. federal agencies, departments, or offices mentioned in the submission. Write out each name fully in standardized format, followed by its abbreviation in parentheses if commonly known (e.g., "Food and Drug Administration (FDA)"). If an abbreviation is not commonly used, omit it. Only include agencies that are substantively mentioned outside the address block. If no agencies are mentioned, return an empty list.
- interesting_quotes: A list of up to 3 interesting direct quotes from the text. Add a newline character (\n) between each quote.
- sentiment_rating: On a 1-5 scale, rate the submission's sentiment towards AI adoption, where:
  - 1 = very worried
  - 2 = somewhat worried
  - 3 = neutral
  - 4 = somewhat enthusiastic
  - 5 = very enthusiastic
  - NA = not applicable (if the submission doesn't express a clear sentiment about AI adoption)
- sentiment_rationale: A brief explanation (1-2 sentences) supporting the sentiment rating assigned.
- main_topics: A list of topics discussed in the submission. Include ONLY those that apply from this list:
  - Application and Use in the Private Sector
  - Application and Use in the Public Sector
  - Biosecurity
  - Cybersecurity
  - Data Centers
  - Data Privacy and Security
  - Energy Consumption and Efficiency
  - Environmental Concerns
  - Export Controls
  - Ethical AI Frameworks and Bias Mitigation
  - Explainability and Assurance of AI Model Outputs
  - Hardware and Chips
  - Impact on Small Businesses
  - Innovation and Competition
  - Intellectual Property Issues
  - International Collaboration
  - Job Displacement
  - Model Development
  - National Security and Defense
  - Open Source Development
  - Procurement
  - Research and Development Funding Priorities
  - Specific Regulatory Approaches (e.g., sector-specific vs. broad)
  - Technical and Safety Standards
  - Workforce Development and Education
- additional_themes: List any important themes discussed that aren't covered by the main_topics list.
- keywords: Provide 5 keywords that best encapsulate the submission's main ideas.
- policy_suggestions: A list of actionable policy suggestions mentioned in the text (e.g., "Implement mandatory model audits" or "Fund AI literacy programs").

Return ONLY a JSON object matching this structure."""

    def analyze(self, text):
        if self.model == "gpt-4o-mini" or self.model=="gpt-4.1-mini":
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": text}
                ],
                "response_format": {"type": "json_object"}
            }

            response = requests.post(self.api_base, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            result = json.loads(data["choices"][0]["message"]["content"])
        elif self.model == "gemini-2.5-flash-preview-04-17":
            client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            response = client.models.generate_content(
                model='gemini-2.5-flash-preview-04-17',
                contents=self.get_system_prompt() + "\n" + text,
                config={
                    "response_mime_type": "application/json",
                },
            )
            result = response.text
        return result    

def analyze_texts(text_dir, processed_data_dir, top_n=None, batch_size=10, model="gpt-4o-mini"):
    """
    Analyze text files and save structured results incrementally with robust error handling.
    Creates a temp folder for intermediate results and aggregates at the end.
    
    Args:
        text_dir: Directory containing text files to analyze
        processed_data_dir: Directory to save processed results
        top_n: Optional limit on number of files to process
        batch_size: Number of files to process before saving interim results
        model: Model to use for analysis
    
    Returns:
        Path to the final aggregated results file
    """
    import os
    import json
    import time
    import shutil
    from datetime import datetime
    from tqdm import tqdm
    
    # Create main output directory if it doesn't exist
    create_directory(processed_data_dir)
    
    # Create a timestamped temp folder for incremental results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir_name = f"temp_analysis_{timestamp}"
    temp_dir = os.path.join(processed_data_dir, temp_dir_name)
    
    # Check if there's an existing temp directory to resume from
    existing_temp_dirs = [d for d in os.listdir(processed_data_dir) 
                          if os.path.isdir(os.path.join(processed_data_dir, d)) 
                          and d.startswith("temp_analysis_")]
    
    # Use the most recent temp directory if it exists
    if existing_temp_dirs:
        existing_temp_dirs.sort(reverse=True)  # Most recent first
        temp_dir = os.path.join(processed_data_dir, existing_temp_dirs[0])
        print(f"Found existing temp directory: {temp_dir}. Will resume processing from there.")
    else:
        # Create a new temp directory
        create_directory(temp_dir)
        print(f"Created temp directory for incremental results: {temp_dir}")
    
    # Initialize analyzer
    analyzer = Analyzer(model=model)
    
    # Get text files to process
    txt_files = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
    txt_files.sort()
    if top_n:
        txt_files = txt_files[:top_n]
    
    # Determine which files have already been processed
    processed_files = set()
    for filename in os.listdir(temp_dir):
        if filename.endswith('.json'):
            processed_files.add(filename.replace('.json', '.txt'))
    
    # Filter out already processed files
    files_to_process = [f for f in txt_files if f not in processed_files]
    print(f"Found {len(files_to_process)} files to process out of {len(txt_files)} total files.")
    
    # Process files in batches with progress tracking
    with tqdm(total=len(files_to_process), desc="Analyzing texts") as pbar:
        for i, txt_file in enumerate(files_to_process):
            file_path = os.path.join(text_dir, txt_file)
            
            # Skip if the file doesn't exist
            if not os.path.exists(file_path):
                print(f"Warning: File {file_path} does not exist. Skipping.")
                pbar.update(1)
                continue
            
            # Read the text file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                pbar.update(1)
                continue
            
            # Analyze the text
            max_retries = 3
            retry_delay = 10  # seconds
            
            for attempt in range(max_retries):
                try:
                    analysis = analyzer.analyze(text)
                    
                    # Save individual result to temp directory
                    result_filename = os.path.splitext(txt_file)[0] + '.json'
                    result_path = os.path.join(temp_dir, result_filename)
                    
                    with open(result_path, 'w', encoding='utf-8') as f:
                        json.dump(analysis, f, indent=2)
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Error analyzing {txt_file} (attempt {attempt+1}/{max_retries}): {e}")
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        print(f"Failed to analyze {txt_file} after {max_retries} attempts: {e}")
            
            pbar.update(1)
            
            # Optional: Add a small delay between API calls to avoid rate limits
            time.sleep(0.5)
    
    # Aggregate results from temp directory
    results = {}
    print("Aggregating results from individual files...")
    
    for filename in os.listdir(temp_dir):
        if filename.endswith('.json'):
            try:
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                
                # Use the original txt filename as the key
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                results[txt_filename] = result
                
            except Exception as e:
                print(f"Error reading result file {filename}: {e}")
    
    # Save aggregated results to timestamped file
    final_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(processed_data_dir, f"analyzed_{final_timestamp}.json")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Final results saved to {output_filename}")
    print(f"Processed {len(results)} files successfully.")
    
    try:
        shutil.rmtree(temp_dir)
        print(f"Removed temp directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Could not remove temp directory {temp_dir}: {e}")
    
    # === AUTO GENERATE HTML REPORT ===
    repo_dir = os.getcwd()
    html_output_dir = os.path.join(repo_dir, 'html_reports')
    create_directory(html_output_dir)
    make_html(output_filename, html_output_dir)
    
    return output_filename

def make_html(json_file_path, output_dir):
    """Generate HTML reports from analyzed data with pagination, filtering, and GitHub Pages support.
    
    Creates two files:
    1. A timestamped report with all data embedded
    2. An index.html file that loads data externally for GitHub Pages
    """
    import os
    import json
    import pandas as pd
    from datetime import datetime
    import jinja2
    
    def create_directory(directory_path):
        """Create a directory if it doesn't exist."""
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Created directory: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")
        return directory_path

    create_directory(output_dir)
    
    print(f"Reading analysis from: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Flatten records
    records = []
    for filename, analysis in data.items():
        record = {'filename': filename}
        record.update(analysis)
        records.append(record)

    df = pd.DataFrame(records)
    
    # Identify enumerated fields (fields where we want checkboxes instead of text search)
    enumerated_fields = [
        'submitter_type', 
        'sentiment_rating',
        'main_topics'
    ]
    
    # Get unique values for each enumerated field
    enumerated_values = {}
    for field in enumerated_fields:
        if field in df.columns:
            # For list fields like main_topics, flatten the lists to get all unique values
            if df[field].dtype == 'object' and isinstance(df[field].iloc[0], list):
                all_values = []
                for item_list in df[field].dropna():
                    if isinstance(item_list, list):
                        all_values.extend(item_list)
                    else:
                        all_values.append(item_list)
                unique_values = sorted(list(set(all_values)))
            else:
                unique_values = sorted(df[field].dropna().unique().tolist(), key=str)
            enumerated_values[field] = unique_values
    
    fields = df.columns.tolist()
    json_data = df.to_json(orient='records')
    submitter_counts = df['submitter_type'].value_counts().to_dict()
    
    # Define which columns to show initially
    initial_columns = ['filename', 'summary', 'submitter_type', 'main_topics']
    
    # First, create the timestamped HTML with embedded data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedded_file_path = os.path.join(output_dir, f"report_{timestamp}.html")
    
    # Save the data as a separate JSON file for GitHub Pages
    data_file_path = os.path.join(output_dir, "data.json")
    with open(data_file_path, 'w', encoding='utf-8') as f:
        f.write(json_data)
    print(f"JSON data file generated: {data_file_path}")
    
    # Template for both versions (embedded and index.html)
    template_str = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Public Comments Analysis</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js"></script>
        <style>
            .filter-panel {
                max-height: 300px;
                overflow-y: auto;
            }
            :root {
                --primary: #4F46E5;
                --primary-hover: #4338CA;
                --secondary: #F59E0B;
                --light: #F3F4F6;
                --dark: #1F2937;
                --success: #10B981;
                --danger: #EF4444;
            }
            .btn-primary {
                background-color: var(--primary);
                color: white;
                transition: all 0.2s;
            }
            .btn-primary:hover {
                background-color: var(--primary-hover);
            }
            .btn-secondary {
                background-color: var(--secondary);
                color: white;
                transition: all 0.2s;
            }
            .btn-secondary:hover {
                background-color: #D97706;
            }
            .card {
                border-radius: 0.5rem;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                transition: all 0.3s;
            }
            .card:hover {
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            }
            .table-header {
                background-color: var(--dark);
                color: white;
            }
            .table-row:nth-child(even) {
                background-color: var(--light);
            }
            .pagination-btn {
                background-color: var(--light);
                border: 1px solid #D1D5DB;
                padding: 0.5rem 1rem;
                transition: all 0.2s;
            }
            .pagination-btn:hover:not(:disabled) {
                background-color: #D1D5DB;
            }
            .pagination-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            .pagination-info {
                background-color: white;
                border: 1px solid #D1D5DB;
                padding: 0.5rem 1rem;
            }
            .dropdown-menu {
                position: absolute;
                background-color: white;
                border: 1px solid #D1D5DB;
                border-radius: 0.25rem;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 0.5rem;
                z-index: 10;
                max-height: 300px;
                overflow-y: auto;
                display: none;
            }
            .dropdown-menu.show {
                display: block;
            }
            .column-filter {
                padding: 4px;
                margin-top: 4px;
                width: 100%;
                font-size: 0.75rem;
                border: 1px solid #D1D5DB;
                border-radius: 0.25rem;
            }
            .filter-dropdown {
                position: absolute;
                background: white;
                border: 1px solid #D1D5DB;
                border-radius: 0.25rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
                z-index: 20;
                display: none;
                width: 250px;
                max-height: 300px;
                overflow-y: auto;
                padding: 0.5rem;
            }
            .filter-toggle {
                cursor: pointer;
                margin-left: 4px;
                display: inline-block;
            }
            .filter-toggle:hover {
                color: var(--primary);
            }
            .th-container {
                display: flex;
                align-items: center;
                justify-content: space-between;
            }
        </style>
    </head>
    <body class="bg-gray-100 p-6">
        <div class="container mx-auto">
            <h1 class="text-3xl font-bold mb-8 text-center text-indigo-700">Public Comments Analysis: AI Action Plan</h1>
            
            <!-- Total Submissions Info -->
            <div class="bg-white p-6 rounded-lg shadow mb-8 card">
                <div class="flex justify-between items-center">
                    <h2 class="text-xl font-semibold text-gray-800">Total Submissions</h2>
                    <p class="text-2xl font-bold text-indigo-600" id="totalSubmissions">...</p>
                </div>
                
                <!-- Main control buttons -->
                <div class="flex justify-end mt-4">
                    <button id="clearFilters" class="btn-secondary py-2 px-4 rounded mr-2">
                        Clear All Filters
                    </button>
                </div>
            </div>

            <!-- Data Table -->
            <div class="bg-white p-6 rounded-lg shadow card">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-xl font-semibold text-gray-800">Data Table</h2>
                    <div class="flex space-x-2">
                        <!-- Column selector dropdown -->
                        <div class="relative" id="columnSelectorContainer">
                            <button id="columnSelector" class="btn-primary py-2 px-4 rounded flex items-center">
                                <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16m-7 6h7"></path>
                                </svg>
                                Show/Hide Columns
                            </button>
                            <div id="columnMenu" class="dropdown-menu">
                                <!-- Column checkboxes will be added here by JavaScript -->
                            </div>
                        </div>
                        
                        <button id="downloadCsv" class="btn-primary py-2 px-4 rounded flex items-center">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path>
                            </svg>
                            Download Filtered CSV
                        </button>
                    </div>
                </div>
                
                <!-- Pagination controls - top -->
                <div class="flex justify-between items-center mb-4">
                    <div class="flex items-center">
                        <span class="text-gray-700 mr-2">Show</span>
                        <select id="rowsPerPage" class="p-2 border border-gray-300 rounded focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500">
                            <option value="10">10</option>
                            <option value="25">25</option>
                            <option value="50" selected>50</option>
                            <option value="100">100</option>
                        </select>
                        <span class="text-gray-700 ml-2">entries</span>
                    </div>
                    <div class="flex">
                        <button id="prevPage" class="pagination-btn rounded-l">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
                            </svg>
                        </button>
                        <span id="pageInfo" class="pagination-info">Page 1 of 1</span>
                        <button id="nextPage" class="pagination-btn rounded-r">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                
                <div class="overflow-x-auto">
                    <table id="dataTable" class="min-w-full bg-white border border-gray-200">
                        <thead id="tableHeader">
                            <!-- Column headers will be added by JavaScript -->
                        </thead>
                        <tbody>
                            <!-- Rows injected by JavaScript -->
                        </tbody>
                    </table>
                </div>
                
                <!-- Pagination controls - bottom -->
                <div class="flex justify-between items-center mt-6">
                    <div id="tableInfo" class="text-sm text-gray-600">Showing 0 to 0 of 0 entries</div>
                    <div class="flex">
                        <button id="prevPageBottom" class="pagination-btn rounded-l">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path>
                            </svg>
                        </button>
                        <span id="pageInfoBottom" class="pagination-info">Page 1 of 1</span>
                        <button id="nextPageBottom" class="pagination-btn rounded-r">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
            
            <footer class="mt-8 text-center text-gray-500 text-sm">
                <p>Generated on {{ timestamp }}</p>
            </footer>
        </div>

        <!-- Filter dropdown template - will be cloned for each column -->
        <div id="filterDropdownTemplate" class="filter-dropdown" style="display: none;">
            <div class="filter-panel">
                <!-- For enumerated fields (checkboxes) -->
                <div class="enumerated-options"></div>
                <!-- For text search -->
                <div class="text-search">
                    <input type="text" class="column-filter" placeholder="Search...">
                </div>
            </div>
        </div>

        <script>
            {% if is_embedded %}
            // Embedded version - data included in the page
            const analysisData = {{ json_data|safe }};
            {% else %}
            // GitHub Pages version - load data from external file
            let analysisData = [];
            fetch('data.json')
                .then(response => response.json())
                .then(data => {
                    analysisData = data;
                    initializeApp();
                })
                .catch(error => {
                    console.error('Error loading data:', error);
                    document.querySelector('#dataTable tbody').innerHTML = 
                        '<tr><td colspan="' + fields.length + '" class="py-4 text-center text-red-500">Error loading data. Please try again.</td></tr>';
                });
            {% endif %}
            
            const fields = {{ fields|tojson }};
            const enumeratedFields = {{ enumerated_fields|tojson }};
            const enumeratedValues = {{ enumerated_values|tojson }};
            const initialColumns = {{ initial_columns|tojson }};
            
            // Pagination state
            let currentPage = 1;
            let rowsPerPage = 50;
            let filteredData = [];
            let visibleColumns = [...initialColumns]; // Start with initial columns
            
            // Filter state
            let activeFilters = {};
            let activeFilterDropdown = null;
            
            {% if is_embedded %}
            // For embedded version, initialize immediately
            document.addEventListener('DOMContentLoaded', initializeApp);
            {% endif %}
            
            function initializeApp() {
                // Initialize filteredData with all data
                filteredData = [...analysisData];
                
                // Display stats
                document.getElementById('totalSubmissions').textContent = analysisData.length;
                
                // Set up column selector
                setupColumnSelector();
                
                // Apply initial filtering and display
                updateTableHeaders();
                applyFiltersAndUpdateTable();
                
                // Set up event listeners
                document.getElementById('clearFilters').addEventListener('click', clearAllFilters);
                document.getElementById('downloadCsv').addEventListener('click', downloadFilteredCsv);
                document.getElementById('rowsPerPage').addEventListener('change', function() {
                    rowsPerPage = parseInt(this.value);
                    currentPage = 1; // Reset to first page
                    applyFiltersAndUpdateTable();
                });
                
                // Column selector toggle
                document.getElementById('columnSelector').addEventListener('click', function(e) {
                    e.stopPropagation();
                    document.getElementById('columnMenu').classList.toggle('show');
                });
                
                // Close dropdown when clicking outside
                document.addEventListener('click', function(e) {
                    if (!document.getElementById('columnSelectorContainer').contains(e.target)) {
                        document.getElementById('columnMenu').classList.remove('show');
                    }
                    
                    // Close any active filter dropdown if clicking outside
                    if (activeFilterDropdown && !activeFilterDropdown.contains(e.target) && 
                        !e.target.classList.contains('filter-toggle')) {
                        activeFilterDropdown.style.display = 'none';
                        activeFilterDropdown = null;
                    }
                });
                
                // Pagination controls
                document.getElementById('prevPage').addEventListener('click', () => changePage(-1));
                document.getElementById('nextPage').addEventListener('click', () => changePage(1));
                document.getElementById('prevPageBottom').addEventListener('click', () => changePage(-1));
                document.getElementById('nextPageBottom').addEventListener('click', () => changePage(1));
            }
            
            function setupColumnSelector() {
                const columnMenu = document.getElementById('columnMenu');
                columnMenu.innerHTML = '';
                
                // "Select All" option
                const selectAllDiv = document.createElement('div');
                selectAllDiv.className = 'flex items-center mb-2 pb-2 border-b border-gray-200';
                
                const selectAllCheckbox = document.createElement('input');
                selectAllCheckbox.type = 'checkbox';
                selectAllCheckbox.id = 'select-all-columns';
                selectAllCheckbox.className = 'mr-2 h-4 w-4 text-indigo-600';
                selectAllCheckbox.checked = visibleColumns.length === fields.length;
                
                selectAllCheckbox.addEventListener('change', function() {
                    const isChecked = this.checked;
                    
                    // Update all checkboxes
                    document.querySelectorAll('.column-checkbox').forEach(checkbox => {
                        checkbox.checked = isChecked;
                    });
                    
                    // Update visibleColumns
                    visibleColumns = isChecked ? [...fields] : [];
                    
                    // Redraw the table
                    updateTableHeaders();
                    displayPagedData();
                });
                
                const selectAllLabel = document.createElement('label');
                selectAllLabel.htmlFor = 'select-all-columns';
                selectAllLabel.textContent = 'Select All';
                selectAllLabel.className = 'font-semibold text-sm text-gray-700';
                
                selectAllDiv.appendChild(selectAllCheckbox);
                selectAllDiv.appendChild(selectAllLabel);
                columnMenu.appendChild(selectAllDiv);
                
                // Add individual column options
                fields.forEach(field => {
                    const div = document.createElement('div');
                    div.className = 'flex items-center mb-2';
                    
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.id = `column-${field}`;
                    checkbox.className = 'column-checkbox mr-2 h-4 w-4 text-indigo-600';
                    checkbox.dataset.column = field;
                    checkbox.checked = visibleColumns.includes(field);
                    
                    checkbox.addEventListener('change', function() {
                        const column = this.dataset.column;
                        
                        if (this.checked && !visibleColumns.includes(column)) {
                            visibleColumns.push(column);
                        } else if (!this.checked && visibleColumns.includes(column)) {
                            visibleColumns = visibleColumns.filter(col => col !== column);
                        }
                        
                        // Update "Select All" checkbox
                        document.getElementById('select-all-columns').checked = 
                            visibleColumns.length === fields.length;
                        
                        // Redraw the table
                        updateTableHeaders();
                        displayPagedData();
                    });
                    
                    const label = document.createElement('label');
                    label.htmlFor = `column-${field}`;
                    label.textContent = field;
                    label.className = 'text-sm text-gray-700';
                    
                    div.appendChild(checkbox);
                    div.appendChild(label);
                    columnMenu.appendChild(div);
                });
            }
            
            function updateTableHeaders() {
                const tableHeader = document.getElementById('tableHeader');
                tableHeader.innerHTML = '';
                
                const headerRow = document.createElement('tr');
                
                visibleColumns.forEach(field => {
                    const th = document.createElement('th');
                    th.className = 'py-3 px-4 border-b text-left text-xs font-semibold uppercase tracking-wider table-header';
                    
                    const thContainer = document.createElement('div');
                    thContainer.className = 'th-container';
                    
                    const fieldTitle = document.createElement('span');
                    fieldTitle.textContent = field;
                    thContainer.appendChild(fieldTitle);
                    
                    // Add filter toggle
                    const filterToggle = document.createElement('span');
                    filterToggle.className = 'filter-toggle';
                    filterToggle.innerHTML = `
                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"></path>
                        </svg>
                    `;
                    
                    filterToggle.addEventListener('click', function(e) {
                        e.stopPropagation();
                        showFilterDropdown(field, e.target);
                    });
                    
                    thContainer.appendChild(filterToggle);
                    th.appendChild(thContainer);
                    headerRow.appendChild(th);
                });
                
                tableHeader.appendChild(headerRow);
            }
            
            function showFilterDropdown(field, element) {
                // Close any currently open dropdown
                if (activeFilterDropdown) {
                    activeFilterDropdown.style.display = 'none';
                }
                
                // Clone the dropdown template
                const dropdown = document.getElementById('filterDropdownTemplate').cloneNode(true);
                dropdown.id = `filterDropdown-${field}`;
                dropdown.dataset.field = field;
                
                // Position the dropdown
                const rect = element.getBoundingClientRect();
                dropdown.style.top = `${rect.bottom + window.scrollY}px`;
                dropdown.style.left = `${rect.left + window.scrollX - 200}px`; // Offset to align better
                
                // Set up content based on field type
                const isEnumerated = enumeratedFields.includes(field);
                const enumeratedOptionsContainer = dropdown.querySelector('.enumerated-options');
                const textSearchContainer = dropdown.querySelector('.text-search');
                
                if (isEnumerated) {
                    // Set up enumerated options (checkboxes)
                    enumeratedOptionsContainer.innerHTML = '';
                    const values = enumeratedValues[field] || [];
                    
                    values.forEach((value, index) => {
                        const div = document.createElement('div');
                        div.className = 'flex items-center mb-2';
                        
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.id = `${field}-value-${index}`;
                        checkbox.className = 'filter-checkbox mr-2 h-4 w-4 text-indigo-600';
                        checkbox.dataset.field = field;
                        checkbox.dataset.value = value;
                        
                        // Check if this filter is active
                        if (activeFilters[field] && 
                            activeFilters[field].type === 'enumerated' && 
                            activeFilters[field].values.includes(value)) {
                            checkbox.checked = true;
                        }
                        
                        checkbox.addEventListener('change', function() {
                            updateEnumeratedFilter(field, value, this.checked);
                        });
                        
                        const label = document.createElement('label');
                        label.htmlFor = `${field}-value-${index}`;
                        label.textContent = value;
                        label.className = 'text-sm text-gray-700';
                        
                        div.appendChild(checkbox);
                        div.appendChild(label);
                        enumeratedOptionsContainer.appendChild(div);
                    });
                    
                    textSearchContainer.style.display = 'none';
                } else {
                    // Set up text search
                    enumeratedOptionsContainer.style.display = 'none';
                    const input = textSearchContainer.querySelector('input');
                    input.dataset.field = field;
                    
                    // Set current value if there's an active filter
                    if (activeFilters[field] && activeFilters[field].type === 'text') {
                        input.value = activeFilters[field].value;
                    }
                    
                    input.addEventListener('input', function() {
                        updateTextFilter(field, this.value);
                    });
                }
                
                // Add to document and show
                document.body.appendChild(dropdown);
                dropdown.style.display = 'block';
                activeFilterDropdown = dropdown;
            }
            
            function updateEnumeratedFilter(field, value, isChecked) {
                // Initialize filter if needed
                if (!activeFilters[field]) {
                    activeFilters[field] = { type: 'enumerated', values: [] };
                }
                
                // Add or remove the value
                if (isChecked && !activeFilters[field].values.includes(value)) {
                    activeFilters[field].values.push(value);
                } else if (!isChecked && activeFilters[field].values.includes(value)) {
                    activeFilters[field].values = activeFilters[field].values.filter(v => v !== value);
                }
                
                // Remove filter if no values are selected
                if (activeFilters[field].values.length === 0) {
                    delete activeFilters[field];
                }
                
                // Apply filters
                applyFiltersAndUpdateTable();
            }
            
            function updateTextFilter(field, value) {
                if (value.trim() === '') {
                    // Remove filter if empty
                    if (activeFilters[field]) {
                        delete activeFilters[field];
                    }
                } else {
                    // Set or update filter
                    activeFilters[field] = { type: 'text', value: value.trim().toLowerCase() };
                }
                
                // Apply filters
                applyFiltersAndUpdateTable();
            }
            
            function clearAllFilters() {
                // Clear all active filters
                activeFilters = {};
                
                // Reset pagination
                currentPage = 1;
                
                // Reapply filters (which will now show all data)
                applyFiltersAndUpdateTable();
                
                // Close any active filter dropdown
                if (activeFilterDropdown) {
                    activeFilterDropdown.style.display = 'none';
                    activeFilterDropdown = null;
                }
            }
            
            function applyFiltersAndUpdateTable() {
                // Apply all filters
                filteredData = analysisData.filter(item => {
                    // Check all active filters
                    return Object.entries(activeFilters).every(([field, filter]) => {
                        const itemValue = item[field];
                        
                        if (itemValue === null || itemValue === undefined) {
                            return false;
                        }
                        
                        if (filter.type === 'text') {
                            // Text filter
                            return String(itemValue).toLowerCase().includes(filter.value);
                        } else if (filter.type === 'enumerated') {
                            // Enumerated filter (checkbox)
                            if (filter.values.length === 0) return true; // No filter selected
                            
                            // Handle array fields (like main_topics)
                            if (Array.isArray(itemValue)) {
                                return filter.values.some(value => itemValue.includes(value));
                            }
                            
                            // Handle scalar fields
                            return filter.values.some(value => String(itemValue) === String(value));
                        }
                        
                        return true;
                    });
                });
                
                // Update pagination info
                updatePaginationControls();
                
                // Display the filtered and paginated data
                displayPagedData();
            }
            
            function updatePaginationControls() {
                const totalPages = Math.max(1, Math.ceil(filteredData.length / rowsPerPage));
                
                // Ensure current page is valid
                if (currentPage > totalPages) {
                    currentPage = totalPages;
                }
                
                // Update page info displays
                document.getElementById('pageInfo').textContent = `Page ${currentPage} of ${totalPages}`;
                document.getElementById('pageInfoBottom').textContent = `Page ${currentPage} of ${totalPages}`;
                
                // Update table info
                const start = (currentPage - 1) * rowsPerPage + 1;
                const end = Math.min(start + rowsPerPage - 1, filteredData.length);
                document.getElementById('tableInfo').textContent = 
                    `Showing ${filteredData.length > 0 ? start : 0} to ${end} of ${filteredData.length} entries`;
                
                // Enable/disable prev/next buttons
                const prevButtons = [document.getElementById('prevPage'), document.getElementById('prevPageBottom')];
                const nextButtons = [document.getElementById('nextPage'), document.getElementById('nextPageBottom')];
                
                prevButtons.forEach(btn => {
                    btn.disabled = currentPage === 1;
                    btn.classList.toggle('opacity-50', currentPage === 1);
                });
                
                nextButtons.forEach(btn => {
                    btn.disabled = currentPage === totalPages;
                    btn.classList.toggle('opacity-50', currentPage === totalPages);
                });
            }
            
            function changePage(direction) {
                const totalPages = Math.ceil(filteredData.length / rowsPerPage);
                const newPage = currentPage + direction;
                
                if (newPage >= 1 && newPage <= totalPages) {
                    currentPage = newPage;
                    displayPagedData();
                    updatePaginationControls();
                    
                    // Scroll to top of table
                    document.getElementById('dataTable').scrollIntoView({ behavior: 'smooth' });
                }
            }
            
            function displayPagedData() {
                const tbody = document.querySelector('#dataTable tbody');
                tbody.innerHTML = '';
                
                if (filteredData.length === 0) {
                    const noDataRow = document.createElement('tr');
                    noDataRow.innerHTML = `<td colspan="${visibleColumns.length}" class="py-4 text-center">No matching records found</td>`;
                    tbody.appendChild(noDataRow);
                    return;
                }
                
                const start = (currentPage - 1) * rowsPerPage;
                const pagedData = filteredData.slice(start, start + rowsPerPage);
                
                pagedData.forEach((item, index) => {
                    const row = document.createElement('tr');
                    row.className = 'table-row hover:bg-gray-100 transition-colors';
                    
                    visibleColumns.forEach(field => {
                        const cell = document.createElement('td');
                        cell.className = 'py-3 px-4 border-b';
                        
                        let content = item[field];
                        
                        // Format array values
                        if (Array.isArray(content)) {
                            content = content.join(', ');
                        }
                        
                        // Handle null/undefined values
                        cell.innerHTML = (content !== undefined && content !== null) ? content : '';
                        
                        row.appendChild(cell);
                    });
                    
                    tbody.appendChild(row);
                });
            }
            
            function downloadFilteredCsv() {
                // Use only currently filtered data and visible columns
                const dataToExport = filteredData.map(item => {
                    const exportItem = {};
                    visibleColumns.forEach(field => {
                        let value = item[field];
                        // Convert arrays to comma-separated strings for CSV
                        if (Array.isArray(value)) {
                            value = value.join(', ');
                        }
                        exportItem[field] = value;
                    });
                    return exportItem;
                });
                
                const csv = Papa.unparse(dataToExport);
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                
                const link = document.createElement('a');
                link.setAttribute('href', url);
                link.setAttribute('download', `public_comments_analysis_${timestamp}.csv`);
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        </script>
    </body>
    </html>
    '''

    env = jinja2.Environment()
    env.filters['tojson'] = lambda obj: json.dumps(obj)
    
    # Create the embedded version (with data included in the HTML)
    embedded_template = env.from_string(template_str)
    embedded_html = embedded_template.render(
        json_data=json_data,
        submitter_counts=submitter_counts,
        fields=fields,
        enumerated_fields=enumerated_fields,
        enumerated_values=enumerated_values,
        initial_columns=initial_columns,
        is_embedded=True,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    with open(embedded_file_path, 'w', encoding='utf-8') as f:
        f.write(embedded_html)
    print(f"Timestamped HTML report generated: {embedded_file_path}")
    
    # Create the index.html version (loads data externally)
    index_html_path = os.path.join(output_dir, "index.html")
    index_template = env.from_string(template_str)
    index_html = index_template.render(
        json_data="[]",  # Empty array, will load data from data.json
        submitter_counts=submitter_counts,
        fields=fields,
        enumerated_fields=enumerated_fields,
        enumerated_values=enumerated_values,
        initial_columns=initial_columns,
        is_embedded=False,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    with open(index_html_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    print(f"Index.html generated for GitHub Pages: {index_html_path}")
    github_pages_dir = os.path.join(os.getcwd(), 'docs')
    create_directory(github_pages_dir)

    # Copy index.html and data.json to docs/
    shutil.move(index_html_path, os.path.join(github_pages_dir, 'index.html'))
    shutil.move(data_file_path, os.path.join(github_pages_dir, 'data.json'))

    print(f"Moved index.html and data.json to GitHub Pages folder: {github_pages_dir}")
    
    return embedded_file_path

def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Download, process, and analyze PDF files')
    parser.add_argument('--download_only', action='store_true', help='Only download and extract files')
    parser.add_argument('--process_only', action='store_true', help='Only process PDFs to extract text')
    parser.add_argument('--analyze', action='store_true', help='Analyze extracted text files')
    parser.add_argument('--top_n', type=int, default=None, help='Analyze only the top N .txt files')
    parser.add_argument('--make_html', action='store_true', help='Generate HTML report from analysis')
    parser.add_argument('--make_github_pages', action='store_true', help='Generate HTML report optimized for GitHub Pages')
    parser.add_argument('--json_file', type=str, help='Path to JSON file for HTML generation (if not using --analyze)')
    parser.add_argument('--model', type=str, default='gpt-4o-mini', choices=['gpt-4o-mini', 'gemini-2.5-flash-preview-04-17', 'gpt-4.1-mini'], help='Specify the AI model to use for analysis')
    args = parser.parse_args()
    
    repo_dir = os.getcwd()
    raw_data_dir = os.path.join(repo_dir, 'raw_data')
    text_dir = os.path.join(repo_dir, 'text')
    processed_data_dir = os.path.join(repo_dir, 'processed_data')
    html_output_dir = os.path.join(repo_dir, 'html_reports')
    github_pages_dir = os.path.join(repo_dir, 'docs')  # GitHub Pages uses the 'docs' folder by default
    
    create_directory(raw_data_dir)
    
    json_file_path = None
    
    if args.download_only:
        download_and_extract(raw_data_dir)

    if args.process_only:
        process_pdfs(raw_data_dir, text_dir)

    # If no flags like --download_only or --process_only or --analyze are given, do download+process by default
    if not args.download_only and not args.process_only and not args.analyze and not args.make_html and not args.make_github_pages:
        download_and_extract(raw_data_dir)
        process_pdfs(raw_data_dir, text_dir)

    if args.analyze:
        json_file_path = analyze_texts(text_dir, processed_data_dir, top_n=args.top_n, model=args.model)
    
    # Find the most recent JSON file if none specified
    if (args.make_html or args.make_github_pages) and not json_file_path:
        if args.json_file:
            json_file_path = args.json_file
        elif os.path.exists(processed_data_dir):
            json_files = [f for f in os.listdir(processed_data_dir) if f.lower().endswith('.json')]
            if json_files:
                json_files.sort(reverse=True)  # Sort by name descending (usually includes timestamp)
                json_file_path = os.path.join(processed_data_dir, json_files[0])
                print(f"Using most recent JSON file: {json_file_path}")
            else:
                print("No JSON files found in processed_data directory. Please run --analyze first or specify a JSON file with --json_file.")
                return
        else:
            print(f"Directory {processed_data_dir} does not exist. Please run --analyze first or specify a JSON file with --json_file.")
            return
        
        if not os.path.exists(json_file_path):
            print(f"JSON file not found: {json_file_path}")
            return
    
    if args.make_html:
        # Use original HTML generation for reports (embedded data)
        make_html(json_file_path, html_output_dir)
            
    if args.make_github_pages:
        # Use our new version for GitHub Pages
        create_directory(github_pages_dir)
        make_html(json_file_path, github_pages_dir)  # This will now create both timestamped and index.html files


if __name__ == "__main__":
    main()