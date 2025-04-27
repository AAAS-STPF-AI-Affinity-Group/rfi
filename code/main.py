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
- agencies: A list of any U.S. federal agencies, departments, or offices mentioned in the submission. Write out each name fully in standardized format, followed by its abbreviation in parentheses if commonly known (e.g., "Food and Drug Administration (FDA)"). If an abbreviation is not commonly used, omit it. If no agencies are mentioned, return an empty list.
- interesting_quotes: A list of up to 3 interesting direct quotes from the text.
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
    """Analyze text files and save structured results incrementally with progress bar. Automatically generates HTML."""
    create_directory(processed_data_dir)
    
    analyzer = Analyzer(model=model)
    txt_files = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
    txt_files.sort()
    if top_n:
        txt_files = txt_files[:top_n]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(processed_data_dir, f"analyzed_{timestamp}.json")

    if os.path.exists(output_filename):
        with open(output_filename, 'r', encoding='utf-8') as f:
            results = json.load(f)
        print(f"Resuming from existing file with {len(results)} items.")
    else:
        results = {}

    with tqdm(total=len(txt_files), desc="Analyzing texts") as pbar:
        for idx, txt_file in enumerate(txt_files):
            if txt_file in results:
                pbar.update(1)
                continue

            file_path = os.path.join(text_dir, txt_file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            try:
                analysis = analyzer.analyze(text)
                results[txt_file] = analysis
            except Exception as e:
                print(f"Failed to analyze {txt_file}: {e}")

            if (idx + 1) % batch_size == 0 or idx == len(txt_files) - 1:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2)
                print(f"Saved progress after {idx+1} files.")

            pbar.update(1)

    print(f"Analysis complete. Final results saved to {output_filename}")

    # === AUTO GENERATE HTML REPORT ===
    repo_dir = os.getcwd()
    html_output_dir = os.path.join(repo_dir, 'html_reports')
    create_directory(html_output_dir)
    make_html(output_filename, html_output_dir)
    # === === ===

    return output_filename

def make_html(json_file_path, output_dir):
    """Generate HTML reports from analyzed data with pagination, better filtering, and GitHub Pages support.
    
    Creates two files:
    1. A timestamped report with all data embedded
    2. An index.html file that loads data externally for GitHub Pages
    """

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
                unique_values = sorted(df[field].dropna().unique().tolist())
            enumerated_values[field] = unique_values
    
    fields = df.columns.tolist()
    json_data = df.to_json(orient='records')
    submitter_counts = df['submitter_type'].value_counts().to_dict()
    
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
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/PapaParse/5.3.0/papaparse.min.js"></script>
        <style>
            .filter-panel {
                max-height: 200px;
                overflow-y: auto;
            }
        </style>
    </head>
    <body class="bg-gray-100 p-6">
        <div class="container mx-auto">
            <h1 class="text-3xl font-bold mb-6 text-center">Public Comments Analysis</h1>
            
            <!-- Stats Overview -->
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Total Submissions</h2>
                    <p class="text-4xl font-bold text-blue-600" id="totalSubmissions">...</p>
                </div>
                <div class="bg-white p-6 rounded-lg shadow md:col-span-2">
                    <h2 class="text-xl font-semibold mb-4">Submitter Types</h2>
                    <canvas id="submitterChart"></canvas>
                </div>
            </div>

            <!-- Filters -->
            <div class="bg-white p-6 rounded-lg shadow mb-8">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-semibold">Filters</h2>
                    <button id="clearFilters" class="bg-gray-300 hover:bg-gray-400 text-gray-800 font-bold py-2 px-4 rounded">
                        Clear All Filters
                    </button>
                </div>
                
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <!-- Enumerated filters - generated dynamically -->
                    {% for field, values in enumerated_values.items() %}
                    <div class="filter-group">
                        <h3 class="font-semibold mb-2">{{ field }}</h3>
                        <div class="filter-panel bg-gray-50 p-3 rounded">
                            {% for value in values %}
                            <div class="flex items-center mb-1">
                                <input type="checkbox" id="{{ field }}_{{ loop.index }}" class="filter-checkbox mr-2" 
                                       data-field="{{ field }}" data-value="{{ value }}">
                                <label for="{{ field }}_{{ loop.index }}" class="text-sm">{{ value }}</label>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endfor %}
                    
                    <!-- Text search fields - for non-enumerated fields -->
                    {% for field in fields %}
                        {% if field not in enumerated_values.keys() %}
                        <div class="filter-group">
                            <h3 class="font-semibold mb-2">{{ field }}</h3>
                            <input type="text" data-field="{{ field }}" class="columnSearch p-2 border rounded w-full" placeholder="Search...">
                        </div>
                        {% endif %}
                    {% endfor %}
                </div>
            </div>

            <!-- Data Table -->
            <div class="bg-white p-6 rounded-lg shadow">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-semibold">Data Table</h2>
                    <div class="flex space-x-2">
                        <button id="downloadCsv" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                            Download Filtered CSV
                        </button>
                    </div>
                </div>
                
                <!-- Pagination controls - top -->
                <div class="flex justify-between items-center mb-4">
                    <div>
                        <span>Show</span>
                        <select id="rowsPerPage" class="mx-2 p-1 border rounded">
                            <option value="10">10</option>
                            <option value="25">25</option>
                            <option value="50" selected>50</option>
                            <option value="100">100</option>
                        </select>
                        <span>entries</span>
                    </div>
                    <div class="pagination-controls">
                        <button id="prevPage" class="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-l disabled:opacity-50">Prev</button>
                        <span id="pageInfo" class="bg-gray-100 px-4 py-1">Page 1 of 1</span>
                        <button id="nextPage" class="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-r disabled:opacity-50">Next</button>
                    </div>
                </div>
                
                <div class="overflow-x-auto">
                    <table id="dataTable" class="min-w-full bg-white">
                        <thead>
                            <tr>
                                {% for field in fields %}
                                <th class="py-2 px-4 border-b bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                                    {{ field }}
                                </th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Rows injected by JavaScript -->
                        </tbody>
                    </table>
                </div>
                
                <!-- Pagination controls - bottom -->
                <div class="flex justify-between items-center mt-4">
                    <div id="tableInfo" class="text-sm text-gray-600">Showing 0 to 0 of 0 entries</div>
                    <div class="pagination-controls">
                        <button id="prevPageBottom" class="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-l disabled:opacity-50">Prev</button>
                        <span id="pageInfoBottom" class="bg-gray-100 px-4 py-1">Page 1 of 1</span>
                        <button id="nextPageBottom" class="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded-r disabled:opacity-50">Next</button>
                    </div>
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
            
            const submitterCounts = {{ submitter_counts|tojson }};
            const fields = {{ fields|tojson }};
            const enumeratedFields = {{ enumerated_fields|tojson }};
            
            // Pagination state
            let currentPage = 1;
            let rowsPerPage = 50;
            let filteredData = [];
            
            {% if is_embedded %}
            // For embedded version, initialize immediately
            document.addEventListener('DOMContentLoaded', initializeApp);
            {% endif %}
            
            function initializeApp() {
                // Initialize filteredData with all data
                filteredData = [...analysisData];
                
                // Display stats
                document.getElementById('totalSubmissions').textContent = analysisData.length;
                
                // Apply initial filtering and display
                applyFiltersAndUpdateTable();
                
                // Set up event listeners
                document.querySelectorAll('.columnSearch').forEach(input => {
                    input.addEventListener('input', applyFiltersAndUpdateTable);
                });
                
                document.querySelectorAll('.filter-checkbox').forEach(checkbox => {
                    checkbox.addEventListener('change', applyFiltersAndUpdateTable);
                });
                
                document.getElementById('clearFilters').addEventListener('click', clearAllFilters);
                document.getElementById('downloadCsv').addEventListener('click', downloadFilteredCsv);
                document.getElementById('rowsPerPage').addEventListener('change', function() {
                    rowsPerPage = parseInt(this.value);
                    currentPage = 1; // Reset to first page
                    applyFiltersAndUpdateTable();
                });
                
                // Pagination controls
                document.getElementById('prevPage').addEventListener('click', () => changePage(-1));
                document.getElementById('nextPage').addEventListener('click', () => changePage(1));
                document.getElementById('prevPageBottom').addEventListener('click', () => changePage(-1));
                document.getElementById('nextPageBottom').addEventListener('click', () => changePage(1));
                
                // Submitter Type Bar Chart
                const submitterCtx = document.getElementById('submitterChart').getContext('2d');
                new Chart(submitterCtx, {
                    type: 'bar',
                    data: {
                        labels: Object.keys(submitterCounts),
                        datasets: [{
                            label: 'Number of Submissions',
                            data: Object.values(submitterCounts),
                            backgroundColor: '#4299E1'
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { display: false } },
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: { precision: 0 }
                            }
                        }
                    }
                });
            }
            
            function clearAllFilters() {
                // Clear all text search inputs
                document.querySelectorAll('.columnSearch').forEach(input => {
                    input.value = '';
                });
                
                // Uncheck all checkboxes
                document.querySelectorAll('.filter-checkbox').forEach(checkbox => {
                    checkbox.checked = false;
                });
                
                // Reset pagination
                currentPage = 1;
                
                // Reapply filters (which will now show all data)
                applyFiltersAndUpdateTable();
            }
            
            function applyFiltersAndUpdateTable() {
                // Get text search filters
                const textFilters = {};
                document.querySelectorAll('.columnSearch').forEach(input => {
                    const field = input.dataset.field;
                    const value = input.value.trim().toLowerCase();
                    if (value) textFilters[field] = value;
                });
                
                // Get checkbox filters grouped by field
                const checkboxFilters = {};
                document.querySelectorAll('.filter-checkbox:checked').forEach(checkbox => {
                    const field = checkbox.dataset.field;
                    const value = checkbox.dataset.value;
                    
                    if (!checkboxFilters[field]) {
                        checkboxFilters[field] = [];
                    }
                    checkboxFilters[field].push(value);
                });
                
                // Apply all filters
                filteredData = analysisData.filter(item => {
                    // Check text filters
                    const textFilterPassed = Object.entries(textFilters).every(([field, searchValue]) => {
                        const itemValue = item[field];
                        if (itemValue === null || itemValue === undefined) return false;
                        return String(itemValue).toLowerCase().includes(searchValue);
                    });
                    
                    if (!textFilterPassed) return false;
                    
                    // Check checkbox filters
                    return Object.entries(checkboxFilters).every(([field, values]) => {
                        if (values.length === 0) return true; // No filter selected
                        
                        const itemValue = item[field];
                        
                        // Handle array fields (like main_topics)
                        if (Array.isArray(itemValue)) {
                            return values.some(value => itemValue.includes(value));
                        }
                        
                        // Handle scalar fields
                        return values.includes(String(itemValue));
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
                    noDataRow.innerHTML = `<td colspan="${fields.length}" class="py-4 text-center">No matching records found</td>`;
                    tbody.appendChild(noDataRow);
                    return;
                }
                
                const start = (currentPage - 1) * rowsPerPage;
                const pagedData = filteredData.slice(start, start + rowsPerPage);
                
                pagedData.forEach(item => {
                    const row = document.createElement('tr');
                    row.className = 'hover:bg-gray-50';
                    
                    fields.forEach(field => {
                        const cell = document.createElement('td');
                        cell.className = 'py-2 px-4 border-b';
                        
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
                // Use only currently filtered data
                const csv = Papa.unparse(filteredData);
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
        is_embedded=True
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
        is_embedded=False
    )
    
    with open(index_html_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    print(f"Index.html generated for GitHub Pages: {index_html_path}")
    
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