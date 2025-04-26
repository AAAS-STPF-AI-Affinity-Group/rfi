import os
import requests
import zipfile
import argparse
import shutil
from pathlib import Path
import PyPDF2
from dotenv import load_dotenv
import datetime
import json
import jinja2
import pandas as pd

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

- themes: A list of main themes discussed.
- keywords: A list of important keywords.
- summary: A short paragraph summarizing the main points.
- submitter_type: The type of submitter (e.g., individual, company, advocacy group, etc.). If unclear, guess based on context.
- interesting_quotes: A list of up to 3 interesting direct quotes from the text.

Return ONLY a JSON object matching this structure."""

    def analyze(self, text):
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

        response = requests.post(self.api_base, headers=headers, json=payload, verify=False)
        response.raise_for_status()
        data = response.json()
        return json.loads(data["choices"][0]["message"]["content"])

def analyze_texts(text_dir, processed_data_dir, top_n=None):
    """Analyze text files and save structured results."""
    create_directory(processed_data_dir)
    
    analyzer = Analyzer()
    txt_files = [f for f in os.listdir(text_dir) if f.lower().endswith('.txt')]
    txt_files.sort()  # Optional: sort alphabetically
    if top_n:
        txt_files = txt_files[:top_n]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = os.path.join(processed_data_dir, f"analyzed_{timestamp}.json")
    
    results = {}
    for txt_file in txt_files:
        file_path = os.path.join(text_dir, txt_file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"Analyzing: {txt_file}")
        try:
            analysis = analyzer.analyze(text)
            results[txt_file] = analysis
        except Exception as e:
            print(f"Failed to analyze {txt_file}: {e}")

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"Analysis complete. Results saved to {output_filename}")
    return output_filename

def make_html(json_file_path, output_dir):
    """Generate HTML report from analyzed data."""
    create_directory(output_dir)
    
    print(f"Reading analysis from: {json_file_path}")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to pandas DataFrame for easier manipulation
    records = []
    for filename, analysis in data.items():
        record = {
            'filename': filename,
            'submitter_type': analysis.get('submitter_type', 'Unknown'),
            'summary': analysis.get('summary', ''),
            'themes': ', '.join(analysis.get('themes', [])),
            'keywords': ', '.join(analysis.get('keywords', [])),
            'interesting_quotes': '<br>'.join(analysis.get('interesting_quotes', []))
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    # Calculate summary statistics
    submitter_counts = df['submitter_type'].value_counts().to_dict()
    
    # We're not using theme counts anymore, so we can remove this section
    
    # Convert DataFrame to JSON for JavaScript
    json_data = df.to_json(orient='records')
    
    # Create HTML with Jinja2 template
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
            .hidden {
                display: none;
            }
        </style>
    </head>
    <body class="bg-gray-100 p-6">
        <div class="container mx-auto">
            <h1 class="text-3xl font-bold mb-6 text-center">Public Comments Analysis Dashboard</h1>
            
            <!-- Summary statistics -->
            <div class="mb-8">
                <div class="bg-white p-6 rounded-lg shadow">
                    <h2 class="text-xl font-semibold mb-4">Submitter Types</h2>
                    <canvas id="submitterChart"></canvas>
                </div>
            </div>
            
            <!-- Filters -->
            <div class="bg-white p-6 rounded-lg shadow mb-8">
                <h2 class="text-xl font-semibold mb-4">Filters</h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Submitter Type</label>
                        <select id="submitterFilter" class="mt-1 block w-full p-2 border border-gray-300 rounded-md">
                            <option value="">All</option>
                        </select>
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Theme</label>
                        <input type="text" id="themeFilter" placeholder="Filter by theme" class="mt-1 block w-full p-2 border border-gray-300 rounded-md">
                    </div>
                    <div>
                        <label class="block text-sm font-medium text-gray-700">Keyword</label>
                        <input type="text" id="keywordFilter" placeholder="Filter by keyword" class="mt-1 block w-full p-2 border border-gray-300 rounded-md">
                    </div>
                </div>
            </div>
            
            <!-- Data Table -->
            <div class="bg-white p-6 rounded-lg shadow">
                <div class="flex justify-between items-center mb-4">
                    <h2 class="text-xl font-semibold">Data Table</h2>
                    <button id="downloadCsv" class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
                        Download as CSV
                    </button>
                </div>
                <div class="overflow-x-auto">
                    <table id="dataTable" class="min-w-full bg-white">
                        <thead>
                            <tr>
                                <th class="py-2 px-4 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Filename</th>
                                <th class="py-2 px-4 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Submitter Type</th>
                                <th class="py-2 px-4 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Summary</th>
                                <th class="py-2 px-4 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Themes</th>
                                <th class="py-2 px-4 border-b border-gray-200 bg-gray-100 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">Keywords</th>
                            </tr>
                        </thead>
                        <tbody>
                            <!-- Data rows will be inserted here by JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        
        <script>
            // Data from backend
            const analysisData = {{ json_data|safe }};
            const submitterCounts = {{ submitter_counts|tojson }};
            
            // Initialize charts
            document.addEventListener('DOMContentLoaded', function() {
                // Submitter Chart
                const submitterCtx = document.getElementById('submitterChart').getContext('2d');
                const submitterChart = new Chart(submitterCtx, {
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
                        plugins: {
                            legend: {
                                display: false
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    precision: 0
                                }
                            }
                        }
                    }
                });
                
                // Populate submitter filter
                const submitterFilter = document.getElementById('submitterFilter');
                const submitterTypes = [...new Set(analysisData.map(item => item.submitter_type))];
                submitterTypes.forEach(type => {
                    const option = document.createElement('option');
                    option.value = type;
                    option.textContent = type;
                    submitterFilter.appendChild(option);
                });
                
                // Populate data table
                populateTable(analysisData);
                
                // Event listeners for filters
                document.getElementById('submitterFilter').addEventListener('change', filterData);
                document.getElementById('themeFilter').addEventListener('input', filterData);
                document.getElementById('keywordFilter').addEventListener('input', filterData);
                
                // Download CSV
                document.getElementById('downloadCsv').addEventListener('click', downloadCsv);
            });
            
            function populateTable(data) {
                const tbody = document.querySelector('#dataTable tbody');
                tbody.innerHTML = '';
                
                data.forEach(item => {
                    const row = document.createElement('tr');
                    row.className = 'hover:bg-gray-50';
                    
                    row.innerHTML = `
                        <td class="py-2 px-4 border-b border-gray-200">${item.filename}</td>
                        <td class="py-2 px-4 border-b border-gray-200">${item.submitter_type}</td>
                        <td class="py-2 px-4 border-b border-gray-200">${item.summary}</td>
                        <td class="py-2 px-4 border-b border-gray-200">${item.themes}</td>
                        <td class="py-2 px-4 border-b border-gray-200">${item.keywords}</td>
                    `;
                    
                    tbody.appendChild(row);
                });
            }
            
            function filterData() {
                const submitterType = document.getElementById('submitterFilter').value;
                const themeText = document.getElementById('themeFilter').value.toLowerCase();
                const keywordText = document.getElementById('keywordFilter').value.toLowerCase();
                
                const filteredData = analysisData.filter(item => {
                    const matchesSubmitter = !submitterType || item.submitter_type === submitterType;
                    const matchesTheme = !themeText || item.themes.toLowerCase().includes(themeText);
                    const matchesKeyword = !keywordText || item.keywords.toLowerCase().includes(keywordText);
                    
                    return matchesSubmitter && matchesTheme && matchesKeyword;
                });
                
                populateTable(filteredData);
            }
            
            function downloadCsv() {
                // Get visible data
                const visibleRows = Array.from(document.querySelectorAll('#dataTable tbody tr')).filter(row => !row.classList.contains('hidden'));
                const visibleData = visibleRows.map(row => {
                    const cells = Array.from(row.cells);
                    return {
                        filename: cells[0].textContent,
                        submitter_type: cells[1].textContent,
                        summary: cells[2].textContent,
                        themes: cells[3].textContent,
                        keywords: cells[4].textContent
                    };
                });
                
                // Convert to CSV
                const csv = Papa.unparse(visibleData);
                
                // Create download link
                const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.setAttribute('href', url);
                link.setAttribute('download', 'public_comments_analysis.csv');
                link.style.visibility = 'hidden';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }
        </script>
    </body>
    </html>
    '''
    
    # Create Jinja2 environment and template
    env = jinja2.Environment()
    # Add the tojson filter
    env.filters['tojson'] = lambda obj: json.dumps(obj)
    template = env.from_string(template_str)
    
    # Render HTML
    html_content = template.render(
        json_data=json_data,
        submitter_counts=submitter_counts
    )
    
    # Save HTML to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file_path = os.path.join(output_dir, f"report_{timestamp}.html")
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {html_file_path}")
    return html_file_path


def main():
    """Main function to parse arguments and run the script."""
    parser = argparse.ArgumentParser(description='Download, process, and analyze PDF files')
    parser.add_argument('--download_only', action='store_true', help='Only download and extract files')
    parser.add_argument('--process_only', action='store_true', help='Only process PDFs to extract text')
    parser.add_argument('--analyze', action='store_true', help='Analyze extracted text files')
    parser.add_argument('--top_n', type=int, default=None, help='Analyze only the top N .txt files')
    parser.add_argument('--make_html', action='store_true', help='Generate HTML report from analysis')
    parser.add_argument('--json_file', type=str, help='Path to JSON file for HTML generation (if not using --analyze)')
    args = parser.parse_args()
    
    repo_dir = os.getcwd()
    raw_data_dir = os.path.join(repo_dir, 'raw_data')
    text_dir = os.path.join(repo_dir, 'text')
    processed_data_dir = os.path.join(repo_dir, 'processed_data')
    html_output_dir = os.path.join(repo_dir, 'html_reports')
    
    create_directory(raw_data_dir)
    
    json_file_path = None
    
    if args.download_only:
        download_and_extract(raw_data_dir)

    if args.process_only:
        process_pdfs(raw_data_dir, text_dir)

    # If no flags like --download_only or --process_only or --analyze are given, do download+process by default
    if not args.download_only and not args.process_only and not args.analyze and not args.make_html:
        download_and_extract(raw_data_dir)
        process_pdfs(raw_data_dir, text_dir)

    if args.analyze:
        json_file_path = analyze_texts(text_dir, processed_data_dir, top_n=args.top_n)
        
    if args.make_html:
        # If json_file is specified, use that, otherwise use the one from analyze_texts
        if args.json_file:
            json_file_path = args.json_file
        elif json_file_path is None:
            # Find the most recent JSON file if none specified
            if os.path.exists(processed_data_dir):
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
            
        make_html(json_file_path, html_output_dir)


if __name__ == "__main__":
    main()