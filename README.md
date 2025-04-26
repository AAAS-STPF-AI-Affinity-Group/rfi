# AI Action Plan Comments Processor

Tool for processing and analyzing the comments received in response to the "Request for Information on the Development of an Artificial Intelligence (AI) Action Plan."

This script downloads, extracts text from, and analyzes the public comments submitted to the government regarding the AI Action Plan.

## Quick Start

1. Clone this repository
2. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_key_here
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. **Important**: Run the script from the root directory:
   ```
   python code/main.py
   ```
   This ensures all subfolders are created in the correct locations.

## Usage Options

### Download & Process Everything
```
python code/main.py
```
Downloads the zip file, extracts PDFs, and converts them to text.

### Analyze Existing Text Files
```
python code/main.py --analyze
```
Runs OpenAI analysis on text files (requires `.env` with API key).

### Generate HTML Report
```
python code/main.py --make_html
```
Generates an interactive HTML dashboard from analysis results.

### Other Options
```
python code/main.py --download_only    # Just download and extract PDFs
python code/main.py --process_only     # Just convert PDFs to text
python code/main.py --analyze --top_n 5 # Analyze only first 5 files
python code/main.py --make_html --json_file path/to/analysis.json # Generate HTML from specific JSON file
```

## Notes
- The `raw_data` folder is not included in the repo due to file size
- Analysis results are saved as JSON in the `processed_data` folder
- HTML reports are saved in the `html_reports` folder
- This script currently uses OpenAI's API with the `response_format={"type": "json_object"}` parameter to ensure proper JSON formatting
- You can swap other OpenAI models but they need to support the JSON response format option
- It's possible to adapt this for other providers (Anthropic, etc.) with some modifications to the `Analyzer` class

## Work in Progress
- **enum_example.py**: Script demonstrating how to constrain LLM responses to a predefined set of values (e.g., submitter types). This approach has the model return only allowed values from an enumeration.

## Test Cases
To begin with, we are using the following test cases:

- AI-RFI-2025-2555
  - This is a public comment submitted by Glenn Parham in his personal capacity.
  - It focuses on government efficiency with sections on Talent & Workforce, Acquisition & Adoption, Authorization & Compliance, Infrastructure, Data Ownership & IP, and Security Research.
  - It is a bit tricky in that the final section is given as "security" in the summary, but it is actually labeled "R&D" in the text as it is security research. 
- OpenAI-RFI-2025
  - This is a public comment submitted by OpenAI in their capacity as a company.
  - It focuses on federal preemption of state AI regulations, export control strategy, building infrastructure, weakening copyright protections, and government adoption of AI.
  - It is a bit tricky in that the final section is given as "security" in the summary, but it is actually labeled "R&D" in the text as it is security research. 
- ASCAP-AI-RFI-2025-1
  - This is a public comment submitted by the American Society of Composers, Authors and Publishers
  - It deals almost exclusively with copyright and has an opposing viewpoint to OpenAI
- DS4Everyone-AI-RFI-2025
  - Data Science 4 Everyone is an academic group focused on Data Science education
  - Their submission specifically calls on NSF to create a national Digital Frontier Teaching Corps and Teacher College Innovation Grants

## Test Questions
For each of the test cases, we will ask the following questions:

1. On a 1-5 scale, where 1 is "very worried" and 5 is "very enthusiastic", how would you rate the submission's sentiment towards AI adoption?
2. Tag the repository with whether it deals with the following:
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
  - International Collaboration and Standards
  - Job Displacement
  - Model Development
  - National Security and Defense
  - Open Source Development
  - Procurement
  - Research and Development Funding Priorities
  - Specific Regulatory Approaches (e.g., sector-specific vs. broad)
  - Workforce Development and Education
3. If there are any topics not given in #2, identify them and add them to the list
4. For each topic cluster, summarize key arguments, proposals, and concerns raised within that theme across all relevant responses.
5. Provide 5 keywords that best encapsulate the submission's main ideas.
6. Identify and extract actionable policy suggestions mentioned in the text. E.g. "Implement mandatory model audits" or "Fund AI literacy programs"