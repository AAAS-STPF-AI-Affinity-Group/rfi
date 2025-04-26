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

### Other Options
```
python code/main.py --download_only    # Just download and extract PDFs
python code/main.py --process_only     # Just convert PDFs to text
python code/main.py --analyze --top_n 5 # Analyze only first 5 files
```

## Notes
- The `raw_data` folder is not included in the repo due to file size
- Analysis results are saved as JSON in the `processed_data` folder
- This script currently uses OpenAI's API with the `response_format={"type": "json_object"}` parameter to ensure proper JSON formatting
- You can swap other OpenAI models but they need to support the JSON response format option
- It's possible to adapt this for other providers (Anthropic, etc.) with some modifications to the `Analyzer` class

## Work in Progress
- **enum_example.py**: Script demonstrating how to constrain LLM responses to a predefined set of values (e.g., submitter types). This approach has the model return only allowed values from an enumeration.