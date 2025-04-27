# AI Action Plan Comments Processor

Tool for processing and analyzing the comments received in response to the "Request for Information on the Development of an Artificial Intelligence (AI) Action Plan."

This script downloads, extracts text from, analyzes, and generates an interactive HTML report from public comments submitted to the government regarding the AI Action Plan.

---

## Quick Start

1. Clone this repository
2. Create a `.env` file with your OpenAI API key:
   ```bash
   OPENAI_API_KEY=your_key_here
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run from the repository root:
   ```bash
   python code/main.py
   ```

---

## Usage Options

### Download, Process, and Analyze Everything
```bash
python code/main.py
```

### Just Analyze Existing Text Files
```bash
python code/main.py --analyze
```

### Generate HTML Report
```bash
python code/main.py --make_html
```

### Other Options
```bash
python code/main.py --download_only       # Only download and extract PDFs
python code/main.py --process_only        # Only extract text from PDFs
python code/main.py --analyze --top_n 5    # Analyze only the first 5 text files and make the html
python code/main.py --make_html            # Just make the html
```

---

## Deployment to GitHub Pages ("Prod")

The generated `index.html` and `data.json` files are automatically places in the the `docs/` folder, which serves as the source for [GitHub Pages](https://aaas-stpf-ai-affinity-group.github.io/rfi/).

**Important:**
- GitHub Pages is configured to build from the `prod` branch.
- `main` is used for ongoing development.
- When ready to update the live site, run:

```bash
./scripts/deploy-to-prod.sh
```

This script:
- Updates `prod` by merging `main` into it.
- Pushes the updated `prod` branch to GitHub.
- Triggers a new GitHub Pages build.

**Note:** Only run the deploy script after verifying changes are ready for production.

---

## Outputs
- `raw_data/`: downloaded PDFs
- `text/`: extracted text from PDFs
- `processed_data/`: structured JSON analysis results
- `html_reports/`: timestamped HTML reports
- `docs/`: `index.html` and `data.json` for GitHub Pages

---

## Testing and Validation

We are validating the processing pipeline using the following known public comments:

- **AI-RFI-2025-2555**: Glenn Parham (individual)
- **OpenAI-RFI-2025**: OpenAI (corporate)
- **ASCAP-AI-RFI-2025-1**: ASCAP (non-profit)
- **DS4Everyone-AI-RFI-2025**: Data Science 4 Everyone (academic group)

Each comment is evaluated for:
- Sentiment toward AI adoption (1-5 scale)
- Topical tags (from a defined list)
- Additional emergent themes
- Summarized key arguments
- Keywords
- Actionable policy suggestions
- Agencies mentioned
- Interesting quotes

---

## Notes
- All interactions with OpenAI use the `response_format={"type": "json_object"}` for structured outputs.
- Other models (e.g., Gemini) can be swapped in with minor changes.
- This repo is actively evolving: workflows and structure may be improved over time.