import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class MinimalAnalyzer:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.api_base = "https://api.openai.com/v1/chat/completions"
        self.submitter_types = ["individual", "company", "nonprofit", "government", "academic", "other"]
    
    def get_system_prompt(self):
        submitter_types_str = ", ".join(self.submitter_types)
        return f"""You are an assistant that reads public comments submitted to the government.
For each text you receive, please extract ONLY the submitter_type field.

- submitter_type: MUST be one of the following types: {submitter_types_str}.
  - "company" means a for-profit business entity
  - "individual" means a person writing in their personal capacity
  - "nonprofit" means a non-governmental organization with charitable/advocacy purpose
  - "government" means a government agency or official writing in official capacity
  - "academic" means a university, research institution, or individual identifying as a researcher/professor
  - "other" should only be used when the submitter clearly doesn't fit the other categories

Return ONLY a JSON object with a single field: {{"submitter_type": "chosen_type"}}"""

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

        response = requests.post(self.api_base, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        result = json.loads(data["choices"][0]["message"]["content"])
        
        # Validate submitter_type
        if result["submitter_type"] not in self.submitter_types:
            result["submitter_type"] = "other"
            
        return result

def test_sample_texts():
    sample_texts = [
        # Company examples
        """As CEO of TechInnovate Solutions, I am writing to express our company's strong support for the 
        proposed AI safety regulations. Our firm has been developing AI systems for over a decade.
        
        Best regards,
        Jane Smith
        CEO, TechInnovate Solutions""",
        
        """Microsoft Corporation appreciates the opportunity to comment on the proposed regulations.
        As one of the world's leading technology companies, we believe...""",
        
        """On behalf of Small Business Innovations LLC, I would like to express concerns about the 
        potential impact of these regulations on startups and small enterprises...""",
        
        # Individual examples
        """I am writing as a concerned citizen regarding the proposed regulations. 
        As someone who uses AI tools daily, I believe...""",
        
        """My name is John Doe and I'm a software engineer with 15 years of experience.
        While I'm employed by BigTech Inc., these views are entirely my own...""",
        
        # Nonprofit examples
        """The Center for Responsible Technology, a 501(c)(3) nonprofit organization, 
        submits the following comments on the proposed regulations...""",
        
        # Government example
        """The Department of Energy's Office of Scientific Research hereby submits these comments
        in response to the call for public input regarding AI regulations...""",
        
        # Academic example
        """As researchers at the University of California AI Ethics Lab, we submit the following
        comments based on our recent studies of algorithmic bias...""",

        # Ambiguous example
        """Thank you for the opportunity to provide feedback on these important regulations.
        I believe that artificial intelligence must be developed responsibly..."""
    ]
    
    analyzer = MinimalAnalyzer()
    results = []
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\nTesting Sample #{i}:")
        print(f"---Text snippet: {text[:50]}...")
        
        result = analyzer.analyze(text)
        results.append(result)
        print(f"---Classification: {result['submitter_type']}")
    
    # Summary
    print("\n\nSUMMARY OF RESULTS:")
    for i, result in enumerate(results, 1):
        print(f"Sample #{i}: {result['submitter_type']}")
        
    # Count by type
    type_counts = {}
    for result in results:
        submitter_type = result["submitter_type"]
        type_counts[submitter_type] = type_counts.get(submitter_type, 0) + 1
        
    print("\nCOUNTS BY TYPE:")
    for submitter_type, count in type_counts.items():
        print(f"{submitter_type}: {count}")

if __name__ == "__main__":
    test_sample_texts()