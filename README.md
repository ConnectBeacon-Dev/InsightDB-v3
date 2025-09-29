## Version 3 of InsightDB
# 1) Create & activate venv
python -m venv .venv\llcpp .\.venv\llcpp\Scripts\activate

# 2) Install deps
python -m pip install --upgrade pip wheel
pip install chromadb sentence-transformers llama-cpp-python

# 3) put the integrated_company_search.json file in same directory (Home directory)

# 4) Download the models "qwen2.5-3b-instruct-q8_0.gguf" in below ways:
     from website : https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF and select 8-bit
     From PowerShell:
     # a) (once) install the CLI
          python -m pip install -U "huggingface_hub[cli]"

     # b) download the exact GGUF from the Qwen org repo
          huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF `
            --include "qwen2.5-3b-instruct-q8_0.gguf" `
            --local-dir ".\models"

# 5) Execute the commands
     python json_rag_win.py index --json .\integrated_company_search.json --db .\db
     python json_rag_win.py query --db .\db --model .\models\qwen2.5-3b-instruct-q8_0.gguf --ask "contact details of MADHYA BHARAT AGRO PRODUCTS LIMITED"

# 6) Execute the shell script for running few queries
     ./run_queries.ps1
