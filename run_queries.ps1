# Paths
$DB_PATH = ".\db"
$SCRIPT = "json_rag_win.py"

# model Path
$MODEL_PATH = ".\models\qwen2.5-3b-instruct-q8_0.gguf"


# List of queries
$queries = @(
  "contact details of MADHYA BHARAT AGRO PRODUCTS LIMITED"
  "location of the HEG LIMITED"
  "list of all companies in madhya pradesh"
  "company based on Bhopal"
  "list of defence startup"
  "list of MSME in india"
  "List of defence MSME in kerala"
  "List of company in goa"
  "show me the companies whose product are of consumable type"
  "Address of FLONEX OIL TECHNOLOGIES PRIVATE LIMITED"
  "MSME Companies"
  "How many products supplied to HAL in Gujarat"
  "List companies having ISO 9001 certificate"
)

# Loop through queries
foreach ($q in $queries) {
    Write-Output "=============================================="
    Get-Date
    Write-Output "Running query: $q"
    python $SCRIPT query --db $DB_PATH --model $MODEL_PATH --ask "$q"
    Get-Date
    Write-Output "=============================================="
    Write-Output ""
}
