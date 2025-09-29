# Paths
$DB_PATH = ".\db"
$SCRIPT = "json_rag_win.py"

# List of models
$models = @(
    ".\models\qwen2.5-3b-instruct-q8_0.gguf"
    ".\models\Qwen2.5-7B-Instruct-Q4_K_M.gguf"
	"D:\InsightDB-v2-main\models\Qwen2.5-14B-Instruct-Q4_K_M.gguf"
)
# Single log file
$logFile = ".\output_all.log"

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
# Loop through models and queries

# Start fresh
"==== Starting Run $(Get-Date) ====" | Out-File -FilePath $logFile

# Loop through models and queries
foreach ($model in $models) {
    "##############################################" | Out-File -FilePath $logFile -Append
    "Running queries with model: $model" | Out-File -FilePath $logFile -Append
    "##############################################" | Out-File -FilePath $logFile -Append

    foreach ($q in $queries) {
        "==============================================" | Out-File -FilePath $logFile -Append
        (Get-Date) | Out-File -FilePath $logFile -Append
        "Running query: $q" | Out-File -FilePath $logFile -Append

        # Run query and append both stdout and stderr to log file
        python $SCRIPT query --db $DB_PATH --model $model --ask "$q" *>> $logFile

        (Get-Date) | Out-File -FilePath $logFile -Append
        "==============================================" | Out-File -FilePath $logFile -Append
        "" | Out-File -FilePath $logFile -Append
    }
}

"==== Run Completed $(Get-Date) ====" | Out-File -FilePath $logFile -Append

