#!/bin/bash

# Function to extract MMLU and MON scores from JSON
extract_scores() {
  local json_file="outputs/gemma-3-12b-it.json"
  
  if [ -f "$json_file" ]; then
    # Skip jq check and use it directly - if it fails, we'll handle it with a custom solution
    
    # Try using jq first
    if jq '.' "$json_file" > /dev/null 2>&1; then
      # jq works, use it
      echo "MMLU Scores:"
      jq -r '.[] | select(.mmlu != null) | "  " + (.name // "Unnamed") + ": " + (.mmlu | tostring)' "$json_file" 2>/dev/null
      
      echo -e "\nMON Scores:"
      jq -r '.[] | select(.match != null or .oppose != null or .neither != null) | 
        "  " + (.name // "Unnamed") + ": Match=" + (.match | tostring) + 
        ", Oppose=" + (.oppose | tostring) + 
        ", Neither=" + (.neither | tostring)' "$json_file" 2>/dev/null
    else
      # Fallback to grep and sed
      echo "Using fallback method (jq not working):"
      echo "MMLU Scores:"
      grep -o '"mmlu":[0-9.]*' "$json_file" | sed 's/"mmlu":/  Score: /g'
      
      echo -e "\nMON Scores:"
      echo "  Match scores:"
      grep -o '"match":[0-9.]*' "$json_file" | sed 's/"match":/  Score: /g'
      echo "  Oppose scores:"
      grep -o '"oppose":[0-9.]*' "$json_file" | sed 's/"oppose":/  Score: /g'
      echo "  Neither scores:"
      grep -o '"neither":[0-9.]*' "$json_file" | sed 's/"neither":/  Score: /g'
    fi
  else
    echo "Error: $json_file not found"
  fi
}

# Main loop
while true; do
  clear
  echo "======================================================="
  echo "Last updated: $(date)"
  echo "======================================================="
  
  # Display contents of progress.txt
  echo -e "\nContents of outputs/progress.txt:"
  echo "-------------------------------------------------------"
  if [ -f "outputs/progress.txt" ]; then
    cat "outputs/progress.txt"
  else
    echo "Error: outputs/progress.txt not found"
  fi
  
  # Extract and display scores
  echo -e "\n-------------------------------------------------------"
  extract_scores
  echo -e "\n-------------------------------------------------------"
  echo "Press Ctrl+C to exit"
  
  # Wait 10 seconds before updating again
  sleep 10
done

