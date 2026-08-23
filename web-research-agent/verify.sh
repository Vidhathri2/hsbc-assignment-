#!/bin/bash
set -e

# Checks output correctness: valid JSON, citations point to real sources, no duplicate report_ids, confidence scores within range.

RESULTS_DIR="results"
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Results directory not found!"
    exit 1
fi

FILES=($RESULTS_DIR/*.json)
if [ ${#FILES[@]} -eq 0 ] || [ "${FILES[0]}" == "$RESULTS_DIR/*.json" ]; then
    echo "No JSON files found in results/"
    exit 1
fi

echo "Verifying ${#FILES[@]} reports..."

declare -A report_ids
valid=true

for file in "${FILES[@]}"; do
    # 1. Valid JSON schema
    if ! jq empty "$file" > /dev/null 2>&1; then
        echo "FAIL: Invalid JSON in $file"
        valid=false
        continue
    fi

    # 2. No duplicate report_ids
    report_id=$(jq -r '.report_id' "$file")
    if [ -n "${report_ids[$report_id]}" ]; then
        echo "FAIL: Duplicate report_id $report_id found in $file"
        valid=false
    fi
    report_ids[$report_id]=1

    # 3. Confidence scores within range
    confidence=$(jq -r '.critique.confidence_score' "$file")
    if (( $(echo "$confidence < 0.0" | bc -l) )) || (( $(echo "$confidence > 1.0" | bc -l) )); then
        echo "FAIL: Confidence score $confidence out of range (0.0-1.0) in $file"
        valid=false
    fi

    # 4. Citations point to real sources
    source_ids=$(jq -r '.sources[].source_id' "$file" | sort | uniq)
    citations=$(jq -r '.sections[].citations[]' "$file" | sort | uniq)
    
    for citation in $citations; do
        if ! echo "$source_ids" | grep -q "^$citation$"; then
            echo "FAIL: Citation $citation in $file does not point to a valid source_id"
            valid=false
        fi
    done
done

if [ "$valid" = true ]; then
    echo "SUCCESS: All reports passed verification."
    exit 0
else
    echo "FAILURE: Some verification checks failed."
    exit 1
fi
