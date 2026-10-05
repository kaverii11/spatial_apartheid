#!/bin/bash
FILE="ieee_manuscript.tex"
LAST_MODIFIED=""

echo "Starting LaTeX watcher for $FILE..."

while true; do
  if [ -f "$FILE" ]; then
    CURRENT_MODIFIED=$(stat -f "%m" "$FILE" 2>/dev/null || stat -c "%Y" "$FILE" 2>/dev/null)
    if [ "$CURRENT_MODIFIED" != "$LAST_MODIFIED" ]; then
      if [ -n "$LAST_MODIFIED" ]; then
        echo "Detected change in $FILE, compiling..."
      fi
      pdflatex -interaction=nonstopmode -halt-on-error "$FILE" < /dev/null > /dev/null
      LAST_MODIFIED=$CURRENT_MODIFIED
    fi
  fi
  sleep 2
done
