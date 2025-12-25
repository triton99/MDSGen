#!/usr/bin/env bash

# Exit on error
set -e

# Folder to commit (change if needed)
TARGET_DIR="./masked_diffusion"

# Optional: commit message prefix
MSG_PREFIX="Add file"

# Make sure we're inside a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not inside a git repository."
  exit 1
fi

# Get list of untracked + modified files
FILES=$(git status --porcelain | awk '{print $2}')

if [ -z "$FILES" ]; then
  echo "✅ No files to commit."
  exit 0
fi

for file in $FILES; do
  # Skip directories
  if [ -d "$file" ]; then
    continue
  fi

  echo "📌 Committing: $file"

  git add "$file"
  git commit -m "$MSG_PREFIX: $file"
done

echo "🎉 All files committed one-by-one."
