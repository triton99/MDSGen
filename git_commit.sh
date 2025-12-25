#!/usr/bin/env bash

set -euo pipefail

MSG_PREFIX="Add file"

# Ensure we are inside a git repo
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not inside a git repository"
  exit 1
fi

# Get all untracked + modified files recursively (null-delimited, safe)
git status --porcelain -z | while IFS= read -r -d '' status; do
  # status format: XY <path>
  file="${status:3}"

  # Skip if directory (git normally does not list dirs, but just in case)
  if [ -d "$file" ]; then
    continue
  fi

  echo "📌 Committing: $file"

  git add "$file"
  git commit -m "$MSG_PREFIX: $file"
done

echo "🎉 All files in all subfolders committed one-by-one."
