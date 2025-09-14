#!/bin/bash

echo "========================================"
echo "Push Airbnb Monitor to GitHub"
echo "========================================"

# Check if gh is authenticated
if ! gh auth status >/dev/null 2>&1; then
    echo "Please authenticate with GitHub first:"
    echo "Running: gh auth login"
    gh auth login
fi

echo ""
echo "Creating GitHub repository..."

# Create the repository
repo_name="airbnb-monitor"
echo "Repository name: $repo_name"

# Create public repository with description
gh repo create $repo_name \
    --public \
    --description "Comprehensive monitoring system for Airbnb/apartment complexes with automatic door detection, person tracking, and real-time notifications" \
    --source=. \
    --remote=origin \
    --push

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! Repository created and code pushed."
    echo ""
    echo "Your repository is now available at:"
    gh repo view --web
else
    echo ""
    echo "If the repository already exists, you can push manually:"
    echo "1. Add remote: git remote add origin https://github.com/YOUR_USERNAME/$repo_name.git"
    echo "2. Push code: git push -u origin main"
fi

echo ""
echo "========================================"
echo "Next steps for your Jetson Nano:"
echo "========================================"
echo ""
echo "On your Jetson Nano, run:"
echo ""
echo "git clone https://github.com/$(gh api user -q .login)/$repo_name.git"
echo "cd $repo_name"
echo "./setup.sh"
echo ""