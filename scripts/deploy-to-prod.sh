#!/bin/bash
# scripts/deploy-to-prod.sh

set -e  # stop immediately if anything fails

echo "🔄 Checking out main and pulling latest..."
git checkout main
git pull origin main

echo "🔄 Checking out prod and pulling latest..."
git checkout prod
git pull origin prod

echo "🔀 Merging main into prod..."
git merge main

echo "🚀 Pushing prod to origin..."
git push origin prod

echo "✅ Deployment to prod branch complete!"

