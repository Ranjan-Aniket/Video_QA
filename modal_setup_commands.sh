#!/bin/bash
# Modal Setup Commands - Activates venv and creates secrets

# Activate virtual environment
source .venv/bin/activate

echo "Creating OpenAI secret..."
modal secret create openai-secret OPENAI_API_KEY=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

echo "Creating Anthropic secret..."
modal secret create anthropic-secret ANTHROPIC_API_KEY=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

echo "Listing secrets..."
modal secret list

echo ""
echo "✅ Done! Now run: modal deploy modal_pipeline.py"
