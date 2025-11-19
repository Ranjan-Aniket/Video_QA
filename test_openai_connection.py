"""
Quick test to verify OpenAI API key is working
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 80)
print("TESTING OPENAI API CONNECTION")
print("=" * 80)

# Check if API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("❌ ERROR: OPENAI_API_KEY not found in environment variables")
    exit(1)

print(f"✅ API key loaded: {api_key[:20]}...{api_key[-4:]}")

# Test API connection
try:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    print("\n🔄 Testing API connection with a simple completion...")

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello! API is working!' in one sentence."}
        ],
        max_tokens=50
    )

    result = response.choices[0].message.content

    print(f"✅ API Response: {result}")
    print("\n" + "=" * 80)
    print("SUCCESS: OpenAI API is configured correctly!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\n" + "=" * 80)
    print("FAILED: Could not connect to OpenAI API")
    print("=" * 80)
    exit(1)
