# Efficient LLMs - Layer Merging

This repository contains tools for efficient layer merging in Large Language Models.

## Setup

### 1. Environment Variables

To use this tool, you need to set up your Hugging Face access token as an environment variable:

```bash
# Set your Hugging Face token
export HF_TOKEN="your_hugging_face_token_here"

# Or create a .env file (make sure it's in .gitignore)
echo "HF_TOKEN=your_hugging_face_token_here" > .env
```

**⚠️ IMPORTANT: Never commit your actual token to version control!**

### 2. Get Your Hugging Face Token

1. Go to [Hugging Face Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Create a new token with appropriate permissions
3. Copy the token and set it as an environment variable

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
python utils/layer_merging.py --help
```

## Security Notes

- API tokens and access keys are never stored in the code
- Use environment variables for sensitive information
- The `.gitignore` file prevents accidental commits of sensitive files
- If you accidentally commit a token, immediately revoke it and clean your git history

## Contributing

When contributing to this repository:

1. Never hardcode API keys or tokens
2. Use environment variables for configuration
3. Follow the existing code style
4. Test your changes thoroughly