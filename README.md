# 🚀 Multi-account LLM Proxy Mixture

> Multi-account proxy for Gemini, Qwen, iFlow and OpenRouter with intelligent response merging for quality or speed

[![Join our Discord](https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/Jum4V2Zh3H)&nbsp;[![Join our Reddit](https://img.shields.io/badge/Reddit-r/XorVibeLife-ff4500?style=for-the-badge&logo=reddit&logoColor=white)](https://reddit.com/r/XorVibeLife)

## Why This Exists

Stop paying \$200/month for LLM APIs. This proxy gives you:

- ✅ **\$0 cost** - Uses free tiers of Gemini, Qwen, iFlow, and OpenRouter
- 🧠 **Better quality** - Mixture-of-Agents combines 3-5 responses → provides superior answers
- ⚡ **3x faster** - Race mode returns fastest response
- 🔄 **Unlimited** - Multi-account rotation bypasses rate limits
- 🔌 **OpenAI-compatible** - Works with Cursor, Continue.dev, LibreChat, etc.
- 🐳 **Self-hosted** - Full privacy and control

## How It Works

### Quality Mode

```
Your Question → [Gemini + Qwen + iFlow + OpenRouter] → Synthesize → Better Answer
```

Asks 3-5 providers simultaneously, combines best insights → **Superior quality**

### Speed Mode

```
Your Question → [Gemini + Qwen + iFlow + OpenRouter] → First Response → Faster Answer
```

Returns fastest response → **3x speed boost**

### Unlimited Mode

```
Account 1 (rate limited) → Auto-switch → Account 2 → Account 3 → ...
```

Rotates through 10+ free accounts → **never hit limits**

## Setup Guide

### Prerequisites

- Docker Desktop ([download](https://www.docker.com/products/docker-desktop))
- Google account (for Gemini)
- Qwen account ([signup](https://chat.qwen.ai/))
- iFlow account ([signup](https://iflow.cn/))
- OpenRouter free keys ([get keys](https://openrouter.ai/keys))

### Step 1: Get Gemini Credentials

```bash
# Install Gemini CLI
https://github.com/google-gemini/gemini-cli

# Login with Google account
gemini

# Find oauth_creds.json
# Windows: %USERPROFILE%\.gemini\
# Linux/Mac: ~/.gemini/

# Copy to project
cp ~/.gemini/oauth_creds.json gemini_oauth_creds_files/oauth_creds_01.json

# Get Project ID from https://aistudio.google.com/
# Add to config.yaml: "oauth_creds_01.json": "gen-lang-client-XXXXXXXXXX"
```

### Step 2: Get Qwen Credentials

```bash
# install and auth https://github.com/QwenLM/qwen-code
# find oauth_creds.json and copy to qwen_oauth_creds_files/oauth_creds_01.json
```

### Step 3: Get iFlow Credentials

iFlow uses a similar authentication system to Qwen. You can obtain credentials by logging in through their official CLI.

```bash
# Install and authenticate with the iFlow CLI
# Follow instructions from: https://github.com/iflow-ai/iflow-cli

# Find the generated oauth_creds.json file
# Linux/Mac: ~/.iflow/oauth_creds.json
# Windows: %USERPROFILE%\.iflow\oauth_creds.json

# Copy it to the iflow proxy directory
cp ~/.iflow/oauth_creds.json iflow_oauth_creds_files/oauth_creds_01.json
```

### Step 4: Get OpenRouter Keys

```bash
# Visit https://openrouter.ai/keys
# Generate free API keys (no credit card needed)
# Add to config.yaml under openrouter.keys
```

### Step 5: Configure

Edit `config.yaml` in each directory.
Add your credentials to the respective folders:
    `gemini/gemini_oauth_creds_files/`
    `qwen/qwen_oauth_creds_files/`
    `iflow/iflow_oauth_creds_files/`
    `openrouter/config.yaml`

### Step 6: Run

```bash
# Start all services
docker-compose down && docker-compose up --build --force-recreate

# Check health
curl http://localhost:8007/health

# View logs
docker-compose logs -f
```

## Usage

### Basic Request

```bash
curl http://localhost:8007/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "temperature": 0.7
  }'
```

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                 MOA Aggregator (Port 8007)                │
│           Combines responses from all sources             │
└─────────────┬──────────┬──────────┬──────────┬─────────────┘
              │          │          │          │
       ┌──────▼─────┐ ┌──▼─────┐ ┌──▼─────┐ ┌──▼────────┐
       │   Gemini   │ │  Qwen  │ │  iFlow │ │ OpenRouter│
       │  (8004)    │ │ (8005) │ │ (8008) │ │  (8006)   │
       │ 3 accts    │ │ 5 accts│ │ 5 accts│ │  5 keys   │
       └────────────┘ └────────┘ └────────┘ └───────────┘
```

## Cost Comparison

| Solution | Monthly Cost | Rate Limits | Setup Time |
| :-- | :-- | :-- | :-- |
| OpenAI API | \$20-200 | 10k RPM | 5 min |
| Anthropic | \$25-150 | 5k RPM | 5 min |
| **Free LLM Proxy** | **\$0** | **Depends on the number of accounts** | **30 min** |

## Monitoring

```bash
# Service health
curl http://localhost:8004/health  # Gemini
curl http://localhost:8005/health  # Qwen
curl http://localhost:8008/health  # iFlow
curl http://localhost:8006/health  # OpenRouter
curl http://localhost:8007/health  # MOA

# Account status
cat gemini_account_states.json
cat qwen_account_states.json
cat iflow_account_states.json

# Live logs
docker-compose logs -f mixtureofllm
```

## Troubleshooting

**Problem: "No accounts available"**

- Add more OAuth credential files for the specific service (Gemini, Qwen, iFlow).
- Check the account state JSON files (`*_account_states.json`).
- Wait for the cooldown period to expire.

**Problem: "All accounts rate limited"**

- Add more accounts (you may need 10+ for continuous use).
- Increase `initial_backoff_seconds` in the `config.yaml` for the affected service.

**Problem: "Token expired" or "Invalid apiKey"**

- Tokens should auto-refresh automatically.
- If refreshing fails consistently, your `refresh_token` may have expired. Re-generate the `oauth_creds.json` file by logging in again with the respective CLI (e.g., `gemini auth`, `iflow auth`).

**Problem: Slow responses**

- Reduce `num_candidates` to 3 in the aggregator's `config.yaml`.
- Use "Flash" or smaller models instead of "Pro" models.
- Increase `soft_timeout_seconds` in the aggregator's `config.yaml`.

## Contributing

PRs welcome! Areas needing help:

- [ ] More free provider with big context window integrations

## License

MIT - Personal and educational use. Respect provider ToS.

## Links

- [GitHub](https://github.com/xor0110xor-prog/free-llm-proxy-mixture)
- [Discord](https://discord.gg/Jum4V2Zh3H) - Xor Vibe—life community
- [Reddit](https://reddit.com/r/XorVibeLife)

***

**Built with vibe coding ⚡ by [xor0110xor](https://github.com/xor0110xor-prog)**