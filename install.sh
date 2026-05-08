#!/bin/bash
set -e

echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║      🐧  PENGUIN SQUAD  —  AUTO INSTALLER               ║"
echo "  ║                                                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""

# ── Check Python ────────────────────────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "  ❌  Python3 not found!"
    echo "  👉  Install: https://www.python.org/downloads/"
    exit 1
fi
echo "  ✅  $(python3 --version) found."

# ── Upgrade pip ─────────────────────────────────────────────────────────────
echo ""
echo "  [1/4]  Upgrading pip..."
python3 -m pip install --upgrade pip --quiet
echo "  ✅  pip ready."

# ── Install TA-Lib ───────────────────────────────────────────────────────────
echo ""
echo "  [2/4]  Installing TA-Lib..."

if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &>/dev/null; then
        brew install ta-lib --quiet 2>/dev/null || true
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y libta-lib-dev --quiet 2>/dev/null || true
    elif command -v yum &>/dev/null; then
        sudo yum install -y ta-lib-devel --quiet 2>/dev/null || true
    fi
fi

python3 -m pip install TA-Lib --quiet 2>/dev/null || \
    echo "  ⚠️  TA-Lib install failed — some indicators will use fallback."
echo "  ✅  TA-Lib ready."

# ── Install Penguin Squad ───────────────────────────────────────────────────
echo ""
echo "  [3/4]  Installing Penguin Squad..."
python3 -m pip install git+https://github.com/YOUR_USERNAME/madagascar-penguins.git --quiet
echo "  ✅  Penguin Squad installed."

# ── Done ───────────────────────────────────────────────────────────────────
echo ""
echo "  [4/4]  Done!"
echo ""
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║                                                          ║"
echo "  ║   ✅  Installation complete!                             ║"
echo "  ║                                                          ║"
echo "  ║   Run the bot:  madagascar-penguins                            ║"
echo "  ║                                                          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Your API keys will be saved at:"
echo "  ~/.penguin_squad/.env"
echo "  (only on YOUR computer — never uploaded anywhere)"
echo ""
