# ============================================================
#  Penguin Squad -- Setup
#  .\setup.ps1
# ============================================================

$Host.UI.RawUI.WindowTitle = "Penguin Squad"

function C($t, $c = "White") { Write-Host $t -ForegroundColor $c }
function Blank { Write-Host "" }
function Line  { C ("=" * 50) "DarkGray" }

# ---- penguin art helpers ------------------------------------
function Show-All($sk="White", $kw="White", $rc="White", $pr="White") {
    C "       .---.     .---.     .---.     .---. " $sk
    C "      ( o o )   ( - - )   ( o o )   ( ^ ^ )" $kw
    C "       \ = /     \ w /     \ ~ /     \ _ / " $rc
    C "      SKIPPER   KOWALSKI    RICO    PRIVATE " $pr
}

function Show-One($who) {
    switch ($who) {
        "skipper"  {
            Blank
            C "                  .---." "Cyan"
            C "                 ( o o )" "Cyan"
            C "                  \ = /" "Cyan"
            C "                 SKIPPER" "Cyan"
            Blank
        }
        "kowalski" {
            Blank
            C "                  .---." "Magenta"
            C "                 ( - - )" "Magenta"
            C "                  \ w /" "Magenta"
            C "                 KOWALSKI" "Magenta"
            Blank
        }
        "rico"     {
            Blank
            C "                  .---." "Green"
            C "                 ( o o )" "Green"
            C "                  \ ~ /" "Green"
            C "                   RICO" "Green"
            Blank
        }
        "private"  {
            Blank
            C "                  .---." "Yellow"
            C "                 ( ^ ^ )" "Yellow"
            C "                  \ _ /" "Yellow"
            C "                 PRIVATE" "Yellow"
            Blank
        }
    }
}

# ---- ask helpers --------------------------------------------
function Ask($prompt, $default = "") {
    $hint = if ($default) { " [Enter = $default]" } else { "" }
    Write-Host "  $prompt$hint : " -NoNewline -ForegroundColor Gray
    $v = Read-Host
    if ($v -eq "" -and $default) { return $default }
    return $v
}

function AskSecret($prompt) {
    Write-Host "  $prompt : " -NoNewline -ForegroundColor Gray
    $ss = Read-Host -AsSecureString
    return [Runtime.InteropServices.Marshal]::PtrToStringAuto(
               [Runtime.InteropServices.Marshal]::SecureStringToBSTR($ss))
}

function AskKey($prompt) {
    Write-Host "  API key : " -NoNewline -ForegroundColor DarkGray
    $v = Read-Host
    return $v
}


# ============================================================
Clear-Host
Blank
Show-All "Cyan" "Magenta" "Green" "Yellow"
Blank
C "            PENGUIN SQUAD" "White"
Line
Blank

# ---- 1. Python ---------------------------------------------
try {
    $pv = python --version 2>&1
    C "  Python: $pv" "DarkGray"
} catch {
    C "  ERROR: Python not found. Install from python.org" "Red"
    pause; exit 1
}
Blank

# ---- 2. Install packages -----------------------------------
C "  Installing packages..." "DarkGray"
Blank

$pkgs = @(
    "MetaTrader5>=5.0.45", "pandas>=2.0.0", "numpy>=1.24.0",
    "scikit-learn>=1.3.0", "xgboost>=2.0.0", "python-dotenv>=1.0.0",
    "pytz>=2024.1", "requests>=2.31.0", "langgraph>=0.2.0",
    "langchain-core>=0.3.0", "anthropic>=0.40.0",
    "openai>=1.50.0", "google-generativeai>=0.8.0"
)

$total = $pkgs.Count; $i = 0
foreach ($p in $pkgs) {
    $i++
    $bar = ("#" * $i) + ("." * ($total - $i))
    Write-Host "`r  [$bar]" -NoNewline -ForegroundColor DarkCyan
    pip install $p --quiet 2>&1 | Out-Null
}
Blank
C "  Done." "Green"
Blank
Line

# ---- 3. MT5 ------------------------------------------------
Blank
C "  MetaTrader 5" "White"
Blank
$mt5Login    = Ask "Login"
$mt5Password = AskSecret "Password"
$mt5Server   = Ask "Server" "Exness-MT5Trial15"
$botMode     = Ask "Mode  (backtest / demo / live)" "demo"
if ($botMode -notin @("backtest","demo","live")) { $botMode = "demo" }

# ---- 4. Agents ---------------------------------------------
Blank
Line
Blank
Show-All "Cyan" "Magenta" "Green" "Yellow"
Blank
C "  Do you want to set up AI agents?" "White"
C "  (without keys the bot trades with built-in rules)" "DarkGray"
Blank

$useAgents = Ask "Set up agents? (y/n)" "y"

$skipperProvider = "none"
$openaiKey       = ""
$anthropicKey    = ""
$googleKey       = ""
$deepseekKey     = ""
$orchEnabled     = "False"

if ($useAgents -match "^[yY]") {
    $orchEnabled = "True"

    # ---- SKIPPER -------------------------------------------
    Clear-Host
    Blank
    Show-One "skipper"
    C "  SKIPPER  --  Market Analyst" "Cyan"
    Blank
    C "  Choose provider:" "White"
    C "    [1]  ChatGPT  (OpenAI GPT-4o)" "Gray"
    C "    [2]  Gemini   (Google)" "Gray"
    C "    [0]  Skip     (rule-based)" "DarkGray"
    Blank

    $sc = Ask "Choice" "0"
    switch ($sc) {
        "1" {
            $skipperProvider = "openai"
            Show-One "skipper"
            $openaiKey = AskKey "OpenAI"
        }
        "2" {
            $skipperProvider = "gemini"
            Show-One "skipper"
            $googleKey = AskKey "Google"
        }
        default { $skipperProvider = "none" }
    }

    # ---- KOWALSKI ------------------------------------------
    Clear-Host
    Blank
    Show-One "kowalski"
    C "  KOWALSKI  --  Risk Manager  [Claude]" "Magenta"
    Blank
    C "  console.anthropic.com" "DarkGray"
    Blank

    $kc = Ask "Set up Kowalski? (y/n)" "y"
    if ($kc -match "^[yY]") {
        $anthropicKey = AskKey "Anthropic"
    }

    # ---- RICO ----------------------------------------------
    Clear-Host
    Blank
    Show-One "rico"
    C "  RICO  --  News Analyst  [Gemini]" "Green"
    Blank
    C "  aistudio.google.com/apikey" "DarkGray"
    Blank

    if ($googleKey -ne "") {
        C "  (using same Google key as Skipper)" "DarkGray"
    } else {
        $rc = Ask "Set up Rico? (y/n)" "y"
        if ($rc -match "^[yY]") {
            $googleKey = AskKey "Google"
        }
    }

    # ---- PRIVATE -------------------------------------------
    Clear-Host
    Blank
    Show-One "private"
    C "  PRIVATE  --  Final Validator  [DeepSeek]" "Yellow"
    Blank
    C "  platform.deepseek.com" "DarkGray"
    Blank

    $pc = Ask "Set up Private? (y/n)" "y"
    if ($pc -match "^[yY]") {
        $deepseekKey = AskKey "DeepSeek"
    }
}

if ($dc -match "^[yY]") {
}

# ---- 6. Write .env -----------------------------------------
@"
MT5_LOGIN=$mt5Login
MT5_PASSWORD=$mt5Password
MT5_SERVER=$mt5Server
BOT_MODE=$botMode


OPENAI_API_KEY=$openaiKey
ANTHROPIC_API_KEY=$anthropicKey
GOOGLE_API_KEY=$googleKey
DEEPSEEK_API_KEY=$deepseekKey

SKIPPER_PROVIDER=$skipperProvider
ORCHESTRATOR_ENABLED=$orchEnabled
AGENT_MIN_QUALITY=3.0
AGENT_MIN_CONFIDENCE=55
$safeDir = Join-Path $env:USERPROFILE ".penguin_squad"
if (-not (Test-Path $safeDir)) { New-Item -ItemType Directory -Path $safeDir | Out-Null }
$envPath = Join-Path $safeDir ".env"
"@ | Out-File -FilePath $envPath -Encoding utf8

# ---- Done --------------------------------------------------
Clear-Host
Blank
Show-All "Cyan" "Magenta" "Green" "Yellow"
Blank
Line
Blank

$sk2 = if ($skipperProvider -ne "none" -and ($openaiKey -or $googleKey)) { "ON  [$($skipperProvider.ToUpper())]" } else { "rule-based" }
$kw2 = if ($anthropicKey) { "ON  [CLAUDE]"  } else { "rule-based" }
$rc2 = if ($googleKey)    { "ON  [GEMINI]"  } else { "rule-based" }
$pr2 = if ($deepseekKey)  { "ON  [DEEPSEEK]"} else { "rule-based" }

C "  Skipper   $sk2" "Cyan"
C "  Kowalski  $kw2" "Magenta"
C "  Rico      $rc2" "Green"
C "  Private   $pr2" "Yellow"
Blank
Line
Blank
C "  python main.py --mode $botMode" "White"
Blank
Line
Blank
