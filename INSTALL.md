# EGW Research Guided Install

This file is the source of truth for a friendly AI-assisted install of the EGW Research package.

It is written mainly for Google Gemini CLI, because Gemini has a free sign-in option for many users. Advanced users can also use Claude Code or OpenAI Codex, but this guide does not teach those tools.

## What The User Is Installing

EGW Research is a local package that lets an AI assistant search the writings of Ellen G. White on the user's own computer.

Main parts:
- Python environment
- Local Qdrant database in Docker
- Prebuilt EGW search snapshots
- Optional offline chat with a local Ollama model

Supported languages:
- `en` = English
- `es` = Spanish
- `pt` = Portuguese
- `fr` = French
- `ko` = Korean

## Copy And Paste This Into Gemini

Use this exact prompt in Gemini after Gemini CLI is installed and working:

```text
Help me install EGW Research on this computer.

Important:
- I am not technical, so explain things simply and do one step at a time.
- Before doing anything, ask me these 4 questions:
  1. Am I using Windows or Linux?
  2. Which folder do I want to install EGW Research into?
  3. Which languages do I want: English (en), Spanish (es), Portuguese (pt), French (fr), Korean (ko)?
  4. Do I want optional offline AI support now, or should we skip it for now and maybe do it later?
- After I answer, do the work for me as much as possible.

Install rules:
- Prefer Windows PowerShell commands on Windows.
- Prefer bash commands on Linux.
- First check whether Python 3.10+ is installed.
- Then check whether Docker is installed and running, because EGW Research uses Qdrant in Docker for local search.
- If something required is missing, guide me clearly and use the simplest safe method available on my system.
- Clone the repository: git clone https://github.com/DanielMunozT/egw-chat.git
- Change into the directory and run setup with my chosen languages:
  python setup.py --lang <my-languages>
- After setup, read INSTALL.md, README.md, and GEMINI.md inside the directory before continuing.
- Run: python start.py
- Verify that the install works by running a small test search.

Gemini configuration:
- Create or update project file .gemini/settings.json inside the EGW Research folder.
- Set contextFileName so Gemini can use GEMINI.md, AGENTS.md, and INSTALL.md.
- Set autoAccept to true so harmless read-only actions are less annoying.
- Add a conservative coreTools list that allows common safe project actions and the shell commands needed on this OS.
- Do not promise that settings.json will remove all permission prompts. If broader permission is needed, tell me clearly that the reliable low-friction option is starting Gemini later with --yolo.
- Never use dangerous delete commands unless I clearly ask for them.

Offline option:
- Ask me whether I want offline AI support now or later.
- If I say not now, tell me I can ask you to install it later.
- If I say yes, install Ollama and choose a model based on this computer:
  - less than 8 GB RAM: qwen2.5:3b
  - 8 to 16 GB RAM: gemma3:4b by default, qwen2.5:3b if the machine seems weak
  - more than 16 GB RAM or a decent GPU: you may offer a larger model, but keep the default choice practical
- After that, explain how I can run offline chat later with the package.

At the end:
- Tell me exactly what was installed.
- Tell me the exact command to use next time.
- Keep instructions short and plain.
```

## Notes For The AI Agent

Use the simplest correct path for the user's operating system.

Windows:
- Prefer PowerShell.
- If `python` does not work, check `py`.
- If `git` is not available, guide the user to install it from https://git-scm.com or suggest downloading the repo as a ZIP from GitHub.

Linux:
- Use normal shell commands.
- Keep install paths simple, such as `~/egw-chat`.

## Recommended Gemini Project Settings

Gemini CLI supports a project settings file at `.gemini/settings.json`.

This is a reasonable starting point for this package:

```json
{
  "contextFileName": ["GEMINI.md", "AGENTS.md", "INSTALL.md"],
  "autoAccept": true,
  "coreTools": [
    "ReadFileTool",
    "GlobTool",
    "ShellTool(python)",
    "ShellTool(py)",
    "ShellTool(pip)",
    "ShellTool(docker)",
    "ShellTool(curl)",
    "ShellTool(tar)",
    "ShellTool(ls)",
    "ShellTool(where)",
    "ShellTool(ollama)",
    "ShellTool(git)"
  ]
}
```

Important:
- The exact shell command list should be adjusted for the user's operating system.
- `autoAccept` only helps with actions Gemini considers safe.
- If the user wants the fewest prompts possible during setup, the practical command is:

```bash
gemini --yolo
```

Use that only if the user is comfortable with Gemini acting without repeated confirmations.

## What To Tell The User For Next Time

After setup, the shortest normal workflow is:

1. Open a terminal in the EGW Research folder.
2. Start Gemini there.
3. Ask Gemini to start the package if needed and help with research.

Normal mode:

```bash
gemini
```

Low-friction mode:

```bash
gemini --yolo
```

Good plain-English prompt for next time:

```text
Please start EGW Research if needed, then help me research this question in the installed languages: [write your question here]
```

## Updating

To get the latest code updates:

```bash
cd ~/egw-chat
git pull
```

Language data packages rarely change. If an update is available, re-run:

```bash
python setup.py --lang en
```

## Optional Offline Use Later

If the user wants to use the package with a local model later, the usual package flow is:

1. Make sure Qdrant is running:

```bash
python start.py
```

2. Run offline chat:

```bash
python scripts/chat.py
```

Or choose a model explicitly:

```bash
python scripts/chat.py --model qwen2.5:3b
python scripts/chat.py --model gemma3:4b
```

The package already documents this in `README.md`, `GEMINI.md`, and `scripts/chat.py`.
