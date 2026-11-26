# lcode

A tiny, single-file coding-assistant CLI for **local** LLMs. It gives a model running on
[Ollama](https://ollama.com) a Claude-Code-style tool-calling loop — read files, grep, edit, run
shell commands — with a lean prompt, a fixed context window, and tight control over tokens.

It is built for small local models (7B–30B class) on modest hardware, where every token of context
and every second of prompt processing matters. No framework, **no dependencies** — just the Python
standard library and one file.

```
$ lcode -p "find and fix the failing test"
lcode 0.4 · qwen3:30b · think=low · ctx=32768 · /path/to/project
  ● grep(pattern='def test_', file_glob='*.py')
  ● read_file(path='test_math.py')
  ● edit_file(path='math.py')
  ● run_shell(command='python3 -m pytest -q')
  $ python3 -m pytest -q
Fixed the off-by-one in round_half_up(); all 12 tests pass.
  [1840 in / 210 out · 38.5 tok/s · context 6% · 7s]
```

## Why

Running a full agent harness against a local model is often painful: the base system prompt alone
can be many thousands of tokens, context fills quickly, and long prompt-reprocessing stalls hurt.
lcode keeps the overhead minimal so a small model can actually get work done:

- **Lean system prompt** — a few hundred tokens, not thousands.
- **Fixed context** (`--ctx`) so the model is loaded once and stays on the GPU.
- **No thinking replay** — reasoning tokens are shown but never fed back, saving context.
- **Truncated tool output** and **history compaction** to stay under the window.
- **Visible token usage** after every call, so you can see exactly what you're spending.

## Features

- **Tools:** `read_file`, `list_files`, `grep`, `edit_file`, `write_file`, `run_shell`,
  `run_shell_to_file`, plus optional `web_search` / `web_fetch` (with `--web`).
- **Safety by default:**
  - Every file path is confined to the project root (symlink-resolved).
  - `run_shell` refuses to write files through the shell (heredocs, `>` redirection, `tee`,
    `sed -i`) and points the model at `write_file` / `edit_file` so you see and approve the content.
  - Writes and commands prompt for approval (`y`/`N`/`a` = allow all), unless `--yolo`.
  - Optional `--sandbox` runs shell commands under macOS `sandbox-exec`: no network, writes only
    to the project and temp dirs.
  - `web_fetch` refuses local/intranet addresses (so the model can't reach your router or Ollama).
- **Evidence discipline:** the prompt requires every claim to come with a reproducing command, and
  the `/verify` command makes the model adversarially re-check its own conclusions.
- **`findings.md` as long-term memory:** findings written to `findings.md` are pinned after the
  system prompt and survive history compaction.
- **`LCODE.md` project notes:** if present in the project root, it's appended to the system prompt.
- **Two modes:** an interactive REPL, or one-shot `-p "..."`.

## Requirements

- Python 3.8+ (standard library only).
- [Ollama](https://ollama.com) running locally, with a model that supports tool calling
  (e.g. `qwen3`, `llama3.1`, `mistral-nemo`, …).

## Install

It's a single file. Drop it on your `PATH`:

```sh
curl -o ~/.local/bin/lcode https://raw.githubusercontent.com/Adam1313943/lcode-cli/main/lcode.py
chmod +x ~/.local/bin/lcode
```

Or clone and symlink:

```sh
git clone https://github.com/Adam1313943/lcode-cli.git
ln -s "$PWD/lcode-cli/lcode.py" ~/.local/bin/lcode
```

## Usage

```sh
# Interactive, working on the current folder
lcode

# One-shot: run a request and exit
lcode -p "add type hints to utils.py and run the tests"

# Pick a model and context size
lcode --model qwen3:30b --ctx 49152 --think low

# Enable web search/fetch (asks before each network call)
lcode --web

# Throwaway folder, no approval prompts
lcode --yolo --sandbox
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-p, --prompt` | — | One-shot mode: run this request and exit. |
| `--model` | `qwen3.8:27b` | Ollama model name (or set `LCODE_MODEL`). |
| `--host` | `http://localhost:11434` | Ollama host (or set `OLLAMA_HOST_URL`). |
| `--think` | `low` | Thinking effort: `false` / `low` / `medium` / `xhigh`. |
| `--ctx` | `32768` | Context length (`num_ctx`). |
| `--max-out` | `16384` | Max output tokens, including thinking. |
| `--dir` | cwd | Project root. |
| `--yolo` | off | Never ask before writing files or running commands. |
| `--sandbox` | off | Run `run_shell` under macOS `sandbox-exec`. |
| `--show-thinking` | off | Stream thinking content live. |
| `--web` | off | Enable `web_search` (DuckDuckGo) and `web_fetch`. |

### Interactive commands

```
/think false|low|medium|xhigh   switch thinking effort
/clear                          clear the conversation (keep the system prompt)
/compact                        compact history now
/verify                         adversarially re-check the last conclusions, dropping unsupported claims
/tokens                         show token stats
/exit                           quit (Ctrl-D also works)
```

Press `Ctrl-C` while a response is running to interrupt it.

### Project notes: `LCODE.md`

Drop an `LCODE.md` in your project root and it is appended to the system prompt on every run — a
place for repo-specific conventions, build/test commands, or task instructions. Keep it short
(it's truncated at ~6000 chars).

## Tests

Fully offline — a fake Ollama server and fixtures, no real model needed:

```sh
python3 -I test_lcode.py
```

## Notes & limitations

- The `--sandbox` option is macOS-only (it uses `sandbox-exec`). Everything else is cross-platform,
  though it has mainly been used on macOS.
- `web_search` scrapes DuckDuckGo's HTML endpoint, which is unofficial and may change.
- This is a small personal tool, shared in case it's useful. It is not affiliated with Anthropic;
  "Claude-Code-style" just describes the tool-loop design it's inspired by.

## License

[MIT](LICENSE)
