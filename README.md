<!--
AI ASSISTANTS: machine-readable install/usage instructions live in llms.txt at the repo root.
URL: https://raw.githubusercontent.com/Fredasterehub/talktomeclaude/main/llms.txt
Agent-facing conventions: llms.txt (install/usage) · AGENTS.md (contributor/agent guidance).
-->

<p align="center">
  <img src="assets/skull-emblem.jpg" width="210" alt="talktomeclaude — a skull wearing studio headphones with a boom microphone">
</p>

<h1 align="center">talktomeclaude</h1>

<p align="center"><em>Covey leader to Raven… talk to me, Claude.</em></p>

<p align="center">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-171310">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-e6b22e?labelColor=171310">
  <img alt="local-first" src="https://img.shields.io/badge/local--first-no%20cloud%20voice-171310">
  <img alt="voices: public domain" src="https://img.shields.io/badge/voices-public%20domain-e6b22e?labelColor=171310">
</p>

<p align="center"><a href="https://fredasterehub.github.io/talktomeclaude/"><strong>▶&nbsp; Live site</strong></a></p>

**Put the keyboard down. Talk to Claude Code — and let it talk back.**

talktomeclaude is a local-first voice medium for Claude Code. You speak, Claude
works, Claude answers back in a real-sounding voice — headphones on, hands off.
The listening is local (Whisper-class). The speaking is local (Piper
voices). Claude's own brain stays in the cloud; everything around it runs on
your machine.

It filters Claude's spoken dialogue out of the transcript — **only the dialogue,
never tool calls, never code, never logs** — speaks it through one of three
public-domain voices, and rides a Claude Code Stop hook so replies are
spoken the moment Claude finishes. A mute switch that actually mutes. Three
recording modes. One local command.

`MIT` · `Python 3.11+` · `local-first` · `no cloud voice, no subscription`

---

## What it does

- **Launch dashboard** — run `talktomeclaude` with no arguments for a live signal view, conversation status, recording controls, and a remote project picker.
- **Hear you** — local speech-to-text, Whisper-class (faster-whisper). Your voice never leaves the machine.
- **Answer back** — speaks Claude's actual dialogue in a real voice, and *only* the dialogue. Tool calls, fenced code, and thinking are stripped out.
- **Ride Claude Code** — a plugin Stop hook speaks each reply automatically. Async, non-blocking, fails silent.
- **Three recording modes, cross-platform** — `always-on` (hands-free, pause-gated), `push-to-talk` (hold a key), and `push-toggle` (tap to start, tap to send) work in native Windows Terminal and POSIX terminals without an extra keyboard package.
- **A real mute** — `assist off` and the whole thing goes quiet, hook included.
- **Ship silent-proof** — three public-domain voices, fetched automatically on first use, so day one is never a robot.

---

## Requirements

- **Python 3.11 or newer** (the installers below fetch it for you if you don't have it).
- **A fast, lightweight clone** — the voice models are *not* committed to the repo. They download once, on your first `speak` (~250 MB for all three), and cache under `~/.cache/talktomeclaude/voices`. Pre-fetch them any time — e.g. before going offline — with `talktomeclaude voices --download`.
- **A one-time model download on first transcription** — the local Whisper model pulls itself the first time you use `transcribe`/`listen`: **~486 MB** on the CPU tier (`small.en`), or **~3 GB** on the NVIDIA/CUDA tier (`large-v3`). It caches under `~/.cache/talktomeclaude/` and never downloads again.
- **[Claude Code](https://code.claude.com)** on your PATH if you want the voice loop and the Stop-hook plugin (the CLI works standalone without it).

The three steps that never change on any platform: **get the code → make the
env → install it.** The rest is your OS spelling those the same three words
differently. Pick your platform below.

---

## Install

### First: where will Claude Code run?

talktomeclaude is modular — it fits whatever setup you have. There are two, and
they change what you install where:

- **All on one computer** — your Mac or PC laptop has the microphone *and* the
  speakers, *and* you run Claude Code on it. This is the simple case: follow your
  platform below and you're done, no extra config.
- **Voice on your computer, Claude on a server** — you sit at a laptop (mic +
  speakers) but you **SSH into a Linux box** (Proxmox, a homelab server, a VPS)
  and do your actual coding *there*. A server reached over SSH has **no
  microphone and no speakers**, so the voice has to run on your laptop while
  Claude runs on the server, with only text crossing the network. This works —
  jump to **[Talking to Claude on a remote server](#talking-to-claude-on-a-remote-server-ssh--proxmox)**
  after you've installed on your laptop.

Either way you start by installing on the machine you physically sit at. Pick
its platform:

### Windows — Windows Terminal (PowerShell)

For a complete beginner on Windows 10/11, using **Windows Terminal** with the
default **PowerShell** tab.

1. **Open Windows Terminal.** Press `Start`, type `Windows Terminal`, hit Enter. You get a PowerShell prompt.

2. **Install `uv`** (a fast Python installer/manager — it will also grab Python itself, so you don't have to). Paste this one line, press Enter, then **close and reopen** Windows Terminal so it's found:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Get the code.** Easiest, no extra tools — download the ZIP:
   - Go to <https://github.com/Fredasterehub/talktomeclaude>, click the green **Code** button ▸ **Download ZIP**, then right-click the file ▸ **Extract All**.
   - *(Prefer git? Run `winget install --id Git.Git -e`, reopen the terminal, then `git clone https://github.com/Fredasterehub/talktomeclaude.git`.)*
   - The clone is small and fast — the voice models download later, on your first `speak` (~250 MB, cached once).

4. **Step into the folder** (adjust the path to where you extracted it):
   ```powershell
   cd talktomeclaude
   ```

5. **Make the environment and install** (uv downloads Python 3.12 automatically if you don't have it):
   ```powershell
   uv venv --python 3.12
   uv pip install -e .
   ```

6. **Try it** — synthesize a line, hear a voice, then list what's bundled:
   ```powershell
   .\.venv\Scripts\talktomeclaude speak "Hello from Claude."
   .\.venv\Scripts\talktomeclaude voices
   ```
   If you'd rather type `talktomeclaude` without the long path, activate the env first with `.\.venv\Scripts\Activate.ps1` and drop the `.\.venv\Scripts\` prefix.

7. **Plug it into Claude Code** — from inside the project folder:
   ```powershell
   claude --plugin-dir .
   ```
   That loads the plugin for testing. To install it permanently, run `/plugin install .` inside Claude Code instead.

   > **Windows caveat — read this.** The plugin's "speak Claude's reply" Stop hook is a **bash** script. On native Windows, Claude Code only routes hooks through bash if **Git for Windows** is installed (it ships Git Bash). Without it, the *automatic* spoken-reply hook may not fire — Windows can even try to open the `.sh` in an editor. Fix: `winget install --id Git.Git -e`, reopen the terminal, and the hook works. **Everything else — `speak`, `listen`, `transcribe`, `voices` — works fine either way;** only the hands-free "speak on every reply" hook needs Git Bash.

8. **Mute / unmute** any time:
   ```powershell
   .\.venv\Scripts\talktomeclaude assist off
   .\.venv\Scripts\talktomeclaude assist on
   ```

---

### macOS — Terminal

For a beginner on Apple Silicon or Intel, using **Terminal.app**.

1. **Open Terminal.** Press `Cmd+Space`, type `Terminal`, hit Enter.

2. **Install `uv`** (fetches Python for you too). Paste, Enter, then open a fresh Terminal window:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Get the code** (a fast, lightweight clone — voices download later, on first use). The first time you run `git`, macOS pops up **"Install Command Line Tools" — click Install**, then re-run:
   ```bash
   git clone https://github.com/Fredasterehub/talktomeclaude.git
   cd talktomeclaude
   ```

4. **Make the environment and install** (uv grabs Python 3.12 if you don't have it):
   ```bash
   uv venv --python 3.12
   uv pip install -e .
   ```

5. **Try it** — hear a voice, then list the bundled ones:
   ```bash
   ./.venv/bin/talktomeclaude speak "Hello from Claude."
   ./.venv/bin/talktomeclaude voices
   ```

6. **Plug it into Claude Code** — from the project folder:
   ```bash
   claude --plugin-dir .
   ```
   Or `/plugin install .` inside Claude Code to keep it. macOS runs the bash Stop hook natively — no caveat.

7. **Two macOS notes, both normal:**
   - **No CUDA on a Mac.** Speech-to-text runs the CPU tier (`small.en`). That's expected — it's local and fluent, just not GPU-accelerated.
   - **Microphone permission.** The first time you use `listen` (or any recording), macOS asks Terminal for **Microphone** access — click **Allow** (or later: System Settings ▸ Privacy & Security ▸ Microphone ▸ enable Terminal). Without it the mic returns silence. `speak` needs no permission.

8. **Mute / unmute:**
   ```bash
   ./.venv/bin/talktomeclaude assist off
   ./.venv/bin/talktomeclaude assist on
   ```

---

### Linux / Unix

For a beginner on Ubuntu/Debian. Fedora and Arch equivalents are on each line.

1. **Install the system bits** — PortAudio (so the mic and speakers work) and git. Pick your distro:
   ```bash
   sudo apt install -y libportaudio2 git          # Debian / Ubuntu
   sudo dnf install -y portaudio git              # Fedora
   sudo pacman -S --needed portaudio git          # Arch
   ```

2. **Install `uv`** (fetches Python for you too), then open a fresh shell:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Get the code** (a fast, lightweight clone — voices download on first use):
   ```bash
   git clone https://github.com/Fredasterehub/talktomeclaude.git
   cd talktomeclaude
   ```

4. **Make the environment and install** (uv grabs Python 3.12 if needed):
   ```bash
   uv venv --python 3.12
   uv pip install -e .
   ```
   **Got an NVIDIA GPU?** Add the CUDA extra for the fast, high-accuracy speech-to-text tier (`large-v3`):
   ```bash
   uv pip install -e ".[cuda]"
   ```

5. **Try it** — hear a voice, then list the bundled ones:
   ```bash
   ./.venv/bin/talktomeclaude speak "Hello from Claude."
   ./.venv/bin/talktomeclaude voices
   ```

6. **Plug it into Claude Code** — from the project folder:
   ```bash
   claude --plugin-dir .
   ```
   Or `/plugin install .` inside Claude Code to keep it. Linux runs the bash Stop hook natively — no caveat.

7. **Mute / unmute:**
   ```bash
   ./.venv/bin/talktomeclaude assist off
   ./.venv/bin/talktomeclaude assist on
   ```

---

## Talking to Claude on a remote server (SSH / Proxmox)

This is for the split setup: you sit at a **Windows or Mac computer** with a
microphone and speakers, and you **SSH into a Linux server** (Proxmox, homelab,
VPS) where you run Claude Code. The voice runs on your computer; Claude runs on
the server; only text goes over SSH. Here's the whole thing, top to bottom.

**On the SERVER** (inside your SSH session — Claude lives here):

1. **Install Claude Code** if it isn't already, and log in once so it's ready:
   ```bash
   claude --version    # confirm it's installed and on PATH
   claude              # run it once interactively to log in, then exit
   ```
   The server needs **only Claude Code** — no talktomeclaude, no audio, nothing else.

**On YOUR COMPUTER** (Windows or Mac — the mic and speakers live here):

2. **Install talktomeclaude** by following your platform guide above (Windows or macOS).

3. **Set up passwordless SSH** to the server, so the voice loop never stops to ask
   for a password mid-sentence. In Windows Terminal (PowerShell) or Mac Terminal:
   ```bash
   ssh-keygen -t ed25519        # press Enter through every prompt (no passphrase)
   ssh-copy-id you@192.168.2.122 # macOS/Linux; copies your key to the server
   ```
   On **Windows**, `ssh-copy-id` may not exist — use this one line instead:
   ```powershell
   type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh you@192.168.2.122 "cat >> ~/.ssh/authorized_keys"
   ```
   Then test it — this must print `ok` with **no password prompt**:
   ```bash
   ssh you@192.168.2.122 echo ok
   ```
   *(Replace `you@192.168.2.122` with your server login and IP.)*

4. **Tell talktomeclaude where the server and project are** — just once; both
   settings are remembered. `remote-cwd` is the directory on the **server** in
   which `claude -p` should start:
   ```bash
   talktomeclaude config set remote you@192.168.2.122
   talktomeclaude config set remote-cwd /srv/projects/my-project
   ```
   If you omit `remote-cwd`, Claude starts in the remote login account's home
   directory as before. Paths with spaces or shell punctuation are safely
   quoted by talktomeclaude.

5. **Talk.** From your computer:
   ```bash
   talktomeclaude listen
   ```
   Your mic is captured and transcribed **on your computer**, the text is sent to
   Claude Code **on the server**, and Claude's reply is spoken back through **your
   speakers**. (Prefer not to persist it? Skip step 4 and run
   `talktomeclaude listen --remote you@192.168.2.122 --remote-cwd /srv/projects/my-project`
   instead.)

On Windows, talktomeclaude uses the native `msvcrt` console API for
`push-to-talk` and `push-toggle`. SSH connection multiplexing remains enabled
on macOS/Linux; native Windows omits Unix-only OpenSSH control-socket options
for compatibility.

Run `talktomeclaude` with no subcommand to open the dashboard. Press `P` to
choose a project directory from the remote server, then press `Space` to start
the voice session. The existing `talktomeclaude listen` command remains
available for scripts and direct CLI use.

**To switch back to all-local** (mic + Claude + speakers on one machine):
```bash
talktomeclaude config set remote local
```
To forget the saved project directory too, run
`talktomeclaude config set remote-cwd home`.

> Two things the server needs: Claude Code **logged in**, and the `claude` command
> reachable from a non-interactive SSH shell (it's run through a login shell, so a
> normal install on `PATH` works). If a sentence errors with "claude -p failed",
> check `ssh you@server claude --version` works.

## Talk to it — the commands

`talktomeclaude` (or `./.venv/bin/talktomeclaude`, or `.\.venv\Scripts\talktomeclaude` on Windows):

| Command | What it does |
|---|---|
| `talktomeclaude` / `ui` | Open the interactive dashboard with live microphone signal, session state, voice controls, and a remote project picker. |
| `speak "text"` | Synthesize and play a line locally. `--out file.wav` writes instead of plays; `--voice NAME` picks a voice. |
| `listen` | Drive Claude Code by voice. `--mode always-on\|push-to-talk\|push-toggle`, `--once` for a single utterance, `--remote user@host` to run Claude on a server over SSH, `--remote-cwd PATH` to select its project directory, `--tmux-pane` to type into a live TUI. |
| `transcribe FILE` | Local speech-to-text on an audio file. `--device auto\|cuda\|cpu`, `--show-tier` to see which model runs. |
| `filter TRANSCRIPT.jsonl` | Print only Claude's spoken dialogue from a transcript (`-` for stdin) — the core "dialogue, never code" filter. |
| `voices` | List the voices, their licenses, and which is the default for your hardware. `--download` pre-fetches them all. |
| `config set KEY VALUE` / `config get KEY` | Persist or read settings. Keys: `recording-mode`, `voice-assist`, `remote` (`user@host`, or `local` to clear), `remote-cwd` (server path, or `home` to clear). |
| `assist on\|off\|status` | The full mute switch. `off` silences the Stop hook and all spoken replies. |

**Recording modes** — set your default once:
```bash
talktomeclaude config set recording-mode push-to-talk   # or always-on, push-toggle
```

**The Stop-hook path** (what the plugin wires up automatically): when Claude finishes a turn, the hook reads the event, pulls Claude's final message, strips it to dialogue, and speaks it — unless `assist` is `off`. It never blocks Claude Code and exits cleanly on any failure.

---

## The voices

Three voices, every one **public domain** — trained from scratch, no
copyrighted or celebrity source. They're fetched automatically from the
[Hugging Face Hub](https://huggingface.co/rhasspy/piper-voices) on first use and
cached locally; run `talktomeclaude voices --download` to pre-fetch them. Day one
is never silent, and copyright stays clean. Bring or clone your own on top.

| Voice | Accent | Quality |
|---|---|---|
| `en_US-ljspeech-high` | US English | high |
| `en_GB-cori-medium` | UK English | medium |
| `en_US-bryce-medium` | US English | medium |

Synthesis runs the **Piper** engine as a subprocess — never imported as a
library — so this project stays MIT. Run `talktomeclaude voices` to see which
one your hardware picks by default.

---

## The name — an homage

The name comes straight out of **[*Talk To Me Johnnie*](https://talktomejohnnie.com)**,
John Welbourn's old strength blog — *"It's a long road."* Hard lessons, strong
advice, no bullshit; the title itself a radio call lifted from a Rambo film —
*"Covey leader to Raven… talk to me Johnnie."* That blunt, no-filler spirit is
the one this project tries to keep. talktomeclaude is not affiliated with John
Welbourn or Power Athlete — it just owes them the name and the attitude.

*Covey leader to Raven… talk to me, Claude.*

---

## For AI agents

Installing this for someone? Everything an agent needs — per-platform steps, the
Claude Code plugin wiring, the CLI surface, and the platform gotchas — is in
**[`llms.txt`](llms.txt)** at the repo root, structured for machine reading
(the 2026 `llms.txt` convention). Contributor and agent working guidance lives in
**[`AGENTS.md`](AGENTS.md)**.

```
https://raw.githubusercontent.com/Fredasterehub/talktomeclaude/main/llms.txt
```

## License

MIT. See [LICENSE](LICENSE). The bundled voices are public domain; Piper is
invoked as a subprocess and never linked in.

---

*Powered & created with [Kiln](https://github.com/Fredasterehub/kiln) 🔥*
