# Music Generator — YuE2 on 8 GB or 16 GB GPUs, with a web UI

Built on [YuE2](https://github.com/multimodal-art-projection/YuE): *frontier music generation with symbolic planning, zero-shot covers, and agentic music editing.* Give it lyrics and a style prompt: it writes a melody-and-chord plan, then realizes that plan as a complete 48 kHz stereo song with vocals and accompaniment — and because the plan is an editable score, you can change the composition and render it again.

Upstream targets **Linux, Python 3.12 and a 24 GB GPU**. This repo runs it on **Windows**, unquantized, on an **8 GB card** (RTX 4060, *low-VRAM mode*: model halves are swapped through system RAM) or a **16 GB+ card** (*normal mode*: the whole model stays on the GPU), at roughly one minute of compute per minute of audio, and wraps it in a local web app (**Music Gen Studio**). The mode is picked automatically from the detected VRAM.

![Music Gen Studio](docs/screenshot.png)

| File | What |
|---|---|
| `webui.py` + `webui/` | Local web app: compose, live stage progress, cancel/queue, in-page playback, score rendered as sheet music, edit-and-re-render, library of past songs |
| `run_lowvram.py` | The runner (also a CLI) with both VRAM modes. Monkey-patches the upstream package at import time; the `YuE/` submodule is never modified |
| `download_models.py` | Fetches the ~7.3 GB of weights into `models/` as plain files |
| `abc_transpose.py` | Transposes a generated score by N semitones (e.g. to hit a requested key) — verify with the upstream `abc_tools.py inspect` |
| `Start Music Gen Studio.cmd` | Double-click launcher for the web app (this PC only) |
| `Start Music Gen Studio (share on network).cmd` | Same, but reachable by everyone on your LAN — see *Share on your network* |
| `songs/` | Example requests: `sahod.json` (a Tagalog OPM song about late salaries), `sahod-prog-metal.json` (same lyrics as progressive metal), `sahod-prog-metal-duet.json` (the metal version re-sung as a male/female duet by supplying its chord-free score with `cot: melody` and singer names in the section tags), `sahod-orchestra-choir.json` (rewritten as a Coldplay-style orchestral anthem with choir), `dark-trap-instrumental-*.json` (instrumentals: lyrics made only of `[... - instrumental]` section tags plus "instrumental, no vocals" leading the style; the planned score comes out with an all-rest vocal voice; `dark-trap-instrumental-3min.json` reaches 2½ minutes by repeating the drop/breakdown blocks of a planned score and supplying it as `abc`) |
| `YuE/` | Upstream repository, pinned as a git submodule |

## Setup (Windows)

Requirements: an NVIDIA GPU with BF16 support and at least 8 GB VRAM (compute capability ≥ 8.0; FP8 mode needs ≥ 8.9), a recent driver, Python 3.10+ (`py -3.11` below), git, ~15 GB of disk, and enough RAM to hold the model between songs (16 GB works, 32 GB+ is comfortable in low-VRAM mode, which keeps most of the model in RAM).

```powershell
git clone --recurse-submodules https://github.com/jrlabanza/music-generator.git
cd music-generator
py -3.11 -m venv YuE\.venv
YuE\.venv\Scripts\python.exe -m pip install --upgrade pip
YuE\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
YuE\.venv\Scripts\python.exe -m pip install -e YuE
YuE\.venv\Scripts\python.exe -m pip install -r requirements.txt
YuE\.venv\Scripts\python.exe download_models.py
```

The venv lives inside `YuE\` because that is where the launcher and scripts look for it. Install PyTorch from the CUDA index *before* `pip install -e YuE` so the pinned `torch==2.10.0` resolves to the CUDA build.

## Use the web app

Double-click **`Start Music Gen Studio.cmd`**, or:

```powershell
YuE\.venv\Scripts\python.exe webui.py --open
```

The page opens at http://127.0.0.1:7860 once the server is up (~10 s; the model is loaded into RAM once and kept there).

- **Compose** — style prompt (genre, instruments, vocal, language, tempo), lyrics with `[Verse]`/`[Chorus]` tags, plan mode, seed. *Load example* fills in the upstream song. `Ctrl+Enter` generates.
- **Progress** — each stage live (plan score → semantic tokens → synthesize → decode) with tokens/s and an audio-length estimate, a *Cancel* button, and a queue for further requests.
- **Result** — plays in the page; FLAC/WAV downloads; the generated score as sheet music (abcjs, bundled) or ABC text; *Load into composer* to remix; *Edit score* to change harmony or melody and re-render — the white-box editing flow from the upstream docs.
- **Plan first** — *Plan score* writes just the score (~15 s) and shows it as sheet music; *Render this score* then generates the audio from exactly that plan, or *Edit in composer* lets you change it first.
- **Takes** — generate up to 4 takes of the same request with different seeds; tick songs in the Library and *Compare selected* to play them side by side.
- **Sampling** — temperature, top-p, top-k, repetition penalty and max length for the score and audio phases, CFG scale, and the number of ODE steps (upstream's `GenerationConfig`; blank = default).
- **Score tools** (on the ABC box) — strip chords, keep only the vocal or instrumental line (upstream `abc_tools.py`), transpose by semitones (`abc_transpose.py`), set the tempo, a *Sections* editor to reorder / repeat / delete / halve sections, *Check* to validate and summarise, and *Compare with source* to prove an edit kept the notes.
- **Cover a recording** — upload a song and SheetSage2 transcribes its melody (chord-free) straight into the score box; add lyrics and a style and generate (upstream's cover workflow). With *instrumental cover* ticked (the default) the lyrics box is filled with section tags from the score, so an instrumental can be restyled without typing anything; *Tags from score* does the same for any score in the box. Requires the SheetSage2 environment (below).
- **Re-decode (legacy)** — re-render a song's saved latents through the benchmark decoder `YuE2-Vae-legacy` in a few seconds for a second listening version.
- **New take / Same score, new style** — one-click variations of any song in the Library.
- **Lyric writing** (buttons under the Lyrics box) — *Continue lyrics* writes the next section in the same language, rhyme scheme and story (pick which section, or let it choose); *Write from title* drafts complete lyrics from the title and style; *undo* restores. Runs on a **local** model through [Ollama](https://ollama.com) (`qwen2.5:7b` by default, `MUSICGEN_LYRICS_MODEL` to change): install Ollama, run `ollama pull qwen2.5:7b`, and the buttons work — on the GPU when no song is generating, on the CPU otherwise, never holding VRAM between requests. Read the output over before generating; it's a 7B model.
- **Sing it in your voice** (on any finished song) — upload a 30–60 s clip of yourself singing (*Voices…*), pick it, press *Sing it in this voice*: Demucs lifts the vocal off the song, [Seed-VC](https://github.com/Plachtaa/seed-vc)'s zero-shot singing model re-sings it with your timbre (pitch-shift it a few semitones if the song sits outside your range), and the new vocal is mixed back over the original backing. Each version appears under the player and stays in the song's folder (`audio-voice-<name>.flac`). YuE2 itself has no voice cloning — this is post-processing, so the melody, words and backing are unchanged; only the voice is yours. Requires the voice environment (below).
- **Harmonies, duets and auto pitch** (same row) — *harmonies* renders extra copies of the vocal a 3rd/5th/octave away and blends them under the lead; *duet* sings the chorus (or verses, bridge, or alternating sections) in a second reference voice, using the karaoke line timings to split the sections; pitch *auto* moves the melody into the reference singer's range before converting.
- **Just describe it** (top of the composer) — one sentence ("a birthday song for my mum, Tagalog, acoustic OPM") and the local model drafts the title, a well-formed style line and complete tagged lyrics for review; *Draft & generate* goes straight to a song, optionally sung in one of your voices when it finishes.
- **Style presets** (under the Style box) — built-in genre prompts plus your own (*Save style*); *Try styles…* queues the same title and lyrics once per ticked preset to hear which genre fits.
- **Karaoke: sync lyrics** — Whisper times the lyrics to the separated vocal (about 15 s); the lyrics follow the player line by line (click a line to jump there), with `.lrc` and `.srt` downloads for players and video editors. *Cover a recording* can use the same model to write down the words of an upload.
- **Split into stems** — Demucs splits any song into vocals, drums, bass and other, plus an instrumental (no vocals) and an acapella, each playable and downloadable.
- **Export** — mastered MP3 (320 kbps, tagged), WAV or FLAC of the original or any voice version: loudness normalised to −14 LUFS for streaming, peak-limited to −1 dBFS, optional fade-out and silence trim.
- **Share** — a listen-only link for a song (valid 1 day to a year, QR code) that works without the password, using the public address when the tunnel is on.
- **Favourites, ratings, tags, notes** on every song, with search and *favourites* / *mine* filters in the Library; *New take*, *Same score, new style*, *Edit score* and *Try styles* record where a song came from, shown as its version family.
- **Settings** (header pill) — your name (shown on the queue and on songs you make, so shared use stays sortable) and a Discord webhook and/or Telegram bot that get a message with a share link whenever a job finishes.
- **Phone** — installable as an app from the share page ("Add to Home Screen"), with lock-screen play/pause/next controls.
- **System** (header pill) — versions, GPU, model files and their hashes, storage, whether the cover feature, the lyric model, the voice tools and notifications are ready.
- **Library** — every song in `outputs/`, newest first. Each folder keeps `audio.flac`, `score.abc`, `plan.json`, `semantic.npy`, `latent.npy`, `request.json`, `result.json`. *Delete* moves a song's folder to `trash/` (restore by moving it back into `outputs/`; empty `trash/` by hand to reclaim disk). It works while the song is loaded in players — files are served from memory so the server never pins them — and if some other program holds a file open, the song is hidden at once and moved as soon as the file is released.

Flags: `--vram low|normal|auto` (see below), `--share` to let other devices on your network use it (below), `--quantization fp8` for an even smaller GPU footprint (slower: eager decoding), `--gpu-reserve-gib 1.5` if you close other GPU apps. The GPU is only used while a song is generating.

### Share on your network

Only one PC needs the GPU. Start the app with **`Start Music Gen Studio (share on network).cmd`** (or `webui.py --share`) and it listens on every interface; the console and the header pill show the address to hand out, e.g. `http://192.168.1.20:7860`. Friends on the same Wi-Fi/LAN open that in any browser — phones included — and their songs queue one at a time on your GPU, with everyone seeing the shared Library.

- **Windows Firewall** must allow inbound TCP on port 7860 for the Python that runs the server (the venv's `python.exe` hands off to your base Python install). The first time it listens, Windows usually shows an *allow access* prompt — tick both *Private* and *Public* if your network shows as Public. Or add the rule once from an **administrator** PowerShell:

  ```powershell
  New-NetFirewallRule -DisplayName "Music Gen Studio" -Direction Inbound -Protocol TCP -LocalPort 7860 -Action Allow -Profile Any
  ```

- There is **no login**: anyone on the network can generate, cancel the running job, and download every song. Keep it to networks you trust and never port-forward it to the internet.
- If a friend cannot connect although the rule exists, the network itself may isolate clients (common on guest/corporate Wi-Fi), or the PC's address changed — check the pill for the current one.

### Use it from anywhere (phone, away from home)

First, **set a password** — it is required for anything beyond your own machine:

```powershell
YuE\.venv\Scripts\python.exe webui.py --share --password "choose-something-long"
```

`--password` (or `--password-file password.txt`, or the `MUSICGEN_PASSWORD` environment variable) protects every page, API call and download: browsers get a sign-in page (`/login`, cookie kept for 30 days, `/logout` to end it), and scripts can use HTTP Basic auth instead. Then pick a route:

| Route | What you get | Notes |
|---|---|---|
| **Tailscale** (recommended) | A private VPN between your devices. Install it on the PC and the phone, sign in with the same account, then open `http://<the PC's 100.x.x.x address>:7860` from the phone anywhere. `tailscale serve --bg 7860` adds HTTPS at a stable `https://<pc>.<tailnet>.ts.net` address. | Nothing is exposed to the public internet; works through corporate/CGNAT networks; free for personal use. Installing needs admin rights. |
| **Cloudflare Tunnel** (built in) | Put a password in `password.txt` and double-click **`Start Music Gen Studio (internet).cmd`** (= `webui.py --share --tunnel --password-file password.txt --open`). It starts `cloudflared`, prints an `https://….trycloudflare.com` address, and the share pill on the page shows it as a **QR code** for your phone. | Needs [cloudflared](https://github.com/cloudflare/cloudflared/releases) installed (or `--cloudflared PATH`). No account or router changes, but the URL is public and changes every start — a free Cloudflare account + your own domain gives a fixed one. `--tunnel` refuses to run without a password. |
| Router port-forwarding + nginx/Caddy | The classic reverse proxy. | Exposes your home network, needs a static IP or dynamic DNS and a certificate; not worth it next to the two above. |

Whatever the route: the PC stays on with the app running, generation still queues one at a time, and a phone on mobile data streams the FLAC fine (a 3-minute song is ~25 MB). If the PC is managed by an employer, check their policy before tunnelling it.

**Leaving it running for days.** With `--share`/`--tunnel` the app asks Windows not to sleep while it runs (`--no-keep-awake` to disable), restarts `cloudflared` by itself if it exits (the public address changes then; the current one is always in `public_url.txt` and the share pill), and the internet launcher restarts the app if it ever crashes. Two things it cannot control: **Windows Update restarts** (pause updates for the period, Settings → Windows Update) and **logging off or closing the console window** (both end it). After a reboot nobody is logged in, so nothing runs until you sign in and start the launcher again; a Task Scheduler entry that runs the launcher at logon covers the sign-in part.

### Cover a recording (SheetSage2)

The cover feature runs upstream's SheetSage2 transcriber in its own environment (it pins a different torch). One-time setup, about 2.7 GB of extra downloads:

```powershell
py -3.11 -m venv .venv-sheetsage2
.venv-sheetsage2\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv-sheetsage2\Scripts\python.exe -m pip install huggingface-hub==0.36.0 transformers==4.45.2 safetensors==0.5.3 numpy==1.24.3 scipy==1.13.1 mir_eval==0.8.2 pretty_midi==0.2.10 mido==1.3.3 setuptools==78.1.1 soundfile
YuE\.venv\Scripts\python.exe download_models.py --sheetsage2 --legacy-vae
```

`download_models.py --sheetsage2` fetches `m-a-p/SheetSage2` and its `MERT-v2-FullSong` encoder into `models/` and points the SheetSage2 config at the local encoder. `sheetsage_transcribe.py` decodes uploads with `soundfile` (wav, flac, mp3, ogg); other containers need FFmpeg on PATH. Transcription runs on the GPU between generations (the two never overlap) and takes about a minute for a 3-minute song.

### Sing it in your voice (Seed-VC)

Voice conversion runs in its own environment too (Seed-VC pins older `transformers`/`numpy`). One-time setup; the Seed-VC checkpoints (~1.5 GB: the singing model, RMVPE pitch tracker, CAMPPlus speaker encoder, BigVGAN vocoder, Whisper) and Demucs download from Hugging Face on the first conversion:

```powershell
py -3.11 -m venv .venv-voice
.venv-voice\Scripts\python.exe -m pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
.venv-voice\Scripts\python.exe -m pip install scipy==1.13.1 librosa==0.10.2 "huggingface-hub>=0.28.1" munch==4.0.0 einops==0.8.0 descript-audio-codec==1.0.0 pydub==0.25.1 transformers==4.46.3 soundfile numpy==1.26.4 hydra-core==1.3.2 pyyaml python-dotenv demucs mutagen
git clone https://github.com/Plachtaa/seed-vc tools\seed-vc
```

Then in the app open *Voices…* on any song and add a recording of yourself: 30–60 s, singing (speaking works, singing works better), alone, in a quiet room, no backing track, no reverb — the model copies whatever it hears. Clips live in `voices/` (gitignored; they never leave the PC). `voice_convert.py` does the work: Demucs `htdemucs` separates the song (stems are cached in the song's `stems/` folder, so trying a second voice skips that step), Seed-VC's f0-conditioned model (`DiT … f0_44k … v2`, 30 diffusion steps) converts the vocal at 44.1 kHz, then it is level-matched to the original vocal, mixed over the accompaniment and written next to `audio.flac`. Roughly real time on an RTX 4060 (a one-minute song takes one to two minutes; the first run also downloads the checkpoints); the GPU is shared with generation, so conversions queue behind songs. Harmonies and duets add one conversion pass each (the renders are cached in `stems/`, so re-mixing with another harmony or duet voice reuses them).

The same environment powers the other audio tools, all sharing the per-song `stems/` cache: `stems.py` (Demucs stems), `export_audio.py` (loudness/limiter/fades → MP3 via `lameenc`, tags via `mutagen`) and `lyrics_sync.py` (Whisper word timestamps aligned to the lyrics → `karaoke.json`, `.lrc`, `.srt`; or a plain transcription for covers). Whisper defaults to `openai/whisper-small`, which Seed-VC had already downloaded; set `MUSICGEN_WHISPER_MODEL=openai/whisper-large-v3-turbo` (1.6 GB, better for Tagalog and mixed languages) before starting the server to use a stronger one.

Notifications (Settings pill) post to a Discord webhook (server settings → Integrations → Webhooks) and/or a Telegram bot (token from @BotFather, your chat id from a bot such as @userinfobot); both are stored in `settings.json`, gitignored. Share links are signed with a random secret in `.share_secret`, also gitignored, so they stay valid across restarts and survive password changes.

```powershell
.venv-voice\Scripts\python.exe voice_convert.py --song outputs\first-song\audio.flac --reference voices\me.wav --name me --output outputs\first-song --semitones -2
```

## Command line

```powershell
YuE\.venv\Scripts\python.exe run_lowvram.py --output outputs\first-song
YuE\.venv\Scripts\python.exe run_lowvram.py --style "English, indie rock, male vocal, 120 BPM" --lyrics-file lyrics.txt --output outputs\rock
YuE\.venv\Scripts\python.exe run_lowvram.py --request songs\sahod-prog-metal.json --output outputs\sahod-metal
YuE\.venv\Scripts\python.exe run_lowvram.py --request my-song.json --cot melody --abc-file YuE\examples\melody.abc --output outputs\melody
```

`--cot full|melody|off`, `--abc-file` (needs `full` or `melody`), `--seed`, `--cfg-scale`, `--vram low|normal|auto`, `--quantization fp8`. Output directories must be fresh. See `YuE/docs/` for the upstream generation, editing and cover guides.

## VRAM modes

| Mode | Picked when | What happens |
|---|---|---|
| `normal` | ≥ 14 GB VRAM detected (e.g. a 16 GB card) | Upstream placement: the whole 6.8 GB model stays on the GPU. Fastest. |
| `low` | < 14 GB (e.g. 8 GB) | Only the model half in use is on the GPU; the rest and the embedding table live in RAM (details below). A few seconds of weight movement per song. |
| `auto` (default) | — | Chooses between the two from the detected card, so the same launcher works on both machines. |

Force a mode with `--vram low` / `--vram normal` on `webui.py` or `run_lowvram.py`; the console banner and the page's status pill show which one is active. The Windows attention fixes apply in both modes.

## How the 8 GB fit works (low-VRAM mode)

The 3B model is 6.8 GB in BF16 and the upstream pipeline caps itself at total VRAM minus 2 GB, so on an 8 GB card the stock scripts cannot even load it. `run_lowvram.py` subclasses the pipeline and keeps only the half of the AR–NAR Mixture-of-Transformers model that is in use on the GPU, parking the rest in system RAM:

| Phase | On the GPU | In RAM |
|---|---|---|
| Plan score / semantic tokens | AR layers, lm_head | NAR layers, embedding table |
| NAR prefill | AR layers | NAR layers, lm_head, embedding table |
| NAR flow matching | NAR layers | AR layers, lm_head, embedding table |
| VAE decode | VAE | transformer |

The 0.7 GiB token-embedding table never needs to be on the GPU: rows are gathered in RAM and copied over (a few KB per decode step; the CUDA-graph decoder reads them from a static buffer), which is numerically identical. Measured: **3.4 GiB** resident during decoding and a **5.1 GiB peak** for a 5½-minute song that uses almost the whole 9000-token budget, with CUDA-graph decoding at ~45 tokens/s on an RTX 4060.

It also works around three quirks of the Windows PyTorch wheel and the upstream code: FlashAttention is not compiled in, so grouped-query SDPA silently falls back to the O(n²) math kernel (about 24 GB at song length) — K/V heads are expanded instead, which selects the fused memory-efficient kernel; the CUDA-graph decoder is pointed at cuDNN attention rather than the missing flash entrypoint; and the VAE's legacy `weight_norm` leaves 254 MiB of computed weights on the GPU after every decode, which is released explicitly. Everything else (BF16 weights, sampling, the exact NAR prefill) is upstream behaviour.

**Reproducibility.** With the default cuDNN graph attention, the same seed does not reproduce the same song run-to-run on this setup (cuDNN appears to pick a different execution plan per graph capture). `--graph-attention sdpa` (web app and CLI) is deterministic but decodes about 2.7× slower.

## Licenses

`YuE/` is the upstream code under Apache-2.0 (see `YuE/LICENSE`). The model weights downloaded by `download_models.py` are under **CC BY-NC 4.0 with an additional creator permission** (see `YuE/MODEL_LICENSE`): free for personal use and content creators, non-commercial for research, and companies need a commercial license from the YuE2 authors. `webui/vendor/abcjs-basic-min.js` is [abcjs](https://abcjs.net) (MIT).
