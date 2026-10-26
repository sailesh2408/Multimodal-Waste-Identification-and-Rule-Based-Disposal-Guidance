# Waste Guide

**Multimodal Waste Identification and Rule-Based Disposal Guidance.** This is a course project for 22ENV231 Environmental Management at Amrita Vishwa Vidyapeetham, Amritapuri.

Take a photo of a waste item on your phone and learn which stream it belongs to, how to prepare it, and where to take it.

```
phone camera ──► Stage 1: VLM on Hugging Face ──► confidence gate ──► Stage 2: rules table ──► guidance
                 (names the item only)            (asks if unsure)     (data/rules.csv decides)
```

The AI only identifies the item. The disposal advice always comes from `data/rules.csv`, so it stays consistent, auditable and editable without retraining.

## Deploy to Vercel

1. Go to <https://vercel.com/new> and import this GitHub repository. Vercel detects FastAPI automatically, so you don't need to change any settings.
2. Before clicking **Deploy**, open **Environment Variables** and add:
   - `HF_TOKEN`: a Hugging Face token. Create a **fine-grained** token at <https://huggingface.co/settings/tokens> and tick **Make calls to Inference Providers**.
3. Click **Deploy**, then open the URL on your phone. To install it like an app, use **Share → Add to Home Screen**.

Every push to `main` redeploys automatically. Optional variables (`VLM_MODELS`, `CONFIDENCE_THRESHOLD`, `DEFAULT_REGION`) are listed in [`.env.example`](.env.example).

On Vercel, request logs, corrections and feedback appear in **Project → Logs** as JSON lines; the filesystem there is read-only. They are kept only for Vercel's log-retention period, so connect a database if you need to keep them permanently.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env            # macOS/Linux: cp .env.example .env
```

Put your token in `.env` and run:

```bash
python run.py
```

It prints a phone URL you can open on the same Wi-Fi. If you set `MOCK_VLM=1` in `.env`, the UI runs without a token and returns canned answers.

## How it works

| Step | File | What happens |
|---|---|---|
| Capture | `static/app.js` | The phone camera opens via `<input capture>`. The photo is shrunk to 1280 px JPEG before upload, which keeps it well under Vercel's 4.5 MB limit. |
| Identify | `app/vlm.py` | The photo and optional text go to a VLM: `google/gemma-4-26B-A4B-it` by default, with `Qwen/Qwen3-VL-8B-Instruct` as fallback. The model must pick `item_type`, `material` and `condition` from `data/taxonomy.json`. |
| Confidence gate | `app/main.py` → `confirmation()` | The user is asked to confirm if the item is unknown, if confidence is below `CONFIDENCE_THRESHOLD`, or if a hazardous alternative (for example a battery) is possible. |
| Decide | `app/rules.py` | If any matching row is hazardous, hazardous rows win. Otherwise the most specific row wins (region > item type > condition > material). |
| Where to take it | `data/collection_points.csv` | Per-region drop-off text, for example Haritha Karma Sena in Kerala. A rule can override it, as medicines do with pharmacy take-back. |
| Learn | logs | Requests, user corrections and 👍/👎 feedback are logged. Photos are never stored. |

**UI extras:**
- Example items to try without a camera.
- A "Know your bins" guide.
- Live loading steps with eco tips.
- Confidence meter.
- Tick-off preparation checklist.
- Share button.
- Haptic feedback on hazardous items.
- "Your impact" counter and recent scans, stored only in your browser.
- Light and dark mode.
- Installable to the home screen.

## Editing the rules

Open `data/rules.csv` in Excel or Google Sheets. `*` means "any". Separate prep steps with `|`. Then run `pytest`. The `test_rules_table_is_healthy` test catches duplicate keys, typos in categories, missing sources, and item types without a rule.

> The rules cite the Solid Waste Management Rules 2016, E-Waste (Management) Rules 2022, Battery Waste Management Rules 2022, Bio-Medical Waste Management Rules 2016 and Plastic Waste Management Rules 2016. Verify them against your local body's current practice.

## Project layout

```
app/          main.py (API + entrypoint) · vlm.py (Stage 1) · rules.py (Stage 2) · taxonomy.py
data/         taxonomy.json · rules.csv · collection_points.csv
static/       index.html · app.js · styles.css (mobile UI)
tests/        rules, VLM-reply parsing, API
pyproject.toml, vercel.json, .python-version   Vercel config (entrypoint app.main:app, 60 s limit, Python 3.12)
run.py        local server that prints the phone URL
```

## Team

Dinesh · Rupesh · Sailesh · Jitin Sai
