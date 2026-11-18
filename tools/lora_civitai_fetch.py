"""Look up every LoRA of a folder on Civitai by its file hash and save what Civitai says about it.

Run with ComfyUI's Python (no extra packages needed):

    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI-VisionPromptAssistant\\tools\\lora_civitai_fetch.py "D:\\ComfyUI_windows_portable\\ComfyUI\\models\\loras\\SexyAI_Studio"

Your Civitai API key is read from the CIVITAI_API_KEY environment variable, or asked once (hidden input). It is
only sent to civitai.com and never written to disk.

For each 'Name.safetensors' it writes 'Name.civitai.json' (model name, version, description, trigger words, tags,
base model, link). Renamed files are still found: the hash does not depend on the name. Nothing else is changed;
existing files are only skipped or refreshed. Hashes are cached in '_civitai_hashes.json' so a second run is fast.
"""
import getpass
import hashlib
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

API = "https://civitai.com/api/v1"


def sha256(path, cache):
    stat = os.stat(path)
    key = f"{os.path.basename(path)}|{stat.st_size}|{int(stat.st_mtime)}"
    if key in cache:
        return cache[key]
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    cache[key] = digest.hexdigest().upper()
    return cache[key]


def get(url, key):
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}",
                                                   "User-Agent": "VisionPromptAssistant-LoRA-notes/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def plain(text):
    """Civitai descriptions are HTML: keep readable text only."""
    text = re.sub(r"(?i)<br\s*/?>|</p>|</li>|</h\d>", "\n", str(text or ""))
    text = html.unescape(re.sub(r"<[^>]+>", "", text))
    return re.sub(r"\n\s*\n+", "\n", re.sub(r"[ \t]+", " ", text)).strip()


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    key = os.environ.get("CIVITAI_API_KEY") or getpass.getpass("Civitai API key (hidden): ").strip()
    if not key:
        print("No API key: stopping.")
        return
    cache_path = os.path.join(folder, "_civitai_hashes.json")
    try:
        cache = json.load(open(cache_path, encoding="utf-8"))
    except Exception:
        cache = {}
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".safetensors"))
    found = missing = 0
    for index, name in enumerate(files, 1):
        path = os.path.join(folder, name)
        base = os.path.splitext(path)[0]
        print(f"[{index}/{len(files)}] {name}: hashing...", end=" ", flush=True)
        try:
            digest = sha256(path, cache)
            json.dump(cache, open(cache_path, "w", encoding="utf-8"), indent=1)
            version = get(f"{API}/model-versions/by-hash/{digest}", key)
            model = {}
            try:
                model = get(f"{API}/models/{version.get('modelId')}", key)
            except Exception:
                pass
            info = {
                "file": name,
                "sha256": digest,
                "model_name": (version.get("model") or {}).get("name") or model.get("name"),
                "version_name": version.get("name"),
                "base_model": version.get("baseModel"),
                "trigger_words": version.get("trainedWords") or [],
                "tags": model.get("tags") or [],
                "model_description": plain(model.get("description"))[:4000],
                "version_description": plain(version.get("description"))[:2000],
                "url": f"https://civitai.com/models/{version.get('modelId')}?modelVersionId={version.get('id')}",
            }
            json.dump(info, open(base + ".civitai.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
            found += 1
            print(f"found: {info['model_name']} ({info['base_model']})")
        except urllib.error.HTTPError as error:
            missing += 1
            print("not on Civitai" if error.code == 404 else f"HTTP {error.code}")
        except Exception as error:
            missing += 1
            print(f"error: {error}")
        time.sleep(0.4)  # gentle with the API
    print(f"\nDone: {found} found, {missing} not found. Info saved next to each LoRA as '.civitai.json'.")


if __name__ == "__main__":
    main()
