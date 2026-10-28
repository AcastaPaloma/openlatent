#!/usr/bin/env bash
# Mirror the research record to the HF checkpoint repo (runs/<name>/ is uploaded by each run itself):
# docs/journal.md, results/*.md, results/dashboard.json, results/figures/, results/audit/, configs/.
# Runs on a VM (it holds the HF token); pulls the branch first.  Usage: bash scripts/push_hf.sh <vm-ip>
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/hosts.sh"
IP="${1:-158.101.16.146}"
$SSH ubuntu@$IP 'cd ~/openlatent/research/privacy-obfuscation && git pull -q --ff-only && python3 - <<EOF
from huggingface_hub import HfApi
from ppv.run import HF_REPO
api = HfApi()
api.upload_folder(repo_id=HF_REPO, folder_path=".", path_in_repo="record",
                  allow_patterns=["docs/journal.md", "results/*.md", "results/dashboard.json", "results/figures/*.png",
                                  "results/audit/*", "configs/**/*.json"],
                  commit_message="research record sync")
print("synced research record to", HF_REPO)
EOF'
