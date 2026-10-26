"""Collect 12-month maintenance metrics for GitHub repos via one GraphQL call each.

Usage: python scripts/repo_health.py docs/research/repos.txt health.json
Needs an authenticated `gh` CLI. Lines starting with # in the repo list are skipped.
"""
import datetime
import json
import subprocess
import sys

SINCE = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

QUERY = """
query($owner:String!, $name:String!, $opened:String!, $closed:String!, $merged:String!, $responded:String!) {
  repository(owner:$owner, name:$name) {
    nameWithOwner stargazerCount forkCount isArchived pushedAt
    licenseInfo { spdxId }
    defaultBranchRef { target { ... on Commit {
      committedDate
      history(since:"%sT00:00:00Z") { totalCount }
    } } }
    releases(last:1, orderBy:{field:CREATED_AT, direction:ASC}) { nodes { tagName publishedAt } }
    openIssues: issues(states:OPEN) { totalCount }
    mentionableUsers { totalCount }
  }
  opened: search(query:$opened, type:ISSUE) { issueCount }
  closed: search(query:$closed, type:ISSUE) { issueCount }
  merged: search(query:$merged, type:ISSUE) { issueCount }
  responded: search(query:$responded, type:ISSUE) { issueCount }
}
""" % SINCE


def health(full: str) -> dict:
    owner, name = full.split("/")
    q = f"repo:{full} is:issue"
    args = [
        "gh", "api", "graphql", "-f", f"query={QUERY}",
        "-F", f"owner={owner}", "-F", f"name={name}",
        "-F", f"opened={q} created:>={SINCE}",
        "-F", f"closed={q} closed:>={SINCE}",
        "-F", f"merged=repo:{full} is:pr is:merged merged:>={SINCE}",
        "-F", f"responded={q} created:>={SINCE} comments:>0",
    ]
    out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode:
        return {"repo": full, "error": out.stderr.strip()[:200]}
    d = json.loads(out.stdout)["data"]
    r = d["repository"]
    commit = (r["defaultBranchRef"] or {}).get("target") or {}
    rel = r["releases"]["nodes"]
    return {
        "repo": r["nameWithOwner"],
        "stars": r["stargazerCount"],
        "archived": r["isArchived"],
        "license": (r["licenseInfo"] or {}).get("spdxId"),
        "last_commit": (commit.get("committedDate") or "")[:10],
        "commits_12m": commit.get("history", {}).get("totalCount"),
        "issues_opened_12m": d["opened"]["issueCount"],
        "issues_with_reply_12m": d["responded"]["issueCount"],
        "issues_closed_12m": d["closed"]["issueCount"],
        "prs_merged_12m": d["merged"]["issueCount"],
        "open_issues": r["openIssues"]["totalCount"],
        "latest_release": f'{rel[-1]["tagName"]} {rel[-1]["publishedAt"][:10]}' if rel else None,
    }


if __name__ == "__main__":
    rows = [health(line.strip()) for line in open(sys.argv[1]) if line.strip() and not line.startswith("#")]
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
    for r in rows:
        if "error" in r:
            print(f'{r["repo"]:48} ERROR {r["error"]}')
            continue
        print(
            f'{r["repo"][:48]:48} ★{r["stars"]:>6} last={r["last_commit"]} c12={r["commits_12m"]!s:>5} '
            f'iss12={r["issues_opened_12m"]:>4} reply={r["issues_with_reply_12m"]:>4} closed={r["issues_closed_12m"]:>4} '
            f'pr12={r["prs_merged_12m"]:>4} lic={r["license"]} rel={r["latest_release"]} arch={r["archived"]}'
        )
