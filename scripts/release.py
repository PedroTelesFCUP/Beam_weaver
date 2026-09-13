"""Plan publication or release-note updates without changing GitHub state."""

import ast
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def project_version():
    """Resolve the version declared by pyproject without importing the package."""
    config = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    version = config["project"].get("version")
    if version is None:
        attribute = config["tool"]["setuptools"]["dynamic"]["version"]["attr"]
        module, name = attribute.rsplit(".", 1)
        source = Path(*module.split(".")).with_suffix(".py")
        for node in ast.parse(source.read_text(encoding="utf-8")).body:
            targets = node.targets if isinstance(node, ast.Assign) else []
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                version = ast.literal_eval(node.value)
                break
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("A stable x.y.z project version is required for automatic release")
    return version


def github_get(path, *, missing_ok=False):
    """Only HTTP 404 means missing; permission and network errors must fail."""
    repository = os.environ["GITHUB_REPOSITORY"]
    request = Request(
        f"{os.environ.get('GITHUB_API_URL', 'https://api.github.com')}/repos/{repository}/{path}",
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.load(response)
    except HTTPError as error:
        if missing_ok and error.code == 404:
            return None
        raise


def release_plan():
    version = project_version()
    tag = f"v{version}"
    notes = Path("docs/releases") / f"{tag}.md"
    if not notes.is_file():
        print(f"No {notes}; this commit does not request a release.")
        return {"publish": "false", "update_notes": "false"}
    notes_text = notes.read_text(encoding="utf-8")
    if not notes_text.strip():
        raise ValueError(f"Release notes are empty: {notes}")

    tested_sha = os.environ["TESTED_SHA"]
    actual_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if actual_sha != tested_sha:
        raise ValueError("Checkout does not match the commit that passed Python checks")
    existing = github_get(f"releases/tags/{tag}", missing_ok=True)
    if existing is not None:
        if existing.get("draft"):
            raise ValueError(f"{tag} has an existing draft; inspect it before retrying publication")
        update_notes = existing.get("body", "") != notes_text
        print(f"{tag} is already published; release-note update needed: {update_notes}.")
        return {
            "publish": "false",
            "update_notes": "true" if update_notes else "false",
            "tag": tag,
            "notes": notes.as_posix(),
        }

    reference = github_get(f"git/ref/tags/{tag}", missing_ok=True)
    if reference is not None:
        target = reference["object"]
        # Annotated tags may point to another tag; compare the final commit.
        for _ in range(10):
            if target["type"] != "tag":
                break
            target = github_get(f"git/tags/{target['sha']}")["object"]
        if target["type"] != "commit" or target["sha"] != tested_sha:
            raise ValueError(f"Existing {tag} does not point to the tested commit; refusing to move it")

    print(f"Ready to publish {tag} from tested commit {tested_sha}.")
    return {"publish": "true", "update_notes": "false", "tag": tag, "notes": notes.as_posix()}


if __name__ == "__main__":
    outputs = release_plan()
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        for key, value in outputs.items():
            stream.write(f"{key}={value}\n")
