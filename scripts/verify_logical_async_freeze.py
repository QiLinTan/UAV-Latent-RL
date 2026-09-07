"""Verify frozen model/data archive and recoverable source dependencies."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.freeze_logical_async import sha256


def verify(manifest_path,require_tag=False):
    manifest=json.loads(Path(manifest_path).read_text())
    archive=Path(manifest["archive"]["path"])
    if sha256(archive)!=manifest["archive"]["sha256"]:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        for artifact in manifest["artifacts"]:
            payload=bundle.read(artifact["archive_member"])
            if len(payload)!=artifact["bytes"] or hashlib.sha256(payload).hexdigest()!=artifact["sha256"]:
                raise ValueError("Archive member mismatch: "+artifact["archive_member"])
        for source in manifest["source_files"]:
            relative=str(Path(source["path"]).relative_to(ROOT))
            if relative=="envs/forest/layout.py":
                payload=bundle.read("workspace_files/"+relative)
            else:
                payload=subprocess.check_output(["git","show",manifest["git_commit"]+":"+relative],cwd=ROOT)
            if hashlib.sha256(payload).hexdigest()!=source["sha256"]:
                raise ValueError("Frozen source mismatch: "+relative)
    if require_tag:
        tagged=subprocess.check_output(["git","show",manifest["git_tag"]+":docs/experiment_contracts/logical_async_reference_pid_v1.json"],cwd=ROOT)
        if json.loads(tagged)!=manifest:
            raise ValueError("Tagged manifest differs from working manifest")
    result={"archive_verified":True,"artifact_members":len(manifest["artifacts"]),"source_files_verified":len(manifest["source_files"]),"tag_verified":require_tag,"logical_only":not manifest["runtime_async"]}
    print(json.dumps(result))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default="docs/experiment_contracts/logical_async_reference_pid_v1.json")
    parser.add_argument("--require-tag",action="store_true")
    args=parser.parse_args()
    verify(args.manifest,args.require_tag)
