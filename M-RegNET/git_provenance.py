"""
Git provenance for every run — single source of truth.

A checkpoint or results folder produced from an unknown/uncommitted code state is
undebuggable later (see the 20260520_170759 dead-checkpoint post-mortem: logged
0.854, reproduced 0.089, root cause unrecoverable for lack of a recorded SHA).
Every run-producing script (train_mri.py, compare_synthseg_vs_gt.py, inference.py,
run_inference_all.py) records its commit via this module.
"""

import os
import subprocess

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def git_provenance():
    """Return {'git_sha', 'git_dirty', 'git_branch'} for the repo containing this file."""
    def _git(*args):
        return subprocess.check_output(
            ['git', *args], cwd=_REPO_DIR, stderr=subprocess.DEVNULL,
        ).decode().strip()

    try:
        return {
            'git_sha': _git('rev-parse', 'HEAD'),
            'git_branch': _git('rev-parse', '--abbrev-ref', 'HEAD'),
            'git_dirty': bool(_git('status', '--porcelain')),
        }
    except Exception as e:
        return {'git_sha': f'UNKNOWN ({e})', 'git_branch': 'UNKNOWN', 'git_dirty': True}


def write_git_sha(output_dir, logger=None):
    """Write GIT_SHA.txt into output_dir and return the provenance dict.
    Loudly warns when the tree is dirty — a dirty run is not reproducible."""
    prov = git_provenance()
    dirty_note = "  *** DIRTY WORKING TREE — NOT reproducible ***" if prov['git_dirty'] else ""
    lines = [
        f"git_sha    : {prov['git_sha']}",
        f"git_branch : {prov['git_branch']}",
        f"git_dirty  : {prov['git_dirty']}{dirty_note}",
    ]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "GIT_SHA.txt"), 'w') as f:
        f.write("\n".join(lines) + "\n")

    emit = (logger.warning if (logger and prov['git_dirty'])
            else logger.info if logger else print)
    emit(" | ".join(lines))
    if prov['git_dirty']:
        (logger.warning if logger else print)(
            "WARNING: run launched from a DIRTY working tree — the exact code cannot "
            "be recovered from the recorded SHA. Commit before running."
        )
    return prov
