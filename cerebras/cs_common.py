"""Shared CS-3 run-configuration helpers for train_cstorch.py / minimal_probe.py.

Everything here is CLIENT-side run plumbing. None of it touches the traced
graph, so changing it never invalidates the cluster's compile cache.
"""
import os

import cerebras.pytorch as cstorch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def add_cluster_args(ap, default_label="name=nubun", default_job_time_sec=82800):
    """CLI flags shared by every CS-3 entry point."""
    ap.add_argument("--num-csx", type=int, default=1,
                    help="Number of CS-3 systems to request (ALCF has 4).")
    ap.add_argument("--num-workers-per-csx", type=int, default=1,
                    help="streaming input workers per CS-3 (ClusterConfig default 1); "
                         "raise for large batches if the log reports input starvation")
    ap.add_argument("--job-time-sec", type=int, default=default_job_time_sec,
                    help="Wall-clock cap in seconds (ALCF hard limit is 24h).")
    ap.add_argument("--job-label", default=default_label,
                    help="label=value tag for `csctl get jobs` filtering.")
    ap.add_argument("--mount-dirs", default=None,
                    help="Comma-separated dirs the workers mount (repo + data). "
                         "Defaults to the repo root. Every dir must exist.")
    ap.add_argument("--python-paths", default=None,
                    help="Comma-separated PYTHONPATH dirs for the workers "
                         "(must include the repo root so `import vqvae` works).")
    ap.add_argument("--artifact-dir", default=None,
                    help="Per-run client artifact dir (run_meta.json, cirh.mlir, "
                         "track.json). Default: cerebras_logs/ (overwritten per run).")
    ap.add_argument("--skip-image-build", action="store_true",
                    help="Skip the custom worker-image build (it always fails on "
                         "ALCF: no internet from the build pod). Also skips "
                         "automount configuration, so verify on the canary first.")
    return ap


def configure_debug_args(skip_image_build=False):
    """Client-side appliance debug args (cstorch.backends.csx.debug.debug_args).

    - auto_log_export_policy=ENABLED: the appliance exports logs itself on an
      FS stall. It does NOT cover ordinary failures — always run
      `cerebras/export_logs.sh <wsjob>` promptly after any failed job; job
      records expire from the cluster in about a week.
    - skip_image_build: optional, see add_cluster_args.
    """
    du = cstorch.backends.csx.debug.debug_args.debug_usr
    # enum cerebras.DebugArgs.DebugUSR.AutoLogExportPolicy: ENABLED == 2
    du.auto_log_export_policy = du.DESCRIPTOR.fields_by_name[
        "auto_log_export_policy"].enum_type.values_by_name[
        "AUTO_LOG_EXPORT_POLICY_ENABLED"].number
    du.skip_image_build = bool(skip_image_build)


def make_cluster_config(args):
    """ClusterConfig from the shared CLI flags. mgmt_address/credentials resolve
    from the ALCF user node's /opt/cerebras/config_v2; we never set them.

    disable_mount_check=False: the client validates every mount dir against the
    cluster's volumes BEFORE submitting. (The default, True, silently accepted a
    nonexistent $REPO/data mount in July 2026 — the leading suspect for the
    real graph's only execute-mode death.)
    """
    mount_dirs = ([d for d in args.mount_dirs.split(",") if d]
                  if args.mount_dirs else [REPO_ROOT])
    python_paths = ([d for d in args.python_paths.split(",") if d]
                    if args.python_paths else [REPO_ROOT])
    for d in mount_dirs + python_paths:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"mount/python path does not exist: {d}")
    return cstorch.distributed.ClusterConfig(
        num_csx=args.num_csx,
        num_workers_per_csx=getattr(args, "num_workers_per_csx", 1),
        job_labels=[args.job_label],
        job_time_sec=args.job_time_sec,
        mount_dirs=mount_dirs,
        python_paths=python_paths,
        disable_mount_check=False,
    )


def make_backend(args, **kwargs):
    """cstorch.backend('CSX', ...) with the shared cluster config, artifact dir
    and debug args applied. Extra kwargs: compile_only=, validate_only=."""
    configure_debug_args(skip_image_build=getattr(args, "skip_image_build", False))
    artifact_dir = getattr(args, "artifact_dir", None)
    if artifact_dir:
        os.makedirs(artifact_dir, exist_ok=True)
    return cstorch.backend("CSX", artifact_dir=artifact_dir,
                           cluster_config=make_cluster_config(args), **kwargs)
