"""launch_sweep.py — distribute Phase 4 sweep configs across two GB10 boxes.

For each config in sweep.yaml:
  1. Pick the next available host round-robin
  2. ssh into that host and `nohup` run `run_phase4.sh --config-name=NAME ...`
  3. Stream remote stdout/stderr to a local log file via the ssh pipe
  4. Wait for all to finish, then summarize

Configs run with `nohup` on the remote so an SSH disconnect doesn't kill the job.
The ssh pipe stays alive only to capture logs; the remote process is independent.
"""

import argparse
import os
import subprocess
import sys
import threading
import time

import yaml


def shell_quote(s: str) -> str:
    return "'" + str(s).replace("'", "'\"'\"'") + "'"


def args_to_cli(defaults: dict, overrides: dict) -> list[str]:
    """Merge defaults + overrides. argparse expects --kebab-case for flags;
    YAML uses snake_case for readability — convert here."""
    merged = dict(defaults)
    merged.update(overrides)
    out = []
    for key, val in merged.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                out.append(flag)
        else:
            out.extend([flag, str(val)])
    return out


def run_serial_on_host(host: str, jobs: list[tuple[str, list[str]]],
                       log_dir: str,
                       remote_dir: str = "/home/wilkie/code/nubun"
                       ) -> tuple[subprocess.Popen, list, str]:
    """Run a sequence of configs on the same host, one after the other.

    jobs: list of (name, cli_args). Each is run as a separate `run_phase4.sh`
    invocation, chained with `;` (so a failure of one doesn't block the rest).
    Logs go to logs/sweep/{host}.serial.log; per-config tee'd into
    logs/sweep/{name}.log so we still get one log per config.

    Returns (popen, [(name, log_path) ...], host_log_path)
    """
    chain = []
    per_config_logs = []
    for name, args in jobs:
        log_path = os.path.join(log_dir, f"{name}.log")
        per_config_logs.append((name, log_path))
        quoted_args = " ".join(shell_quote(a) for a in args)
        # Each step writes its own log via tee, but also flows through to the host log.
        chain.append(
            f"echo '====== START {name} (host {host}) ======' && "
            f"cd {remote_dir} && bash run_phase4.sh --config-name={shell_quote(name)} "
            f"{quoted_args} 2>&1 | tee {shell_quote(remote_dir + '/' + log_path)}; "
            f"echo '====== END {name} ======'"
        )
    remote_cmd = " ; ".join(chain)
    if host in ("localhost", "127.0.0.1"):
        cmd = ["bash", "-lc", remote_cmd]
    else:
        cmd = ["ssh", "-o", "BatchMode=yes", host, remote_cmd]

    host_log_path = os.path.join(log_dir, f"{host.replace('/', '_')}.serial.log")
    host_log = open(host_log_path, "w")
    host_log.write(f"# host: {host}\n# configs: {[n for n, _ in jobs]}\n")
    host_log.write(f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    host_log.flush()
    p = subprocess.Popen(cmd, stdout=host_log, stderr=subprocess.STDOUT, text=True)
    return p, per_config_logs, host_log_path


def wait_for_all(host_jobs: list[tuple], poll_seconds: float = 30):
    """host_jobs: list of (host, popen, per_config_logs, host_log_path)."""
    while True:
        any_running = False
        for host, p, _, _ in host_jobs:
            if p.poll() is None:
                any_running = True
        if not any_running:
            return
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="scripts/sweep.yaml")
    parser.add_argument("--hosts", default="localhost,spark2",
                        help="Comma-separated hosts (round-robin assignment)")
    parser.add_argument("--remote-dir", default="/home/wilkie/code/nubun")
    parser.add_argument("--log-dir", default="logs/sweep")
    parser.add_argument("--only", default="",
                        help="Comma-separated subset of config names")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        sweep = yaml.safe_load(f)
    defaults = sweep.get("defaults", {})
    configs = sweep.get("configs", [])
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in configs if c["name"] in wanted]

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    if not hosts:
        sys.exit("no hosts")

    os.makedirs(args.log_dir, exist_ok=True)

    print(f"Sweep: {len(configs)} configs across {len(hosts)} hosts")
    print(f"  hosts: {hosts}")
    print(f"  log dir: {args.log_dir}")
    print()

    # Group configs per host (round-robin assignment, each host runs serially)
    by_host: dict[str, list[tuple[str, list[str]]]] = {h: [] for h in hosts}
    for i, cfg in enumerate(configs):
        host = hosts[i % len(hosts)]
        cli_args = args_to_cli(defaults, cfg.get("args", {}))
        by_host[host].append((cfg["name"], cli_args))

    print("Per-host serial schedule:")
    for host, jobs in by_host.items():
        print(f"  {host}:")
        for name, cli_args in jobs:
            print(f"    {name}")
            if args.dry_run:
                print(f"      cmd: bash run_phase4.sh --config-name={name} {' '.join(cli_args)}")

    if args.dry_run:
        print("\n--dry-run: no jobs launched")
        return

    print("\nLaunching one serial chain per host...")
    host_jobs = []
    for host, jobs in by_host.items():
        if not jobs:
            continue
        p, per_config_logs, host_log = run_serial_on_host(
            host, jobs, args.log_dir, args.remote_dir)
        names = ", ".join(n for n, _ in jobs)
        print(f"  {host:>10s}  pid {p.pid}  → {host_log}  (configs: {names})")
        host_jobs.append((host, p, per_config_logs, host_log))

    print(f"\nWaiting for {len(host_jobs)} host chain(s) to finish...")
    wait_for_all(host_jobs)
    print("\nAll done.")

    # Summary
    print("\nResult summary:")
    for host, p, per_config_logs, host_log in host_jobs:
        rc = p.returncode
        host_status = "OK" if rc == 0 else f"FAIL({rc})"
        print(f"  Host {host}: {host_status}")
        for name, log_path in per_config_logs:
            final_acc = "?"
            try:
                with open(log_path) as f:
                    tail = f.read().splitlines()[-200:]
                for line in reversed(tail):
                    if "same-lang avg acc:" in line:
                        final_acc = line.split(":")[1].strip()
                        break
            except Exception:
                pass
            print(f"    {name:>20s}  same-lang={final_acc}")


if __name__ == "__main__":
    main()
