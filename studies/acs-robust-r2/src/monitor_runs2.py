"""Training monitor for acs-robust-r2 (F1 + C1 manual-loop runs).

Polls each run's manual/result.json every ~120 s; appends one compact line per
NEW training iteration to logs/monitor.log. EXITS (0) as soon as EITHER
train_robust2.py process is gone (completion or crash) so the session gets a
notification; exits (1) immediately on an error signature in a driver log.
Start AFTER both trainings are launched:
    python monitor_runs2.py <pid_f1> <pid_c1>
"""
import json
import os
import sys
import time

RUNS = {
    "F1": "/workspace/test_results/c2F1_ft60_lmix_260808/manual",
    "C1": "/workspace/test_results/c2C1_ft40_lmix_260808/manual",
}
LOG = "/workspace/studies/acs-robust-r2/logs/monitor.log"
DRIVER_LOGS = {
    "F1": "/workspace/studies/acs-robust-r2/logs/train_f1.log",
    "C1": "/workspace/studies/acs-robust-r2/logs/train_c1.log",
}
ERR_SIGS = ("Traceback (most recent call last)", "ERROR trial_runner",
            "CUDA out of memory", "RayTaskError")


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def last_metrics(run_dir):
    path = os.path.join(run_dir, "result.json")
    if not os.path.exists(path):
        return None
    last = None
    with open(path, errors="ignore") as f:
        for line in f:
            if line.strip():
                last = line
    if last is None:
        return None
    try:
        d = json.loads(last)
    except json.JSONDecodeError:
        return None
    cm = d.get("custom_metrics") or {}
    ev = (d.get("evaluation") or {}).get("custom_metrics") or {}
    return dict(
        it=d.get("training_iteration"),
        len=d.get("episode_len_mean"),
        succ=cm.get("c2_success_mean"),
        J=cm.get("J_success_mean"),
        ev_succ=ev.get("c2_success_mean"),
        ev_J=ev.get("J_success_mean"),
    )


def fmt(m):
    def r(x, n=0):
        return "-" if x is None else round(float(x), n)
    s = f"it{m['it']:>3} len {r(m['len'])} succ {r(m['succ'], 2)} J {r(m['J'])}"
    if m.get("ev_succ") is not None:
        s += f" | eval {r(m['ev_succ'], 2)}/{r(m['ev_J'])}"
    return s


def main():
    pids = {"F1": int(sys.argv[1]), "C1": int(sys.argv[2])}
    seen = {"F1": -1, "C1": -1}
    with open(LOG, "a") as log:
        log.write(f"--- monitor started {time.strftime('%F %T')} pids {pids} ---\n")
        log.flush()
        while True:
            for tag, run_dir in RUNS.items():
                m = last_metrics(run_dir)
                if m and m["it"] != seen[tag]:
                    seen[tag] = m["it"]
                    log.write(f"[{time.strftime('%T')}] {tag} {fmt(m)}\n")
                    log.flush()
                dl = DRIVER_LOGS[tag]
                if os.path.exists(dl):
                    with open(dl, errors="ignore") as f:
                        txt = f.read()[-20000:]
                    if any(s in txt for s in ERR_SIGS):
                        log.write(f"[{time.strftime('%T')}] {tag} ERROR SIGNATURE in {dl}\n")
                        log.flush()
                        print(f"{tag}: error signature — check {dl}")
                        sys.exit(1)
            dead = [t for t, p in pids.items() if not alive(p)]
            if dead:
                log.write(f"[{time.strftime('%T')}] process exit: {dead} — monitor done\n")
                log.flush()
                print(f"exited: {dead}; last iters {seen}")
                sys.exit(0)
            time.sleep(120)


if __name__ == "__main__":
    main()
