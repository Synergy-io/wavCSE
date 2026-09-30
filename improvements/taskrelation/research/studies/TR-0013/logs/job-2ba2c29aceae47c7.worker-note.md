# Failed attempt 2 — job-2ba2c29aceae47c7 (arm `mssl-l2-0p1`)

`infra job logs --local` holds no copy for this job: the ARC runner was killed while
starting the command, before it recorded a log or an outcome. What the provider and the
worker held at the time is recorded here.

**Provider record (`infra job status job-2ba2c29aceae47c7 --json`):**

```
state            = FAILED
exit_code        = None
failure_reason   = worker al5b4m763ci2bu recorded a job process for
                   job-2ba2c29aceae47c7 that is no longer running, and recorded no
                   outcome; the command result cannot be verified
preparation_phase = starting_command
finished_at      = 2026-09-29T23:34:58.597119Z
```

**Worker evidence (read directly on the worker, 2026-09-29):**

```
$ df -h /            -> overlay 100G 100G 176K 100% /     (container disk full)
$ du -sh /workspace/wavcse-jobs/*
    45G  /workspace/wavcse-jobs/job-2ba2c29aceae47c7
    50G  /workspace/wavcse-jobs/job-3fb19bd3a1dd4493
$ tail /workspace/wavcse-jobs/job-2ba2c29aceae47c7/logs/*.log
    with open(temporary, "w", encoding="utf-8") as handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    OSError: [Errno 28] No space left on device:
        '/workspace/wavcse-jobs/job-2ba2c29aceae47c7/state/pid.json.partial.2284'
```

**Diagnosis.** The 100 GB container disk held two job workspaces at once — the first
attempt's (~50 GB, retained after it failed) and this one's (~45 GB) — so the runner could
not write its own state file and the command never produced an outcome. No training step,
no MLflow run and no output existed for this attempt. Removing the first attempt's scratch
freed 50 GB (`df` 51 % used). The runner's pid write is on the critical path *before* the
training command, so nothing scientific was affected.

**Disposition.** Pre-science infrastructure failure. The registered configuration is
unchanged and re-run; the orchestrator resets the entry with an audit event, and each job's
scratch is now removed as soon as its job is terminal, so one live workspace is the norm.
