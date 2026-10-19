# Execute from the extracted submission directory. This file contains no secrets.
$env:PYTHONUTF8 = '1'
$env:EMAIL_GUARD_ARTIFACT_ROOT = (Resolve-Path .\artifacts).Path
# CPU/E0 only; no neural packages required in .venv-app.
.\.venv-app\Scripts\python.exe -m app.run --disable-laya

# Optional Laya: first merge separately supplied, verified Laya files into artifacts.
# .\.venv-app\Scripts\python.exe scripts\prepare_model_artifacts.py --source ..\email_guard_laya_artifacts --destination artifacts --only-laya
# Then use an existing trusted runtime with versions from requirements-app-laya.lock.txt:
# $env:EMAIL_GUARD_LAYA_PYTHON = (Resolve-Path ..\trusted-laya-runtime\Scripts\python.exe).Path
# .\.venv-app\Scripts\python.exe -m app.run
