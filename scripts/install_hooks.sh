#!/bin/sh
# Install git hooks that run scripts/check_sensitive.py before every commit, on every commit message, and before every push.
set -e
cd "$(git rev-parse --show-toplevel)"
for h in pre-commit:--staged commit-msg:--msg pre-push:--push; do
  name=${h%%:*}; mode=${h#*:}
  cat > .git/hooks/$name <<HOOK
#!/bin/sh
PY=.venv/bin/python; [ -x \$PY ] || PY=python3
exec \$PY scripts/check_sensitive.py $mode "\$@"
HOOK
  chmod +x .git/hooks/$name
done
echo "hooks installed: pre-commit, commit-msg, pre-push"
