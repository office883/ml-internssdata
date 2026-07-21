#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
apt-get update -qq
apt-get install -y -qq git cmake g++ make libboost-program-options-dev libboost-graph-dev python3 python3-pip python3-venv time curl
cd /tmp
git clone --recursive --depth 1 https://github.com/markirch/sat-modulo-symmetries sms
cd sms
chmod +x build-and-install.sh
./build-and-install.sh -l >/tmp/sms_build.log 2>&1
test -x build/src/smsg
python3 -m venv /tmp/v
/tmp/v/bin/pip -q install 'python-sat>=1.8.dev13'
curl -fsSL 'https://raw.githubusercontent.com/office883/ml-internssdata/conway99-sat-20260722/scratch/conway99/direct_shell_cnf.py' -o /tmp/direct_shell_cnf.py
mkdir -p /tmp/conwaysms
for b in $BRANCHES; do /tmp/v/bin/python /tmp/direct_shell_cnf.py --branch "$b" --out "/tmp/conwaysms/b${b}.cnf" | tee "/tmp/conwaysms/b${b}.meta"; done
# 27 singleton cells preserve the fixed labelled shell.  Only the 72 outside
# vertices may be permuted by SMS, which is WLOG because they are unlabeled.
PARTITION='1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 72'
cat > /tmp/run_sms_one.sh <<'SH2'
#!/usr/bin/env bash
set -uo pipefail
b="$1";log="/tmp/conwaysms/b${b}.sms.log";tim="/tmp/conwaysms/b${b}.sms.time"
set +e
/usr/bin/time -v timeout --signal=INT --kill-after=20s "${SOLVE_SECONDS}s" /tmp/sms/build/src/smsg -v 99 --dimacs "/tmp/conwaysms/b${b}.cnf" --initial-partition $PARTITION --frequency 1 --cutoff 0 --timeout "$SOLVE_SECONDS" >"$log" 2>"$tim"
rc=$?
set -e
status=$(grep -E 'Result:|s SATISFIABLE|s UNSATISFIABLE|Instance is unknown' "$log" | tail -3 | tr '\n' ';' || true)
echo "SMS_FIXED_SHELL_BRANCH $b RC $rc STATUS ${status:-NONE}" | tee "/tmp/conwaysms/b${b}.summary"
tail -40 "$log" || true
tail -25 "$tim" || true
SH2
export PARTITION
chmod +x /tmp/run_sms_one.sh
printf '%s\n' $BRANCHES | xargs -P"$PARALLEL" -n1 /tmp/run_sms_one.sh
cat /tmp/conwaysms/b*.summary
sha256sum /tmp/conwaysms/* | sort
cat /tmp/conwaysms/b*.summary
