#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
apt-get update -qq
apt-get install -y -qq git cmake gcc g++ make libboost-program-options-dev libboost-graph-dev python3 python3-pip python3-venv time curl coreutils
python3 -m venv /tmp/v
/tmp/v/bin/pip -q install 'python-sat>=1.8.dev13'
cd /tmp
git clone --depth 1 https://github.com/arminbiere/kissat.git
cd kissat && ./configure >/tmp/kissat_config.log && make -j"$(nproc)" >/tmp/kissat_build.log
cd /tmp
git clone --recursive --depth 1 https://github.com/markirch/sat-modulo-symmetries sms
cd sms && chmod +x build-and-install.sh && ./build-and-install.sh -l >/tmp/sms_build.log 2>&1
test -x /tmp/sms/build/src/smsg
curl -fsSL 'https://raw.githubusercontent.com/office883/ml-internssdata/conway99-sat-20260722/scratch/conway99/direct_independent13_cnf.py' -o /tmp/direct_independent13_cnf.py
mkdir -p /tmp/i13
/tmp/v/bin/python /tmp/direct_independent13_cnf.py --out /tmp/i13/conway_i13.cnf | tee /tmp/i13/meta.json
sha256sum /tmp/i13/conway_i13.cnf
# Run plain Kissat and symmetry-aware SMS in parallel.  The initial partition
# S_13 x S_86 is WLOG because the first cell is only required to be an
# independent 13-set and all other vertices are unlabeled.
set +e
(
 /usr/bin/time -v timeout --signal=INT --kill-after=20s "${SOLVE_SECONDS}s" /tmp/kissat/build/kissat --no-colors /tmp/i13/conway_i13.cnf > /tmp/i13/kissat.out 2> /tmp/i13/kissat.time
 echo $? > /tmp/i13/kissat.rc
) & p1=$!
(
 /usr/bin/time -v timeout --signal=INT --kill-after=20s "${SOLVE_SECONDS}s" /tmp/sms/build/src/smsg -v 99 --dimacs /tmp/i13/conway_i13.cnf --initial-partition 13 86 --frequency 1 --cutoff 0 --timeout "$SOLVE_SECONDS" > /tmp/i13/sms.out 2> /tmp/i13/sms.time
 echo $? > /tmp/i13/sms.rc
) & p2=$!
wait $p1;wait $p2
set -e
echo '=== KISSAT ==='
grep -E '^s |^v ' /tmp/i13/kissat.out | tail -5 || true
tail -35 /tmp/i13/kissat.out || true
tail -25 /tmp/i13/kissat.time || true
echo '=== SMS ==='
grep -E 'Result:|s SATISFIABLE|s UNSATISFIABLE|Instance is unknown' /tmp/i13/sms.out | tail -5 || true
tail -40 /tmp/i13/sms.out || true
tail -25 /tmp/i13/sms.time || true
echo '=== SUMMARY ==='
echo "KISSAT_RC $(cat /tmp/i13/kissat.rc 2>/dev/null || echo missing) STATUS $(grep '^s ' /tmp/i13/kissat.out | tail -1 || echo NONE)"
echo "SMS_RC $(cat /tmp/i13/sms.rc 2>/dev/null || echo missing) STATUS $(grep 'Result:' /tmp/i13/sms.out | tail -1 || echo NONE)"
sha256sum /tmp/i13/* | sort
