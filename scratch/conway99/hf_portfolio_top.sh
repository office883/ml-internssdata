#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1
apt-get update -qq
apt-get install -y -qq git gcc g++ make python3 python3-pip python3-venv time coreutils curl
python3 -m venv /tmp/v
/tmp/v/bin/pip -q install 'numpy>=2.0' 'python-sat>=1.8.dev13'
cd /tmp
git clone --depth 1 https://github.com/arminbiere/kissat.git
cd kissat && ./configure >/tmp/kissat_config.log && make -j"$(nproc)" >/tmp/kissat_build.log
cd /tmp
curl -fsSL 'https://raw.githubusercontent.com/office883/ml-internssdata/conway99-sat-20260722/scratch/conway99/shell_full_sat_corrected.py' -o /tmp/shell_full_sat_corrected.py
mkdir -p /tmp/conwaytop
cat > /tmp/gen_one_top.py <<'PY'
import sys,hashlib,json,time
sys.path.insert(0,'/tmp')
from shell_full_sat_corrected import Encoder
b=int(sys.argv[1]); out=f'/tmp/conwaytop/b{b}.cnf'
t=time.time(); e=Encoder(b,'top').build(); e.cnf.to_file(out)
rec=dict(e.stats); rec['sha256']=hashlib.sha256(open(out,'rb').read()).hexdigest(); rec['total_generate_seconds']=time.time()-t
open(f'/tmp/conwaytop/b{b}.meta.json','w').write(json.dumps(rec,sort_keys=True,indent=2)+'\n')
print('GENERATED_TOP',b,json.dumps(rec,sort_keys=True),flush=True)
PY
cat > /tmp/verify_top.py <<'PY'
import sys,hashlib,json
sys.path.insert(0,'/tmp')
from shell_full_sat_corrected import Encoder
b=int(sys.argv[1]); fn=sys.argv[2]; model=[]
for line in open(fn,errors='replace'):
    if line.startswith('v '):
        model.extend(int(x) for x in line[2:].split() if int(x))
assert model
e=Encoder(b,'top').build(); B,D=e.verify(model,full=False)
ssel,psel,dedges=e.decode(model)
rec={'branch':b,'sigma':ssel,'phi':psel,'D_edges':dedges,'B_sha256':hashlib.sha256(B.tobytes()).hexdigest(),'D_sha256':hashlib.sha256(D.tobytes()).hexdigest()}
print('TOP_SAT_VERIFIED',json.dumps(rec,separators=(',',':')))
PY
for b in $BRANCHES; do /tmp/v/bin/python /tmp/gen_one_top.py "$b"; done
cat > /tmp/run_one_top.sh <<'SH2'
#!/usr/bin/env bash
set -uo pipefail
b="$1"; cnf="/tmp/conwaytop/b${b}.cnf"; log="/tmp/conwaytop/b${b}.solver.out"; tim="/tmp/conwaytop/b${b}.solver.time"
set +e
/usr/bin/time -v timeout --signal=INT --kill-after=20s "${SOLVE_SECONDS}s" /tmp/kissat/build/kissat --no-colors "$cnf" >"$log" 2>"$tim"
rc=$?
set -e
status=$(grep '^s ' "$log" | tail -1 || true)
echo "TOP_BRANCH $b RC $rc STATUS ${status:-NONE}" | tee "/tmp/conwaytop/b${b}.summary"
if grep -q '^s SATISFIABLE' "$log"; then /tmp/v/bin/python /tmp/verify_top.py "$b" "$log" | tee "/tmp/conwaytop/b${b}.verified"; fi
if grep -q '^s UNSATISFIABLE' "$log"; then echo "TOP_UNSAT_NO_PROOF_BRANCH $b" | tee "/tmp/conwaytop/b${b}.unsat"; fi
tail -20 "$log" || true
tail -20 "$tim" || true
SH2
chmod +x /tmp/run_one_top.sh
printf '%s\n' $BRANCHES | xargs -P"$PARALLEL" -n1 /tmp/run_one_top.sh
cat /tmp/conwaytop/b*.summary
sha256sum /tmp/conwaytop/* | sort
