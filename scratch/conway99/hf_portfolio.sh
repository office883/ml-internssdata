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
mkdir -p /tmp/conway
cat > /tmp/gen_one.py <<'PY'
import sys,hashlib,json,time
sys.path.insert(0,'/tmp')
from shell_full_sat_corrected import Encoder
b=int(sys.argv[1]); out=f'/tmp/conway/b{b}.cnf'
t=time.time(); e=Encoder(b,'full').build(); e.cnf.to_file(out)
rec=dict(e.stats); rec['sha256']=hashlib.sha256(open(out,'rb').read()).hexdigest(); rec['total_generate_seconds']=time.time()-t
open(f'/tmp/conway/b{b}.meta.json','w').write(json.dumps(rec,sort_keys=True,indent=2)+'\n')
print('GENERATED',b,json.dumps(rec,sort_keys=True),flush=True)
PY
cat > /tmp/verify_sat.py <<'PY'
import sys,hashlib,json,gzip,base64
sys.path.insert(0,'/tmp')
from shell_full_sat_corrected import Encoder
b=int(sys.argv[1]); fn=sys.argv[2]; model=[]
for line in open(fn,errors='replace'):
    if line.startswith('v '):
        for x in line[2:].split():
            z=int(x)
            if z:model.append(z)
assert model,'no model lines'
e=Encoder(b,'full').build(); A=e.verify(model,full=True)
raw='\n'.join(''.join(map(str,row.tolist())) for row in A).encode()+b'\n'
edges=[(i,j) for i in range(99) for j in range(i+1,99) if A[i,j]]
assert len(edges)==693
print('SAT_VERIFIED_BRANCH',b,'edges',len(edges),'matrix_sha256',hashlib.sha256(raw).hexdigest())
print('MATRIX_GZIP_BASE64',base64.b64encode(gzip.compress(raw,compresslevel=9,mtime=0)).decode())
print('EDGE_LIST_JSON',json.dumps(edges,separators=(',',':')))
PY
for b in $BRANCHES; do /tmp/v/bin/python /tmp/gen_one.py "$b"; done
cat > /tmp/run_one.sh <<'SH2'
#!/usr/bin/env bash
set -uo pipefail
b="$1"; cnf="/tmp/conway/b${b}.cnf"; log="/tmp/conway/b${b}.solver.out"; tim="/tmp/conway/b${b}.solver.time"
set +e
/usr/bin/time -v timeout --signal=INT --kill-after=20s "${SOLVE_SECONDS}s" /tmp/kissat/build/kissat --no-colors "$cnf" >"$log" 2>"$tim"
rc=$?
set -e
status=$(grep '^s ' "$log" | tail -1 || true)
echo "BRANCH $b RC $rc STATUS ${status:-NONE}" | tee "/tmp/conway/b${b}.summary"
if grep -q '^s SATISFIABLE' "$log"; then /tmp/v/bin/python /tmp/verify_sat.py "$b" "$log" | tee "/tmp/conway/b${b}.verified"; fi
if grep -q '^s UNSATISFIABLE' "$log"; then echo "UNSAT_NO_PROOF_BRANCH $b" | tee "/tmp/conway/b${b}.unsat"; fi
tail -30 "$log" || true
tail -25 "$tim" || true
SH2
chmod +x /tmp/run_one.sh
printf '%s\n' $BRANCHES | xargs -P"$PARALLEL" -n1 /tmp/run_one.sh
cat /tmp/conway/b*.summary
sha256sum /tmp/conway/* | sort
