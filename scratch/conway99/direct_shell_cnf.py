#!/usr/bin/env python3
import argparse,itertools,hashlib,json,time
from pysat.formula import IDPool
from pysat.card import CardEnc,EncType

N=99

def parts(n,m=None):
    if n==0:
        yield ();return
    if m is None or m>n:m=n
    for x in range(m,0,-1):
        for r in parts(n-x,x):yield(x,)+r

def matching(part):
    pairs=[(3+2*i,4+2*i) for i in range(6)];out=[];s=0
    for m in part:
        b=pairs[s:s+m]
        if m==1:out.append(b[0])
        else:
            for r in range(m):out.append((b[r][1],b[(r+1)%m][0]))
        s+=m
    return [tuple(sorted(e)) for e in out]

def shell(part):
    H=[[0]*27 for _ in range(27)]
    def add(i,j):H[i][j]=H[j][i]=1
    add(0,1);add(0,2);add(1,2)
    for x in range(3,15):add(0,x);add(x,x+12)
    for x in range(3,15,2):add(x,x+1)
    for y in range(15,27):add(1,y)
    for x,z in matching(part):add(x+12,z+12)
    return H

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--branch',type=int,required=True,choices=range(1,12));ap.add_argument('--out',required=True);a=ap.parse_args()
    part=list(parts(6))[a.branch-1];H=shell(part)
    E=[[0]*N for _ in range(N)];v=1
    for i in range(N):
        for j in range(i+1,N):E[i][j]=E[j][i]=v;v+=1
    assert v-1==N*(N-1)//2
    pool=IDPool(start_from=v);body=a.out+'.body';f=open(body,'w');clauses=0
    def add(c):
        nonlocal clauses
        f.write(' '.join(map(str,c))+' 0\n');clauses+=1
    def exact(lits,k):
        enc=CardEnc.equals(lits=lits,bound=k,vpool=pool,encoding=EncType.seqcounter)
        for c in enc.clauses:add(c)
    # Fix the complete 27-vertex induced edge shell.
    for i in range(27):
        for j in range(i+1,27):add([E[i][j] if H[i][j] else -E[i][j]])
    # Degrees.
    for i in range(N):exact([E[i][j] for j in range(N) if j!=i],14)
    # For each pair: common-neighbor count plus adjacency is exactly 2.
    for i in range(N):
        for j in range(i+1,N):
            ps=[]
            for k in range(N):
                if k==i or k==j:continue
                z=pool.id(f'p_{i}_{j}_{k}');x=E[i][k];y=E[j][k]
                add([-z,x]);add([-z,y]);add([z,-x,-y]);ps.append(z)
            exact(ps+[E[i][j]],2)
    f.close();nv=pool.top
    with open(a.out,'w') as out:
        out.write(f'p cnf {nv} {clauses}\n')
        with open(body) as b:
            for line in b:out.write(line)
    import os;os.remove(body)
    rec={'branch':a.branch,'partition':part,'vertices':N,'edge_variables':N*(N-1)//2,'variables':nv,'clauses':clauses,'sha256':hashlib.sha256(open(a.out,'rb').read()).hexdigest()}
    print(json.dumps(rec,sort_keys=True),flush=True)
if __name__=='__main__':main()
