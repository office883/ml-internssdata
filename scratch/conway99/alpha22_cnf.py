#!/usr/bin/env python3
"""Complete CNF for the alpha=22 (Hoffman coclique) branch.

Vertices 0..21 form an independent set.  Equality in the Hoffman bound forces
every remaining vertex to have exactly four neighbours in it and ten neighbours
among vertices 22..98.  The common-neighbour equations are encoded exactly, so
SAT is equivalent to a full srg(99,14,1,2) in this branch.
"""
import argparse, hashlib, json, os
from pysat.formula import IDPool
from pysat.card import CardEnc, EncType
N=99; M=22

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); a=ap.parse_args()
    E=[[0]*N for _ in range(N)]; nxt=1
    for i in range(N):
        for j in range(i+1,N): E[i][j]=E[j][i]=nxt; nxt+=1
    pool=IDPool(start_from=nxt); body=a.out+'.body'; fh=open(body,'w'); clauses=0
    def add(c):
        nonlocal clauses
        fh.write(' '.join(map(str,c))+' 0\n'); clauses+=1
    def exact(lits,k):
        enc=CardEnc.equals(lits=list(lits),bound=k,vpool=pool,encoding=EncType.seqcounter)
        for c in enc.clauses:add(c)
    # Coclique.
    for i in range(M):
        for j in range(i+1,M):add([-E[i][j]])
    # Each coclique point has 14 outside neighbours.  Every outside point has
    # four coclique neighbours and ten outside neighbours.
    for i in range(M):exact((E[i][j] for j in range(M,N)),14)
    for u in range(M,N):
        exact((E[i][u] for i in range(M)),4)
        exact((E[u][v] for v in range(M,N) if v!=u),10)
    # Unified SRG equation: common(i,j)+edge(i,j)=2.
    for i in range(N):
        for j in range(i+1,N):
            terms=[E[i][j]]
            for k in range(N):
                if k==i or k==j: continue
                z=pool.id(f'p_{i}_{j}_{k}'); x=E[i][k]; y=E[j][k]
                add([-z,x]); add([-z,y]); add([z,-x,-y]); terms.append(z)
            exact(terms,2)
    fh.close(); nv=pool.top
    with open(a.out,'w') as out:
        out.write(f'p cnf {nv} {clauses}\n')
        with open(body) as src:
            for line in src: out.write(line)
    os.remove(body)
    rec={'vertices':N,'independent_set':M,'outside':N-M,'variables':nv,
         'clauses':clauses,'sha256':hashlib.sha256(open(a.out,'rb').read()).hexdigest()}
    print(json.dumps(rec,sort_keys=True),flush=True)
if __name__=='__main__':main()
