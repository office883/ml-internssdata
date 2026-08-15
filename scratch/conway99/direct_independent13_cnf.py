#!/usr/bin/env python3
"""Complete direct CNF for srg(99,14,1,2) with vertices 0..12 independent.

The normalization is WLOG: for any edge xy, the 12 neighbors of x but not y
and the 12 neighbors of y but not x induce two internal perfect matchings and
one cross perfect matching.  Their union is a disjoint union of even cycles,
so one bipartition class is independent of size 12.  The unique common
neighbor of x,y is nonadjacent to all 24 cycle vertices, extending it to an
independent 13-set.
"""
import argparse,itertools,hashlib,json,os
from pysat.formula import IDPool
from pysat.card import CardEnc,EncType
N=99

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    E=[[0]*N for _ in range(N)];v=1
    for i in range(N):
        for j in range(i+1,N):E[i][j]=E[j][i]=v;v+=1
    pool=IDPool(start_from=v); body=a.out+'.body'; f=open(body,'w'); clauses=0
    def add(c):
        nonlocal clauses
        f.write(' '.join(map(str,c))+' 0\n');clauses+=1
    def exact(lits,k):
        enc=CardEnc.equals(lits=lits,bound=k,vpool=pool,encoding=EncType.seqcounter)
        for c in enc.clauses:add(c)
    def atmost(lits,k):
        enc=CardEnc.atmost(lits=lits,bound=k,vpool=pool,encoding=EncType.seqcounter)
        for c in enc.clauses:add(c)
    # Guaranteed independent set.
    for i in range(13):
        for j in range(i+1,13):add([-E[i][j]])
    # Exact degrees.
    for i in range(N):exact([E[i][j] for j in range(N) if j!=i],14)
    # Any outside vertex sees at most one endpoint of each local edge in N(v),
    # hence at most seven independent-set vertices.  This is redundant but
    # useful propagation and follows from lambda=1.
    for vtx in range(13,N):atmost([E[i][vtx] for i in range(13)],7)
    # Exact adjacent/nonadjacent common-neighbor equation in unified form:
    # |N(i) cap N(j)| + edge(i,j) = 2.
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
    os.remove(body)
    rec={'vertices':N,'independent_set':13,'edge_variables':N*(N-1)//2,'variables':nv,'clauses':clauses,'sha256':hashlib.sha256(open(a.out,'rb').read()).hexdigest()}
    print(json.dumps(rec,sort_keys=True),flush=True)
if __name__=='__main__':main()
