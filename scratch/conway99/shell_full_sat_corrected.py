#!/usr/bin/env python3
"""Complete SAT encoding of an srg(99,14,1,2) extending one of the 11 edge shells.

Vertices 0..26 form the fixed shell H.  The 72 outside vertices are canonically
labelled: 12 Z_x vertices by x in X and 60 W_e vertices by non-matching X-edge e.
The B incidence factorization is encoded by a matching sigma and an edge
bijection phi.  D is the graph on the 72 outside vertices.

mode=top enforces top-left and top-right block equations plus outside degrees.
mode=full additionally enforces every bottom-right off-diagonal equation, hence
is equivalent to the full SRG adjacency identity for the chosen shell type.
"""
import argparse, itertools, json, time, hashlib
from pathlib import Path
import numpy as np
from pysat.formula import CNF, IDPool
from pysat.card import CardEnc, EncType
from pysat.solvers import Solver


def parts(n,m=None):
    if n==0:
        yield (); return
    if m is None or m>n:m=n
    for x in range(m,0,-1):
        for r in parts(n-x,x):yield(x,)+r


def matching(part):
    pairs=[(3+2*i,4+2*i) for i in range(6)];o=[];s=0
    for m in part:
        b=pairs[s:s+m]
        if m==1:o.append(b[0])
        else:
            for r in range(m):o.append((b[r][1],b[(r+1)%m][0]))
        s+=m
    return [tuple(sorted(e)) for e in o]


def shell(part):
    H=np.zeros((27,27),dtype=np.int8)
    def add(i,j):H[i,j]=H[j,i]=1
    add(0,1);add(0,2);add(1,2)
    for x in range(3,15):add(0,x);add(x,x+12)
    for x in range(3,15,2):add(x,x+1)
    for y in range(15,27):add(1,y)
    for x,z in matching(part):add(x+12,z+12)
    return H


class Encoder:
    def __init__(self,branch,mode):
        self.branch=branch; self.part=list(parts(6))[branch-1]; self.mode=mode
        self.H=shell(self.part).astype(int)
        self.X=list(range(3,15)); self.Y=list(range(15,27))
        MX={tuple(sorted((x,x+1))) for x in range(3,15,2)}
        MY={tuple(sorted((x+12,z+12))) for x,z in matching(self.part)}
        self.EX=[e for e in itertools.combinations(self.X,2) if e not in MX]
        self.EY=[e for e in itertools.combinations(self.Y,2) if e not in MY]
        assert len(self.EX)==len(self.EY)==60
        self.G=12*np.eye(27,dtype=int)-self.H+2*np.ones((27,27),dtype=int)-self.H@self.H
        self.pool=IDPool(); self.cnf=CNF(); self.named={}; self.and_cache={}
        self.s={}; self.p={}; self.by={}; self.d={}
        self.stats={}

    def new(self,key):
        z=self.pool.id(str(key)); self.named[key]=z; return z

    def exact(self,terms,k):
        lits=[]; const=0
        for t in terms:
            if t is True: const+=1
            elif t is False: pass
            else: lits.append(int(t))
        k-=const
        if k<0 or k>len(lits): self.cnf.append([]); return
        if not lits:
            if k!=0:self.cnf.append([])
            return
        if k==0:
            self.cnf.extend([[-z] for z in lits]); return
        if k==len(lits):
            self.cnf.extend([[z] for z in lits]); return
        enc=CardEnc.equals(lits=lits,bound=k,vpool=self.pool,encoding=EncType.seqcounter)
        self.cnf.extend(enc.clauses)

    def AND(self,a,b,key):
        if a is False or b is False:return False
        if a is True:return b
        if b is True:return a
        a=int(a);b=int(b)
        if a==b:return a
        kk=(key,min(a,b),max(a,b))
        if kk in self.and_cache:return self.and_cache[kk]
        z=self.new(('and',key,len(self.and_cache)))
        self.cnf.append([-z,a]);self.cnf.append([-z,b]);self.cnf.append([z,-a,-b])
        self.and_cache[kk]=z
        return z

    def setup_factor(self):
        for x in self.X:
            for y in self.Y:
                if self.G[x,y]>=1:self.s[x,y]=self.new(('s',x,y))
        for ei,e in enumerate(self.EX):
            for fi,f in enumerate(self.EY):
                if all(self.G[x,y]>=1 for x in e for y in f):
                    self.p[ei,fi]=self.new(('p',ei,fi))
        for x in self.X:self.exact([self.s[x,y] for y in self.Y if (x,y) in self.s],1)
        for y in self.Y:self.exact([self.s[x,y] for x in self.X if (x,y) in self.s],1)
        for ei in range(60):self.exact([self.p[ei,fi] for fi in range(60) if (ei,fi) in self.p],1)
        for fi in range(60):self.exact([self.p[ei,fi] for ei in range(60) if (ei,fi) in self.p],1)
        for x in self.X:
            for y in self.Y:
                terms=[]
                if (x,y) in self.s:terms.append(self.s[x,y])
                for ei,e in enumerate(self.EX):
                    if x not in e:continue
                    for fi,f in enumerate(self.EY):
                        if y in f and (ei,fi) in self.p:terms.append(self.p[ei,fi])
                self.exact(terms,int(self.G[x,y]))
        for xi,x in enumerate(self.X):
            for y in self.Y:
                self.by[y,xi]=self.s.get((x,y),False)
        for ei in range(60):
            u=12+ei
            for y in self.Y:
                opts=[self.p[ei,fi] for fi,f in enumerate(self.EY) if y in f and (ei,fi) in self.p]
                if not opts:
                    self.by[y,u]=False;continue
                b=self.new(('by',y,u));self.by[y,u]=b
                for z in opts:self.cnf.append([-z,b])
                self.cnf.append([-b]+opts)

    def B(self,i,u):
        if i in (0,1):return False
        if i==2:return bool(u<12)
        if 3<=i<=14:
            if u<12:return self.X[u]==i
            return i in self.EX[u-12]
        return self.by[i,u]

    def setup_D(self):
        for u in range(72):
            for v in range(u+1,72):self.d[u,v]=self.new(('d',u,v))

    def D(self,u,v):
        if u==v:return False
        return self.d[(u,v) if u<v else (v,u)]

    def setup_degrees_and_top(self):
        for u in range(72):self.exact([self.D(u,v) for v in range(72) if v!=u],11 if u<12 else 10)
        for i in range(27):
            neigh=[j for j in range(27) if self.H[i,j]]
            for u in range(72):
                terms=[self.B(i,u)]
                terms.extend(self.B(j,u) for j in neigh)
                for v in range(72):
                    if v==u:continue
                    terms.append(self.AND(self.B(i,v),self.D(u,v),('bd',i,u,v)))
                self.exact(terms,2)

    def setup_bottom(self):
        for u in range(72):
            for v in range(u+1,72):
                terms=[self.D(u,v)]
                for i in range(27):
                    terms.append(self.AND(self.B(i,u),self.B(i,v),('bb',i,u,v)))
                for w in range(72):
                    if w==u or w==v:continue
                    terms.append(self.AND(self.D(u,w),self.D(v,w),('dd',u,v,w)))
                self.exact(terms,2)

    def build(self):
        t=time.time();self.setup_factor();self.setup_D();self.setup_degrees_and_top()
        if self.mode=='full':self.setup_bottom()
        self.cnf.nv=max(self.cnf.nv,self.pool.top)
        self.stats={'branch':self.branch,'partition':self.part,'mode':self.mode,
                    'named_variables':len(self.named),'and_variables':len(self.and_cache),
                    'cnf_variables':self.cnf.nv,'clauses':len(self.cnf.clauses),
                    'build_seconds':time.time()-t}
        return self

    def decode(self,model):
        pos=set(z for z in model if z>0)
        ssel=[(x,y) for (x,y),z in self.s.items() if z in pos]
        psel=[(ei,fi) for (ei,fi),z in self.p.items() if z in pos]
        dedges=[(u,v) for (u,v),z in self.d.items() if z in pos]
        return ssel,psel,dedges

    def verify(self,model,full=False):
        ssel,psel,dedges=self.decode(model)
        assert len(ssel)==12 and len(psel)==60
        smap=dict(ssel);pmap=dict(psel)
        B=np.zeros((27,72),dtype=int)
        for u,x in enumerate(self.X):
            y=smap[x];B[2,u]=B[x,u]=B[y,u]=1
        for ei,e in enumerate(self.EX):
            f=self.EY[pmap[ei]];u=12+ei
            for i in e+f:B[i,u]=1
        assert np.array_equal(B@B.T,self.G)
        D=np.zeros((72,72),dtype=int)
        for u,v in dedges:D[u,v]=D[v,u]=1
        assert np.array_equal(self.H@B+B@D,2*np.ones((27,72),dtype=int)-B)
        assert np.array_equal(D.sum(1),14-B.sum(0))
        if full:
            assert np.array_equal(B.T@B+D@D,12*np.eye(72,dtype=int)-D+2*np.ones((72,72),dtype=int))
            A=np.block([[self.H,B],[B.T,D]])
            assert A.shape==(99,99) and np.array_equal(A@A,12*np.eye(99,dtype=int)-A+2*np.ones((99,99),dtype=int))
            return A
        return B,D


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--branch',type=int,required=True,choices=range(1,12));ap.add_argument('--mode',choices=['top','full'],default='top');ap.add_argument('--solver',default='cadical195');ap.add_argument('--seconds',type=int,default=30);ap.add_argument('--dec-budget',type=int,default=0);ap.add_argument('--conf-budget',type=int,default=0);ap.add_argument('--out',required=True);ap.add_argument('--cnf')
    a=ap.parse_args(); enc=Encoder(a.branch,a.mode).build(); print('BUILD',enc.stats,flush=True)
    if a.cnf:
        enc.cnf.to_file(a.cnf);print('CNF',a.cnf,hashlib.sha256(Path(a.cnf).read_bytes()).hexdigest(),flush=True)
    s=Solver(name=a.solver,bootstrap_with=enc.cnf.clauses);timed=[False]
    if a.dec_budget:
        try:s.dec_budget(a.dec_budget)
        except NotImplementedError:pass
    if a.conf_budget:
        try:s.conf_budget(a.conf_budget)
        except NotImplementedError:pass
    import threading
    timer=None
    if a.seconds>0:
        def stop():
            timed[0]=True
            try:s.interrupt()
            except Exception:pass
        timer=threading.Timer(a.seconds,stop); timer.daemon=True; timer.start()
    t=time.time()
    try:res=s.solve_limited(expect_interrupt=True)
    finally:
        if timer:timer.cancel()
    dt=time.time()-t
    if res is None:timed[0]=True
    try:stats=s.accum_stats()
    except NotImplementedError:stats={}
    print('RESULT',res,'timeout',timed[0],'seconds',dt,'stats',stats,flush=True)
    rec=dict(enc.stats);rec.update({'solver':a.solver,'seconds_limit':a.seconds,'result':res,'timed_out':timed[0],'solve_seconds':dt,'solver_stats':stats})
    if res:
        model=s.get_model();ssel,psel,dedges=enc.decode(model);rec['s']=ssel;rec['p']=psel;rec['D_edges']=dedges
        enc.verify(model,full=a.mode=='full');print('VERIFIED model',len(dedges),'D edges',flush=True)
        if a.mode=='full':
            A=enc.verify(model,full=True)
            np.savetxt(str(Path(a.out).with_suffix('.matrix.txt')),A,fmt='%d')
    p=Path(a.out);p.write_text(json.dumps(rec,indent=2)+'\n');print('OUT',p,hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
    s.delete()
if __name__=='__main__':main()
