#!/usr/bin/env python3
"""Iff-complete Hoffman-coclique search split into four WLOG local cases.

No graph automorphism is assumed. Choose a coclique point 0, label its 14
outside neighbours and their forced 7K2 local graph, then normalize three
blocks. `branch` and `case` cover the four residual local orbits.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
class CNF:
 def __init__(self,path):self.path=Path(path);self.body=self.path.with_suffix('.body');self.f=self.body.open('w');self.nv=0;self.nc=0;self.cache={}
 def var(self):self.nv+=1;return self.nv
 def add(self,*c):self.f.write(' '.join(map(str,c))+' 0\n');self.nc+=1
 def unit(self,x):
  if x is True:return
  self.add() if x is False else self.add(int(x))
 def OR(self,a,b):
  if a is True or b is True:return True
  if a is False:return b
  if b is False:return a
  if a==b:return a
  z=self.var();a=int(a);b=int(b);self.add(-a,z);self.add(-b,z);self.add(-z,a,b);return z
 def thr(self,a,b,x):
  if a is True:return True
  if b is False or x is False:return a
  if b is True:return self.OR(a,x)
  if x is True:return self.OR(a,b)
  b=int(b);x=int(x)
  if a is False:
   z=self.var();self.add(-z,b);self.add(-z,x);self.add(z,-b,-x);return z
  a=int(a);z=self.var();self.add(-a,z);self.add(-b,-x,z);self.add(-z,a,b);self.add(-z,a,x);return z
 def exact(self,ts,k):
  xs=[];c=0
  for x in ts:
   if x is True:c+=1
   elif x is not False:xs.append(int(x))
  k-=c
  if k<0 or k>len(xs):self.add();return
  if not xs:
   if k:self.add()
   return
  prev=[True]+[False]*(k+1)
  for x in xs:
   cur=[True]
   for j in range(1,k+2):cur.append(self.thr(prev[j],prev[j-1],x))
   prev=cur
  self.unit(prev[k]);u=prev[k+1];self.unit(False if u is True else True if u is False else -int(u))
 def AND(self,a,b):
  if a is False or b is False:return False
  if a is True:return b
  if b is True:return a
  a=int(a);b=int(b)
  if a==b:return a
  k=(min(a,b),max(a,b))
  if k in self.cache:return self.cache[k]
  z=self.var();self.cache[k]=z;self.add(-z,a);self.add(-z,b);self.add(z,-a,-b);return z
 def lexle(self,a,b):
  eq=True
  for x,y in zip(a,b):
   self.add(-x,y) if eq is True else self.add(-eq,-x,y)
   z=self.var()
   if eq is True:self.add(-z,-x,y);self.add(-z,x,-y);self.add(-x,-y,z);self.add(x,y,z)
   else:self.add(-z,eq);self.add(-z,-x,y);self.add(-z,x,-y);self.add(-eq,-x,-y,z);self.add(-eq,x,y,z)
   eq=z
 def finish(self):
  self.f.close()
  with self.path.open('w') as o:
   o.write(f'p cnf {self.nv} {self.nc}\n')
   with self.body.open() as f:
    for l in f:o.write(l)
  self.body.unlink()
class E:
 def __init__(self,b,c,out):
  self.branch=b;self.case=c;self.f=CNF(out);self.B=[[None]*77 for _ in range(22)];self.C={}
  for u in range(77):self.B[0][u]=u<14
  for p in range(1,22):
   for u in range(77):self.B[p][u]=self.f.var()
  M={tuple(sorted(x)) for x in [(0,3),(1,4),(2,5),(6,7),(8,9),(10,11),(12,13)]}
  for u in range(77):
   for v in range(u+1,77):self.C[u,v]=((u,v) in M) if v<14 else self.f.var()
 def edge(self,u,v):
  if u==v:return False
  if u>v:u,v=v,u
  return self.C[u,v]
 def fix(self,u,pts):
  pts=set(pts)
  for p in range(22):
   x=self.B[p][u];self.f.unit(x if p in pts else (not x if isinstance(x,bool) else -x))
 def build(self):
  f=self.f;self.fix(0,{0,1,2,3});self.fix(1,{0,1,4,5});self.fix(2,{0,2,4,6} if self.branch==0 else {0,2,6,7});f.unit(self.B[3][4 if self.case==0 else 6])
  for u in range(14,76):f.lexle([self.B[p][u] for p in range(1,22)],[self.B[p][u+1] for p in range(1,22)])
  for u in range(14):f.exact((self.B[p][u] for p in range(1,22)),3)
  for u in range(14,77):f.exact((self.B[p][u] for p in range(1,22)),4)
  for p in range(1,22):f.exact((self.B[p][u] for u in range(14)),2)
  for p in range(1,22):
   for q in range(p+1,22):f.exact((f.AND(self.B[p][u],self.B[q][u]) for u in range(77)),2)
  for u in range(14,77):f.exact((self.edge(u,v) for v in range(14)),2)
  for p in range(1,22):
   for u in range(77):f.exact([self.B[p][u]]+[f.AND(self.edge(u,v),self.B[p][v]) for v in range(77) if v!=u],2)
  for u in range(77):
   for v in range(u+1,77):f.exact([self.edge(u,v)]+[f.AND(self.B[p][u],self.B[p][v]) for p in range(22)]+[f.AND(self.edge(u,w),self.edge(v,w)) for w in range(77) if w not in (u,v)],2)
  f.finish();return {'branch':self.branch,'case':self.case,'variables':f.nv,'clauses':f.nc,'and_variables':len(f.cache)}
def main():
 a=argparse.ArgumentParser();a.add_argument('--branch',type=int,choices=[0,1],required=True);a.add_argument('--case',type=int,choices=[0,1],required=True);a.add_argument('--out',type=Path,required=True);x=a.parse_args();t=time.time();r=E(x.branch,x.case,x.out).build();r['seconds']=time.time()-t;r['sha256']=hashlib.sha256(x.out.read_bytes()).hexdigest();print(json.dumps(r,sort_keys=True))
if __name__=='__main__':main()
