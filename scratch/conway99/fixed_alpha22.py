#!/usr/bin/env python3
"""Exact fixed-design SAT encoding for a Hoffman coclique in srg(99,14,1,2).

Given one 2-(22,4,2) block design with 77 (possibly repeated) blocks,
vertices 0..21 are fixed as a coclique and outside vertex 22+i has
neighborhood block i in the coclique. The remaining edge variables are C.
The encoding is iff-complete and assumes no graph automorphism.
"""
from __future__ import annotations
import argparse,bz2,hashlib,itertools,json,xml.etree.ElementTree as ET
from pathlib import Path
class CNF:
 def __init__(self):self.nv=0;self.clauses=[]
 def var(self):self.nv+=1;return self.nv
 def add(self,*x):self.clauses.append(list(x))
 def unit(self,x):
  if x is True:return
  self.clauses.append([] if x is False else [int(x)])
 def thr(self,a,b,x):
  if a is True:return True
  if b is False or x is False:return a
  if b is True:
   if a is False:return x
   if a==x:return a
   z=self.var();a=int(a);x=int(x);self.add(-a,z);self.add(-x,z);self.add(-z,a,x);return z
  if x is True:
   if a is False:return b
   if a==b:return a
   z=self.var();a=int(a);b=int(b);self.add(-a,z);self.add(-b,z);self.add(-z,a,b);return z
  if a is False:
   z=self.var();b=int(b);x=int(x);self.add(-z,b);self.add(-z,x);self.add(z,-b,-x);return z
  z=self.var();a=int(a);b=int(b);x=int(x);self.add(-a,z);self.add(-b,-x,z);self.add(-z,a,b);self.add(-z,a,x);return z
 def exact(self,terms,k):
  xs=[];c=0
  for x in terms:
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
 def write(self,p):
  with open(p,'w') as f:
   f.write(f'p cnf {self.nv} {len(self.clauses)}\n')
   for c in self.clauses:f.write(' '.join(map(str,c))+' 0\n')
def parse(path):
 data=bz2.open(path,'rb').read() if str(path).endswith('.bz2') else Path(path).read_bytes();r=ET.fromstring(data);ns={'d':'http://designtheory.org/xml-namespace'};out=[]
 for bd in r.findall('.//d:block_design',ns):
  e=bd.find('d:blocks',ns);bs=[]
  if e is None:continue
  for b in e.findall('d:block',ns):bs.append(tuple(int(z.text) for z in b.findall('d:z',ns)))
  if bs:out.append((bd.attrib.get('id','unknown'),bs))
 return out
def validate(bs):
 pc={p:0 for p in itertools.combinations(range(22),2)}
 ok=len(bs)==77 and all(len(b)==4 and len(set(b))==4 and set(b)<=set(range(22)) for b in bs)
 if ok:
  for b in bs:
   for p in itertools.combinations(sorted(b),2):pc[p]+=1
 h={i:0 for i in range(5)};mx=0;dup=0
 for a,b in itertools.combinations(bs,2):
  t=len(set(a)&set(b));h[t]=h.get(t,0)+1;mx=max(mx,t);dup+=t==4
 return {'shape_ok':ok,'pair_counts_ok':ok and set(pc.values())=={2},'max_intersection':mx,'duplicates':dup,'intersection_histogram':h}
class Encoder:
 def __init__(self,bs):self.bs=[frozenset(b) for b in bs];self.f=CNF();self.e={};self.av={};self.reason=None
 def E(self,u,v):
  if u==v:return False
  if u>v:u,v=v,u
  return self.e.get((u,v),False)
 def AND(self,a,b):
  if a is False or b is False:return False
  if a is True:return b
  if b is True:return a
  a=int(a);b=int(b)
  if a==b:return a
  k=(min(a,b),max(a,b))
  if k in self.av:return self.av[k]
  z=self.f.var();self.av[k]=z;self.f.add(-z,a);self.f.add(-z,b);self.f.add(z,-a,-b);return z
 def build(self):
  v=validate([tuple(sorted(b)) for b in self.bs])
  if not v['shape_ok'] or not v['pair_counts_ok']:self.reason='invalid design';self.f.add();return self
  if v['max_intersection']>=3:self.reason=f"intersection {v['max_intersection']}";self.f.add();return self
  for u in range(77):
   for w in range(u+1,77):
    if len(self.bs[u]&self.bs[w])<=1:self.e[u,w]=self.f.var()
  through=[[u for u,b in enumerate(self.bs) if p in b] for p in range(22)]
  for u,b in enumerate(self.bs):
   for p in range(22):self.f.exact((self.E(u,v) for v in through[p] if v!=u),1 if p in b else 2)
  for u in range(77):
   for v in range(u+1,77):
    terms=[self.E(u,v)]+[self.AND(self.E(u,w),self.E(v,w)) for w in range(77) if w not in (u,v)]
    self.f.exact(terms,2-len(self.bs[u]&self.bs[v]))
  return self
 def stats(self):return {'edge_variables':len(self.e),'and_variables':len(self.av),'variables':self.f.nv,'clauses':len(self.f.clauses),'immediate_reason':self.reason}
def main():
 a=argparse.ArgumentParser();a.add_argument('xml',type=Path);a.add_argument('--index',type=int);a.add_argument('--out-dir',type=Path,default=Path('fixed43'));x=a.parse_args();ds=parse(x.xml);print('DESIGNS',len(ds));x.out_dir.mkdir(parents=True,exist_ok=True);ids=[x.index] if x.index is not None else range(len(ds));res=[]
 for i in ids:
  did,bs=ds[i];e=Encoder(bs).build();p=x.out_dir/f'design_{i:02d}.cnf';e.f.write(p);r={'index':i,'id':did,'validation':validate(bs),**e.stats(),'cnf':str(p),'cnf_sha256':hashlib.sha256(p.read_bytes()).hexdigest()};res.append(r);print(json.dumps(r,sort_keys=True))
 m=x.out_dir/'manifest.json';m.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n');print('MANIFEST_SHA256',hashlib.sha256(m.read_bytes()).hexdigest())
if __name__=='__main__':main()
