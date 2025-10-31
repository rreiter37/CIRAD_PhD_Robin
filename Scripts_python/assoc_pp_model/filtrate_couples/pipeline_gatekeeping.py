#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative gatekeeping selection (dynamic S) with adaptive correlation test.

Modification:
  - Gate 2 (correlation test) is adaptive:
        * If number of datasets < 30: permutation test (robust for small n)
        * Else: classical Spearman correlation test (approximate p-value)
"""

from __future__ import annotations
import os, re, math, argparse, random
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

# Optional SciPy
try:
    from scipy.stats import wilcoxon, binom, spearmanr
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# ------------------------- Config -------------------------

RESULTS_ROOT_DEFAULT = "/Results/asso_pp_model"
MODEL_COL = "Model"
FILENAME_PREFIX = "results_"
FILENAME_EXCLUDE_SUBSTRINGS = ["NICON","CNN","Ridge","LGBM","PLS"]
EXCLUDE_PREPS = {"PCA"}

ALPHA_DEFAULT = 0.05
DELTA_MARGIN_DEFAULT = 0.01
N_PERM_DEFAULT = 10000
SEED_DEFAULT = 42
MAX_EPOCHS_DEFAULT = 10
OUT_PREFIX_DEFAULT = "gatekeeping_iter_adaptive"

# ------------------------- Discovery utilities -------------------------

def discover_dataset_csvs(root: str) -> List[Tuple[str,str]]:
    pairs=[]
    if not os.path.isdir(root):
        return pairs
    for ds in sorted(os.listdir(root)):
        ds_dir=os.path.join(root,ds)
        if not os.path.isdir(ds_dir): continue
        cands=[f for f in os.listdir(ds_dir)
               if f.startswith(FILENAME_PREFIX) and f.lower().endswith(".csv")]
        cands=[f for f in cands if not any(excl in f for excl in FILENAME_EXCLUDE_SUBSTRINGS)]
        if not cands: continue
        cands.sort()
        pairs.append((ds,os.path.join(ds_dir,cands[0])))
    return pairs

def detect_task_from_models(models:List[str])->str:
    has_reg=any(isinstance(m,str) and m.endswith("_reg") for m in models)
    has_clf=any(isinstance(m,str) and m.endswith("_classif") for m in models)
    if has_reg and not has_clf: return "regression"
    if has_clf and not has_reg: return "classification"
    if has_reg and has_clf: return "ambiguous"
    return "unknown"

# ------------------------- Build Δ profiles -------------------------

def collect_delta_profiles_by_task(root:str)->Tuple[pd.DataFrame,pd.DataFrame]:
    reg_rows,clf_rows=[],[]
    for ds,path in discover_dataset_csvs(root):
        try: df=pd.read_csv(path)
        except Exception: continue
        if MODEL_COL not in df.columns: continue
        models=[str(x) for x in df[MODEL_COL].dropna().tolist()]
        task=detect_task_from_models(models)
        if task in ("ambiguous","unknown"): continue
        tidy=df.melt(id_vars=[MODEL_COL],var_name="prep",value_name="metric").rename(columns={MODEL_COL:"model"})
        if EXCLUDE_PREPS: tidy=tidy[~tidy["prep"].isin(EXCLUDE_PREPS)]
        tidy["metric"]=pd.to_numeric(tidy["metric"],errors="coerce")
        tidy=tidy.dropna(subset=["metric"])
        if tidy.empty: continue
        pivot=tidy.pivot_table(index="model",columns="prep",values="metric",aggfunc="mean")
        for prep in pivot.columns:
            col=pivot[prep].dropna()
            if col.empty: continue
            if task=="regression":
                best=float(col.min())
                for cand in col.index:
                    reg_rows.append((ds,cand,str(prep),float(col.loc[cand]-best)))
            else:
                best=float(col.max())
                for cand in col.index:
                    clf_rows.append((ds,cand,str(prep),float(best-col.loc[cand])))
    diffs_reg=pd.DataFrame(reg_rows,columns=["dataset","candidate_model","prep","delta"])
    diffs_clf=pd.DataFrame(clf_rows,columns=["dataset","candidate_model","prep","delta"])
    return diffs_reg,diffs_clf

# ------------------------- Tests -------------------------

def exact_sign_test_inferiority(deltas:np.ndarray,delta_margin:float)->Tuple[Optional[float],int]:
    y=deltas-delta_margin
    eff=y[np.abs(y)>0]; n_eff=int(eff.size)
    if n_eff==0: return None,0
    s_pos=int(np.sum(eff>0))
    if SCIPY_AVAILABLE:
        from scipy.stats import binom
        p=1.0-binom.cdf(s_pos-1,n_eff,0.5)
    else:
        from math import comb
        p=sum(comb(n_eff,k)*(0.5**n_eff) for k in range(s_pos,n_eff+1))
    return float(p),n_eff

def wilcoxon_inferiority(deltas:np.ndarray,delta_margin:float)->Tuple[Optional[float],int]:
    if not SCIPY_AVAILABLE: return None,0
    y=deltas-delta_margin
    y=y[np.abs(y)>0]; n=len(y)
    if n==0: return None,0
    try:
        stat,p=wilcoxon(y,alternative="greater",zero_method="wilcox")
        return float(p),n
    except Exception:
        return None,n

def spearman_perm_or_classical(x:np.ndarray,y:np.ndarray,n_perm:int=10000,seed:int=42)->Tuple[float,float]:
    """
    Adaptive Spearman test:
      * n < 30 -> permutation test (H1: rho>0)
      * n >= 30 -> classical Spearman test (approx, H1: rho>0)
    Returns (rho_obs, p_value)
    """
    rng=np.random.default_rng(seed)
    mask=np.isfinite(x)&np.isfinite(y)
    x=x[mask]; y=y[mask]
    n=len(x)
    if n<3: return np.nan,np.nan
    # Compute Spearman correlation
    if SCIPY_AVAILABLE:
        rho_obs,_=spearmanr(x,y)
    else:
        rx=pd.Series(x).rank().to_numpy(); ry=pd.Series(y).rank().to_numpy()
        rho_obs=float(np.corrcoef(rx,ry)[0,1])
    # Small sample => permutation
    if n<30:
        rx=pd.Series(x).rank().to_numpy()
        ry=pd.Series(y).rank().to_numpy()
        rx_c=rx-rx.mean(); ry_c=ry-ry.mean()
        denom=np.sqrt(np.dot(rx_c,rx_c)*np.dot(ry_c,ry_c))
        rho_obs=float(np.dot(rx_c,ry_c)/denom)
        ge=1
        for _ in range(n_perm):
            rng.shuffle(ry)
            ry_c2=ry-ry.mean()
            rho_perm=np.dot(rx_c,ry_c2)/np.sqrt(np.dot(rx_c,rx_c)*np.dot(ry_c2,ry_c2))
            if rho_perm>=rho_obs-1e-15: ge+=1
        p=ge/(n_perm+1.0)
    else:
        # Approximate one-sided p-value for rho>0
        if not SCIPY_AVAILABLE: return rho_obs,np.nan
        rho_obs, p_two = spearmanr(x,y)
        p_one = p_two/2 if rho_obs>0 else 1-p_two/2
        p = max(min(p_one,1.0),0.0)
    return float(rho_obs),float(p)

def benjamini_hochberg(pvals:List[Optional[float]])->List[Optional[float]]:
    idx=[(i,p) for i,p in enumerate(pvals) if p is not None and np.isfinite(p)]
    m=len(idx); q=[None]*len(pvals)
    if m==0: return q
    idx.sort(key=lambda t:t[1])
    raw=[min(1.0,idx[k][1]*m/(k+1)) for k in range(m)]
    for k in range(m-2,-1,-1):
        raw[k]=min(raw[k],raw[k+1])
    for k,(i,_) in enumerate(idx): q[i]=raw[k]
    return q

# ------------------------- Core gatekeeping -------------------------

def envelope_profile(block:pd.DataFrame,datasets:List[str])->np.ndarray:
    if block.empty: return np.full(len(datasets),np.nan)
    return np.nanmin(block[datasets].to_numpy(float),axis=0)

def iterative_gatekeeping(diffs:pd.DataFrame,alpha:float,delta:float,n_perm:int,seed:int,max_epochs:int=10):
    if diffs.empty: return pd.DataFrame(),pd.DataFrame(columns=["candidate_model","prep"])
    mat=diffs.pivot_table(index=["candidate_model","prep"],columns="dataset",values="delta",aggfunc="mean")
    mat=mat.sort_index()
    datasets=list(mat.columns)
    med=mat.median(axis=1,skipna=True)
    order=med.sort_values().index
    mat=mat.loc[order]; med=med.loc[order]
    kept_mask=pd.Series(False,index=mat.index)
    if len(kept_mask)>0: kept_mask.iloc[0]=True
    records={idx:{"median_delta":float(med.loc[idx])} for idx in mat.index}
    prev=None
    for epoch in range(1,max_epochs+1):
        gate2_idx=[]
        for idx in mat.index:
            if kept_mask.loc[idx]: continue
            deltas=mat.loc[idx].to_numpy(float)
            deltas=deltas[np.isfinite(deltas)]
            p1,n1=exact_sign_test_inferiority(deltas,delta)
            p1w,n1w=wilcoxon_inferiority(deltas,delta)
            rec=records[idx]
            rec.update({"gate1_perf_sign_p":p1,"gate1_perf_n_eff":n1,
                        "gate1_wilcoxon_p":p1w,"gate1_wilcoxon_n_eff":n1w})
            if p1 is None or p1>=alpha:
                kept_mask.loc[idx]=True
                rec["gate1_pass"]=False
            else:
                rec["gate1_pass"]=True
                gate2_idx.append(idx)
        vS=envelope_profile(mat.loc[kept_mask],datasets)
        p_list=[]; idx_list=[]
        for idx in gate2_idx:
            vec=mat.loc[idx].to_numpy(float)
            rho,p2=spearman_perm_or_classical(vec,vS,n_perm=n_perm,seed=seed+epoch)
            records[idx]["gate2_corr_rho"]=rho
            records[idx]["gate2_corr_p"]=p2
            p_list.append(p2); idx_list.append(idx)
        qvals=benjamini_hochberg(p_list) if p_list else []
        for idx,q2 in zip(idx_list,qvals):
            records[idx]["gate2_corr_q"]=q2
        for idx in gate2_idx:
            q2=records[idx].get("gate2_corr_q",None)
            if q2 is not None and np.isfinite(q2) and q2<alpha:
                kept_mask.loc[idx]=False; records[idx]["decision"]="EXCLUDE"
            else:
                kept_mask.loc[idx]=True; records[idx]["decision"]="KEEP"
        now=tuple(kept_mask[kept_mask].index)
        if prev is not None and now==prev: break
        prev=now
    rows=[]
    for idx in mat.index:
        rec=records[idx]; model,prep=idx
        rows.append({
            "candidate_model":model,"prep":prep,
            "median_delta":rec.get("median_delta",np.nan),
            "gate1_perf_sign_p":rec.get("gate1_perf_sign_p",None),
            "gate1_perf_n_eff":rec.get("gate1_perf_n_eff",None),
            "gate1_wilcoxon_p":rec.get("gate1_wilcoxon_p",None),
            "gate1_wilcoxon_n_eff":rec.get("gate1_wilcoxon_n_eff",None),
            "gate2_corr_rho":rec.get("gate2_corr_rho",None),
            "gate2_corr_p":rec.get("gate2_corr_p",None),
            "gate2_corr_q":rec.get("gate2_corr_q",None),
            "decision":"KEEP" if kept_mask.loc[idx] else "EXCLUDE"})
    summary=pd.DataFrame(rows).sort_values(["decision","median_delta"])
    kept=summary[summary["decision"]=="KEEP"][["candidate_model","prep"]].reset_index(drop=True)
    return summary,kept

# ------------------------- Main -------------------------

ap=argparse.ArgumentParser(description="Iterative gatekeeping with adaptive correlation test.")
ap.add_argument("--root",type=str,default=RESULTS_ROOT_DEFAULT)
ap.add_argument("--alpha",type=float,default=ALPHA_DEFAULT)
ap.add_argument("--delta",type=float,default=DELTA_MARGIN_DEFAULT)
ap.add_argument("--n_perm",type=int,default=N_PERM_DEFAULT)
ap.add_argument("--seed",type=int,default=SEED_DEFAULT)
ap.add_argument("--max_epochs",type=int,default=MAX_EPOCHS_DEFAULT)
ap.add_argument("--out_prefix",type=str,default=OUT_PREFIX_DEFAULT)
args=ap.parse_args()

np.random.seed(args.seed); random.seed(args.seed)
diffs_reg,diffs_clf=collect_delta_profiles_by_task(args.root)
out_dir=os.path.join(args.root,"All_datasets"); os.makedirs(out_dir,exist_ok=True)
if not diffs_reg.empty:
    print("[INFO] Regression task...")
    s_reg,k_reg=iterative_gatekeeping(diffs_reg,args.alpha,args.delta,args.n_perm,args.seed,args.max_epochs)
    s_reg.to_csv(os.path.join(out_dir,f"{args.out_prefix}_reg_report.csv"),index=False)
    k_reg.to_csv(os.path.join(out_dir,f"{args.out_prefix}_reg_keep.csv"),index=False)
if not diffs_clf.empty:
    print("[INFO] Classification task...")
    s_clf,k_clf=iterative_gatekeeping(diffs_clf,args.alpha,args.delta,args.n_perm,args.seed,args.max_epochs)
    s_clf.to_csv(os.path.join(out_dir,f"{args.out_prefix}_classif_report.csv"),index=False)
    k_clf.to_csv(os.path.join(out_dir,f"{args.out_prefix}_classif_keep.csv"),index=False)
if diffs_reg.empty and diffs_clf.empty:
    print("[WARN] No usable datasets found.")