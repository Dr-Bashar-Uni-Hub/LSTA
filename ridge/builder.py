import numpy as np, pandas as pd
TRAIN=(pd.Timestamp('2015-01-01 00:00',tz='UTC'),pd.Timestamp('2018-12-31 23:00',tz='UTC'))
VAL=(pd.Timestamp('2019-01-01 00:00',tz='UTC'),pd.Timestamp('2019-12-31 23:00',tz='UTC'))
TEST=(pd.Timestamp('2020-01-01 00:00',tz='UTC'),pd.Timestamp('2020-09-30 23:00',tz='UTC'))
SEQ=168
def prepare_frame(df, load_col, solar_col):
    """df: utc_timestamp + raw columns (MW). Returns frame (>= 2015-01-01 UTC) with calendar features and scaler."""
    d=df[['utc_timestamp',load_col,solar_col]].copy()
    d['utc_timestamp']=pd.to_datetime(d['utc_timestamp'],utc=True)
    d=d[(d.utc_timestamp>=TRAIN[0])&(d.utc_timestamp<=TEST[1])].sort_values('utc_timestamp').reset_index(drop=True)
    ts=d.utc_timestamp
    assert (ts.diff().dropna()==pd.Timedelta(hours=1)).all(), 'non-hourly timestamps'
    h,dw,dy=ts.dt.hour,ts.dt.dayofweek,ts.dt.dayofyear
    cal=np.c_[np.sin(2*np.pi*h/24),np.cos(2*np.pi*h/24),np.sin(2*np.pi*dw/7),np.cos(2*np.pi*dw/7),
              np.sin(2*np.pi*(dy-1)/365.25),np.cos(2*np.pi*(dy-1)/365.25)]
    raw=d[[load_col,solar_col]].to_numpy(np.float64)
    tr=((ts>=TRAIN[0])&(ts<=TRAIN[1])).to_numpy()
    fit=raw[tr]; fit=fit[np.isfinite(fit).all(1)]
    mean=fit.mean(0); scale=fit.std(0,ddof=0)            # sklearn StandardScaler convention
    feats=np.c_[(raw-mean)/scale,cal].astype(np.float32)
    return dict(ts=ts.reset_index(drop=True),feats=feats,mean=mean,scale=scale,raw=raw)
def build_horizon(P,h):
    """Window = the 168 hours ending h hours before the target; every hour from window start to target must be observed."""
    F=P['feats']; ts=P['ts']; ok=np.isfinite(F).all(1); n=len(F)
    cs=np.r_[0,np.cumsum(~ok)]                          # prefix count of missing rows
    out={k:{'X':[],'y':[],'t':[]} for k in ('train','val','test')}; cand={k:0 for k in out}; rej={k:0 for k in out}
    for i in range(SEQ+h-1,n):
        t=ts[i]
        part='train' if TRAIN[0]<=t<=TRAIN[1] else 'val' if VAL[0]<=t<=VAL[1] else 'test' if TEST[0]<=t<=TEST[1] else None
        if part is None: continue
        cand[part]+=1; s=i-h+1-SEQ
        if cs[i+1]-cs[s]>0: rej[part]+=1; continue
        out[part]['X'].append(F[s:i-h+1]); out[part]['y'].append(F[i,:2]); out[part]['t'].append(t.value//10**9)
    res={}
    for k,v in out.items():
        res[f'X_{k}']=np.asarray(v['X'],np.float32); res[f'y_{k}']=np.asarray(v['y'],np.float32); res[f't_{k}']=np.asarray(v['t'],np.int64)
    return res,cand,rej


