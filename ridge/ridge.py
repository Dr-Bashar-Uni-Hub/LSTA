import numpy as np, pandas as pd
ALPHAS=10.0**np.arange(-5,6)
FSETS=['lags','lags+hx1','lags+hx3','lags+hx3both']
def feats(d,split,h,s,fset):
    X=d[f'X_{split}']; t=pd.to_datetime(d[f't_{split}'],unit='s',utc=True)
    lags=X[:,:,:2].reshape(len(X),-1)
    hr=np.eye(24,dtype=np.float32)[t.hour]; dw=np.eye(7,dtype=np.float32)[t.dayofweek]
    dy=np.c_[np.sin(2*np.pi*(t.dayofyear-1)/365.25),np.cos(2*np.pi*(t.dayofyear-1)/365.25)].astype(np.float32)
    F=[lags,hr,dw,dy]
    idx_last,idx_d24,idx_w168=167,143+h,h-1
    series=[s] if fset!='lags+hx3both' else [0,1]
    if fset!='lags':
        for ss in series:
            vals=[X[:,idx_last,ss]] if fset=='lags+hx1' else [X[:,idx_last,ss],X[:,idx_d24,ss],X[:,idx_w168,ss]]
            for v in vals: F.append(hr*v[:,None])
    return np.concatenate(F,1).astype(np.float64)
def fit_select(d,h,s,mean,scale):
    ytr=d['y_train'][:,s].astype(np.float64); yva=d['y_val'][:,s]*scale[s]+mean[s]
    best=None; log=[]
    for fs in FSETS:
        Xtr=feats(d,'train',h,s,fs); Xva=feats(d,'val',h,s,fs)
        mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd==0]=1
        A=(Xtr-mu)/sd; B=(Xva-mu)/sd; ym=ytr.mean()
        G=A.T@A; c=A.T@(ytr-ym)
        for a in ALPHAS:
            beta=np.linalg.solve(G+a*np.eye(len(G)),c)
            pv=(B@beta+ym)*scale[s]+mean[s]; r=float(np.sqrt(np.mean((pv-yva)**2)))
            log.append((fs,a,r))
            if best is None or r<best[2]: best=(fs,a,r,beta,mu,sd,ym)
    return best,log
def predict(d,split,h,s,best,mean,scale):
    fs,a,_,beta,mu,sd,ym=best
    return ((feats(d,split,h,s,fs)-mu)/sd@beta+ym)*scale[s]+mean[s]
