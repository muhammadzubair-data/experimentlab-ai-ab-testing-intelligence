"""Statistical utilities for randomized binary-outcome A/B experiments."""
import math
from statistics import NormalDist
import numpy as np
import pandas as pd

N = NormalDist()
REQUIRED = {'variant','converted'}

def validate(df):
    if not REQUIRED.issubset(df.columns):
        raise ValueError('CSV requires variant and converted columns')
    d = df.copy()
    d['variant'] = d['variant'].astype(str).str.strip()
    if d['variant'].isna().any() or (d['variant']=='').any():
        raise ValueError('Missing variant')
    if not d['converted'].isin([0,1,True,False]).all():
        raise ValueError('converted must contain binary 0/1 values')
    d['converted'] = d['converted'].astype(int)
    if d['variant'].nunique()!=2:
        raise ValueError('Exactly two variants required')
    if 'user_id' in d.columns and d['user_id'].duplicated().any():
        raise ValueError('Duplicate user_id: one row per randomized unit required')
    return d

def wilson(k,n,alpha=.05):
    if n<=0: raise ValueError('n must be positive')
    z=N.inv_cdf(1-alpha/2); p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return max(0,center-half),min(1,center+half)

def analyze(df, control='A', alpha=.05):
    if not 0<alpha<1: raise ValueError('alpha must be between 0 and 1')
    d=validate(df)
    labels=sorted(d.variant.unique())
    if control not in labels: raise ValueError('Control not found')
    treatment=next(x for x in labels if x!=control)
    a=d[d.variant==control].converted; b=d[d.variant==treatment].converted
    n0,n1=len(a),len(b); k0,k1=int(a.sum()),int(b.sum())
    p0,p1=k0/n0,k1/n1; diff=p1-p0
    pooled=(k0+k1)/(n0+n1)
    se0=math.sqrt(pooled*(1-pooled)*(1/n0+1/n1))
    z=diff/se0 if se0 else 0.
    pval=2*(1-N.cdf(abs(z))) if se0 else 1.
    se=math.sqrt(p0*(1-p0)/n0+p1*(1-p1)/n1)
    margin=N.inv_cdf(1-alpha/2)*se
    expected=n0+n1; chi=((n0-expected/2)**2/(expected/2)+(n1-expected/2)**2/(expected/2))
    srm_p=2*(1-N.cdf(math.sqrt(chi)))
    return dict(control=control,treatment=treatment,n_control=n0,n_treatment=n1,
      conversions_control=k0,conversions_treatment=k1,rate_control=p0,rate_treatment=p1,
      absolute_uplift=diff,relative_uplift=diff/p0 if p0 else None,
      ci_low=diff-margin,ci_high=diff+margin,p_value=pval,significant=pval<alpha,
      srm_p_value=srm_p,srm_warning=srm_p<.001,
      control_ci=wilson(k0,n0,alpha),treatment_ci=wilson(k1,n1,alpha))

def sample_size(baseline,relative_mde,alpha=.05,power=.8):
    if not (0<baseline<1 and relative_mde>0 and 0<alpha<1 and 0<power<1):
        raise ValueError('Invalid sample size inputs')
    p1=baseline*(1+relative_mde)
    if p1>=1: raise ValueError('Target conversion rate must be below 1')
    z_a=N.inv_cdf(1-alpha/2); z_b=N.inv_cdf(power)
    pbar=(baseline+p1)/2
    numerator=(z_a*math.sqrt(2*pbar*(1-pbar))+z_b*math.sqrt(baseline*(1-baseline)+p1*(1-p1)))**2
    return math.ceil(numerator/(p1-baseline)**2)

def generate(seed=42,n_per_variant=2500,baseline=.10,relative_lift=.12):
    if n_per_variant<1 or not 0<=baseline<=1 or not 0<=baseline*(1+relative_lift)<=1:
        raise ValueError('Invalid generator parameters')
    rng=np.random.default_rng(seed)
    variants=np.repeat(['A','B'],n_per_variant)
    probabilities=np.where(variants=='A',baseline,baseline*(1+relative_lift))
    return pd.DataFrame({'user_id':[f'U{i:06d}' for i in range(len(variants))],
      'variant':variants,'converted':rng.binomial(1,probabilities),
      'device':rng.choice(['Mobile','Desktop','Tablet'],len(variants),p=[.65,.30,.05])})
