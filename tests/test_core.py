import pandas as pd
import pytest
from core import generate,validate,analyze,wilson,sample_size

def test_reproducible():
    assert generate().equals(generate())
def test_dataset_shape():
    assert len(generate(n_per_variant=12))==24
def test_validation_missing():
    with pytest.raises(ValueError): validate(pd.DataFrame({'x':[1]}))
def test_validation_nonbinary():
    with pytest.raises(ValueError): validate(pd.DataFrame({'variant':['A','B'],'converted':[0,2]}))
def test_validation_duplicate_users():
    with pytest.raises(ValueError): validate(pd.DataFrame({'variant':['A','B'],'converted':[0,1],'user_id':[1,1]}))
def test_equal_rates():
    d=pd.DataFrame({'variant':['A']*10+['B']*10,'converted':[1]*5+[0]*5+[1]*5+[0]*5})
    r=analyze(d); assert r['absolute_uplift']==0 and r['p_value']==1
def test_perfect_separation():
    d=pd.DataFrame({'variant':['A']*100+['B']*100,'converted':[0]*100+[1]*100})
    assert analyze(d)['significant']
def test_wilson_bounds():
    a,b=wilson(2,10); assert 0<=a<=.2<=b<=1
def test_sample_size():
    assert sample_size(.1,.1)>sample_size(.1,.2)
def test_bad_sample_size():
    with pytest.raises(ValueError): sample_size(.9,.5)
def test_srm_warning():
    d=pd.DataFrame({'variant':['A']*1000+['B']*100,'converted':[0]*1100})
    assert analyze(d)['srm_warning']
def test_custom_control():
    assert analyze(generate(n_per_variant=20),control='B')['control']=='B'
