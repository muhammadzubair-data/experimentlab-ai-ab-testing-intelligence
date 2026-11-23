import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from core import generate,validate,analyze,sample_size

st.set_page_config(page_title='ExperimentLab AI | ZubairLabs',page_icon='🧪',layout='wide')
st.markdown('''<style>
.stApp{background:linear-gradient(145deg,#081329,#112444);color:#eaf5ff}
[data-testid="stMetric"]{background:#152d4e;border:1px solid #295477;border-radius:14px;padding:18px}
h1,h2,h3{color:#b4f2ef!important}
</style>''',unsafe_allow_html=True)
st.title('🧪 ExperimentLab AI')
st.caption('ZubairLabs • A/B Testing, Conversion Uplift & Experimentation Intelligence • Synthetic demonstration')
with st.sidebar:
    st.header('Experiment setup')
    upload=st.file_uploader('Upload experiment CSV',type='csv')
    st.caption('Required columns: variant, converted (0/1); optional unique user_id')
    alpha=st.select_slider('Significance level (α)',options=[.01,.05,.10],value=.05)
    page=st.radio('Explore',['Experiment Overview','Segment Explorer','Sample Size Planner','Methodology'])
try:
    df=validate(pd.read_csv(upload) if upload else generate())
except Exception as e:
    st.error(f'Invalid experiment dataset: {e}')
    st.stop()
labels=sorted(df.variant.unique())
control=st.sidebar.selectbox('Control variant',labels)
r=analyze(df,control,alpha)
if page=='Experiment Overview':
    st.subheader('Experiment results')
    cols=st.columns(4)
    for col,label,val in zip(cols,['Control conversion','Treatment conversion','Absolute uplift','Two-sided p-value'],[f"{r['rate_control']:.2%}",f"{r['rate_treatment']:.2%}",f"{r['absolute_uplift']:+.2%}",f"{r['p_value']:.4f}"]):
        col.metric(label,val)
    if r['srm_warning']:
        st.error('Sample Ratio Mismatch detected (p < 0.001). Investigate assignment and tracking before interpreting effects.')
    st.info('Statistically significant at the selected α.' if r['significant'] else 'Not statistically significant at the selected α. This does not prove no effect.')
    fig=go.Figure()
    fig.add_bar(x=[r['control'],r['treatment']],y=[r['rate_control'],r['rate_treatment']],marker_color=['#42c4d6','#a6f2c2'])
    fig.update_layout(title='Observed conversion rates',yaxis_tickformat='.1%',paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#eaf5ff')
    st.plotly_chart(fig,use_container_width=True)
    st.write(f"**95% CI for absolute difference (when α=.05):** {r['ci_low']:+.2%} to {r['ci_high']:+.2%}" if alpha==.05 else f"**{(1-alpha):.0%} CI for absolute difference:** {r['ci_low']:+.2%} to {r['ci_high']:+.2%}")
    st.write(f"Control: {r['conversions_control']:,}/{r['n_control']:,}; Treatment: {r['conversions_treatment']:,}/{r['n_treatment']:,}")
    st.download_button('Download experiment data',df.to_csv(index=False),'experiment_data.csv','text/csv')
elif page=='Segment Explorer':
    if 'device' not in df.columns:
        st.warning('No device column found in uploaded dataset. Add an optional device column to explore segments.')
    else:
        st.warning('Exploratory segment comparisons involve multiple testing; do not interpret unadjusted p-values as confirmatory evidence.')
        out=[]
        for segment,g in df.groupby('device'):
            if g.variant.nunique()==2:
                rr=analyze(g,control,alpha)
                out.append({'Device':segment,'Control N':rr['n_control'],'Treatment N':rr['n_treatment'],'Uplift':rr['absolute_uplift'],'p-value (unadjusted)':rr['p_value']})
        st.dataframe(pd.DataFrame(out),use_container_width=True)
elif page=='Sample Size Planner':
    st.subheader('Plan a future randomized experiment')
    base=st.number_input('Baseline conversion rate (%)',min_value=.1,max_value=99.,value=10.)/100
    mde=st.number_input('Minimum detectable relative lift (%)',min_value=.1,max_value=200.,value=10.)/100
    power=st.slider('Desired statistical power',.70,.95,.80,.05)
    try:
        n=sample_size(base,mde,alpha,power)
        st.metric('Estimated users per group',f'{n:,}')
        st.metric('Total estimated users',f'{2*n:,}')
    except ValueError as e: st.error(str(e))
else:
    st.markdown('''### Statistical methodology
- Randomized two-arm binary-conversion experiment; independent assignment units.
- Two-sided pooled two-proportion z-test for the null hypothesis of equal conversion rates.
- Unpooled Wald confidence interval for the absolute rate difference; Wilson intervals for each rate.
- Sample ratio mismatch check assumes a 50/50 assignment; p < 0.001 is flagged.
- Sample size uses a normal approximation for a two-sided superiority test and equal allocation.
- Sample size is an approximation; use simulation or exact methods for small or extreme rates.
- Avoid peeking and optional stopping without sequential-testing corrections.
- No causal interpretation unless randomization, stable exposure, and accurate tracking are established.
- Synthetic demonstration data only; no production or real-world impact claims.
''')
