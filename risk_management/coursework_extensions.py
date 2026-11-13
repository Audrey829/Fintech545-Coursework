"""Reusable extensions for option diagnostics, regression and portfolio analysis.

Existing pricing, fitting and risk engines remain in their original modules.
Rates/volatilities use decimals; theta is calendar-time decay per year.
"""
from dataclasses import asdict
import numpy as np
import pandas as pd
from scipy import stats, optimize
import statsmodels.api as sm
from .options import gbsm, american_continuous, american_discrete_dividends, _american_no_dividend_batch
from .statistics import sample_moments
from .distributions import fit_normal, fit_student_t, aicc, NormalFit, StudentTFit
from .risk_metrics import value_at_risk, expected_shortfall, simulate_fitted_distribution
from .portfolio import maximize_sharpe_ratio


def parity_diagnostics(chain, spot, maturity, rate, dividend_yield):
    c=chain.query("Type == 'Call'").set_index('Strike')
    p=chain.query("Type == 'Put'").set_index('Strike')
    out=pd.DataFrame(index=c.index)
    out['Gap']=(c.Bid+c.Ask-p.Bid-p.Ask)/2-(spot*np.exp(-dividend_yield*maturity)-c.index*np.exp(-rate*maturity))
    out['Band']=c.Ask-c.Bid+p.Ask-p.Bid
    out['Flag']=out.Gap.abs()>out.Band
    return out


def parity_gap_in_vol_points(parity, implied_vols):
    """Translate flagged put-call parity gaps into local IV-point diagnostics."""
    rows=[]
    for strike in parity.index[parity.Flag]:
        call=implied_vols.loc[(implied_vols.Strike==strike)&(implied_vols.Type=='Call')].iloc[0]
        put=implied_vols.loc[(implied_vols.Strike==strike)&(implied_vols.Type=='Put')].iloc[0]
        gap=parity.loc[strike,'Gap']
        rows.append({
            'Strike':strike,
            'Parity gap':gap,
            'Gap / call vega (vol points)':100*gap/call.Vega,
            'Gap / put vega (vol points)':100*gap/put.Vega,
            'Observed call-put IV (vol points)':100*(call.MidIV-put.MidIV),
            'Parity-consistent put mid':put.Mid+gap,
            'Parity-consistent call mid':call.Mid-gap,
        })
    return pd.DataFrame(rows)


def implied_volatility(price, call, spot, strike, maturity, rate, carry):
    forward_pv=spot*np.exp((carry-rate)*maturity)
    strike_pv=strike*np.exp(-rate*maturity)
    lower=max(0., (forward_pv-strike_pv)*(1 if call else -1))
    upper=forward_pv if call else strike_pv
    if price<lower-1e-10: return np.nan, 'Below no-arbitrage lower bound'
    if price>=upper: return np.nan, 'At/above no-arbitrage upper bound'
    if abs(price-lower)<1e-10: return 0., 'Zero-volatility boundary'
    f=lambda vol: gbsm(call,spot,strike,maturity,rate,carry,vol).value-price
    return optimize.brentq(f,1e-8,20.,xtol=1e-12), 'OK'


def chain_implied_volatilities(chain, spot, maturity, rate, dividend_yield):
    rows=[]
    for x in chain.itertuples():
        row={'Type':x.Type,'Strike':x.Strike,'Mid':(x.Bid+x.Ask)/2}
        for label,price in [('Bid',x.Bid),('Mid',row['Mid']),('Ask',x.Ask)]:
            row[label+'IV'],row[label+'Status']=implied_volatility(price,x.Type=='Call',spot,x.Strike,maturity,rate,rate-dividend_yield)
        v=row['MidIV']
        row['Vega']=gbsm(x.Type=='Call',spot,x.Strike,maturity,rate,rate-dividend_yield,v,include_greeks=True).vega if v>0 else np.nan
        row['OneCentVolPoints']=1/row['Vega'] # 100 * 0.01 / Vega
        rows.append(row)
    return pd.DataFrame(rows)


def dividend_exercise_conditions(strike, maturity, rate, times, amounts):
    threshold=strike*(1-np.exp(-rate*(maturity-np.asarray(times))))
    return pd.DataFrame({'Time':times,'Dividend':amounts,'Interest threshold':threshold,'Condition met':np.asarray(amounts)>threshold})


def discrete_dividend_tree(call, spot, strike, maturity, rate, volatility, steps, amounts, times, american=True):
    """Two cash dividends: exact non-recombining CRR subtrees, no interpolation.

    American path reuses the original tested library. European terminal values
    integrate each terminal subtree with binomial probabilities, then roll back
    across both cash-dividend dates without early exercise.
    """
    dates=np.rint(np.asarray(times)/maturity*steps).astype(int)
    if not np.allclose(dates*maturity/steps,times): raise ValueError('Dividend dates must lie exactly on grid')
    if american:
        return american_discrete_dividends(call,spot,strike,maturity,rate,amounts,dates,volatility,steps)
    if len(dates)!=2: raise ValueError('This European extension accepts exactly two dividends')
    dt=maturity/steps; u=np.exp(volatility*np.sqrt(dt)); d=1/u
    p=(np.exp(rate*dt)-d)/(u-d); discount=np.exp(-rate*dt)
    if not 0<p<1: raise ValueError('Invalid risk-neutral probability')
    n1,n2=int(dates[0]),int(dates[1]-dates[0]); n3=steps-int(dates[1])
    first=spot*u**np.arange(n1+1)*d**(n1-np.arange(n1+1))-amounts[0]
    second=first[:,None]*u**np.arange(n2+1)*d**(n2-np.arange(n2+1))-amounts[1]
    if second.min()<=0: raise ValueError('Cash dividends produce nonpositive nodes')
    j=np.arange(n3+1)
    terminal=second[:,:,None]*u**j*d**(n3-j)
    option=(np.maximum((1 if call else -1)*(terminal-strike),0)*stats.binom.pmf(j,n3,p)).sum(axis=2)*discount**n3
    for k in range(n2): option=discount*(p*option[:,1:]+(1-p)*option[:,:-1])
    option=option[:,0]
    for k in range(n1): option=discount*(p*option[1:]+(1-p)*option[:-1])
    return float(option[0])


def dividend_option_comparison(spot,strike,maturity,rate,volatility,steps,amounts,times):
    adjusted=spot-np.dot(amounts,np.exp(-rate*np.asarray(times)))
    rows=[]
    for call in [True,False]:
        eu=gbsm(call,spot,strike,maturity,rate,rate,volatility).value
        eu_tree=discrete_dividend_tree(call,spot,strike,maturity,rate,volatility,steps,np.zeros(len(amounts)),times,False)
        am=american_continuous(call,spot,strike,maturity,rate,rate,volatility,steps)
        ed=discrete_dividend_tree(call,spot,strike,maturity,rate,volatility,steps,amounts,times,False)
        ad=discrete_dividend_tree(call,spot,strike,maturity,rate,volatility,steps,amounts,times,True)
        approx=gbsm(call,adjusted,strike,maturity,rate,rate,volatility).value
        rows.append({'Type':'Call' if call else 'Put','European no div':eu,'European no-div tree':eu_tree,'American no div':am,'European dividends':ed,'American dividends':ad,'PV-adjusted GBSM':approx,'PV approx minus tree':approx-ed,'EEP no div':am-eu_tree,'EEP dividends':ad-ed,'European tree minus GBSM no div':eu_tree-eu,'American tree minus GBSM no div':am-eu})
    return pd.DataFrame(rows).set_index('Type')


def dividend_tree_convergence(spot,strike,maturity,rate,volatility,amounts,times,steps):
    return pd.DataFrame([{'N':n,**{name:discrete_dividend_tree(call,spot,strike,maturity,rate,volatility,n,amounts,times,True) for name,call in [('Call',True),('Put',False)]}} for n in steps])


def dividend_tree_greeks(spot,strike,maturity,rate,volatility,steps,amounts,times,bumps):
    rows=[]
    for call in [True,False]:
        f=lambda s:discrete_dividend_tree(call,s,strike,maturity,rate,volatility,steps,amounts,times,True)
        base=f(spot)
        for h in bumps:
            up,down=f(spot+h),f(spot-h)
            rows.append({'Type':'Call' if call else 'Put','Bump':h,'Delta':(up-down)/(2*h),'Gamma':(up-2*base+down)/h**2})
    return pd.DataFrame(rows)


def dividend_sensitivities(spot,strike,maturity,rate,volatility,steps,amounts,times,bump=0.001):
    rows=[]
    for call in [True,False]:
        for american in [False,True]:
            row={'Option':('American ' if american else 'European ')+('Call' if call else 'Put')}
            for i in range(2):
                up=np.array(amounts,float); down=up.copy();up[i]+=bump;down[i]-=bump
                row[f'dV/dD{i+1}']=(discrete_dividend_tree(call,spot,strike,maturity,rate,volatility,steps,up,times,american)-discrete_dividend_tree(call,spot,strike,maturity,rate,volatility,steps,down,times,american))/(2*bump)
            rows.append(row)
    return pd.DataFrame(rows)


def last_dividend_call_boundary(strike,remaining,rate,volatility,dividend,steps):
    # After the final dividend, no-dividend American call continuation equals a European call.
    f=lambda s:american_continuous(True,s-dividend,strike,remaining,rate,rate,volatility,steps)-(s-strike)
    return optimize.brentq(f,max(strike,dividend)+1e-6,strike*3)


def gbsm_scenario_values(call,spots,strike,maturity,rate,carry,volatility):
    """Vector form of the existing scalar GBSM, verified against that function."""
    s=np.asarray(spots,float)
    if np.any(s<=0): raise ValueError('Nonpositive simulated stock price')
    if maturity<=0: return np.maximum((1 if call else -1)*(s-strike),0.)
    d1=(np.log(s/strike)+(carry+volatility**2/2)*maturity)/(volatility*np.sqrt(maturity))
    d2=d1-volatility*np.sqrt(maturity); sign=1 if call else -1
    return sign*(s*np.exp((carry-rate)*maturity)*stats.norm.cdf(sign*d1)-strike*np.exp(-rate*maturity)*stats.norm.cdf(sign*d2))


def option_book_greeks(holdings,spot,rate,days_per_year=255):
    rows=[]
    for name,book in holdings.groupby('Portfolio',sort=False):
        row={'Portfolio':name,'Value':0.,'Delta':0.,'Gamma':0.,'Theta':0.}
        for x in book.itertuples():
            if x.Type=='Stock': values={'Value':spot,'Delta':1.,'Gamma':0.,'Theta':0.}
            else:
                g=gbsm(x.Type=='Call',spot,x.Strike,x.DaysToExpiry/days_per_year,rate,rate,x.ImpliedVol,include_greeks=True)
                values={'Value':g.value,'Delta':g.delta,'Gamma':g.gamma,'Theta':g.theta}
            for key in values: row[key]+=x.Holding*values[key]
        rows.append(row)
    return pd.DataFrame(rows).set_index('Portfolio')


def option_book_pnl(holdings,spot,terminal_spots,rate,horizon=10,days_per_year=255):
    greeks=option_book_greeks(holdings,spot,rate,days_per_year)
    result={}
    for name,book in holdings.groupby('Portfolio',sort=False):
        value=np.zeros(np.asarray(terminal_spots).shape)
        for x in book.itertuples():
            v=terminal_spots if x.Type=='Stock' else gbsm_scenario_values(x.Type=='Call',terminal_spots,x.Strike,(x.DaysToExpiry-horizon)/days_per_year,rate,rate,x.ImpliedVol)
            value+=x.Holding*v
        result[name]=value-greeks.loc[name,'Value']
    return pd.DataFrame(result)


def distribution_comparison(returns):
    models={'Normal':fit_normal(returns),'Student-t':fit_student_t(returns)}
    table=pd.DataFrame([{'Model':k,**asdict(m),'AICc':aicc(m,returns)} for k,m in models.items()]).set_index('Model')
    return models,table


def zero_mean_terminal_spots(model,spot,horizon,n_paths,seed):
    centered=NormalFit(0,model.sigma) if isinstance(model,NormalFit) else StudentTFit(0,model.sigma,model.nu)
    draws=simulate_fitted_distribution(centered,n_paths*horizon,seed=seed).reshape(n_paths,horizon)
    if np.any(draws<=-1): raise ValueError('Arithmetic model generated a return <= -100%; cannot silently clip it')
    return spot*np.prod(1+draws,axis=1)


def pnl_risk_table(pnl,levels=(.05,.01)):
    rows=[]
    for col in pnl:
        x=pnl[col].to_numpy()
        row={'Portfolio':col,'Mean':x.mean()}
        for a in levels:
            row[f'VaR {a:.0%}']=value_at_risk(x,a);row[f'ES {a:.0%}']=expected_shortfall(x,a)
            row[f'ES/VaR {a:.0%}']=row[f'ES {a:.0%}']/row[f'VaR {a:.0%}']
        rows.append(row)
    return pd.DataFrame(rows).set_index('Portfolio')


def delta_normal_risk(greeks,spot,daily_sigma,horizon=10,levels=(.05,.01)):
    rows=[]
    for name,g in greeks.iterrows():
        sigma=abs(g.Delta)*spot*daily_sigma*np.sqrt(horizon)
        row={'Portfolio':name,'P&L std':sigma}
        for a in levels:
            row[f'VaR {a:.0%}']=-sigma*stats.norm.ppf(a)
            row[f'ES {a:.0%}']=sigma*stats.norm.pdf(stats.norm.ppf(a))/a
        rows.append(row)
    return pd.DataFrame(rows).set_index('Portfolio')


def delta_only_pnl(greeks,spot,terminal_spots):
    ds=np.asarray(terminal_spots,float)-spot
    return pd.DataFrame({name:g.Delta*ds for name,g in greeks.iterrows()})


def taylor_book_pnl(greeks,spot,terminal_spots,horizon=10,days_per_year=255):
    ds=terminal_spots-spot
    return pd.DataFrame({name:g.Delta*ds+.5*g.Gamma*ds**2+g.Theta*horizon/days_per_year for name,g in greeks.iterrows()})


def taylor_mean_components(greeks,spot,terminal_spots,horizon=10,days_per_year=255):
    """Mean delta, gamma and theta contributions for a terminal-spot sample."""
    ds=np.asarray(terminal_spots,float)-spot
    rows=[]
    for name,g in greeks.iterrows():
        rows.append({
            'Portfolio':name,
            'Delta contribution':g.Delta*ds.mean(),
            'Gamma contribution':.5*g.Gamma*np.mean(ds**2),
            'Theta contribution':g.Theta*horizon/days_per_year,
        })
    out=pd.DataFrame(rows).set_index('Portfolio')
    out['Taylor mean']=out.sum(axis=1)
    return out


def asset_diagnostics(returns,rf,periods):
    rows=[]
    for name in returns:
        x=returns[name].to_numpy(); row={'Asset':name,**sample_moments(x)}
        std=x.std(ddof=1); mean_se=std/np.sqrt(len(x))
        row.update({'Std':std,'Annual Sharpe':np.sqrt(periods)*(x.mean()-rf)/std,'Mean SE':mean_se,'Mean / SE':x.mean()/mean_se})
        j=np.argmax(abs(x-x.mean()));rest=np.delete(x,j)
        row.update({'Extreme date':str(returns.index[j]),'Extreme return':x[j],'Fourth moment share':(x[j]-x.mean())**4/np.sum((x-x.mean())**4),'Excess kurtosis without extreme':sample_moments(rest)['Excess Kurtosis']})
        rows.append(row)
    return pd.DataFrame(rows).set_index('Asset')


def tangency_portfolios(returns,rf,cap_name,cap):
    cov=returns.cov().to_numpy(); mu=returns.mean().to_numpy()
    direction=np.linalg.solve(cov,mu-rf)
    if direction.sum()<=0: raise ValueError('Unconstrained tangency normalization does not maximize positive Sharpe')
    bounds=np.tile([0.,1.],(len(mu),1));bounds[list(returns.columns).index(cap_name),1]=cap
    return pd.DataFrame({'Unconstrained':direction/direction.sum(),'Long only':maximize_sharpe_ratio(cov,mu,rf),'Capped':maximize_sharpe_ratio(cov,mu,rf,bounds)},index=returns.columns)


def portfolio_risk_comparison(returns,weights,rf,periods):
    pnl=returns@weights; hist=pnl_risk_table(pnl)
    hist['Annual Sharpe']=np.sqrt(periods)*(pnl.mean()-rf)/pnl.std(ddof=1)
    for name in pnl:
        mu,sigma=pnl[name].mean(),pnl[name].std(ddof=1)
        for a in [.05,.01]:
            z=stats.norm.ppf(a)
            hist.loc[name,f'Normal VaR {a:.0%}']=-mu-sigma*z
            hist.loc[name,f'Normal ES {a:.0%}']=-mu+sigma*stats.norm.pdf(z)/a
    return hist


def efficient_frontier(returns,n=60):
    mu=returns.mean().to_numpy();cov=returns.cov().to_numpy();p=len(mu)
    minv=optimize.minimize(lambda w:w@cov@w,np.ones(p)/p,bounds=[(0,1)]*p,constraints=[{'type':'eq','fun':lambda w:w.sum()-1}],method='SLSQP',options={'ftol':1e-14})
    if not minv.success: raise RuntimeError(minv.message)
    rows=[]
    for target in np.linspace(mu@minv.x,max(mu),n):
        fit=optimize.minimize(lambda w:w@cov@w,minv.x,bounds=[(0,1)]*p,constraints=[{'type':'eq','fun':lambda w:w.sum()-1},{'type':'eq','fun':lambda w:w@mu-target}],method='SLSQP',options={'ftol':1e-13,'maxiter':1000})
        if not fit.success: raise RuntimeError(fit.message)
        rows.append({'Monthly return':fit.x@mu,'Monthly std':np.sqrt(fit.x@cov@fit.x)})
    return pd.DataFrame(rows)


def ols_factor_model(y,factors):
    y=np.asarray(y,float);X=sm.add_constant(factors,has_constant='add');fit=sm.OLS(y,X).fit()
    k=X.shape[1]+1 # Regression coefficients PLUS fitted Gaussian residual variance.
    return {'fit':fit,'coefficients':pd.DataFrame({'Coefficient':fit.params,'SE':fit.bse,'t':fit.tvalues}), 'R2':fit.rsquared,'AICc':-2*fit.llf+2*k+2*k*(k+1)/(len(y)-k-1)}


def capm_with_downside(returns,rf,asset,market,threshold=-0.05):
    """Fit full-sample and downside-only CAPM regressions on excess returns."""
    x=returns[[market]]-rf
    y=returns[asset]-rf
    downside=x[market]<threshold
    return {
        'full':ols_factor_model(y,x),
        'downside':ols_factor_model(y.loc[downside],x.loc[downside]),
        'downside_mask':downside,
    }


def bootstrap_tangency(returns,rf,replications=200,seed=545):
    rng=np.random.default_rng(seed);rows=[]
    for i in range(replications):
        sample=returns.iloc[rng.integers(0,len(returns),len(returns))]
        try: rows.append(maximize_sharpe_ratio(sample.cov(),sample.mean(),rf))
        except RuntimeError: continue
    x=pd.DataFrame(rows,columns=returns.columns)
    return x, pd.DataFrame({'Mean':x.mean(),'5%':x.quantile(.05),'Median':x.median(),'95%':x.quantile(.95)})


def load_factor_files(ff_path,mom_path):
    ff=pd.read_csv(ff_path); mom=pd.read_csv(mom_path)
    ff.columns=ff.columns.str.strip();mom.columns=mom.columns.str.strip()
    for df in [ff,mom]:
        df['Date']=pd.to_datetime(df.Date.astype(str),format='%Y%m%d',errors='raise')
        for col in df.columns.drop('Date'): df[col]=pd.to_numeric(df[col],errors='raise')/100
        if df.Date.duplicated().any(): raise ValueError('Duplicate factor dates')
    return ff.merge(mom,on='Date',how='inner',validate='one_to_one').set_index('Date').sort_index()


def factor_model_comparison(funds,factors,periods=255):
    joined=funds.join(factors,how='inner'); fits={};coef=[];summary=[];corr=[];perf=[]
    for name in funds:
        y=joined[name]-joined.RF
        perf.append({'Fund':name,'Daily excess mean':y.mean(),'Daily volatility':joined[name].std(ddof=1),'Annual Sharpe':np.sqrt(periods)*y.mean()/y.std(ddof=1)})
        for model,cols in {'CAPM':['Mkt-RF'],'FF3':['Mkt-RF','SMB','HML'],'Carhart4':['Mkt-RF','SMB','HML','Mom']}.items():
            result=ols_factor_model(y,joined[cols]);fits[(name,model)]=result
            table=result['coefficients'].reset_index(names='Term');table.insert(0,'Model',model);table.insert(0,'Fund',name);coef.append(table)
            summary.append({'Fund':name,'Model':model,'Alpha':result['fit'].params['const'],'Alpha t':result['fit'].tvalues['const'],'R2':result['R2'],'AICc':result['AICc']})
        residual=fits[(name,'CAPM')]['fit'].resid
        corr.append({'Fund':name,**{c:residual.corr(joined[c]) for c in ['SMB','HML','Mom']}})
    return joined,fits,pd.concat(coef,ignore_index=True),pd.DataFrame(summary),pd.DataFrame(corr).set_index('Fund'),pd.DataFrame(perf).set_index('Fund')


def factor_projections(funds,factors,fits,periods=255,notional=1e7):
    cutoff=factors.index.max()-pd.DateOffset(years=10);recent=factors.loc[factors.index>cutoff]
    premia=recent.mean();worst=factors.Mom.idxmin();shock=factors.loc[worst,'Mom'];rows=[]
    for name in funds:
        p=fits[(name,'Carhart4')]['fit'].params; cols=['Mkt-RF','SMB','HML','Mom']
        annual=periods*(premia.RF+p[cols]@premia[cols])
        loss=-notional*p.Mom*shock
        rows.append({'Fund':name,'Expected annual without alpha':annual,'Expected annual with alpha':annual+periods*p['const'],'Realized annual arithmetic':periods*funds[name].mean(),'Realized CAGR':np.prod(1+funds[name])**(periods/len(funds))-1,'Momentum beta':p.Mom,'Stress loss USD':loss,'Stress loss / daily std':loss/(notional*funds[name].std(ddof=1))})
    return pd.DataFrame(rows).set_index('Fund'),recent, worst,shock


def omitted_factor_alpha_decomposition(joined,fits,fund):
    full=fits[(fund,'Carhart4')]['fit'].params; rows=[]
    for name in ['SMB','HML','Mom']:
        auxiliary=ols_factor_model(joined[name],joined[['Mkt-RF']])['fit']
        rows.append({'Factor':name,'Sample premium':joined[name].mean(),'Market-adjusted premium':auxiliary.params['const'],'Full beta':full[name],'Alpha contribution':full[name]*auxiliary.params['const']})
    return pd.DataFrame(rows).set_index('Factor')


def omitted_factor_alpha_reconciliation(joined,fits,fund):
    decomposition=omitted_factor_alpha_decomposition(joined,fits,fund)
    capm_alpha=fits[(fund,'CAPM')]['fit'].params['const']
    full_alpha=fits[(fund,'Carhart4')]['fit'].params['const']
    change=capm_alpha-full_alpha
    summary=pd.Series({
        'CAPM alpha':capm_alpha,
        'Carhart4 alpha':full_alpha,
        'Alpha decrease':change,
        'Decomposition sum':decomposition['Alpha contribution'].sum(),
        'Reconciliation residual':change-decomposition['Alpha contribution'].sum(),
    },name=fund)
    return decomposition,summary


def capm_fund_diagnostics(funds,factors,periods=255):
    """Fit only CAPM so omitted-factor predictions can precede multifactor fitting."""
    joined=funds.join(factors,how='inner');summary=[];correlations=[];performance=[]
    for name in funds:
        y=joined[name]-joined.RF
        fitted=ols_factor_model(y,joined[['Mkt-RF']])
        f=fitted['fit']
        summary.append({'Fund':name,'Model':'CAPM','Alpha':f.params['const'],'Alpha t':f.tvalues['const'],'R2':fitted['R2'],'AICc':fitted['AICc']})
        correlations.append({'Fund':name,**{c:f.resid.corr(joined[c]) for c in ['SMB','HML','Mom']}})
        performance.append({'Fund':name,'Daily excess mean':y.mean(),'Daily volatility':joined[name].std(ddof=1),'Annual Sharpe':np.sqrt(periods)*y.mean()/y.std(ddof=1)})
    return joined,pd.DataFrame(summary),pd.DataFrame(correlations).set_index('Fund'),pd.DataFrame(performance).set_index('Fund')
