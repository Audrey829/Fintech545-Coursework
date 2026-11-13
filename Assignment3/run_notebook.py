import os,sys,io,contextlib,traceback,base64,json
from pathlib import Path
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import types
ipd=types.ModuleType("IPython.display")
ip=types.ModuleType("IPython")
ip.get_ipython=lambda:None
ip.version_info=(9,0,0)
ip.display=ipd
sys.modules["IPython"]=ip
sys.modules["IPython.display"]=ipd
p=Path(sys.argv[1]).resolve();os.chdir(p.parent)
nb=nbformat.read(p,as_version=4);ns={'__name__':'__main__'};outputs=[];figs=[]
def display(*objs,**kwargs):
    for obj in objs:
        data={'text/plain':str(obj)}
        if hasattr(obj,'_repr_html_'): data['text/html']=obj._repr_html_()
        outputs.append(nbformat.v4.new_output('display_data',data=data,metadata={}))
def show(*args,**kwargs):
    for number in plt.get_fignums():
        fig=plt.figure(number);buf=io.BytesIO();fig.savefig(buf,format='png',dpi=140,bbox_inches='tight')
        outputs.append(nbformat.v4.new_output('display_data',data={'image/png':base64.b64encode(buf.getvalue()).decode()},metadata={}))
        image_path=p.parent/f'figure_{len(figs)+1}.png';image_path.write_bytes(buf.getvalue());figs.append(str(image_path))
    plt.close('all')
ipd.display=display;plt.show=show
for count,cell in enumerate([c for c in nb.cells if c.cell_type=='code'],1):
    print('Executing',count,flush=True);outputs=[];buf=io.StringIO();cell.execution_count=count
    try:
        with contextlib.redirect_stdout(buf): exec(compile(cell.source,f'{p.name}:cell{count}','exec'),ns)
    except Exception:
        cell.outputs=outputs+[nbformat.v4.new_output('stream',name='stdout',text=buf.getvalue()),nbformat.v4.new_output('error',ename=sys.exc_info()[0].__name__,evalue=str(sys.exc_info()[1]),traceback=traceback.format_exc().splitlines())]
        nbformat.write(nb,p);raise
    # Plain stdout is grouped at the end of each cell; tables/figures retain display order.
    cell.outputs=outputs+([nbformat.v4.new_output('stream',name='stdout',text=buf.getvalue())] if buf.getvalue() else [])
    nbformat.write(nb,p)
summary={}
for key in ['parity','iv','prices2','convergence2','greeks2','sens2','greeks3','flat3','fits3','dn3','diag4','weights4','risk4','boot_summary4','perf5','summary5','projection5','alpha_decomp5','rescorr5']:
    summary[key]=ns[key].to_json(orient='split')
summary['mean_components3']=ns['mean_components3'].to_json(orient='split')
summary['alpha_recon5']=ns['alpha_recon5'].to_json()
summary['risk3']={k:v.to_json(orient='split') for k,v in ns['risk3'].items()}
summary['approx3']={k:v.to_json(orient='split') for k,v in ns['approx3'].items()}
summary['boundary2']=ns['boundary2'];summary['shock5']=ns['shock5'];summary['worst_date5']=str(ns['worst_date5'])
summary['capm4']=ns['capm4']['coefficients'].to_json(orient='split');summary['tailcapm4']=ns['tailcapm4']['coefficients'].to_json(orient='split')
summary['capm4_R2']=ns['capm4']['R2'];summary['tailcapm4_R2']=ns['tailcapm4']['R2'];summary['down_months']=int(ns['down4'].sum())
summary['recent_window']=[str(ns['recent5'].index.min()),str(ns['recent5'].index.max())]
(p.parent/'results.json').write_text(json.dumps(summary,indent=2))
nbformat.validate(nb)
print('ALL CELLS EXECUTED',flush=True)
