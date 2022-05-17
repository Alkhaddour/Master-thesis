
import os, sys
sys.path.append('../..')
from configs.global_config import OUTPUT_PATH
import pandas as pd

exp_out_dir = os.path.join(OUTPUT_PATH, 'exp_1')

# combine run files
n_runs = 4
csvs=[]
for r in range(n_runs):
    run_file = os.path.join(exp_out_dir ,f'summary_run{r}.txt')
    csvs.append(pd.read_csv(run_file,names=['exp name', 'loss']))
    run_results = pd.concat(csvs, axis=0)

print("min loss")
print(run_results.loc[run_results['loss']==min(run_results['loss'])]['exp name'].item())

print("max loss")
print(run_results.loc[run_results['loss']==max(run_results['loss'])])

print("small loss")
print(run_results.loc[run_results['loss']< 3].to_csv('tmppp.csv'))