from logging_output_scripts.utils import check_and_create_dir, get_dataframe, get_all_runs, get_df
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import mlflow
import os
from sklearn.preprocessing import MinMaxScaler

"""
Uses seaborn-package to create MSE-Time and Complexity-Time Plots comparing model performances
on multiple datasets
"""
REGRESSOR_CONFIG_PATH = 'logging_output_scripts/config_regression.json'
CLASSIFIER_REGRESSOR_CONFIG_PATH = 'logging_output_scripts/config_classification.json'
sns.set_style("whitegrid")
sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
sns.set_theme(style="whitegrid",
              font="Times New Roman",
              font_scale=1,
              rc={
                  "lines.linewidth": 1,
                  "pdf.fonttype": 42,
                  "ps.fonttype": 42
              })

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.tight_layout()

def get_histogram(experiment_name, dataset_name, metric_name, steps, isClassifier = False):
    config_path = REGRESSOR_CONFIG_PATH
    if isClassifier:
        config_path = CLASSIFIER_REGRESSOR_CONFIG_PATH
    with open(config_path) as f:
        config = json.load(f)

    client = mlflow.tracking.MlflowClient()
    all_runs = [item for item in next(os.walk(config['data_directory']))[1] if item != '.trash']
    for run in all_runs:
        exp = client.get_experiment(run)
        if dataset_name in exp.name and experiment_name in exp.name and "n:" not in exp.name:

            run_ids = [item for item in next(os.walk(config['data_directory']+ '/' + str(run)))[1] if item != '.trash']
            exp_res = []
            for i_run, id in enumerate(run_ids):
                run = client.get_run(id)
                if 'fold' in run.data.tags and run.data.tags['fold'] == 'True':
                    metrics = client.get_metric_history(id, metric_name)
                    run_res = np.zeros((len(metrics)))
                    for i, metric in enumerate(metrics):
                        run_res[i] = metric.value
                    exp_res.append(run_res)
            exp_res = np.average(np.array(exp_res), axis=0)
            return exp_res[:steps]
    print(f"Could not find {experiment_name} for {dataset_name}")
    pass


def create_plots(metric_name='elitist_complexity', steps=64, isClassifier=False):
    print("STARTING timeline-plots")
    config_path = REGRESSOR_CONFIG_PATH
    if isClassifier:
        config_path = CLASSIFIER_REGRESSOR_CONFIG_PATH
    with open(config_path) as f:
        config = json.load(f)
    final_output_dir = f"{config['output_directory']}"
    output_dir = "time_plots"
    check_and_create_dir(final_output_dir, output_dir)

    for dataset_name in config['datasets']:
        results = [[],[],[]]
        legend_labels = []
        for model in config['model_names']:
            model_name = f"l:{model}"
            result = get_histogram(model_name, dataset_name, metric_name, steps, isClassifier=isClassifier)
            if result is None:
                continue
            for i, res in enumerate(result):
                results[0].append(res)
                results[1].append(i)
                results[2].append(model_name)
            legend_labels.append(config['model_names'][model])

        results = {metric_name: results[0], 'step': results[1], 'model_name': results[2]}
        res_data = pd.DataFrame(results)

        def ax_config(axis):
            axis.set_xlabel('Iteration', weight="bold")
            axis.set_ylabel(config['metrics'][metric_name], weight="bold")
            axis.legend(title='Local models', labels=legend_labels)
  
        title_dict = {"concrete_strength": "Concrete Strength",
                      "combined_cycle_power_plant": "Combined Cycle Power Plant",
                      "airfoil_self_noise": "Airfoil Self Noise",
                      "energy_heat": "Energy Efficiency Heating",
                      "breastcancer": "Breast Cancer",
                      "raisin": "Raisin",
                      "abalone": "Abalone"}

        fig, ax = plt.subplots()
        ax.set_title(title_dict[dataset_name], style="italic")
        ax = sns.lineplot(x='step', y=metric_name, data=res_data, style='model_name', hue='model_name')
        ax_config(ax)
        fig.savefig(f"{final_output_dir}/{output_dir}/{dataset_name}_{metric_name}.png")
    



if __name__ == '__main__':
    create_plots()#metric_name="elitist_error")