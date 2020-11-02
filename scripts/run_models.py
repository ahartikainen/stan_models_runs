"""Sample models from posteriordb."""
import gzip
import logging
import os
import pickle
import re
import shutil
import signal
import tempfile
from contextlib import contextmanager
from pathlib import Path
from time import time

import arviz as az
import click
import cmdstanpy
import numpy as np
import pandas as pd
import posteriordb
import ujson as json

POSTERIORDB_PATH = os.environ.get("POSTERIORDB")

DB = posteriordb.PosteriorDatabase(POSTERIORDB_PATH)

logging.basicConfig(level=logging.WARNING)


def get_timing(path):
    capture = 0
    timing = {}
    with open(path) as f:
        for line in f:
            if capture or (line.startswith("#") and "Elapsed Time" in line):
                capture += 1
                duration = float(
                    re.search(
                        r"(\d+.\d*e?-?\d+)\s+seconds", line, flags=re.IGNORECASE
                    ).group(1)
                )
                if "warm-up" in line.lower():
                    key = "warm-up"
                elif "sampling" in line.lower():
                    key = "sampling"
                timing[key] = duration
            if capture > 1:
                break
    return timing


def get_timing_from_fit(fit):
    timing_chains = {}
    for i, path in enumerate(fit.runset.csv_files, 1):
        timing_chains[i] = get_timing(path)
    return pd.DataFrame.from_dict(timing_chains, orient="index")


def get_gradient_timing(path):
    gradient_time = None
    with open(path, "r") as f:
        match = re.search(
            r"Gradient evaluation took (\d+.\d*e?-?\d+) seconds", f.read()
        )
        if match:
            gradient_time = float(match.group(1))
    return gradient_time


def get_gradient_timing_from_fit(fit):
    gradient_times = []
    for path in fit.runset.stdout_files:
        gradient_times.append(get_gradient_timing(path))
    return gradient_times


def get_max_treedepth(path):
    depth = None
    with open(path, "r") as f:
        match = re.search(r"max_depth = (\d+)", f.read())
        if match:
            depth = int(match.group(1))
    return depth


def get_max_treedepth_from_fit(fit):
    depths = []
    for path in fit.runset.stdout_files:
        depths.append(get_max_treedepth(path))
    return depths


def get_model_and_data(offset=0, num_models=-1):
    """Get model and data from posteriordb."""
    fail_or_not = {"FAIL_INFO": True}
    model_count = 0
    with tempfile.TemporaryDirectory(prefix="stan_testing_") as tmpdir:
        for i, p in enumerate(DB.posteriors(), 0):
            if i < offset:
                continue
            if (num_models > 0) and ((model_count + 1) > num_models):
                break
            model_count += 1
            try:
                model_name = p.posterior_info["model_name"]
                data_name = p.posterior_info["data_name"]
                model_data = f"{data_name}-{model_name}"
                try:
                    model_code = Path(DB.get_model_code_path(model_name, "stan"))
                except FileNotFoundError:
                    print(f"{model_name}: Missing Stan code")
                    fail_or_not[model_data] = (False, "missing code")
                    continue
                try:
                    data = DB.data(data_name)
                except FileNotFoundError:
                    print(f"{data_name}: Missing data")
                    fail_or_not[model_data] = (False, "missing data")
                    continue

                tmpdir_path = Path(tmpdir)
                (tmpdir_path / model_data).mkdir(exist_ok=True)

                new_model_code = tmpdir_path / model_data / model_code.name
                new_data = tmpdir_path / model_data / f"{data_name}.json"

                shutil.copy2(str(model_code), str(new_model_code))
                with new_model_code.open("r") as f:
                    stan_code = f.read()
                stan_code = stan_code.replace("<-", "=")
                with new_model_code.open("w") as f:
                    print(stan_code, file=f)

                with new_data.open("w") as f:
                    json.dump(data.values(), f)

                yield {
                    "model_name": model_name,
                    "data_name": data_name,
                    "model_code": new_model_code,
                    "data": new_data,
                }
                fail_or_not[model_data] = (True, None)
            except:
                fail_or_not[model_data] = (False, "unknown reason")
                continue
        yield fail_or_not


def run(offset=0, num_models=-1):
    """Compile and sample models."""
    fit_info = {}

    chains = 4
    warmup_draws = 500
    draws = 500

    for i, information in enumerate(
        get_model_and_data(offset=offset, num_models=num_models), offset
    ):
        if "FAIL_INFO" in information:
            break
        try:
            model_name = f'{information["data_name"]}-{information["model_name"]}'
            print(f"Starting process for model: {model_name}", flush=True)
            start_build_model = time()
            model = cmdstanpy.CmdStanModel(
                model_name=model_name, stan_file=information["model_code"]
            )
            end_build_model_start_fit = time()

            if i in {
                9,  # stuck warmup
                32,  # slow sampling
                60,  # seed 42 -> chain 1 stuck warmup
                76,  # slow sampling
                77,  # slow sampling
            }:

                fit_info[model_name] = {
                    "cmdstanpy_model_duration": end_build_model_start_fit
                    - start_build_model,
                }
            else:
                fit = model.sample(
                    data=str(information["data"]),
                    chains=chains,
                    seed=42,
                    iter_warmup=warmup_draws,
                    iter_sampling=draws,
                    show_progress=True,
                    save_warmup=True,
                )

                end_fit = time()

                # Stan timings
                stan_timing_info = get_timing_from_fit(fit)

                # Diagnostics
                divergent = fit.draws()[
                    :, :, np.array(fit.column_names) == "divergent__"
                ].astype(bool)
                n_divergent = divergent.sum(0)

                treedepth = fit.draws()[
                    :, :, np.array(fit.column_names) == "treedepth__"
                ]
                n_tree_depth = treedepth.sum(0).ravel()

                max_tree_depth_value = get_max_treedepth_from_fit(fit)
                if len(set(max_tree_depth_value)) > 1:
                    print("WHAT", max_tree_depth_value)
                max_tree_depth_value = max_tree_depth_value[0]
                n_max_tree = (treedepth == max_tree_depth_value).sum(0)

                n_leapfrogs = (
                    fit.draws()[:, :, np.array(fit.column_names) == "n_leapfrog__"]
                    .astype(int)
                    .sum(0)
                )

                stan_gradient_timing_info = get_gradient_timing_from_fit(fit)
                stan_gradient_timing_info_sampling = (
                    n_tree_depth / stan_timing_info["sampling"].values
                )

                summary = az.summary(fit)

                lp_ess_bulk = az.ess(
                    fit.draws()[:, :, np.array(fit.column_names) == "lp__"].squeeze().T,
                    method="bulk",
                )
                lp_ess_tail = az.ess(
                    fit.draws()[:, :, np.array(fit.column_names) == "lp__"].squeeze().T,
                    method="tail",
                )

                min_ess_bulk_par = (
                    summary["ess_bulk"].idxmin()
                    if summary["ess_bulk"] < lp_ess_bulk
                    else "lp__"
                )
                min_ess_bulk = min(summary["ess_bulk"].min(), lp_ess_bulk)
                min_ess_tail_par = (
                    summary["ess_tail"].idxmin()
                    if summary["ess_tail"] < lp_ess_tail
                    else "lp__"
                )
                min_ess_tail = min(summary["ess_tail"].min(), lp_ess_tail)

                stan_total_time = stan_timing_info.values.sum()

                fit_info[model_name] = {
                    "cmdstanpy_model_duration": end_build_model_start_fit
                    - start_build_model,
                    "cmdstanpy_fit_duration": end_fit - end_build_model_start_fit,
                    "stan_timing": stan_timing_info,
                    "stan_gradient_timing": stan_gradient_timing_info,
                    "stan_gradient_timing_sampling": stan_gradient_timing_info_sampling,
                    "n_divergent": n_divergent,
                    "n_tree_depth": n_tree_depth,
                    "n_max_tree": n_max_tree,
                    "n_leapfrogs": n_leapfrogs,
                    "min_ess_bulk": (min_ess_bulk_par, min_ess_bulk),
                    "min_ess_tail": (min_ess_tail_par, min_ess_tail),
                    "min_ess_bulk_per_draw": min_ess_bulk / (draws * chains),
                    "min_ess_bulk_per_second": min_ess_bulk / stan_total_time,
                    "min_ess_tail_per_draw": min_ess_tail / (draws * chains),
                    "min_ess_tail_per_second": min_ess_tail / stan_total_time,
                    "stepsize": fit.stepsize,
                    "chains": chains,
                    "draws": draws,
                    "warmup_draws": warmup_draws,
                }
        except Exception as e:
            print(e)
            continue

    information.pop("FAIL_INFO")
    fit_info["FAIL_INFO"] = information
    return fit_info


@click.command()
@click.option("--offset", default=0, help="Skip # models.")
@click.option("--num_models", default=-1, help="Iterate through # of models")
def process_models(offset, num_models):
    """Run models and get results."""
    print(DB.posterior_names()[offset : offset + num_models])
    fits = run(offset=offset, num_models=num_models)
    os.makedirs("./results", exist_ok=True)
    save_path = f"./results/results_offset_{offset}_num_models_{num_models}.pickle.gz"
    with gzip.open(save_path, "wb") as f:
        pickle.dump(fits, f)

    for i, (key, values) in enumerate(fits.items(), offset):
        if key == "FAIL_INFO":
            print("\n\nFAIL / SUCCESS")
            print(
                f"{sum(not item for item, _ in values.values())} / {sum(item for item, _ in values.values())}\n\n"
            )
            for model, success in values.items():
                if success[0]:
                    print(f"model: {success[0]} <- {model}")
                else:
                    print(f"model: {success[0]} <- {model} <- {success[1]}")
            print("\n\n")
            continue
        print(f"\n\n\nmodel: {i} {key}")
        for stat, val in values.items():
            if "duration" in stat:
                print(stat, val)
            else:
                print(stat)
                print(val)
            print("\n")
    print("Selected models run")


if __name__ == "__main__":
    import datetime

    print(datetime.datetime.now())
    print(POSTERIORDB_PATH)
    process_models()
    print(datetime.datetime.now())
