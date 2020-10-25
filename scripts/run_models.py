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


class TimeoutException(Exception):
    pass


def get_timing(path):
    capture = 0
    timing = {}
    with open(path) as f:
        for line in f:
            if capture or (line.startswith("#") and "Elapsed Time" in line):
                capture += 1
                duration = float(
                    re.search(r"(\d+\.\d*)\s+seconds", line, flags=re.IGNORECASE).group(
                        1
                    )
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
    for i, information in enumerate(
        get_model_and_data(offset=offset, num_models=num_models), offset
    ):
        if "FAIL_INFO" in information:
            break
        try:
            model_name = f'{information["data_name"]}-{information["model_name"]}'
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
            }:
                fit_info[model_name] = {
                    "summary": None,
                    "duration_model_seconds": end_build_model_start_fit
                    - start_build_model,
                    "duration_fit_seconds": 60 * 60 * 5,
                    "stan_timing": None,
                }
            else:
                fit = model.sample(
                    data=str(information["data"]),
                    chains=6,
                    seed=42,
                    iter_warmup=500,
                    iter_sampling=500,
                    show_progress=True,
                )

                end_fit = time()

                timing_info = get_timing_from_fit(fit)

                fit_info[model_name] = {
                    "summary": az.summary(fit),
                    "duration_model_seconds": end_build_model_start_fit
                    - start_build_model,
                    "duration_fit_seconds": end_fit - end_build_model_start_fit,
                    "stan_timing": timing_info,
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
    print(DB.posterior_names())
    process_models()
    print(datetime.datetime.now())
