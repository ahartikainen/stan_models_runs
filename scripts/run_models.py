import gzip
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import arviz as az
import click
import cmdstanpy
import posteriordb
import ujson as json

POSTERIORDB_PATH = os.environ.get("POSTERIORDB")

DB = posteriordb.PosteriorDatabase(POSTERIORDB_PATH)


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
                model_data = f"{model_name}_{data_name}"
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
    fit_info = {}
    for information in get_model_and_data(offset=offset, num_models=num_models):
        if "FAIL_INFO" in information:
            break
        try:
            model_name = f'{information["model_name"]}_{information["data_name"]}'
            model = cmdstanpy.CmdStanModel(
                model_name=model_name, stan_file=information["model_code"]
            )
            fit = model.sample(data=str(information["data"]))
            fit_info[model_name] = {
                "posterior": az.from_cmdstanpy(posterior=fit),
                "rhat": az.rhat(fit),
                "ess_bulk": az.ess(fit, method="bulk"),
                "ess_tail": az.ess(fit, method="tail"),
                "summary": az.summary(fit),
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
    save_path = "./results.pickle.gz"
    with gzip.open(save_path, "wb") as f:
        pickle.dump(fits, f)

    for i, (key, values) in enumerate(fits, offset):
        if key == "FAIL_INFO":
            print("\n\nFAIL vs SUCCESS")
            print(f"Total: {sum(values.values())}\n\n")
            for model, success in values.items():
                print(f"model: {model} -> {success}")
            print("\n\n")
            continue
        print(f"\n\n\nmodel: {i} {key}")
        for stat, val in values.items():
            if stat == "posterior":
                print(stat, val.posterior)
                print(stat, val.sample_stats)
            else:
                print(stat, val)
            print("\n")
    print("Models run")


if __name__ == "__main__":
    import datetime

    print(datetime.datetime.now())
    print(POSTERIORDB_PATH)
    print(DB.posterior_names())
    process_models()
    print(datetime.datetime.now())
