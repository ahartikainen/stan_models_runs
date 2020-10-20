import gzip
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import arviz as az
import cmdstanpy
import posteriordb
import ujson as json

POSTERIORDB_PATH = os.environ.get("POSTERIORDB")

DB = posteriordb.PosteriorDatabase(POSTERIORDB_PATH)


def get_model_and_data():
    """Get model and data from posteriordb."""
    fail_or_not = {"FAIL_INFO": True}
    with tempfile.TemporaryDirectory(prefix="stan_testing_") as tmpdir:
        for p in DB.posteriors():
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


def run():
    fit_info = {}
    for information in get_model_and_data():
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
    return fit_info


def process_models():
    """Run models and get results."""
    fits = run()
    save_path = "./results.pickle.gz"
    with gzip.open(save_path, "wb") as f:
        pickle.dump(fits, f)

    for key, values in fits:
        print(f"\n\n\nmodel: {key}")
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
    process_models()
    print(datetime.datetime.now())
