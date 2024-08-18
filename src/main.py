# This will be the main file that runs different jobs.

from preprocessing.preprocessing import DataProcessor
from training.run_training import train
from training.training_functions import load_data_train_test_split
from resources.helper import save_model, load_configs
from inference.run_inference import run_inference
from datetime import datetime as dt


def main(job):
    if job == "save_initial_data":
        data_processor = DataProcessor()
        data_processor.save_initial_data(start_year=2015, update_raw_data=False)

    elif job == "create_training_set":
        data_processor = DataProcessor()
        id, id_time = data_processor.create_training_set(only_fbs=True)
        print(id, id_time)

    elif job == "train_model":
        id = "f1fb347a5d9c11ef8edf3e22fb810d5d"
        id_time = "20240818_160332"
        metadata = dict()
        model_configs = load_configs("configs/training_config.yml")
        feature_configs = load_configs("configs/features.yml")

        metadata["model_name"] = model_configs.get("model_name")
        metadata["model_id"] = id
        metadata["data_datetime"] = id_time
        metadata["time_trained"] = str(dt.now())

        X_train, X_test, y_train, y_test = load_data_train_test_split(id, id_time)

        model, metrics = train(X_train, X_test, y_train, y_test)
        metadata["metrics"] = metrics
        metadata["features"] = list(X_train.columns)
        metadata["target"] = feature_configs["TARGET"]
        save_model(model, metadata)

    elif job == "update_data":
        data_processor = DataProcessor()
        data_processor.update_data()

    elif job == "run_inference":
        run_inference()


if __name__ == "__main__":
    main("run_inference")
