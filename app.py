from src.preprocessing.preprocessing import DataProcessor


def main(job):
    if job == "save_inital_data":
        data_processor = DataProcessor()
        data_processor.save_initial_data()


if __name__ == "__main__":
    main("save_initial_data")
