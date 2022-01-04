import data
from trainer import Trainer

if __name__ == "__main__":
    dataset_filtered = data.get_recipes_string_list()
    lstm = Trainer()
    model = lstm.get_model(dataset_filtered)
    lstm.generate_combinations(model)