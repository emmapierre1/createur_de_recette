class Trainer(object):
    def __init__(self):
        """
        Class constructor
        """
        pass

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        pass

    def run(self):
        "create the pipeline then fit the model on features"
        self.set_pipeline()
        pass

    def save_model_locally(self):
        """Save the model into a .joblib format"""
        # joblib.dump(self.pipeline, 'model.joblib')
        # print(colored("model.joblib saved locally", "green"))


    def load_model(filename):
        "load model"
        # return joblib.load(filename)
        pass

if __name__ == "__main__":
    "process"
    pass
