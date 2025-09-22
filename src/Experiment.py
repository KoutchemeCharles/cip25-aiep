import os
import pandas as pd 
from src.data.CIP import CIPDataset
from src.data.Annotated import AnnotatedDataset
from src.utils.files import create_dir, save_json

class Experiment():

    def __init__(self, config, test_run) -> None:
        self.config = config 
        self.test_run = test_run

        self.__init_directories()

    def __init_directories(self):
        if self.test_run: self.config.name = self.config.name + "_test_run"
        self.save_dir = os.path.join(self.config.save_dir, self.config.name)
        create_dir(self.save_dir)
        save_path = os.path.join(self.save_dir, "experiment_configuration.json")
        save_json(self.config, save_path)
        self.results_save_path = os.path.join(self.save_dir, "generations.csv")

    def run(self):
        raise NotImplementedError()
    
    def load_dataframe(self):
        """ 
        Instantiate the object responsible for processing the
        dataset and handling evaluation functionalities. 
        """

        dataframe = []
        for ds in self.config.dataset:
            if ds.name.startswith("cip"):
                df = CIPDataset(ds).get_data()
            elif ds.name.startswith("annotated"):
                df = AnnotatedDataset(ds).get_data()
            else:
                exp = Experiment(ds, test_run=False)
                df = pd.read_csv(exp.results_save_path)
            
            if self.test_run: df = df.iloc[:1]
            dataframe.append(df)
        
        return pd.concat(dataframe, axis=0, ignore_index=True)


