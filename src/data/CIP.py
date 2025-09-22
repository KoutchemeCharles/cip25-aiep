import pandas as pd

from src.data.sampling import drop_duplicates
from src.data.sampling import sample_zipf

class CIPDataset():

    def __init__(self, config):
        self.config = config 

    def get_data(self):

        student_data = pd.read_csv(self.config.student_data_path, index_col=False)
        rubrics_data = pd.read_csv(self.config.rubrics_data_path)
        data = student_data.join(rubrics_data.set_index("diag_exercise"), 
                                 on="diag_exercise")
        data = data.dropna(how="all", axis=0).dropna(how="all", axis=1)

        if self.config.exclude_karel:
            data = data[data["diag_exercise"] != "diagnostic3"]

        if self.config.subset:
            print(data["diag_exercise"].unique())
            data = data[data["diag_exercise"].isin(self.config.subset)]
            
        if self.config.drop_duplicates:
            data = drop_duplicates(data)
        elif self.config.zipf_sampling:
            data = sample_zipf(data, total=self.config.zipf_sampling.total, 
                               head=self.config.zipf_sampling.head)

        columns = [c for c in data.columns if "Unnamed" not in c]
        data = data[columns]

        data = data.reset_index(drop=True)
        start, end = 0, len(data)
        if self.config.iloc.start:
            start = self.config.iloc.start
        if self.config.iloc.end:
            end = self.config.iloc.end
        data = data.iloc[start: end]

        return data.reset_index(drop=True)