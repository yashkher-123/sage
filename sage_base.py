import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression



class Sage_Explainer:
    def __init__(self, predict_func):
        self.predict_func = predict_func # user input prediction function

    def fit(self, data_X: pd.DataFrame):

        self.perturbation_factor = 0.3 # perturb feature in range (f_value - (f_std*factor) , f_value + (f_std*factor))
        self.data_X = data_X
        self.std_dict = self.get_scaled_std_ranges(data_X, 0.3) # get feature + scaled std for range radius
        
    def explain(self, instance: dict):
        self.instance = instance
        ranges_dict = {col: (instance[col]-val,instance[col]+val) for col, val in self.std_dict.items()} # dict with perturbation ranges
        self.perturbations = self.get_perturbations(ranges_dict, 10) # dict with feature + all perturbations

    

    def get_scaled_std_ranges(self, data: pd.DataFrame, perturbation_factor):
        std_dict = data.std(ddof=0).to_dict() # assume population level variance
        std_dict = {col: val * perturbation_factor for col, val in std_dict.items()} # multiply std by perturbation factor
        return std_dict

    def get_perturbations(self, ranges: dict, num_samples):
        perturbation_dict = {}

        for col, (low, high) in ranges.items():
            points = np.linspace(low, high, num_samples) # evenly space perturbations based on range+unm_samples
            perturbation_dict[col] = points.tolist() # convert to list and add to dict
            
        return perturbation_dict






"""


diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
target = diabetes.target
model = LinearRegression()
model.fit(df, target)

explainer = Sage_Explainer(model.predict)
explainer.fit(df)
"""
