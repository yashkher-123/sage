import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression



class Sage_Explainer:
    def __init__(self, predict_func):
        self.predict_func = predict_func # user input prediction function

    def fit(self, data_X: pd.DataFrame, perturbation_strength=0.3):

        self.perturbation_factor = 0.3 # perturb feature in range (f_value - (f_std*factor) , f_value + (f_std*factor))
        self.data_X = data_X
        self.std_dict = self.get_scaled_std_ranges(data_X, perturbation_strength) # get feature + scaled std for range radius
        
    def explain(self, instance: dict):
        self.instance = instance

        instance_df = pd.DataFrame([self.instance])
        self.original_pred = self.predict_func(instance_df)[0]

        ranges_dict = {col: (instance[col]-val,instance[col]+val) for col, val in self.std_dict.items()} # dict with perturbation ranges
        self.perturbations = self.get_perturbations(ranges_dict, 10) # dict with feature + all perturbations

        self.sensitivities = {}
        for feature, perturbation_list in self.perturbations.items(): # this is where only continuous features can be chosen
            self.sensitivities[feature] = self.get_sensitivity(feature)

        return self.sensitivities
    
    def graph(self):
        # sort sensitivities by absolute gradient
        sorted_sensitivities = dict(sorted(self.sensitivities.items(), key=lambda item: abs(item[1])))
        
        features = list(sorted_sensitivities.keys())
        values = list(sorted_sensitivities.values())


        colors = ["red" if x < 0 else "green" for x in values]
        
        plt.barh(features, values, color=colors)
        plt.axvline(0, color="black", linewidth=1) # center line
        plt.xlabel("sensitivity")
        plt.ylabel("features")
        plt.title("Feature sensitivities")
        plt.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.show()


    def get_sensitivity(self, feature_name): 

            perturbation_values = self.perturbations[feature_name]
            

            batch_df = pd.DataFrame([self.instance] * len(perturbation_values)) # make new df of num_samples copies of instance
            batch_df[feature_name] = perturbation_values # change given feature series to perturbations
            
            batch_preds = np.ravel(self.predict_func(batch_df)) # single predict function per feature
            # ravel() in case model returns 2d array
            
            original_val = self.instance[feature_name]
            
            # slope = (pred - original_pred) / (perturbed_val - original_val) (secant slope)
            slopes = (batch_preds - self.original_pred) / (np.array(perturbation_values) - original_val)
            # take each prediction, subtract baseline, divide by perturbation minus original feature value

            perturbation_pred_list = np.column_stack((perturbation_values, slopes))

            regressed_sensitivity = self.regress_sensitivity(perturbation_pred_list, feature_name) # get sensitivites using x=perturbation, y=slope
            return regressed_sensitivity
        

    def regress_sensitivity(self, perturbation_pred_list: np.ndarray, feature_name, uniformness_factor = 1):
        #reshape array so it works with linear regression
        x_vals = perturbation_pred_list[:, 0].reshape(-1, 1)
        y_slopes = perturbation_pred_list[:, 1]


        target_val = self.instance[feature_name]

        # normal distribution around true feature value, farther out points have less weight in regression
        std = self.std_dict[feature_name] / self.perturbation_factor # undo the scaling factor/perturbation strength

        # factor > 1: more uniform, 0<factor<1: center more important
        uniformness_strength = std * uniformness_factor

        weights = np.exp(-0.5 * ((x_vals.flatten() - target_val) / uniformness_strength)**2)


        model = LinearRegression()
        model.fit(x_vals, y_slopes, sample_weight=weights)

        target_x = np.array([[self.instance[feature_name]]])

        sensitivity_pred = model.predict(target_x)[0]
        return sensitivity_pred

        # x=perturbation, y = slope (perturbed_pred-original_pred / perturbed_instance[feature_name]-instance[feature_name])
        # linear regression of x vs y, secant slope vs perturbation


    def get_scaled_std_ranges(self, data: pd.DataFrame, perturbation_factor):
        std_dict = data.std(ddof=0).to_dict() # assume population level variance
        std_dict = {col: val * perturbation_factor for col, val in std_dict.items()} # multiply std by perturbation factor
        return std_dict

    def get_perturbations(self, ranges: dict, num_samples):
        perturbation_dict = {}

        for col, (low, high) in ranges.items():
            original_val = (low + high) / 2
            points = np.linspace(low, high, num_samples) # evenly space perturbations based on range+unm_samples
            points = [p for p in points if not np.isclose(p, original_val)] # avoid divide by zero (delta x) when getting slope
            perturbation_dict[col] = points # convert to list and add to dict
            
            
        return perturbation_dict








# UNIT TEST: linear model coefficients should match sage sensitivities

diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
target = diabetes.target
model = LinearRegression()
model.fit(df, target)

explainer = Sage_Explainer(model.predict)
explainer.fit(df)

instance = df.iloc[0].to_dict()
sensitivities = explainer.explain(instance)
print("sensitivities from explainer")
for feature, val in sensitivities.items():
    print(f"{feature:8}: {val:>10.4f}")

print("actual linear model coefficients")
for feature, coef in zip(df.columns, model.coef_):
    print(f"{feature:8}: {coef:>10.4f}")

explainer.graph()


# potential changes: 
# batch predictions rather than one at a time (!!)
# add option to find relative slopes (normalize features before fit) or just absolute slope (raw data)
# do not account for discrete features