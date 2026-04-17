import pandas as pd
import numpy as np
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
        for feature, perturbation_list in self.perturbations.items():
            self.sensitivities[feature] = self.get_sensitivity(feature)

        return self.sensitivities

    def get_sensitivity(self, feature_name): # gets sentitivity for single inputted feature, uses existing perturbations
        perturbation_pred_list = []
        for perturbation in self.perturbations[feature_name]:
            perturbed_instance = self.instance.copy()

            perturbed_instance[feature_name] = perturbation # update feature for each perturbation and run model
            input_df = pd.DataFrame([perturbed_instance])
            
            perturbed_pred = self.predict_func(input_df)[0] # only first row in array
            slope = (perturbed_pred - self.original_pred) / (perturbation - self.instance[feature_name]) # secant slope
            perturbation_pred_list.append([perturbation, slope])

        regressed_sensitivity = self.regress_sensitivity(perturbation_pred_list, feature_name)
        return regressed_sensitivity

    def regress_sensitivity(self, perturbation_pred_list: list, feature_name, uniformness_factor = 1):
        data = np.array(perturbation_pred_list)
        #reshape array so it works with linear regression
        x_vals = data[:, 0].reshape(-1, 1)
        y_slopes = data[:, 1]


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


# potential changes: 
# perturbations with close to zero delta x will result in unstable slope
# batch predictions rather than one at a time (!!)
# add option to find relative slopes (normalize features before fit) or just absolute slope (raw data)
# visualization option with matplotlib