# SAGE
### Sensitivity Analysis with Gradient Estimation

This project represents an extension of a project made to address wildfire severity prediction in Southern California (see my other repos with "fire" in the name).


## Overview

**SAGE finds model sensitivity to continuous features.** In the context of wildfire prediction, it explains what features to target to *most effectively* change/reduce fire potential.
It works through weighted secant slope regression within the local window of an instance. Unlike current XAI methods that focus on understanding *why* a model
made a given prediction (feature attribution), SAGE focuses on how to most effectively *change* predictions.

Paired with explainers such as SHAP/LIME, SAGE allows users to act on predictions, going a step beyond trust to allow for active risk management.


## Capabilities
- Model-agnostic sensitivity analysis: works with any model via a user-supplied prediction function + background dataset
- Weighted secant slope regression: perturbs each continuous feature within a local window around the instance, computes secant slopes, and fits a gaussian-weighted linear regression to estimate sensitivity at the exact instance value
- Relative sensitivity mode: optionally scales sensitivities by each feature's standard deviation to produce a change in prediction per one standard deviation metric, making feature sensitivities directly comparable
- Selective feature analysis: user can supply features to ignore or include in sensitivity analysis, SAGE automatically filters out non-numeric features
- Sensitivity visualization:  horizontal bar chart showing signed sensitivities per feature, sorted by magnitude, with positive/negative coloring


## Usage

```python
import pandas as pd
from sage import Sage_Explainer

explainer = Sage_Explainer(predict_func=model.predict) # provide predict function

# fit on background dataset
explainer.fit(
    data_X=X_train,
    perturbation_strength=0.3,      # optional, default 0.3
    relative_sensitivities=False,   # optional, set True for comparable scaling
    ignore_features=["feature_a"],  # optional, defaults to none
    used_features=["feature_b", "feature_c"]  # optional, defaults to all
)

# explain a single instance
instance = X_test.iloc[0]
sensitivities = explainer.explain(instance)
print(sensitivities)  # dict of {feature: sensitivity}

# visualize with bar chart
explainer.graph()
```


## Parameters

**`Sage_Explainer(predict_func)`** - must supply a model's prediction function to initialize explainer

**`fit(self, data_X, perturbation_strength, relative_sensitivities, ignore_features, used_features)`**
- `data_X` - required, must be pandas DataFrame
- `perturbation_strength` - optional (default 0.3), scales window of perturbations. Strength=1 indicates perturbations range within 1 standard deviation of each feature's value
- `relative_sensitivities` - optional (default False), scales sensitivities by feature std to make feature sensitivity magnitudes comparable with each other
- `ignore_features` - optional (default empty list), allows SAGE to ignore certain features in sensitivity analysis
- `used_features` - optional (default all features list), allows SAGE to only use certain features in sensitivity analysis

**`explain(self, instance)`** - must supply an instance in the form of a dict or Series to get local feature sensitivities
- Returns a dict with feature sensitivities `{feature: sensitivity}`

**`graph(self, instance)`** - supplying instance (dict or Series) is optional, if not given it defaults to graphing a previously-given instance from explain()
- Returns a matplotlib bar chart of ranked feature sensitivities


## How it works

1. **Fit to dataset:** Compute the standard deviation of each continuous feature across the background dataset. Scale each std by `perturbation_strength` to define the local window radius around any given instance.

2. **Perturb features:** For a given instance, generate evenly spaced perturbation values within the local window `(feature_value - scaled_std, feature_value + scaled_std)` for each feature. The original feature value is excluded from perturbations to avoid division by zero.

3. **Compute secant slopes:** For each perturbation, replace the feature value in the instance and run it through the model. Compute the secant slope between the perturbed prediction and the original prediction: `slope = (perturbed_pred - original_pred) / (perturbed_value - original_value)`. This gives a set of local slope estimates across the window.

4. **Regress to estimate sensitivity:** Fit a gaussian-weighted linear regression over the perturbation values (x) and their corresponding secant slopes (y). Points closer to the original feature value are weighted more heavily. The regression is then evaluated at the original feature value to produce a single sensitivity estimate: the approximate rate of change of the model's prediction with respect to that feature at that instance.

5. **Optional relative scaling:** If `relative_sensitivities=True`, each sensitivity is multiplied by the feature's standard deviation, converting units to *change in prediction per one standard deviation of the feature*. This makes sensitivities directly comparable across features with different scales.


## Limitations

- Compute: for each feature in an instance, SAGE perturbs values, gets model predictions, computes secant slopes, fits a linear model, and predicts sensitivity for original value. Compute scales with the number of features, as a separate regression is fit per feature per instance.
- The current implementation of SAGE only works on continuous features
- Sensitivity is local to instance, SAGE cannot find global feature sensitivities


## Requirements

- pandas
- numpy
- sklearn
- matplotlib
- python 3


## Contact

yashpkher@gmail.com, atharvasoni08@gmail.com
