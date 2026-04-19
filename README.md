SAGE (Sensitivity Analysis with Gradient Estimation)

This project represents an extension of my science fair project (1st place city fair, 1st place county fair, California Science and Engineering Fair qualifier). 
Project developed by me (yashpkher@gmail.com) and Atharva Soni (atharvasoni09@gmail.com) as 11th graders. 
Project addressed wildfire severity prediction in Southern California (see my other repos with "fire" in the name).

SAGE finds model sensitivity to continuous features. In the context of wildfire prediction, it explains what features to target to *most effectively* change fire potential. 
It works through weighted secant slope regression within the local window of an instance. Unlike current XAI methods that focus on understanding *why* a model 
made a given prediction (feature attribution), SAGE focuses on how to most effectively *change* predictions.

Paired with explainers such as SHAP/LIME, SAGE allows users to act on predictions, going a step beyond trust. 
