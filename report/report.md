# Data mining report
   - Group members: Xinjie Shen, Fanpu Cao, Guanlin Chen, Yang Wu
## 1. Background
   - Project Background and Objectives
   - Introduction to the Dataset

---
![info](info.png)


## 2. Data Cleaning and Exploratory Data Analysis
   - Handling Missing Values
   - Data Visualization (Data Distribution, Correlation, Boxcox Transformation)

---
![corr](corr.png)

## 3. Model Building and Prediction
   - **Selected Models**
   
     In this section, the objective is to predict the 'Life expectancy' column using other features as input. Five models that have very good performance on regression prediction problems and have large differences in algorithmic principles were selected for comparison:
      - Linear Regression
      - ElasticNet
      - K-Nearest Neighbors
      - Support Vector Machine
      - LightGBM

   - **Model Training and Evaluation**
   
     For all models, during the training and testing phase, the following configurations were made:

     - 'Country' and 'Status' columns were label-encoded.
     - 'Year' column was one-hot encoded.
     - 20% of the data was divided as the test dataset.
     - Using Mean Squared Error (MSE) as the objective function.
     - Plotted 2-dimensional scatter plots with fitted lines to visualize the predictive performance.
   
     Following are the detailed configurations of the five models during training and testing:

     - **Linear Regression, ElasticNet, SVM:**
       - Applied standardization for numerical features.
       - ElasticNet's hyperparameter: alpha set as 0.01, l1 ratio value set as 0.5.
       - Support vector Machine's hyperparameter:nu_value set as 0.5.
       
     - **K-Nearest Neighbors (KNN):**
       - K neighbors set as 10.
       - Applied min-max normalization.
       
     - **LightGBM:**
       - Applied necessary preprocessing (label encoding, one-hot encoding).
       - Using early stopping for 200 rounds.
       - Applied 10000 estimators, learning rate is 0.01.
       
     
|       Models       | ElasticNet | Linear Regression |  KNN  | SVM  | LightGBM |
|:------------------:|:----------:|:-----------------:|:-----:|:----:|:--------:|
| Mean Squared Error |   16.43    |       15.60       | 12.33 | 7.50 |   2.81   |
*Note: Identified LightGBM as the best-performing model with an MSE of 2.81.*

---
![Linear_regression](lr.png)

---
![Elastic](elastic.png)

---
![SVM](svm.png)

---
![KNN](knn.png)

---
![LGBM](lgbm.png)
*From the 2D scatter plot, it can be seen that the prediction results of LGBMregressor are closest to the fitted lines, so LGBM is finally chosen as the prediction model.*


## 4. Model Interpretability Analysis
   - Utilization of Interpretability Tool (SHAP)
   - Key Findings in Model Interpretability

---
![](shap1.png)

---
![](shap2.png)

## 5. Feature Engineering
   - Importance of Feature Engineering (Including the Introduction of 100 Redundant Features)
   - Model's Capability in Feature Selection
   - Performance Enhancement through Feature Engineering

---
![](importance.png)

---
![](noise.png)

## 6. Conclusion
   - Summary of the Project
   - Lessons Learned and Insights
   
## 7. Appendix
   - Code snippets, charts, or additional materials (if applicable)
