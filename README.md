# Project-2
Ames Housing Data and Kaggle Challenge

## Problem Statement
To identify the most relevant features from the Ames Housing Dataset and create a regression model that will most accurately generate predictions of the sales price of each house. Thereafter, submit predictions for test dataset to Kaggle to see how the model does against unknown data and find insights from the analysis.

## Executive Summary
This report aims to build a regression model to predict the sale price of the house. Selected features from train dataset were fit into the models and via cross validation, their root mean square errors (rmse) were compared. The score with the lowest rmse was used to make prediction of the house price.

Several data visualisation like histograms, regplot, box plot, pair plots and count plot were plotted to understand the data better, show trends and outliers.

Feature engineering, creation of dummies variables were rendered to generate features to fit into the model.

Four types of regression models - linear, ridge, lasso and elastic were tested and cross validated.

## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|**Id**|*integer*|train/test|Id of each house| 
|**PID**|*integer*|train/test|Parcel identification number - can be used with city web site for parcel review|
|**MS Subclass**|*integer*|train/test|Identifies the type of dwelling involed in the sale| 
|**MS Zoning**|*object*|train/test|Identifies the general zoning classification of the sale| 
|**Lot Frontage**|*float*|train/test|Linear feet of street connected to property| 
|**Lot Area**|*object*|train/test|Lot size in square feet| 
|**Street**|*object*|train/test|Type of road access to property| 
|**Alley**|*object*|train/test|Type of alley access to property| 
|**Lot Shape**|*object*|train/test|General shape of property| 
|**Land Contour**|*object*|train/test|Flatness of the property| 
|**Utilities**|*object*|train/test|Type of utilities available| 
|**Lot Config**|*object*|train/test|Lot configuration| 
|**Land Slope**|*object*|train/test|Slope of property| 
|**Neighbourhood**|*object*|train/test|Physical locations within Ames city limits| 
|**Condition 1**|*object*|train/test|Proximity to various conditions| 
|**Condition 2**|*object*|train/test|Proximity to various conditions (for second condition)| 
|**Bldg Type**|*object*|train/test|Type of dwelling| 
|**House Style**|*object*|train/test|Style of dwelling| 
|**Overall Qual**|*integer*|train/test|Rates the overall materials and finish of the house| 
|**Overall Cond**|*integer*|train/test|Rates the overall condition of the house| 
|**Year Built**|*integer*|train/test|Original construction date| 
|**Year Remod/Add**|*integer*|train/test|Remodel date (same as construction date if there is no remodeling or additions| 
|**Roof Style**|*object*|train/test|Type of roof| 
|**Roof Mat1**|*object*|train/test|Roof material| 
|**Exterior 1**|*object*|train/test|Exterior covering on house| 
|**Exterior 2**|*object*|train/test|Exterior covering on house (if there is a second material|
|**Mas Vnr Type**|*object*|train/test|Masonry veneer type| 
|**Mas Vnr Area**|*float*|train/test|Masonry veneer area in square feet| 
|**Exter Qual**|*object*|train/test|Evaluates the quality of the material on the exterior| 
|**Exter Cond**|*object*|train/test|Evaluates the present condition of the material on the exterior|
|**Foundation**|*object*|train/test|Type of foundation| 
|**Bsmt Qual**|*object*|train/test|Evaluates the height of the basement| 
|**Bsmt Cond**|*object*|train/test|Evaluates the general condition of the basement| 
|**Bsmt Exposure**|*object*|train/test|Refers to walkout or garden level walls|
|**BsmtFin Type 1**|*object*|train/test|Rating of basement finished area|
|**BsmtFin SF 1**|*float*|train/test|Type 1 finished square feet| 
|**BsmtFin Type 2**|*object*|train/test|Rating of basement finished area (if multiple types)| 
|**BsmtFin SF 2**|*float*|train/test|Type 2 finished square feet| 
|**Bsmt Unf SF**|*float*|train/test|Unfinished square feet of basement area| 
|**Total Bsmt SF**|*float*|train/test|Total square feet of basement area| 
|**Heating**|*object*|train/test|Type of heating| 
|**HeatingQC**|*object*|train/test|Heating quality and condition| 
|**Central Air**|*object*|train/test|Central air conditioning| 
|**Electrical**|*object*|train/test|Electrical system| 
|**1st Flr SF**|*integer*|train/test|First floor square feet| 
|**2nd Flr SF**|*integer*|train/test|Second floor square feet| 
|**Low Qual Fin SF**|*integer*|train/test|Low quality finished square feet (all floors)| 
|**Gr Liv Area**|*integer*|train/test|Above grade (ground) living area square feet| 
|**Bsmt Full Bath**|*float*|train/test|Basement full bathrooms|
|**Bsmt Half Bath**|*float*|train/test|Basement half bathrooms|
|**Full Bath**|*integer*|train/test|Full bathrooms above grade| 
|**Half Bath**|*integer*|train/test|Half baths above grade|
|**Bedroom**|*integer*|train/test|Bedrooms above grade (does not include basement bedrooms)|
|**Kitchen**|*integer*|train/test|Kitchens above grade|
|**KitchenQual**|*object*|train/test|Kitchen quality|
|**TotRmsAbvGrd**|*integer*|train/test|Total rooms above grade (does not include bathrooms)|
|**Functional**|*object*|train/test|Home functionality (assume typical unless deductions are warranted)|
|**Fireplaces**|*integer*|train/test|Number of fireplaces|
|**FireplaceQu**|*object*|train/test|Fireplace quality|
|**Garage Type**|*object*|train/test|Garage location| 
|**Garage Yr Blt**|*float*|train/test|Years garage was built| 
|**Garage Finish**|*object*|train/test|Interior finish of the garage| 
|**Garage Cars**|*float*|train/test|Size of garage in car capacity| 
|**Garage Area**|*float*|train/test|Size of garage in square feet| 
|**Garage Qual**|*object*|train/test|Garage quality| 
|**Garage Cond**|*object*|train/test|Garage condition| 
|**Paved Drive**|*object*|train/test|Paved driveway| 
|**Wood Deck SF**|*integer*|train/test|Wood deck area in square feet| 
|**Open Porch SF**|*integer*|train/test|Open porch area in square feet| 
|**Enclosed Porch**|*integer*|train/test|Enclosed porch area in square feet| 
|**3-Ssn Porch**|*integer*|train/test|Three season porch area in square feet| 
|**Screen Porch**|*integer*|train/test|Screen porch area in square feet| 
|**Pool Area**|*integer*|train/test|Pool area in square feet| 
|**Pool QC**|*object*|train/test|Pool quality| 
|**Fence**|*object*|train/test|Fence quality| 
|**Misc Feature**|*object*|train/test|Miscellaneous feature not covered in other categories| 
|**Misc Val**|*integer*|train/test|Money value of miscellaneous feature|
|**Mo Sold**|*integer*|train/test|Month sold| 
|**Yr Sold**|*integer*|train/test|Year sold| 
|**Sale Type**|*integer*|train/test|Type of sale| 
|**Sale Condition**|*object*|train/test|Condition of sale| 
|**SalePrice**|*integer*|train|Sale price|

## Conclusions
Root mean squared errors for cross validation and for residuals are similar across all 4 models. By using 5-fold cross validation, it allows to train and test the models 5 times of different subsets of training data and build up an estimate of the performance of the model on unseen data. By comparing the cross validation scores across the models, lasso and elastic net showed the same and best score. The final re-trained lasso model was used to predict the new test set and saved as a csv file under the datasets folder.

The root mean squared error for all models are in close competition with one another. Elastic Net gave an optimal l1 ratio of 1, which equate to the lasso regression. The lower the root mean squared error, the better the model is able to predict the sale price of the house. For prediction task, either lasso and elastic net worked the best. However, as the rmse errors values do not differ vastly, any of the models could be used to understand the relationship between the features and sale price. The RMSE of the validation sets are also in agreement to those of the train set, hence, the model is suitable for predicting sales prices of houses in the Ames Housing dataset and not overfitted to the test dataset.

Neighborhoods were found to have the largest coefficients and hence, the greatest impact on sale price, followed by grade living area and then overall quality. The negative coefficients of numerical features may be due to their poorer correlation to sale price compared to the rest of the features.

The limitations of this mode include:

1) Limited to houses located in neighborhoods that exist in the original training dataset
2) Limited to data provided in the same format as the original training dataset, where compulsory data/parameters are the variables used in the model

Submission to kaggle was made and a public score of 33317.70280 was obtained.
