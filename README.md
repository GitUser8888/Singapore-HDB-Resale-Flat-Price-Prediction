#  ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) Project 2 - Singapore Housing Data and Kaggle Challenge

### Project Objectives:
 This project is done as part of the Data Science Immersive course in General Assembly, and was completed in a short time frame of only 2 weeks. We are tasked with creating a regression model based on the [Singapore Public Housing Dataset](https://www.kaggle.com/competitions/dsi-sg-project-2-regression-challenge-hdb-price) to predict the price of resale HDB (Housing and Development Board) flats.

HDB resale prices in Singapore refer to the market value of public housing flats that are sold by existing homeowners to prospective buyers. The HDB is Singapore's public housing authority and plays a crucial role in providing affordable housing for the country's citizens.

Over the years, HDB resale prices have generally experienced an upward trend due to Singapore's economic growth, increasing demand for housing, and limited land availability. The government has implemented various measures to ensure affordability and stability in the public housing market.

However, the HDB resale market remains a big market, and it is important to get ahead in one’s planning when looking to buy or sell a HDB resale flat. For this project, we tackled it in the perspective of a property agent looking to find the best deals for both sellers and buyers.



## The Modeling Process
1. The train dataset has all of the columns needed to generate and refine the models. The test dataset has all of those columns except for the target variable (resale price).
2. Generate regression model using the training data. This process consists of:
    - Data Cleaning
    - EDA
    - Data Visualization
    - Feature Engineering
    - Train-test split
    - Linear Regression
    - Lasso Regression
    - Ridge Regression
    - Refining Models
    - _as a revisit to this project after the end of the Data Science Immersive course, I added in modelling with PyCaret to build a better model with higher R2 and lower RMSE_ 

3. Predict the values for target column (resale price) in the test dataset 
    -  Use of train-test split, cross-validation, and unseen data

4. Evaluate models

---

### Data used:
This dataset is exceptionally detailed with over 70 columns of different features relating to the parameters of the resale flat.

For the purpose of the analysis, we are provided with the `train` and `test` datasets. The `train` dataset contains the resale price of the flats and their relevant information from 2012 to 2021. `train` datasets will be use model building purposes. Information found in the `test` datasets contains the same fields as those found in the `train` dataset, except for the resale prices. Resales prices will be predicted using the trained model

Information found in the `train` datasets includes information such as resale prices, planning_area, flat_model, hdb_age, full_flat_type, mrt_nearest_distance, hawker_nearest_distance and pri_sch_nearest_distance and so on.
The full information could be found in the data dictionary below.



---

### Data Dictionary:

<br>This dataset contains data of resale flats that have been sold from 2012 to 2021. 

| **Column names** | **Descriptions** |
|---|---|
| resale_price | the property's sale price in Singapore dollars. This is the target variable that you're trying to predict for this challenge. |
| Tranc_YearMonth | year and month of the resale transaction e.g. 2015-02 |
| town | HDB township where the flat is located e.g. BUKIT MERAH |
| flat_type | type of the resale flat unit e.g. 3 ROOM |
| block | block number of the resale flat e.g. 454 |
| street_name | street name where the resale flat resides e.g. TAMPINES ST 42 |
| storey_range | floor level (range) of the resale flat unit e.g. 07 TO 09 |
| floor_area_sqm | floor area of the resale flat unit in square metres |
| price_per_sqft | Price per Square Foot of the unit |
| flat_model | HDB model of the resale flat e.g. Multi Generation |
| lease_commence_date | commencement year of the flat units 99-year lease |
| Tranc_Year | year of resale transaction |
| Tranc_Month | month of resale transaction |
| mid_storey | median value of storey_range |
| lower | lower value of storey_range |
| 2room_rental | 2 room rental flat |
| 3room_rental | 3 room rental flat |
| 4room_rental | 4 room rental flat |
| postal | postal code |
| other_room_rental | other room rental flat |
| upper | upper value of storey_range |
| mid | middle value of storey_range |
| full_flat_type | combination of flat_type and flat_model |
| address | combination of block and street_name |
| floor_area_sqft | floor area of the resale flat unit in square feet |
| hdb_age | number of years from lease_commence_date to present year |
| max_floor_lvl | highest floor of the resale flat |
| year_completed | year which construction was completed for resale flat |
| residential | boolean value if resale flat has residential units in the same block |
| commercial | boolean value if resale flat has commercial units in the same block |
| market_hawker | boolean value if resale flat has a market or hawker centre in the same block |
| multistorey_carpark | boolean value if resale flat has a multistorey carpark in the same block |
| precinct_pavilion | boolean value if resale flat has a pavilion in the same block |
| total_dwelling_units | total number of residential dwelling units in the resale flat |
| Latitude | Latitude of the unit |
| Longitude | Longitude of the unit |
| planning_area | planning area of the unit |
| pri_sch_nearest_distance | distance of unit to the nearest primary school |
| 1room_sold | number of 1-room residential units in the resale flat |
| 2room_sold | number of 2-room residential units in the resale flat |
| 3room_sold | number of 3-room residential units in the resale flat |
| 4room_sold | number of 4-room residential units in the resale flat |
| 5room_sold | number of 5-room residential units in the resale flat |
| exec_sold | number of executive type residential units in the resale flat block |
| pri_sch_name | name of the nearest primary school |
| vacancy | vacancy of the unit |
| pri_sch_affiliation | affiliation of primary school |
| pri_sch_latitude | latitude of primary school |
| pri_sch_longitude | longitude of primary school |
| sec_sch_nearest_dist | distance to nearest secondary school |
| sec_sch_name | name of nearest secondary school |
| cutoff_point | PSLE cutoff point of nearest secondary school |
| affiliation | if there is affiliation for the nearest secondary school |
| sec_sch_latitude | latitude of secondary school |
| sec_sch_longitude | longitude of secondary school |
| multigen_sold | number of multi-generational type residential units in the resale flat block |
| mrt_nearest_distance | distance to nearest mrt |
| mrt_name | name of nearest mrt |
| bus_interchange | if there is a bus interchange |
| mrt_interchange | if there is an mrt interchange |
| mrt_latitude | latitude of mrt |
| mrt_longitude | longitude of mrt |
| bus_stop_nearest_distance | distance to nearest bus stop |
| bus_stop_name | name of bus stop |
| bus_stop_latitude | latitude of bus stop |
| bus_stop_longitude | longitude of bus stop |
| Mall_Nearest_Distance | Distance to the nearest mall |
| Mall_Within_500m | How many malls within 500m of the unit |
| Mall_Within_1km | How many malls within 1km of the unit |
| Mall_Within_2km | How many malls within 2km of the unit |
| Hawker_Nearest_Distance | Distance to nearest Hawker Center |
| Hawker_Within_500m | How many Hawker Centers within 500m of the unit |
| Hawker_Within_1km | How many Hawker Centers within 1km of the unit |
| Hawker_Within_2km | How many Hawker Centers within 2km of the unit |
| studio_apartment_sold | number of studio apartment type residential units in the resale flat block |
| 1room_rental | number of 1-room rental residential units in the resale flat block |
| hawker_food_stalls | number of stalls at nearest hawker centre |
| hawker_market_stalls | number of market stalls at nearest hawker centre |

---

### Key takeaways from the project:
1) The Random Forest Regressor model using PyCaret is the best at predicting resale price with the highest R^2 score of 0.9687 and lowest RMSE at 25327.7575

2) Top important features are: floor area, mrt, flat model, max floor level and primary school

---

The value in being able to predict HDB resale prices with machine learning models comes in the following:

1) Plan budget and expectations better.
2) The more information you have, the more you can make an informed choice. 
3) Minimise time and money wastage.
4) Get a better deal in the perspective of either the buyer or seller
