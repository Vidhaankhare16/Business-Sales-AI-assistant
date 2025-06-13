Checkout the web app here -> [http://51.20.116.180:8501/](http://51.20.116.180:8501/) 

[https://business-sales-ai-assistant-sgzze2zamuapalxxbtbpg6.streamlit.app/](https://business-sales-ai-assistant-sgzze2zamuapalxxbtbpg6.streamlit.app/)

## About-Business Sales AI Assistant

A model that uses Bayesian Networks(pgmpy) for causal inference and generates synthetic business sales data for scenario simulation and decision analysis.

##  Synthetic Data Description

The assistant generates a dataset with the following fields:

 BasePrice -  Random base price between ₹500 and ₹2000 |
 Discount  -  Random discount between 0% and 30% applied to each sale |
 Sales     -  Sales quantity derived based on base price and discount using a defined relationship |

 Parameters Calculated
- Sales prediction given values of BasePrice and/or Discount
- Marginal probability distributions for Sales conditioned on user input
- Visualizations of how Sales responds to changes in Discount or BasePrice via probability plots

