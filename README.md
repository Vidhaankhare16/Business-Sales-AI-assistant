Checkout the web app here -> [http://51.20.116.180:8501/](http://51.20.116.180:8501/) 
(hosted on aws for now because memory is limited on streamlit cloud services , will be deploying the app using django or flask soon)

[https://business-sales-ai-assistant-sgzze2zamuapalxxbtbpg6.streamlit.app/](https://business-sales-ai-assistant-sgzze2zamuapalxxbtbpg6.streamlit.app/)

## About-Business Sales AI Assistant

An AI assistant that predicts sales, optimizes discounts, and prevents stockouts using causal time-series modeling with pgmpy and sktime.

##  Synthetic Data Description-
The model generates realistic retail scenarios with:
Temporal trends (growth/seasonality) and noise to mimic real sales fluctuations.
Causal relationships like discount curves (diminishing returns) and marketing spikes.
Controlled variables (inventory levels, lead times) to simulate supply chain constraints.

## Features-
Providing prescriptive insights (not just predictions).
Offering templates for causal forecasting in retail/supply chains.
Demonstrating interoperability between sktime (forecasting) and pgmpy (causal AI)

## Parameters Calculated- 
Optimal Discount: Finds the revenue-maximizing price point using quadratic scaling.
Campaign Lift: Measures incremental sales from marketing using causal difference-in-means.
Stockout Risk: Bayesian probability of inventory shortages given demand forecasts.
Safety Stock: Minimum inventory needed to keep stockout probability below 5%.
