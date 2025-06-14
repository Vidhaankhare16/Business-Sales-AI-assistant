Checkout the web app here -> [http://51.20.116.180:8501/](http://51.20.116.180:8501/) 
(hosted on aws for now because memory is limited on streamlit cloud services)

[https://business-sales-ai-assistant-sgzze2zamuapalxxbtbpg6.streamlit.app/](https://business-sales-ai-assistant-sgzze2zamuapalxxbtbpg6.streamlit.app/)

## About-Business Sales AI Assistant

An AI assistant that predicts sales, optimizes discounts, and prevents stockouts using causal time-series modeling with pgmpy and sktime.

##  Synthetic Data Description-
The model generates realistic retail scenarios with:<br>
Temporal trends (growth/seasonality) and noise to mimic real sales fluctuations.<br>
Causal relationships like discount curves (diminishing returns) and marketing spikes.<br>
Controlled variables (inventory levels, lead times) to simulate supply chain constraints.<br>

## Features-
Providing prescriptive insights (not just predictions).<br>
Offering templates for causal forecasting in retail/supply chains.<br>
Demonstrating interoperability between sktime (forecasting) and pgmpy (causal AI)<br>

## Parameters Calculated- 
Optimal Discount: Finds the revenue-maximizing price point using quadratic scaling.<br>
Campaign Lift: Measures incremental sales from marketing using causal difference-in-means.<br>
Stockout Risk: Bayesian probability of inventory shortages given demand forecasts.<br>
Safety Stock: Minimum inventory needed to keep stockout probability below 5%.<br>
