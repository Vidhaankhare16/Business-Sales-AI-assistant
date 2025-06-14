import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.ets import AutoETS
from pgmpy.inference import VariableElimination
from sklearn.preprocessing import KBinsDiscretizer

# App title and description
st.title("Electrolux Causal Time Series Demo")
st.markdown("""
This demo showcases how causal learning with time series can be applied to retail and supply chain scenarios.
We combine `sktime` for time series forecasting and `pgmpy` for causal modeling.
""")

# Tabs for different functionalities
tab1, tab2, tab3 = st.tabs([
    "Optimal Discount Analysis",
    "Marketing Campaign Impact",
    "Supply Chain Optimization"
])

# ================== Tab 1: Optimal Discount Analysis ==================
with tab1:
    st.header("Optimal Discount Level Analysis")
    st.write("""
    This analysis helps determine the best discount level to maximize revenue while maintaining profitability.
    We use historical sales data and a causal model to estimate the impact of different discount levels.
    """)
    
    # Generating synthetic data
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", end="2023-01-01", freq="W")
    base_sales = np.random.normal(1000, 100, len(dates))
    discounts = np.random.uniform(0, 0.3, len(dates))
    sales_impact = 5000 * discounts - 8000 * discounts**2
    sales = base_sales + sales_impact + np.random.normal(0, 50, len(dates))
    
    df_discount = pd.DataFrame({
        "Date": dates,
        "Sales": sales,
        "Discount": discounts,
        "BasePrice": 1000
    })
    
    if st.checkbox("Show raw discount data"):
        st.dataframe(df_discount)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_discount, x="Discount", y="Sales", ax=ax)
    ax.set_title("Sales vs Discount Level")
    st.pyplot(fig)
    
    st.subheader("Causal Model for Discount Impact")
    
    # Discretize data for Bayesian Network
    discount_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    sales_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    
    df_disc = df_discount.copy()
    df_disc['Discount_bin'] = discount_bins.fit_transform(df_disc[['Discount']])
    df_disc['Sales_bin'] = sales_bins.fit_transform(df_disc[['Sales']])
    
    model = DiscreteBayesianNetwork([('Discount_bin', 'Sales_bin'), ('BasePrice', 'Sales_bin')])
    model.fit(df_disc.drop(columns=["Date", "Discount", "Sales"]), estimator=MaximumLikelihoodEstimator)
    
    infer = VariableElimination(model)
    discount_levels = np.linspace(0, 0.3, 31)
    expected_sales = []
    
    for d in discount_levels:
        d_bin = int(discount_bins.transform([[d]])[0][0])
        query = infer.query(variables=['Sales_bin'], evidence={'Discount_bin': d_bin, 'BasePrice': 1000})
        # Convert back to sales value using bin midpoints
        expected_sales.append(np.sum(query.values * sales_bins.inverse_transform([[i] for i in range(5)]).flatten()))
    
    optimal_discount = discount_levels[np.argmax(expected_sales)]
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(discount_levels, expected_sales)
    ax2.axvline(optimal_discount, color='r', linestyle='--', label=f'Optimal Discount: {optimal_discount:.2%}')
    ax2.set_xlabel("Discount Level")
    ax2.set_ylabel("Expected Sales")
    ax2.set_title("Expected Sales vs Discount Level")
    ax2.legend()
    st.pyplot(fig2)
    
    st.success(f"The optimal discount level is approximately {optimal_discount:.1%}")

# ================== Tab 2: Marketing Campaign Impact ==================
with tab2:
    st.header("Marketing Campaign Impact Analysis")
    st.write("""
    This analysis estimates the causal impact of marketing campaigns on weekly sales.
    We combine time series forecasting (sktime) with causal inference (pgmpy).
    """)
    
    np.random.seed(42)
    # Generate 3 years of data (156 weeks) to ensure enough seasonal cycles
    dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="W")
    n = len(dates)
    
    trend = np.linspace(1000, 1500, n)
    seasonality = 100 * np.sin(np.linspace(0, 6*np.pi, n))  # 3 full cycles for 3 years
    base_sales = trend + seasonality + np.random.normal(0, 50, n)
    
    campaigns = np.zeros(n)
    campaign_weeks = np.random.choice(n, size=15, replace=False)
    campaigns[campaign_weeks] = 1
    
    campaign_impact = 200 * campaigns + np.random.normal(0, 30, n)
    sales = base_sales + campaign_impact
    
    df_campaign = pd.DataFrame({
        "Date": dates,
        "Sales": sales,
        "Campaign": campaigns,
        "Trend": trend,
        "Seasonality": seasonality
    }).set_index("Date")
    
    if st.checkbox("Show raw campaign data"):
        st.dataframe(df_campaign)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_campaign.index, df_campaign['Sales'], label='Actual Sales')
    campaign_dates = df_campaign[df_campaign['Campaign'] == 1].index
    for date in campaign_dates:
        ax.axvline(date, color='r', alpha=0.3)
    ax.set_title("Sales Data with Marketing Campaigns (red lines)")
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Time Series Forecasting with Campaign Impact")
    y = df_campaign['Sales'].to_frame()
    train = y.iloc[:-20]
    test = y.iloc[-20:]
    
    # Use try-except to handle AutoETS initialization
    try:
        forecaster = AutoETS(auto=True, sp=52, n_jobs=-1)  # Weekly seasonality (52 weeks/year)
        forecaster.fit(train)
    except ValueError as e:
        st.warning(f"Warning: {str(e)}. Falling back to non-seasonal ETS model.")
        forecaster = AutoETS(auto=True, sp=1, n_jobs=-1)  # Non-seasonal model as fallback
        forecaster.fit(train)
    
    fh = ForecastingHorizon(test.index, is_relative=False)
    y_pred = forecaster.predict(fh)
    
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    train['Sales'].plot(ax=ax2, label='Train')
    test['Sales'].plot(ax=ax2, label='Test')
    y_pred.plot(ax=ax2, label='Forecast')
    ax2.set_title("Sales Forecast without Campaign Information")
    ax2.legend()
    st.pyplot(fig2)
    
    st.subheader("Causal Impact Estimation")
    
    # Discretize data for Bayesian Network
    campaign_bins = KBinsDiscretizer(n_bins=2, encode='ordinal', strategy='uniform')
    sales_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    trend_bins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    
    df_camp = df_campaign.reset_index().copy()
    df_camp['Campaign_bin'] = campaign_bins.fit_transform(df_camp[['Campaign']])
    df_camp['Sales_bin'] = sales_bins.fit_transform(df_camp[['Sales']])
    df_camp['Trend_bin'] = trend_bins.fit_transform(df_camp[['Trend']])
    
    causal_model = DiscreteBayesianNetwork([('Campaign_bin', 'Sales_bin'), ('Trend_bin', 'Sales_bin')])
    causal_model.fit(df_camp[['Campaign_bin', 'Sales_bin', 'Trend_bin']], estimator=MaximumLikelihoodEstimator)
    
    infer = VariableElimination(causal_model)
    campaign_effect = infer.query(variables=['Sales_bin'], evidence={'Campaign_bin': 1, 'Trend_bin': 1}).values[0] - \
                      infer.query(variables=['Sales_bin'], evidence={'Campaign_bin': 0, 'Trend_bin': 1}).values[0]
    campaign_effect_value = campaign_effect * sales_bins.inverse_transform([[1]]).flatten()[0]
    
    st.info(f"Estimated average campaign impact: {campaign_effect_value:.0f} additional sales per campaign week")

# ================== Tab 3: Supply Chain Optimization ==================
with tab3:
    st.header("Supply Chain Optimization")
    st.write("""
    This analysis helps configure a supply chain to minimize stockout events.
    We model the relationship between inventory levels, lead times, and stockouts.
    """)
    
    np.random.seed(42)
    n_weeks = 100
    lead_time = np.random.randint(1, 4, n_weeks)
    inventory_level = np.random.uniform(50, 150, n_weeks)
    demand = np.random.poisson(lam=30, size=n_weeks) + (inventory_level * 0.2)
    stockout = (demand > inventory_level).astype(int)
    
    df_supply = pd.DataFrame({
        "LeadTime": lead_time,
        "InventoryLevel": inventory_level,
        "Demand": demand,
        "Stockout": stockout
    })
    
    if st.checkbox("Show raw supply chain data"):
        st.dataframe(df_supply)
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.scatterplot(data=df_supply, x="InventoryLevel", y="Stockout", ax=ax[0])
    ax[0].set_title("Stockout vs Inventory Level")
    
    sns.boxplot(data=df_supply, x="LeadTime", y="Stockout", ax=ax[1])
    ax[1].set_title("Stockout by Lead Time")
    st.pyplot(fig)
    
    st.subheader("Causal Model for Stockout Probability")
    
    # Discretize data for Bayesian Network
    inv_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    lead_bins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    
    df_supply_disc = df_supply.copy()
    df_supply_disc['InventoryLevel_bin'] = inv_bins.fit_transform(df_supply_disc[['InventoryLevel']])
    df_supply_disc['LeadTime_bin'] = lead_bins.fit_transform(df_supply_disc[['LeadTime']])
    
    model = DiscreteBayesianNetwork([('LeadTime_bin', 'Stockout'), ('InventoryLevel_bin', 'Stockout')])
    model.fit(df_supply_disc[['LeadTime_bin', 'InventoryLevel_bin', 'Stockout']], estimator=MaximumLikelihoodEstimator)
    
    st.subheader("What-if Analysis")
    col1, col2 = st.columns(2)
    with col1:
        inv_level = st.slider("Inventory Level", 50, 150, 100)
    with col2:
        lt = st.slider("Lead Time (weeks)", 1, 3, 2)
    
    # Convert to bin indices
    inv_bin = int(inv_bins.transform([[inv_level]])[0][0])
    lt_bin = int(lead_bins.transform([[lt]])[0][0])
    
    infer = VariableElimination(model)
    query = infer.query(variables=['Stockout'], evidence={'InventoryLevel_bin': inv_bin, 'LeadTime_bin': lt_bin})
    
    # FIX: Handle cases where query.values might have only 1 class
    if len(query.values) == 2:  # Both classes (0 and 1) are present
        stockout_prob = query.values[1]
    else:  # Only one class is present (either all 0s or all 1s)
        if query.state_names['Stockout'] == [1]:  # Only stockouts
            stockout_prob = 1.0
        else:  # Only no stockouts
            stockout_prob = 0.0
    
    st.metric("Probability of Stockout", f"{stockout_prob:.1%}")
    
    inventory_levels = np.linspace(50, 150, 101)
    probs = []
    for inv in inventory_levels:
        inv_bin = int(inv_bins.transform([[inv]])[0][0])
        query = infer.query(variables=['Stockout'], evidence={'InventoryLevel_bin': inv_bin, 'LeadTime_bin': lt_bin})
        # Same handling as above
        if len(query.values) == 2:
            probs.append(query.values[1])
        else:
            if query.state_names['Stockout'] == [1]:
                probs.append(1.0)
            else:
                probs.append(0.0)
    
    target_prob = 0.05
    try:
        optimal_inv = inventory_levels[next(i for i, p in enumerate(probs) if p < target_prob)]
    except StopIteration:
        optimal_inv = inventory_levels[-1]
        st.warning("Could not find inventory level below target probability. Using maximum value.")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(inventory_levels, probs)
    ax2.axhline(target_prob, color='r', linestyle='--', label=f'{target_prob:.0%} target')
    ax2.axvline(optimal_inv, color='g', linestyle='--', label=f'Optimal Inventory: {optimal_inv:.0f}')
    ax2.set_xlabel("Inventory Level")
    ax2.set_ylabel("Stockout Probability")
    ax2.set_title(f"Stockout Probability vs Inventory Level (Lead Time = {lt} weeks)")
    ax2.legend()
    st.pyplot(fig2)

# Sidebar project info
with st.sidebar:
    st.image("https://www.electroluxgroup.com/wp-content/themes/electrolux/assets/images/logo.svg", width=200)
    st.markdown("""
    **Electrolux Causal Time Series Project**
    
    Key features demonstrated:
    - Optimal discount analysis
    - Marketing campaign impact
    - Supply chain optimization
    """)
    st.markdown("[GitHub Repository](https://github.com/Vidhaankhare16/Business-Sales-AI-assistant)")

# Run the app
if __name__ == '__main__':

    plt.rcParams.update({'figure.raise_window': False})
    
    st.write("App ready! Use the tabs above to explore different causal time series analyses.")
