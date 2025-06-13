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
We combine sktime for time series forecasting and pgmpy for causal modeling.
""")

# Create tabs for different functionalities
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

    # Discretize continuous variables properly
    disc_df = df_discount.copy()
    sales_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    discount_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    
    disc_df['Sales_bin'] = sales_bins.fit_transform(disc_df[['Sales']]).astype(int)
    disc_df['Discount_bin'] = discount_bins.fit_transform(disc_df[['Discount']]).astype(int)
    
    # Convert bins to string labels for pgmpy
    disc_df['Sales_bin'] = 's_' + disc_df['Sales_bin'].astype(str)
    disc_df['Discount_bin'] = 'd_' + disc_df['Discount_bin'].astype(str)
    disc_df['BasePrice'] = 'bp_1000'  # Fixed as string

    model = DiscreteBayesianNetwork([('Discount_bin', 'Sales_bin'), ('BasePrice', 'Sales_bin')])
    model.fit(disc_df[['Discount_bin', 'Sales_bin', 'BasePrice']], estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(model)
    discount_levels = np.linspace(0, 0.3, 31)
    expected_sales = []

    for d in discount_levels:
        d_bin = 'd_' + str(int(discount_bins.transform([[d]])[0][0]))
        try:
            query = infer.query(variables=['Sales_bin'], evidence={'Discount_bin': d_bin, 'BasePrice': 'bp_1000'})
            # Calculate expected value by mapping back to original scale
            bin_centers = sales_bins.bin_edges_[0][:-1] + np.diff(sales_bins.bin_edges_[0])/2
            expected_value = np.sum([prob * center for prob, center in zip(query.values, bin_centers)])
            expected_sales.append(expected_value)
        except:
            expected_sales.append(np.nan)

    # Remove any NaN values that might have occurred
    valid_indices = ~np.isnan(expected_sales)
    discount_levels = discount_levels[valid_indices]
    expected_sales = np.array(expected_sales)[valid_indices]

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
    dates = pd.date_range(start="2021-01-01", end="2023-01-01", freq="W")
    n = len(dates)
    trend = np.linspace(1000, 1500, n)
    seasonality = 100 * np.sin(np.linspace(0, 4 * np.pi, n))
    base_sales = trend + seasonality + np.random.normal(0, 50, n)

    campaigns = np.zeros(n)
    campaign_weeks = np.random.choice(n, size=10, replace=False)
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
    for date in df_campaign[df_campaign['Campaign'] == 1].index:
        ax.axvline(date, color='r', alpha=0.3)
    ax.set_title("Sales Data with Marketing Campaigns (red lines)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Time Series Forecasting with Campaign Impact")
    y = df_campaign['Sales'].to_frame()
    train = y.iloc[:-20]
    test = y.iloc[-20:]

    forecaster = AutoETS(auto=True, sp=52, n_jobs=-1)
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

    # Discretize sales properly
    sales_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
    disc_campaign = df_campaign.reset_index().copy()
    disc_campaign['Sales_bin'] = 's_' + sales_bins.fit_transform(disc_campaign[['Sales']]).astype(int).astype(str)
    disc_campaign['Campaign'] = disc_campaign['Campaign'].astype(str)
    
    # Discretize trend
    trend_bins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='quantile')
    disc_campaign['Trend_bin'] = 't_' + trend_bins.fit_transform(disc_campaign[['Trend']]).astype(int).astype(str)

    causal_model = DiscreteBayesianNetwork([('Campaign', 'Sales_bin'), ('Trend_bin', 'Sales_bin')])
    causal_model.fit(disc_campaign[['Campaign', 'Sales_bin', 'Trend_bin']], estimator=MaximumLikelihoodEstimator)

    infer = VariableElimination(causal_model)
    
    # Calculate expected sales with campaign
    query_with = infer.query(variables=['Sales_bin'], evidence={'Campaign': '1'})
    bin_centers = sales_bins.bin_edges_[0][:-1] + np.diff(sales_bins.bin_edges_[0])/2
    effect_with = np.sum([prob * center for prob, center in zip(query_with.values, bin_centers)])
    
    # Calculate expected sales without campaign
    query_without = infer.query(variables=['Sales_bin'], evidence={'Campaign': '0'})
    effect_without = np.sum([prob * center for prob, center in zip(query_without.values, bin_centers)])
    
    campaign_effect = effect_with - effect_without

    st.info(f"Estimated average campaign impact: {campaign_effect:.0f} additional sales")

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

    # Discretize inventory levels
    inv_bins = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
    disc_supply = df_supply.copy()
    disc_supply['Inventory_bin'] = 'inv_' + inv_bins.fit_transform(disc_supply[['InventoryLevel']]).astype(int).astype(str)
    disc_supply['LeadTime'] = 'lt_' + disc_supply['LeadTime'].astype(str)
    disc_supply['Stockout'] = disc_supply['Stockout'].astype(str)

    model = DiscreteBayesianNetwork([('LeadTime', 'Stockout'), ('Inventory_bin', 'Stockout')])
    model.fit(disc_supply[['LeadTime', 'Inventory_bin', 'Stockout']], estimator=MaximumLikelihoodEstimator)

    st.subheader("What-if Analysis")
    col1, col2 = st.columns(2)
    with col1:
        inv_level = st.slider("Inventory Level", 50, 150, 100)
    with col2:
        lt = st.slider("Lead Time (weeks)", 1, 3, 2)

    inv_bin = 'inv_' + str(int(inv_bins.transform([[inv_level]])[0][0]))
    lt_str = 'lt_' + str(lt)
    
    infer = VariableElimination(model)
    query = infer.query(variables=['Stockout'], evidence={'Inventory_bin': inv_bin, 'LeadTime': lt_str})
    stockout_prob = float(query.values[1])  # Probability of stockout=1

    st.metric("Probability of Stockout", f"{stockout_prob:.1%}")

    inventory_levels = np.linspace(50, 150, 101)
    probs = []
    for inv in inventory_levels:
        inv_bin = 'inv_' + str(int(inv_bins.transform([[inv]])[0][0]))
        query = infer.query(variables=['Stockout'], evidence={'Inventory_bin': inv_bin, 'LeadTime': lt_str})
        probs.append(float(query.values[1]))

    target_prob = 0.05
    optimal_inv = inventory_levels[next(i for i, p in enumerate(probs) if p < target_prob)]

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