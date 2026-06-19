import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load dataset
df = pd.read_csv("data/processed/final_data.csv")

# =========================
# 1–5: PRICE & SIZE ANALYSIS
# =========================

def Price_distribution(df):
    fig = px.histogram(
        df,
        x="Price_in_Lakhs",
        nbins=30,
        title="Distribution of Property Prices"
    )
    st.plotly_chart(fig, use_container_width=True)

def Size_distribution(df):
    fig = px.histogram(
        df,
        x="Size_in_SqFt",
        nbins=30,
        title="Distribution of Property Sizes"
    )
    st.plotly_chart(fig, use_container_width=True)

def Price_per_sqft_by_property_type(df):
    fig = px.box(
        df,
        x="Property_Type",
        y="Price_per_SqFt",
        title="Price per SqFt by Property Type"
    )
    st.plotly_chart(fig, use_container_width=True)

def Size_vs_Price(df):
    fig = px.scatter(
        df,
        x="Size_in_SqFt",
        y="Price_in_Lakhs",
        color="Property_Type",
        title="Property Size vs Price"
    )
    st.plotly_chart(fig, use_container_width=True)

def Outliers(df):
    fig = px.box(
        df,
        y="Price_per_SqFt",
        title="Outliers in Price per SqFt"
    )
    st.plotly_chart(fig, use_container_width=True)


# =========================
# 6–10: LOCATION ANALYSIS
# =========================

def Avg_price_per_sqft_by_state(df):
    data = df.groupby("State")["Price_per_SqFt"].mean().reset_index()

    fig = px.bar(
        data,
        x="State",
        y="Price_per_SqFt",
        title="Average Price per SqFt by State"
    )

    st.plotly_chart(fig, use_container_width=True)

def Avg_price_by_city(df):
    data = df.groupby("City")["Price_in_Lakhs"].mean().reset_index()

    fig = px.bar(
        data,
        x="City",
        y="Price_in_Lakhs",
        title="Average Property Price by City"
    )

    st.plotly_chart(fig, use_container_width=True)

def BHK_distribution_across_cities(df):
    fig = px.box(
        df,
        x="City",
        y="BHK",
        title="BHK Distribution Across Cities"
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# 11–15: CORRELATION
# =========================

def Correlation_heatmap(df):

    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)

def Schools_vs_price_per_sqft(df):

    fig = px.scatter(
        df,
        x="Nearby_Schools",
        y="Price_per_SqFt",
        title="Schools vs Price per SqFt"
    )

    st.plotly_chart(fig, use_container_width=True)

def Hospitals_vs_price_per_sqft(df):

    fig = px.scatter(
        df,
        x="Nearby_Hospitals",
        y="Price_per_SqFt",
        title="Hospitals vs Price per SqFt"
    )

    st.plotly_chart(fig, use_container_width=True)

def Furnishing_vs_price(df):

    fig = px.box(
        df,
        x="Furnished_Status",
        y="Price_in_Lakhs",
        title="Price by Furnished Status"
    )

    st.plotly_chart(fig, use_container_width=True)

def Facing_vs_price_per_sqft(df):

    fig = px.box(
        df,
        x="Facing",
        y="Price_per_SqFt",
        title="Price per SqFt by Facing Direction"
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================
# 16–20: INVESTMENT ANALYSIS
# =========================

def Owner_type(df):

    fig = px.pie(
        df,
        names="Owner_Type",
        title="Properties by Owner Type"
    )

    st.plotly_chart(fig, use_container_width=True)

def Availability_status(df):

    fig = px.pie(
        df,
        names="Availability_Status",
        title="Availability Status Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

def Parking_vs_price(df):

    fig = px.box(
        df,
        x="Parking_Space",
        y="Price_in_Lakhs",
        title="Parking Space vs Property Price"
    )

    st.plotly_chart(fig, use_container_width=True)

def Amenities_vs_price_per_sqft(df):

    fig = px.box(
        df,
        x="Amenities",
        y="Price_per_SqFt",
        title="Amenities vs Price per SqFt"
    )

    st.plotly_chart(fig, use_container_width=True)

def Transport_vs_price_per_sqft():


    fig = px.box(
        df,
        x="Public_Transport_Accessibility",
        y="Price_per_SqFt",
        title="Transport Accessibility vs Price per SqFt"
    )

    st.plotly_chart(fig, use_container_width=True)