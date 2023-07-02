import streamlit as st
from prophet import Prophet
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

page_by_img = """
<style>
[data-testid="stAppViewContainer"] {
    background: rgb(2,0,36);
    background: linear-gradient(0deg, rgba(2,0,36,0.9) 0%, rgba(9,53,121,0.9) 35%, rgba(0,212,255,0.9) 100%);
}
</style>
"""
st.markdown(page_by_img, unsafe_allow_html=True)

st.markdown("<style>.drop-shadow { text-align: center; color: white; font-size: 76px; -webkit-text-stroke: 2px black; font-weight: bold;}</style>", unsafe_allow_html=True)
st.markdown("<h1 class='drop-shadow'>PROPHET FORECAST</h1>", unsafe_allow_html=True)
ticker = st.text_input('Stock Ticker')

if st.button('Predict'):
    # Download data from Yahoo Finance
    df = yf.download(ticker, period='5y')

    # Reset the index to make Date a column
    df = df.reset_index()

    # Rename the columns to match the expected format
    df = df.rename(columns={'Date': 'ds', 'Close': 'y'})

    model = Prophet()
    model.fit(df)

    future = model.make_future_dataframe(periods=365)
    forecast = model.predict(future)

    # Plot the forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='markers', name='Actual', marker=dict(color='orange', size=4)))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='Bounds', fillcolor='rgba(135, 206, 235, 0.2)', line=dict(color='rgba(135, 206, 235, 0.6)')))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='Bounds', fillcolor='rgba(135, 206, 235, 0.2)', line=dict(color='rgba(135, 206, 235, 0.6)')))
    #fig.update_traces(mode='lines', line=dict(color='rgba(135, 206, 235, 0.6)'))

    fig.update_layout(title='Prophet Forecast',
                      xaxis_title='Date',
                      yaxis_title='Price',
                      showlegend=True,
                      width=800,
                      height=600)
    st.plotly_chart(fig)