import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF; /* White background */
        color: #000000; /* Black text */
        font-family: Arial, sans-serif;
    }
    .title {
        text-align: center;
        padding-top: 20px;
        padding-bottom: 20px;
        font-size: 36px;
        font-weight: bold;
        color: #000000; /* Black title text */
    }
    .subheader {
        text-align: center;
        padding-bottom: 20px;
        font-size: 24px;
        color: #000000; /* Black subheader text */
    }
    .selectbox-container {
        margin-top: 20px;
        margin-bottom: 20px;
    }
    .sidebar .sidebar-content {
        background-color: #111111; /* Dark sidebar color */
    }
    .sidebar .sidebar-content .block-container {
        padding: 10px;
    }
    .block-container {
        padding: 20px;
        background-color: #F3F8FF; /* Lighter block background color */
        border-radius: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #000000; /* Black text in block containers */
    }
    /* Changing the rangeslider color */
    .rangeslider .rangeslider__fill {
        background-color: #000000; /* Black rangeslider */
    }
    .rangeslider .rangeslider__handle {
        background-color: #000000; /* Black handle */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<h1 class="title">WISENN</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subheader">Web Interface for Stock Exploration using Neural Network</h2>', unsafe_allow_html=True)

stocks = ('GOOG', 'AAPL', 'MSFT', 'GME', 'AMZN', 'FB', 'TSLA', 'NFLX', 'NVDA', 'JPM', 'V', 'DIS', 'PYPL', 'BABA')
selected_stock = st.selectbox('Select stock for prediction:', stocks, key='selectbox', help='Choose the stock for prediction.')

n_years = st.slider('Time frame:', 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

def plot_time_series():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    
    # Adjusting the text color on the Plotly chart
    fig.update_layout({
        'font': {
            'color': '#000000'  # Black text on the chart
        }
    })
    
    # Changing background color to white and graph colors
    fig.update_layout({
        'plot_bgcolor': 'black',
        'paper_bgcolor': 'black',
    })
    fig.update_traces(marker=dict(color='blue'), line=dict(color='blue'))
    
    st.plotly_chart(fig)
	
def plot_moving_averages():
    # Calculating moving averages
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    # Creating a new figure for moving averages
    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA10'], name="MA10", line=dict(color='red')))
    fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], name="MA50", line=dict(color='green')))
    fig_ma.layout.update(title_text='Moving Averages (MA10 and MA50)', xaxis_rangeslider_visible=False)
    
    # Adjusting the text color on the Plotly chart for moving averages
    fig_ma.update_layout({
        'font': {
            'color': '#000000'  # Black text on the chart
        }
    })
    
    # Changing background color to white and graph colors
    fig_ma.update_layout({
        'plot_bgcolor': 'black',
        'paper_bgcolor': 'black',
    })
    
    st.plotly_chart(fig_ma)

# Displaying the time series plot
plot_time_series()

# Displaying the moving averages plot
plot_moving_averages()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)