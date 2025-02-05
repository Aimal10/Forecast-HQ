import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet_forecaster import process_and_forecast
from chatbot import get_marketing_insights
from utils import load_and_validate_data, calculate_metrics
import time

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Product Forecasting Tool",
                  page_icon="📈",
                  layout="wide",
                  initial_sidebar_state="collapsed")

# Load custom CSS
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Hide all Streamlit elements and remove their spacing
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .css-18e3th9 {
            padding-top: 0;
        }
        .css-1d391kg {
            padding-top: 0;
        }
        section[data-testid="stSidebar"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Add Font Awesome
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# Add logo with no top margin
st.image("logo.png", width=160)

st.markdown("""
<div class="intro-section">
    <p>Upload your historical data and get AI-powered forecasting insights.</p>
</div>
""", unsafe_allow_html=True)

def handle_chat_input():
    """Callback to handle chat input submission"""
    if st.session_state.user_input and st.session_state.user_input.strip():
        user_message = st.session_state.user_input.strip()

        if user_message.lower() == 'clear':
            st.session_state.chat_history = []
            st.session_state.has_generated_initial_insight = False
            st.rerun()
            return

        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_message
        })

        # Clear input immediately
        st.session_state.user_input = ""

        # Generate response
        with st.spinner('<i class="fas fa-brain"></i> Analyzing...'):
            response = get_marketing_insights(
                st.session_state.selected_product,
                st.session_state.forecast_df,
                st.session_state.mae,
                st.session_state.rmse,
                context=user_message,
                chat_history=st.session_state.chat_history)

            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })

def process_data_and_generate_forecast(raw_df, date_col, product_col, selected_product, forecast_col):
    """Process data and generate forecast"""
    try:
        # Process the selected columns
        processed_df = raw_df[raw_df[product_col] == selected_product].copy()
        processed_df = processed_df[[date_col, forecast_col]].copy()

        # Convert date column to datetime
        processed_df[date_col] = pd.to_datetime(processed_df[date_col])

        # Convert forecast column to numeric
        processed_df[forecast_col] = pd.to_numeric(processed_df[forecast_col], errors='coerce')

        # Drop rows where conversion failed
        processed_df = processed_df.dropna()

        # Aggregate data by date
        processed_df = processed_df.set_index(date_col)
        processed_df = processed_df.resample('D').sum()  # Daily aggregation
        processed_df = processed_df.reset_index()

        # Forward fill any missing days after resampling
        processed_df = processed_df.fillna(method='ffill')

        if len(processed_df) < 10:
            st.error(
                "Not enough valid data points for the selected product. Please ensure your selected columns contain appropriate data."
            )
            return None

        # Rename columns for Prophet
        processed_df = processed_df.rename(columns={
            date_col: 'ds',
            forecast_col: 'y'
        })

        return processed_df
    except Exception as e:
        st.error(
            f"Error processing selected columns: {str(e)}\n\n"
            "Please ensure:\n"
            "1. The date column contains valid dates\n"
            "2. The forecast column contains numeric values\n"
            "3. There are enough valid data points")
        return None

def main():
    # Initialize session state
    if 'forecast_df' not in st.session_state:
        st.session_state.forecast_df = None
    if 'selected_product' not in st.session_state:
        st.session_state.selected_product = None
    if 'mae' not in st.session_state:
        st.session_state.mae = None
    if 'rmse' not in st.session_state:
        st.session_state.rmse = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'has_generated_initial_insight' not in st.session_state:
        st.session_state.has_generated_initial_insight = False
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    # File upload section with icon
    st.markdown("""
    <div class="upload-section">
        Select your data file
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "",  # Remove label as we're using custom header
        type=['csv', 'xlsx'],
        help="File should contain date column and product sales data")

    if uploaded_file:
        try:
            # Load data without processing
            raw_df = load_and_validate_data(uploaded_file, validate_only=True)

            # Get all columns
            all_columns = raw_df.columns.tolist()

            # Column selection with icons
            st.markdown('<i class="fas fa-calendar"></i> Date Configuration',
                       unsafe_allow_html=True)
            date_col = st.selectbox("Select date column",
                                  options=all_columns,
                                  help="Choose the column containing dates")

            st.markdown('<i class="fas fa-box"></i> Product Configuration',
                       unsafe_allow_html=True)
            product_col = st.selectbox(
                "Select product column",
                options=[col for col in all_columns if col != date_col],
                help="Choose the column containing product identifiers")

            if product_col:
                unique_products = raw_df[product_col].unique().tolist()
                selected_product = st.selectbox(
                    "Select product to forecast",
                    options=unique_products,
                    help="Choose the specific product")

                st.session_state.selected_product = selected_product

                forecast_options = [
                    col for col in all_columns
                    if col not in [date_col, product_col]
                ]

                st.markdown(
                    '<i class="fas fa-chart-bar"></i> Forecast Configuration',
                    unsafe_allow_html=True)
                forecast_col = st.selectbox(
                    "Select metric to forecast",
                    options=forecast_options,
                    help="Choose the metric to forecast")

                if date_col and forecast_col:
                    processed_df = process_data_and_generate_forecast(
                        raw_df, date_col, product_col, selected_product, forecast_col)

                    if processed_df is not None:
                        # Display data info
                        st.info(
                            f"Processing {len(processed_df)} daily data points from {processed_df['ds'].min().date()} to {processed_df['ds'].max().date()}"
                        )

                        # Forecast parameters
                        col1, col2 = st.columns(2)
                        with col1:
                            forecast_period = st.slider(
                                "Forecast period (days)",
                                min_value=1,
                                max_value=30,
                                value=7)

                        with col2:
                            confidence_interval = st.slider(
                                "Confidence Interval",
                                min_value=0.7,
                                max_value=0.95,
                                value=0.8,
                                step=0.05)

                        if st.button("Generate Forecast", type="primary"):
                            with st.spinner("Generating forecast and insights..."):
                                # Process data and generate forecast
                                forecast_df, model = process_and_forecast(
                                    processed_df, forecast_period,
                                    confidence_interval)

                                # Calculate metrics
                                mae, rmse = calculate_metrics(model, processed_df)

                                # Store results in session state
                                st.session_state.forecast_df = forecast_df
                                st.session_state.mae = mae
                                st.session_state.rmse = rmse
                                st.session_state.chat_history = []
                                st.session_state.has_generated_initial_insight = False

                                # Generate initial insights in the same spinner
                                initial_insights = get_marketing_insights(
                                    st.session_state.selected_product,
                                    st.session_state.forecast_df,
                                    st.session_state.mae,
                                    st.session_state.rmse)
                                st.session_state.chat_history.append({
                                    'role': 'assistant',
                                    'content': initial_insights
                                })
                                st.session_state.has_generated_initial_insight = True

                        # Display forecast if available
                        if st.session_state.forecast_df is not None:
                            # Display metrics
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric("Mean Absolute Error (MAE)",
                                        f"{st.session_state.mae:.2f}")
                            with metric_col2:
                                st.metric("Root Mean Square Error (RMSE)",
                                        f"{st.session_state.rmse:.2f}")

                            # Create plot with increased height
                            fig = make_subplots(rows=1, cols=1)

                            # Plot the actual data
                            fig.add_trace(
                                go.Scatter(
                                    x=processed_df['ds'],
                                    y=processed_df['y'],
                                    mode='markers',
                                    name='Actual Data',
                                    marker=dict(color='black')))

                            # Plot the forecasted data
                            fig.add_trace(
                                go.Scatter(
                                    x=st.session_state.forecast_df['ds'],
                                    y=st.session_state.forecast_df['yhat'],
                                    mode='lines',
                                    name='Forecasted Data',
                                    line=dict(color='blue')))

                            # Plot the confidence intervals
                            fig.add_trace(
                                go.Scatter(
                                    x=st.session_state.forecast_df['ds'],
                                    y=st.session_state.forecast_df['yhat_upper'],
                                    mode='lines',
                                    name='Upper Bound',
                                    line=dict(color='lightblue',
                                            dash='dash')))
                            fig.add_trace(
                                go.Scatter(
                                    x=st.session_state.forecast_df['ds'],
                                    y=st.session_state.forecast_df['yhat_lower'],
                                    mode='lines',
                                    name='Lower Bound',
                                    line=dict(color='lightblue', dash='dash'),
                                    fill='tonexty'))

                            # Update layout
                            fig.update_layout(
                                height=600,
                                title=f"Sales Forecast for {selected_product}",
                                title_x=0.5,
                                xaxis_title="Date",
                                yaxis_title=forecast_col,
                                template="plotly",
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                showlegend=True,
                                xaxis=dict(showgrid=True, zeroline=False),
                                yaxis=dict(showgrid=True,
                                         zeroline=True,
                                         range=[0, None]))

                            # Display plot
                            st.plotly_chart(fig, use_container_width=True)

                            # Chat interface
                            st.markdown("""
                            <div class="chat-container">
                                <div class="chat-header">
                                    <i class="fas fa-robot"></i>
                                    <h2>AI Analytics Assistant</h2>
                                </div>
                                <p class="chat-subtitle">
                                    Your intelligent forecasting companion that analyzes market trends, provides strategic recommendations, 
                                    identifies growth opportunities, and suggests optimal campaign timing.
                                </p>
                            </div>
                            """, unsafe_allow_html=True)

                            # Chat container
                            chat_container = st.container()

                            # Display chat history
                            with chat_container:
                                for message in st.session_state.chat_history:
                                    if message['role'] == 'assistant':
                                        st.markdown(f"""
                                        <div class="chat-message assistant">
                                            <div class="avatar">🤖</div>
                                            <div class="message">
                                                {message['content']}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div class="chat-message user">
                                            <div class="avatar">👤</div>
                                            <div class="message">
                                                {message['content']}
                                            </div>
                                        </div>
                                        """, unsafe_allow_html=True)

                            # Input area
                            st.text_input(
                                "",
                                placeholder=
                                "Ask about the forecast, trends, or marketing strategies...",
                                key="user_input",
                                on_change=handle_chat_input,
                                label_visibility="collapsed")

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info(
                "Please ensure your data is in the correct format and contains appropriate columns for forecasting."
            )

if __name__ == "__main__":
    main()