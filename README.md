# AI-Powered Product Forecasting & Analysis Tool 🚀

A sophisticated time series forecasting application combining advanced data analytics with an interactive AI Marketing Assistant. This tool enables users to generate, visualize, and gain actionable insights from product demand predictions.

## Features ✨

- **Interactive Data Upload**: Easy CSV/Excel file upload functionality
- **Advanced Forecasting**: Powered by Prophet forecasting engine
- **Smart Visualizations**: Interactive Plotly charts with confidence intervals
- **AI Marketing Assistant**: GPT-4 powered analysis and recommendations
- **Responsive Design**: Fully responsive interface that works on all devices
- **Real-time Insights**: Instant marketing strategies and trend analysis

## Technology Stack 🛠️

- **Frontend**: Streamlit
- **Forecasting Engine**: Prophet
- **AI Integration**: OpenAI GPT-4
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Styling**: Custom CSS with brand theming

## Getting Started 🌟

### Prerequisites

- Python 3.11+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
OPENAI_API_KEY=your-api-key-here
```

4. Run the application:
```bash
streamlit run main.py
```

## Usage Guide 📖

1. **Upload Data**
   - Prepare your time series data in CSV or Excel format
   - Ensure your data includes date column and values to forecast
   - Upload through the file uploader interface

2. **Configure Forecast**
   - Select the date column
   - Choose the product column
   - Pick the metric to forecast
   - Set forecast period and confidence interval

3. **Generate Insights**
   - Click "Generate Forecast" to process data
   - View the forecast visualization
   - Read AI-generated insights
   - Ask follow-up questions to the AI Assistant

4. **Interact with Results**
   - Explore the interactive plot
   - Review key metrics (MAE, RMSE)
   - Get marketing recommendations
   - Export insights for presentation

## Key Features Explained 🔑

### Forecast Visualization
- Interactive time series plot
- Confidence intervals
- Historical vs. predicted data
- Adjustable forecast periods

### AI Marketing Assistant
- Natural language interaction
- Strategic recommendations
- Trend analysis
- Market opportunity identification

### Data Processing
- Automatic data validation
- Missing value handling
- Date parsing
- Aggregation capabilities

## Contributing 🤝

We welcome contributions! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Prophet team for the forecasting engine
- OpenAI for GPT-4 API
- Streamlit team for the amazing framework
- All contributors who helped shape this project
