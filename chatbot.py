import os
import pandas as pd
from openai import OpenAI
import json

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_marketing_insights(product_name, forecast_df, mae, rmse, context=None, chat_history=None):
    """
    Generate marketing insights using OpenAI GPT-4 with improved context management
    and response formatting
    """
    # Extract key metrics and trends
    latest_date = forecast_df['ds'].max()
    forecast_end = latest_date + pd.Timedelta(days=30)
    trend = "increasing" if forecast_df['yhat'].iloc[-1] > forecast_df['yhat'].iloc[0] else "decreasing"
    trend_strength = abs(forecast_df['yhat'].iloc[-1] - forecast_df['yhat'].iloc[0]) / forecast_df['yhat'].iloc[0]

    system_message = {
        "role": "system",
        "content": """You are ForecastGPT, an expert AI strategist specializing in sales forecasting and marketing analytics.
        You maintain conversation context and provide data-driven insights.

        Guidelines for your responses:
        1. Present clear titles without any special formatting
        2. Highlight metrics using HTML spans with class="metric"
        3. Emphasize insights using HTML spans with class="highlight-primary"
        4. Present key findings in div with class="insight"
        5. Use clear bullet points for lists
        6. Keep technical terms simple
        7. Be concise but informative
        8. Focus on actionable recommendations

        Structure your analysis with:
        1. A clear title
        2. Brief overview
        3. Key insights (using highlight classes)
        4. Metrics (using metric class)
        5. Recommendations
        6. Next steps

        Use HTML spans for emphasis:
        - <span class="highlight-primary">primary emphasis</span>
        - <span class="highlight-secondary">secondary emphasis</span>
        - <span class="highlight-accent">accent emphasis</span>
        - <span class="metric">metrics</span>
        - <div class="insight">key insights</div>
        """
    }

    # Build conversation history context
    conversation_context = ""
    if chat_history:
        last_exchanges = chat_history[-4:]  # Get last 2 exchanges (4 messages)
        conversation_context = "\n".join([
            f"{'Bot' if msg['role'] == 'assistant' else 'User'}: {msg['content']}"
            for msg in last_exchanges
        ])

    if context:
        # Follow-up question format
        user_message = {
            "role": "user",
            "content": f"""
            Context:
            - Product: {product_name}
            - Trend: {trend} (strength: {trend_strength:.1%})
            - Forecast Period: Until {forecast_end.strftime('%Y-%m-%d')}
            - Accuracy: MAE={mae:.2f}, RMSE={rmse:.2f}

            Previous Conversation:
            {conversation_context}

            New Question: {context}

            Provide a focused response that:
            1. Addresses the specific question
            2. References relevant data points using metric spans
            3. Highlights key insights using appropriate highlight classes
            4. Maintains conversation continuity
            """
        }
    else:
        # Initial analysis format
        user_message = {
            "role": "user",
            "content": f"""
            Please analyze this forecast data:

            Product Details:
            - Name: {product_name}
            - Trend Direction: {trend}
            - Trend Strength: {trend_strength:.1%}
            - Forecast Range: Until {forecast_end.strftime('%Y-%m-%d')}

            Model Performance:
            - Mean Absolute Error: {mae:.2f}
            - Root Mean Square Error: {rmse:.2f}

            Provide an initial analysis that includes:
            1. A clear title without special formatting
            2. Overview with highlighted insights
            3. Key metrics using metric spans
            4. Growth opportunities with highlight-primary
            5. Risk factors with highlight-secondary
            6. Recommended actions with highlight-accent

            Format the response professionally with clear sections and spacing.
            Use the provided HTML classes for emphasis and highlighting.
            """
        }

    try:
        # Generate response with enhanced context
        messages = [system_message, user_message]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=1000,
            temperature=0.7  # Slightly creative while maintaining professionalism
        )

        return response.choices[0].message.content

    except Exception as e:
        return """AI Assistant Error

I apologize, but I encountered an error while analyzing the data. Please try again or rephrase your question.

<span class="highlight-secondary">Error details: {str(e)}</span>
"""