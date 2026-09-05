
import pandas as pd

def generate_economic_snapshot(df, model, feature_list, lookback_months=6):
    """
    df: full cleaned economic dataframe (with derived indicators)
    model: trained recession classification pipeline
    feature_list: exact feature names/order the model expects
    lookback_months: window used to judge "trend" (rising/falling/stable)
    """
    latest = df.iloc[-1]
    recent = df.iloc[-lookback_months:]

    input_row = {
        'Unemployment Change_lag1': df['Unemployment Change'].iloc[-2],
        'Inflation Rate_lag1': df['Inflation Rate'].iloc[-2],
        'Industrial Production Growth_lag1': df['Industrial Production Growth'].iloc[-2],
        'Yield Curve Spread_lag12': df['Yield Curve Spread'].iloc[-13],
    }
    X_latest = pd.DataFrame([input_row])[feature_list]

    recession_prob = model.predict_proba(X_latest)[0][1]

    if recession_prob >= 0.66:
        risk_label = "HIGH"
    elif recession_prob >= 0.33:
        risk_label = "MODERATE"
    else:
        risk_label = "LOW"

    def trend(series, threshold=0.05):
        change = series.iloc[-1] - series.iloc[0]
        if change > threshold:
            return "RISING"
        elif change < -threshold:
            return "FALLING"
        return "STABLE"

    inflation_trend = trend(recent['Inflation Rate'])
    unemployment_trend = trend(recent['Unemployment Rate'])
    ip_trend = trend(recent['Industrial Production Growth'])
    yield_curve_status = "INVERTED" if latest['Yield Curve Spread'] < 0 else "NORMAL"

    snapshot = {
        "date": df.index[-1].strftime('%Y-%m-%d'),
        "recession_risk": risk_label,
        "recession_probability": round(float(recession_prob), 3),
        "inflation_trend": inflation_trend,
        "inflation_rate": round(float(latest['Inflation Rate']), 2),
        "unemployment_trend": unemployment_trend,
        "unemployment_rate": round(float(latest['Unemployment Rate']), 2),
        "industrial_production_trend": ip_trend,
        "yield_curve_status": yield_curve_status,
        "yield_curve_spread": round(float(latest['Yield Curve Spread']), 2),
    }
    return snapshot
