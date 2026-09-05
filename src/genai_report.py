
def generate_executive_report(snapshot, client):
    prompt = f"""You are an economic analyst preparing a monthly executive report for
company leadership. Use ONLY the verified data below — do not invent any numbers,
statistics, or claims not present in this snapshot.

ECONOMIC SNAPSHOT (as of {snapshot['date']}):
- Recession Risk: {snapshot['recession_risk']} (model probability: {snapshot['recession_probability']*100:.1f}%)
- Inflation: {snapshot['inflation_rate']}%, trend is {snapshot['inflation_trend']}
- Unemployment: {snapshot['unemployment_rate']}%, trend is {snapshot['unemployment_trend']}
- Industrial Production trend: {snapshot['industrial_production_trend']}
- Yield Curve: {snapshot['yield_curve_status']} (spread: {snapshot['yield_curve_spread']} percentage points)

Write a MONTHLY ECONOMIC INTELLIGENCE REPORT with these exact section headers:

Executive Summary
Employment
Inflation
Interest Rates & Yield Curve
Industrial Activity
ML Assessment
Key Risks
Key Indicators to Monitor

Keep each section to 1-3 sentences. Under "Key Risks" and "Key Indicators to Monitor",
use a numbered list of up to 3 items each, grounded only in the data given.
"""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text
