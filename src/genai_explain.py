
def explain_economic_snapshot(snapshot, client):
    prompt = f"""You are an economic analyst assistant. You will be given a structured
economic snapshot with real, verified data. Your job is ONLY to explain and interpret
these exact numbers in plain business language — you must NOT invent any statistics,
numbers, or claims that are not present in the data below.

ECONOMIC SNAPSHOT (as of {snapshot['date']}):
- Recession Risk: {snapshot['recession_risk']} (model probability: {snapshot['recession_probability']*100:.1f}%)
- Inflation: {snapshot['inflation_rate']}%, trend is {snapshot['inflation_trend']}
- Unemployment: {snapshot['unemployment_rate']}%, trend is {snapshot['unemployment_trend']}
- Industrial Production trend: {snapshot['industrial_production_trend']}
- Yield Curve: {snapshot['yield_curve_status']} (spread: {snapshot['yield_curve_spread']} percentage points)

Write a short, clear explanation (4-6 sentences) covering:
1. Overall recession risk assessment
2. The 2-3 key indicators driving that assessment
3. What a business leader should take away from this

Do not use markdown headers. Write it as flowing paragraph(s), business-report tone.
"""
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text
