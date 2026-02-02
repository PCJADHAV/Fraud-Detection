import streamlit as st
import pandas as pd
import pickle


def ai_transaction_summary(probability, input_data, feature_importance):
    reasons = []

    amount = input_data["amount"].values[0]
    balance_diff = input_data["balance_diff"].values[0]
    tx_type = input_data["type_encoded"].values[0]

    if amount > 200000:
        reasons.append("🔹 High transaction amount")

    if abs(balance_diff) > 100000:
        reasons.append("🔹 Large balance difference after transaction")

    if tx_type in [1, 2]:  
        reasons.append("🔹 Risky transaction type (TRANSFER / CASH_OUT)")

    top_features = feature_importance.head(3)["Feature"].tolist()

    summary = f"""
**Fraud Probability:** `{probability:.2f}`

**Detected Risk Signals:**
"""

    if reasons:
        for r in reasons:
            summary += f"\n- {r}"
    else:
        summary += "\n- No strong fraud indicators detected"

    summary += "\n\n📌 **Final Decision:** "
    if probability >= 0.5:
        summary += "🚨 High risk transaction"
    elif probability >= 0.3:
        summary += "⚠️ Medium risk transaction"
    else:
        summary += "✅ Low risk transaction"

    return summary



with open("C:\\Users\\Prashant\\Fraud_Detection\\Model\\rf_fraud_model.pkl", "rb") as f:
    model = pickle.load(f)



feature_importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)



st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Fraud Detection System")
st.write("Random Forest Model | Feature-safe deployment")

st.divider()



st.subheader("⚙️ Risk Threshold Settings")

threshold = st.slider(
    "Fraud Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

st.write(f"Transactions with fraud probability ≥ {threshold} will be treated as FRAUD.")



step = st.number_input("Step (Time Step)", min_value=0)

transaction_type = st.selectbox(
    "Transaction Type",
    ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT']
)

amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Old Balance (Sender)", min_value=0.0)
newbalanceOrig = st.number_input("New Balance (Sender)", min_value=0.0)
oldbalanceDest = st.number_input("Old Balance (Receiver)", min_value=0.0)
newbalanceDest = st.number_input("New Balance (Receiver)", min_value=0.0)



type_mapping = {
    'TRANSFER': 1,
    'CASH_OUT': 2,
    'PAYMENT': 3,
    'CASH_IN': 4,
    'DEBIT': 5
}

type_encoded = type_mapping[transaction_type]



balance_diff = oldbalanceOrg - newbalanceOrig



input_data = pd.DataFrame([{
    "step": step,
    "type_encoded": type_encoded,
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "balance_diff": balance_diff
}])


input_data = input_data[model.feature_names_in_]




if st.button("🔍 Predict Fraud"):

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🧾 Prediction Result")

        if probability >= threshold:
            st.error(f"🚨 FRAUD ALERT")
            st.metric("Fraud Probability", f"{probability:.2f}", delta="High Risk")
            st.warning("⚠️ Action: Transaction BLOCKED & Security Team Notified")

        elif probability >= threshold * 0.6:
            st.warning("⚠️ Suspicious Transaction")
            st.metric("Fraud Probability", f"{probability:.2f}", delta="Medium Risk")
            st.info("🕵️ Action: Sent for MANUAL REVIEW")

        else:
            st.success("✅ Legitimate Transaction")
            st.metric("Fraud Probability", f"{probability:.2f}", delta="Low Risk")
            st.info("✔️ Action: Transaction APPROVED")

    with col2:
        st.subheader("🤖 AI Assistant")
        ai_summary = ai_transaction_summary(probability, input_data, feature_importance)
        st.markdown(ai_summary)



st.subheader("📊 Feature Importance (Model Explanation)")

st.bar_chart(
    feature_importance.set_index("Feature")
)
