import streamlit as st
from predict_bert import predict_bert

st.set_page_config(
    page_title="Fake News Detector (BERT)",
    layout="wide",
    page_icon="📰"
)

# ---------- Custom CSS for Professional UI ----------
st.markdown("""
<style>
.card {
    background: #ffffff;
    border-radius: 18px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.12);
    border: 1px solid rgba(148, 163, 184, 0.4);
}
.badge-real {
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(16,185,129,0.15);
    color: #047857;
    font-weight: 600;
}
.badge-fake {
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    background: rgba(239,68,68,0.15);
    color: #b91c1c;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📰 Fake News Detector (BERT Powered)")
st.write("This detector uses a **BERT Transformer model** for high accuracy fake-news classification.")

left, right = st.columns([1.4, 1])

# ---- INPUT SECTION ----
with left:
    st.markdown("### ✍️ Paste News Content")
    text = st.text_area(
        "",
        height=260,
        placeholder="Paste any long news article or sentence here to test authenticity..."
    )

    analyze = st.button("🔍 Analyze News")

# ---- OUTPUT SECTION ----
with right:
    st.markdown("### 📊 Prediction Result")

    if not analyze:
        st.markdown("""
        <div class="card">
            <p style="color:#6b7280;">Results will appear here after you enter a news article.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        label, prob = predict_bert(text)
        conf = prob * 100

        if label == 1:
            verdict = "Fake News"
            badge = "badge-fake"
            emoji = "🚨"
            explanation = "This text resembles patterns of misinformation or fake news."
        else:
            verdict = "Real News"
            badge = "badge-real"
            emoji = "✅"
            explanation = "This text resembles patterns of factual, reliable news."

        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div style="font-size:1.3rem; font-weight:700;">{emoji} {verdict}</div>
                <div class="{badge}">{verdict.upper()}</div>
            </div>

            <p style="margin-top:1rem;"><b>Fake Probability:</b> {conf:.2f}%</p>
        """, unsafe_allow_html=True)

        st.progress(float(prob))

        st.markdown(
            f"""
            <p style="margin-top:1rem; font-size:0.95rem;">{explanation}</p>
            <p style="color:#9ca3af; font-size:0.75rem; margin-top:0.7rem;">
            ⚠️ Note: BERT predictions reflect patterns learned from training data.
            Always verify critical news from trusted sources.
            </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---- SIDEBAR ----
with st.sidebar:
    st.header("ℹ️ Model Info")
    st.write("**Model:** BERT (bert-base-uncased)")
    st.write("**Type:** Transformer (NLP)")
    st.write("**Task:** Fake News Classification (Real / Fake)")
    st.write("**Threshold:** 0.5 for fake")
    st.write("**Training:** Fine-tuned on your dataset")

    st.markdown("---")
    st.write("Paste any **long news sentence or article** to test its authenticity.")
