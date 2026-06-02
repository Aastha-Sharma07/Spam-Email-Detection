import streamlit as st
import joblib
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(
    page_title="Spam Email Detector",
    page_icon="🛡️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load('spam_detector.pkl')

model = load_model()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\b\d{10,}\b', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_data
def load_data():
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'message'})
    df['clean_message'] = df['message'].apply(preprocess_text)
    df['label_enc']     = (df['label'] == 'spam').astype(int)
    df['msg_length']    = df['message'].apply(len)
    df['word_count']    = df['message'].apply(lambda x: len(x.split()))
    return df

st.sidebar.title("🛡️ Spam Detector")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Detect Spam", "📊 Model Performance", "📈 Data Analysis", "ℹ️ About"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Logistic Regression")
st.sidebar.markdown("**Accuracy:** 98.30%")
st.sidebar.markdown("**Dataset:** SMS Spam Collection")

# ── PAGE 1: DETECT SPAM ──────────────────────────────────────
if page == "🔍 Detect Spam":

    st.title("🛡️ Spam Email Detector")
    st.markdown("Powered by **TF-IDF + Logistic Regression** · 98.30% Accuracy")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        subject = st.text_input("📧 Subject Line (optional)",
                                 placeholder="e.g. You've won a $1000 prize!")
        body    = st.text_area("✉️ Message / Email Body", height=200,
                                placeholder="Paste your email or SMS message here...")

        st.markdown("**Try an example:**")
        c1, c2, c3, c4 = st.columns(4)

        if c1.button("💰 Spam: Prize"):
            st.session_state['ex_subject'] = "YOU WON $5000!!!"
            st.session_state['ex_body']    = "CONGRATULATIONS! You have been selected as our GRAND PRIZE WINNER! Claim your FREE $5000 cash prize NOW. Call 1-800-WIN-CASH immediately. Limited time offer!!!"
            st.rerun()

        if c2.button("🏦 Spam: Bank"):
            st.session_state['ex_subject'] = "URGENT: Account Suspended"
            st.session_state['ex_body']    = "Your bank account has been SUSPENDED due to suspicious activity. Verify your details IMMEDIATELY or your account will be permanently closed."
            st.rerun()

        if c3.button("📅 Ham: Meeting"):
            st.session_state['ex_subject'] = "Team standup tomorrow at 10am"
            st.session_state['ex_body']    = "Hi everyone, just a reminder about our weekly standup tomorrow at 10am in the conference room. Please come prepared with your updates. See you then!"
            st.rerun()

        if c4.button("📦 Ham: Order"):
            st.session_state['ex_subject'] = "Your order has shipped"
            st.session_state['ex_body']    = "Hi, your order #12345 has been dispatched. Expected delivery in 2-3 business days. Track your package using the link in your account."
            st.rerun()

        if 'ex_subject' in st.session_state:
            subject = st.session_state['ex_subject']
            body    = st.session_state['ex_body']

        analyze = st.button("🔍 Analyze Message", type="primary",
                             use_container_width=True)

    with col2:
        st.markdown("### ⚙️ NLP Pipeline")
        for icon, title, desc in [
            ("1️⃣", "Input Text",           "Raw email / SMS"),
            ("2️⃣", "Preprocessing",        "Lowercase, remove URLs"),
            ("3️⃣", "TF-IDF Vectorization", "5000 features, bigrams"),
            ("4️⃣", "Logistic Regression",  "Tuned with GridSearchCV"),
            ("5️⃣", "Verdict",              "Spam probability 0-100%"),
        ]:
            st.markdown(f"""
            <div style='padding:8px 12px;margin:5px 0;border-radius:8px;
                        border-left:3px solid #667eea;background:#1e2130'>
                <b>{icon} {title}</b><br>
                <small style='color:#aaa'>{desc}</small>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏆 Model Stats")
        st.metric("Accuracy", "98.30%")
        st.metric("ROC-AUC",  "0.9878")
        st.metric("F1-Score", "0.9199")

    if analyze:
        if not body and not subject:
            st.warning("⚠️ Please enter a message to analyze.")
        else:
            full_text = (f"Subject: {subject}\n\n" if subject else "") + body
            clean     = preprocess_text(full_text)
            pred      = model.predict([clean])[0]
            prob      = model.predict_proba([clean])[0][1]
            is_spam   = pred == 1

            st.markdown("---")
            st.markdown("## 🎯 Analysis Result")

            if is_spam:
                st.error("## 🚫 SPAM DETECTED")
            else:
                st.success("## ✅ LEGITIMATE MESSAGE (HAM)")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Verdict",          "SPAM" if is_spam else "HAM")
            m2.metric("Spam Probability", f"{prob*100:.1f}%")
            m3.metric("Ham Probability",  f"{(1-prob)*100:.1f}%")

            if prob >= 0.85:
                risk = "🔴 Very High Risk"
            elif prob >= 0.60:
                risk = "🟠 High Risk"
            elif prob >= 0.35:
                risk = "🟡 Moderate Risk"
            else:
                risk = "🟢 Low Risk"
            m4.metric("Risk Level", risk)

            st.markdown("### 📊 Confidence Breakdown")
            st.markdown(f"🔴 **Spam: {prob*100:.1f}%**")
            st.progress(float(prob))
            st.markdown(f"🟢 **Ham: {(1-prob)*100:.1f}%**")
            st.progress(float(1 - prob))

            with st.expander("🔬 See preprocessed text"):
                st.code(clean)

# ── PAGE 2: MODEL PERFORMANCE ────────────────────────────────
elif page == "📊 Model Performance":

    st.title("📊 Model Performance")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Model",    "LR + GridSearchCV")
    c2.metric("Test Accuracy", "98.30%")
    c3.metric("ROC-AUC",       "0.9878")
    c4.metric("Best CV F1",    "0.9199")

    st.markdown("---")
    st.subheader("📋 Model Comparison")
    st.dataframe(pd.DataFrame({
        'Model':       ['Naive Bayes', 'LR + SMOTE', 'LR Tuned ✅'],
        'Accuracy':    ['98.12%', '97.94%', '98.30%'],
        'ROC-AUC':     ['0.9886', '0.9867', '0.9878'],
        'Spam Recall': ['87%', '94%', '88%'],
        'Notes':       ['Baseline', 'Better recall', 'Best overall ✅']
    }), use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔲 Confusion Matrix")
        try:
            df = load_data()
            _, X_test, _, y_test = train_test_split(
                df['clean_message'], df['label_enc'],
                test_size=0.2, random_state=42, stratify=df['label_enc']
            )
            y_pred = model.predict(X_test)
            cm     = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Ham','Spam'], yticklabels=['Ham','Spam'])
            ax.set_title('Confusion Matrix', fontweight='bold')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            plt.tight_layout()
            st.pyplot(fig)
            st.markdown(f"""
            - ✅ **True Ham:** {cm[0][0]}
            - ✅ **True Spam:** {cm[1][1]}
            - ⚠️ **Ham → Spam (false pos):** {cm[0][1]}
            - ❌ **Spam missed (false neg):** {cm[1][0]}
            """)
        except Exception as e:
            st.error(f"Error: {e}")

    with col2:
        st.subheader("📄 Classification Report")
        try:
            report_df = pd.DataFrame(
                classification_report(y_test, y_pred,
                    target_names=['Ham','Spam'], output_dict=True)
            ).transpose().round(3)
            st.dataframe(report_df, use_container_width=True)
            st.markdown("---")
            st.subheader("⚙️ Best Hyperparameters")
            st.json({
                "clf__C": 10.0,
                "tfidf__max_features": 5000,
                "tfidf__ngram_range": "(1, 2)",
                "tfidf__sublinear_tf": True,
                "tfidf__stop_words": "english"
            })
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("🔄 5-Fold Cross-Validation Results")
    st.dataframe(pd.DataFrame({
        'Model':       ['Naive Bayes', 'Logistic Regression (SMOTE)'],
        'CV Accuracy': ['98.38% ± 0.22%', '95.56% ± 0.28%'],
        'CV F1-Score': ['0.9387 ± 0.0086', '0.9543 ± 0.0029'],
        'Status':      ['Consistent ✅', 'Consistent ✅']
    }), use_container_width=True, hide_index=True)

# ── PAGE 3: DATA ANALYSIS ────────────────────────────────────
elif page == "📈 Data Analysis":

    st.title("📈 Data Analysis")
    st.markdown("---")

    try:
        df = load_data()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Messages",  f"{len(df):,}")
        c2.metric("Ham Messages",    f"{(df['label']=='ham').sum():,}")
        c3.metric("Spam Messages",   f"{(df['label']=='spam').sum():,}")
        c4.metric("Spam %",          f"{(df['label']=='spam').mean()*100:.1f}%")

        st.markdown("---")
        tab1, tab2, tab3 = st.tabs(["📊 Distribution", "📏 Message Length", "🔤 Top Terms"])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(5, 4))
                df['label'].value_counts().plot(kind='bar', ax=ax,
                    color=['#2ecc71','#e74c3c'], edgecolor='black')
                ax.set_title('Class Distribution', fontweight='bold')
                ax.tick_params(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.pie([4825, 747], labels=['Ham 86.6%','Spam 13.4%'],
                       colors=['#2ecc71','#e74c3c'], autopct='%1.1f%%')
                ax.set_title('Class Balance', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

        with tab2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            df.groupby('label')['msg_length'].plot(
                kind='hist', bins=50, alpha=0.6, ax=axes[0], legend=True)
            axes[0].set_title('Message Length (chars)', fontweight='bold')
            df.groupby('label')['word_count'].plot(
                kind='hist', bins=40, alpha=0.6, ax=axes[1], legend=True)
            axes[1].set_title('Word Count', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Avg Message Length**")
                st.dataframe(df.groupby('label')['msg_length'].mean()
                               .round(1).reset_index(), hide_index=True)
            with col2:
                st.markdown("**Avg Word Count**")
                st.dataframe(df.groupby('label')['word_count'].mean()
                               .round(1).reset_index(), hide_index=True)

        with tab3:
            tfidf_viz = TfidfVectorizer(max_features=5000, ngram_range=(1,2),
                                         stop_words='english', sublinear_tf=True)
            X_all     = tfidf_viz.fit_transform(df['clean_message'])
            fn        = tfidf_viz.get_feature_names_out()
            spam_mean = X_all[df['label_enc'].values==1].toarray().mean(axis=0)
            ham_mean  = X_all[df['label_enc'].values==0].toarray().mean(axis=0)
            top_spam  = [fn[i] for i in spam_mean.argsort()[-15:][::-1]]
            top_ham   = [fn[i] for i in ham_mean.argsort()[-15:][::-1]]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🔴 Top 15 Spam Terms")
                for i, t in enumerate(top_spam, 1):
                    st.markdown(f"`{i:02d}.` **{t}**")
            with col2:
                st.markdown("#### 🟢 Top 15 Ham Terms")
                for i, t in enumerate(top_ham, 1):
                    st.markdown(f"`{i:02d}.` **{t}**")

    except FileNotFoundError:
        st.error("⚠️ spam.csv not found. Place spam.csv in the same folder as app.py")

# ── PAGE 4: ABOUT ────────────────────────────────────────────
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")
    st.markdown("---")
    st.markdown("""
    ## 🛡️ Spam Email Detection using Machine Learning

    ### 📌 Project Summary
    End-to-end NLP classification system detecting spam using TF-IDF
    vectorization and tuned Logistic Regression. Internship-level ML project.

    ---
    ### 🛠️ Tech Stack
    | Component | Technology |
    |---|---|
    | Language | Python 3.x |
    | ML Framework | scikit-learn |
    | Imbalance Handling | imbalanced-learn (SMOTE) |
    | Data Processing | pandas, numpy |
    | Visualization | matplotlib, seaborn |
    | Deployment | Streamlit |
    | Model Saving | joblib |

    ---
    ### 🔁 ML Pipeline
    1. Text Preprocessing — lowercase, remove URLs, emails, punctuation
    2. TF-IDF Vectorization — 5000 features, unigrams + bigrams
    3. Class Balancing — SMOTE (598 → 3859 spam samples)
    4. Model Training — Naive Bayes + Logistic Regression
    5. Hyperparameter Tuning — GridSearchCV, 5-fold CV
    6. Evaluation — Accuracy, F1, ROC-AUC, Confusion Matrix

    ---
    ### 📊 Final Results
    | Model | Accuracy | ROC-AUC |
    |---|---|---|
    | Naive Bayes | 98.12% | 0.9886 |
    | LR + SMOTE | 97.94% | 0.9867 |
    | **LR Tuned — Best** | **98.30%** | **0.9878** |

    ---
    ### 📂 Dataset
    - **Source:** UCI SMS Spam Collection
    - **Size:** 5,572 messages (4,825 ham + 747 spam)
    - **Split:** 80% train / 20% test (stratified)
    """)