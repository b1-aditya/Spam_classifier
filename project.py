import streamlit as st
import joblib
import pandas as pd

model=joblib.load('spam_clf.pkl')

st.set_page_config(layout="wide")
# ---------------------- Sidebar -----------------------
with st.sidebar:
    
    st.markdown("""
    <div style="
        background: linear-gradient(180deg, #4A148C, #1565C0);
        padding: 20px;
        border-radius: 12px;
        text-align:center;
        color:white;
        margin-bottom:15px;
    ">
        <h2 style="margin:0; font-size:26px;">👤 Developer Info</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    🔗 <b>LinkedIn:</b><br>
<a style="text-decoration:none; color:#1565C0;" 
   href="https://www.linkedin.com/in/aditya-sharma" target="_blank">
linkedin.com/in/aditya-sharma
</a>
<br><br>

✉️ <b>Email:</b><br>
<a style="text-decoration:none; color:#1565C0;" 
   href="mailto:aditya@gmail.com">
aditya.sharma@gmail.com
</a>
<br><br>

📞 <b>Phone:</b><br>
+91-6397XXXXXX
<br><br>

🐙 <b>GitHub:</b><br>
<a style="text-decoration:none; color:#1565C0;" 
   href="https://github.com/adityasharma" target="_blank">
github.com/adityasharma
</a>

    """, unsafe_allow_html=True)



st.markdown("""
    <div style="
        background-color:#4A148C;
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    ">
        <h1 style="color: white; text-transform: uppercase; letter-spacing: 2px;">
            🚫Spam Classifier Project
        </h1>
    </div>
""", unsafe_allow_html=True)


col1,col3=st.columns([1.5,2],gap='large')

with col1:
    # Single Message Spam Prediction banner
    st.markdown("""
<div style="
    background: linear-gradient(90deg, #4A148C, #1565C0);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
">
    <h2 style="color:white; margin:0; font-size:24px;">
        𓂃🖊Single Message Spam Prediction
    </h2>
</div>
""", unsafe_allow_html=True)



    text=st.text_area("Enter your message:",height=100)
    if st.button("Predict"):
        result=model.predict([text])[0]
        if result=='spam':
            st.error("Spam->Irrelevant Message🚫")
        else:
            st.success("Ham->Relevant Message✅️")

with col3:

    # Bulk Message Spam Prediction banner
    st.markdown("""
<div style="
    background: linear-gradient(90deg, #4A148C, #1565C0);
    padding: 15px;
    border-radius: 10px;
    text-align: center;
">
    <h2 style="color:white; margin:0; font-size:24px;">
        𝄜 Bulk Message Spam Prediction
    </h2>
</div>
""", unsafe_allow_html=True)



    file = st.file_uploader('Select file conataining bulk messages', type=['csv','txt'])
        
    if file != None:
        df = pd.read_csv(file, header=None, names=['Msg'])
        place = st.empty()
        Ps = st.success('File Uploaded Successfully')
        place.write(df)

        if st.button('Predict', key='b2'):
            df['result'] = model.predict(df.Msg)
            place.write(df)
            Ps.info('Prediction Completed')

            # 🔥 FIXED download button
            csv_data = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="⬇ Download Predicted File",
                data=csv_data,
                file_name="predicted_results.csv",
                mime="text/csv",
                type="primary",
                icon="📥")
                        

# ============================================================
# 🔥 Global Animated Footer (Always Visible)
# ============================================================

footer_html = """
<style>
@keyframes glow {
    0% { box-shadow: 0 0 5px #ff4081; }
    50% { box-shadow: 0 0 18px #ff80ab; }
    100% { box-shadow: 0 0 5px #ff4081; }
}

.footer-tag {
    position: fixed;
    left: 9px;
    bottom: 15px;
    background: linear-gradient(90deg, #4A148C, #1565C0);
    padding: 10px 18px;
    border-radius: 10px;
    color: white;
    font-size: 15px;
    font-weight: 600;
    animation: glow 2.8s infinite ease-in-out;
    z-index: 9999;
}
</style>

<div class="footer-tag">
    👨‍💻 Developed by <b>Aditya Sharma</b>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)

        