import streamlit as st
import pickle
import pandas as pd
from io import StringIO
import time
import sys

# Page configuration
st.set_page_config(
    page_title="Food Sentiment Analyzer",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with multiple methods
@st.cache_resource
def load_model_from_file(filepath='food_sentiment_reg.pkl'):
    """Try multiple methods to load the model"""
    
    # Method 1: Standard pickle
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        return model, "Loaded with pickle"
    except Exception as e:
        pass
    
    # Method 2: Pickle with latin1 encoding
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        return model, "Loaded with pickle (latin1)"
    except Exception as e:
        pass
    
    # Method 3: Try joblib
    try:
        import joblib
        model = joblib.load(filepath)
        return model, "Loaded with joblib"
    except Exception as e:
        pass
    
    # Method 4: Try with different unpickler
    try:
        import pickle
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except:
                    return None
        
        with open(filepath, 'rb') as f:
            model = CustomUnpickler(f).load()
        return model, "Loaded with custom unpickler"
    except Exception as e:
        pass
    
    return None, None

def load_model_from_upload(uploaded_file):
    """Load model from uploaded file"""
    
    # Method 1: Standard pickle
    try:
        model = pickle.load(uploaded_file)
        return model, "Loaded with pickle"
    except Exception as e:
        uploaded_file.seek(0)  # Reset file pointer
    
    # Method 2: Pickle with latin1 encoding
    try:
        model = pickle.load(uploaded_file, encoding='latin1')
        return model, "Loaded with pickle (latin1)"
    except Exception as e:
        uploaded_file.seek(0)
    
    # Method 3: Try joblib
    try:
        import joblib
        model = joblib.load(uploaded_file)
        return model, "Loaded with joblib"
    except Exception as e:
        uploaded_file.seek(0)
    
    return None, None

# Prediction function
def predict_sentiment(model, text):
    try:
        prediction = model.predict([text])[0]
        # Assuming 1 = Positive, 0 = Negative
        sentiment = "Positive" if prediction == 1 else "Negative"
        return sentiment, prediction
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

# Header
st.markdown('<div class="main-header">🍔 Food Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze customer reviews and feedback with AI-powered sentiment detection</div>', unsafe_allow_html=True)

# Load model
model = None
load_method = None

# Try loading from file first
model, load_method = load_model_from_file()

if model is None:
    st.warning("⚠️ Could not load model from file system.")
    st.info("👇 Please upload your model file below")
    
    uploaded_model = st.file_uploader(
        "Upload food_sentiment_reg.pkl or your trained model file", 
        type=['pkl', 'pickle', 'joblib']
    )
    
    if uploaded_model is not None:
        with st.spinner("Loading model..."):
            model, load_method = load_model_from_upload(uploaded_model)
            
            if model is not None:
                st.success(f"✅ Model loaded successfully! ({load_method})")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Failed to load model with all available methods.")
                st.code("""
# Please re-save your model using one of these methods:

# Method 1: Using pickle
import pickle
with open('food_sentiment_reg.pkl', 'wb') as f:
    pickle.dump(your_model, f)

# Method 2: Using joblib (recommended for sklearn models)
import joblib
joblib.dump(your_model, 'food_sentiment_reg.pkl')
                """, language='python')
else:
    st.success(f"✅ Model loaded successfully! ({load_method})")

if model is not None:
    # Sidebar
    st.sidebar.title("📊 About")
    st.sidebar.info(
        "This app analyzes food-related messages and determines whether "
        "the sentiment is **Positive** or **Negative**.\n\n"
        "**Features:**\n"
        "- Single message analysis\n"
        "- Bulk message processing\n"
        "- CSV export for results\n"
        "- Real-time predictions"
    )
    
    st.sidebar.title("📈 Statistics")
    if 'total_predictions' not in st.session_state:
        st.session_state.total_predictions = 0
    if 'positive_count' not in st.session_state:
        st.session_state.positive_count = 0
    if 'negative_count' not in st.session_state:
        st.session_state.negative_count = 0
    
    st.sidebar.metric("Total Predictions", st.session_state.total_predictions)
    col1, col2 = st.sidebar.columns(2)
    col1.metric("✅ Positive", st.session_state.positive_count)
    col2.metric("❌ Negative", st.session_state.negative_count)
    
    # Developer contact information
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 1rem 0;'>
            <p style='font-weight: 600; font-size: 1.1rem; margin: 0.5rem 0; color: #1f77b4;'>
                👨‍💻 Developed by
            </p>
            <p style='font-weight: 700; font-size: 1.2rem; margin: 0.5rem 0;'>
                Aditya Sharma
            </p>
            <div style='margin-top: 1rem; font-size: 0.9rem; color: #666;'>
                <p style='margin: 0.3rem 0;'>
                    📞 <a href='tel:+1234567890' style='color: #666; text-decoration: none;'>+91 63970 15921</a>
                </p>
                <p style='margin: 0.3rem 0;'>
                    📧 <a href='mailto:aditya@example.com' style='color: #666; text-decoration: none;'>adityasharma51123@gmail.com</a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📝 Single Message", "📋 Bulk Analysis", "ℹ️ How to Use"])
    
    # Tab 1: Single Message Analysis
    with tab1:
        st.subheader("Analyze a Single Message")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter your food review or message:",
                height=150,
                placeholder="Example: The pizza was absolutely delicious! Best I've ever had."
            )
            
            analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("### Quick Examples")
            if st.button("😊 Positive Example"):
                user_input = "The food was amazing! Great taste and excellent service."
                st.rerun()
            if st.button("😞 Negative Example"):
                user_input = "Terrible experience. The food was cold and tasteless."
                st.rerun()
        
        if analyze_btn and user_input:
            with st.spinner("Analyzing sentiment..."):
                time.sleep(0.5)  # Visual feedback
                sentiment, prediction = predict_sentiment(model, user_input)
                
                if sentiment:
                    # Update statistics
                    st.session_state.total_predictions += 1
                    if sentiment == "Positive":
                        st.session_state.positive_count += 1
                    else:
                        st.session_state.negative_count += 1
                    
                    # Display result
                    st.markdown("---")
                    if sentiment == "Positive":
                        st.markdown(f"""
                            <div class="sentiment-positive">
                                <h3>✅ Sentiment: Positive</h3>
                                <p><strong>Message:</strong> {user_input}</p>
                                <p><strong>Prediction Value:</strong> {prediction}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                            <div class="sentiment-negative">
                                <h3>❌ Sentiment: Negative</h3>
                                <p><strong>Message:</strong> {user_input}</p>
                                <p><strong>Prediction Value:</strong> {prediction}</p>
                            </div>
                        """, unsafe_allow_html=True)
    
    # Tab 2: Bulk Analysis
    with tab2:
        st.subheader("Bulk Message Analysis")
        
        option = st.radio(
            "Choose input method:",
            ["Upload CSV File", "Paste Multiple Messages"]
        )
        
        results_df = None
        
        if option == "Upload CSV File":
            st.info("📁 Upload a CSV file with a column containing food reviews/messages.")
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.write("**Preview of uploaded data:**")
                    st.dataframe(df.head())
                    
                    # Select column
                    text_column = st.selectbox("Select the column containing messages:", df.columns)
                    
                    if st.button("🚀 Analyze All Messages", type="primary"):
                        with st.spinner(f"Analyzing {len(df)} messages..."):
                            progress_bar = st.progress(0)
                            
                            sentiments = []
                            predictions = []
                            
                            for idx, text in enumerate(df[text_column]):
                                sentiment, prediction = predict_sentiment(model, str(text))
                                sentiments.append(sentiment)
                                predictions.append(prediction)
                                progress_bar.progress((idx + 1) / len(df))
                            
                            df['Sentiment'] = sentiments
                            df['Prediction_Value'] = predictions
                            results_df = df
                            
                            st.success(f"✅ Analysis complete! Processed {len(df)} messages.")
                            
                            # Update statistics
                            st.session_state.total_predictions += len(df)
                            st.session_state.positive_count += sum([1 for s in sentiments if s == "Positive"])
                            st.session_state.negative_count += sum([1 for s in sentiments if s == "Negative"])
                
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
        
        else:  # Paste Multiple Messages
            st.info("📋 Paste multiple messages (one per line)")
            bulk_input = st.text_area(
                "Enter messages:",
                height=200,
                placeholder="The food was great!\nService was terrible.\nLoved the atmosphere!"
            )
            
            if st.button("🚀 Analyze All Messages", type="primary") and bulk_input:
                messages = [msg.strip() for msg in bulk_input.split('\n') if msg.strip()]
                
                with st.spinner(f"Analyzing {len(messages)} messages..."):
                    progress_bar = st.progress(0)
                    
                    results = []
                    for idx, msg in enumerate(messages):
                        sentiment, prediction = predict_sentiment(model, msg)
                        results.append({
                            'Message': msg,
                            'Sentiment': sentiment,
                            'Prediction_Value': prediction
                        })
                        progress_bar.progress((idx + 1) / len(messages))
                    
                    results_df = pd.DataFrame(results)
                    
                    st.success(f"✅ Analysis complete! Processed {len(messages)} messages.")
                    
                    # Update statistics
                    st.session_state.total_predictions += len(messages)
                    st.session_state.positive_count += sum([1 for r in results if r['Sentiment'] == "Positive"])
                    st.session_state.negative_count += sum([1 for r in results if r['Sentiment'] == "Negative"])
        
        # Display results
        if results_df is not None:
            st.markdown("---")
            st.subheader("📊 Results Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Messages", len(results_df))
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                positive_count = len(results_df[results_df['Sentiment'] == 'Positive'])
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("✅ Positive", positive_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                negative_count = len(results_df[results_df['Sentiment'] == 'Negative'])
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("❌ Negative", negative_count)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.subheader("📋 Detailed Results")
            st.dataframe(results_df, use_container_width=True)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Tab 3: How to Use
    with tab3:
        st.subheader("📖 How to Use This App")
        
        st.markdown("""
        ### Single Message Analysis
        1. Navigate to the **Single Message** tab
        2. Enter your food review or message in the text area
        3. Click **Analyze Sentiment** to get instant results
        4. Try the quick examples for demonstration
        
        ### Bulk Analysis
        1. Navigate to the **Bulk Analysis** tab
        2. Choose your input method:
           - **Upload CSV**: Upload a CSV file with your messages
           - **Paste Messages**: Copy and paste multiple messages (one per line)
        3. Click **Analyze All Messages** to process all entries
        4. View the summary statistics and detailed results
        5. Download the results as a CSV file for further analysis
        
        ### Tips
        - The model works best with clear, food-related messages
        - For bulk analysis, ensure your messages are properly formatted
        - Check the sidebar for real-time statistics
        - Results include both sentiment label and prediction value
        
        ### Example Messages
        **Positive:**
        - "The pizza was absolutely delicious!"
        - "Great service and amazing food quality"
        - "Best restaurant experience ever!"
        
        **Negative:**
        - "Food was cold and tasteless"
        - "Terrible service, won't come back"
        - "Disappointed with the quality"
        """)

else:
    st.error("⚠️ Please upload your model file to continue.")
    st.markdown("""
    ### 📝 How to prepare your model file:
    
    Run this code in your Python environment where you trained the model:
    
    ```python
    import joblib
    # or import pickle
    
    # Save your trained model
    joblib.dump(your_trained_model, 'food_sentiment_reg.pkl')
    # or pickle.dump(your_trained_model, open('food_sentiment_reg.pkl', 'wb'))
    ```
    
    Then upload the generated `.pkl` file using the uploader above.
    """)