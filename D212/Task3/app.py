import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

# Set page config
st.set_page_config(page_title="D212 Task 3 - Market Basket Analysis", layout="wide")

# Initialize session state for data storage
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Function to load and process all data at startup
def initialize_data():
    if not st.session_state.initialized:
        try:
            with st.spinner("Loading and processing data..."):
                # Load initial data
                df = pd.read_csv('medical_market_basket.csv')
                df = df.dropna(subset=['Presc01'])
                
                # Create melted dataframe
                melted_df = pd.melt(df, value_name='Medication')
                unique_meds = sorted(melted_df['Medication'].dropna().unique())
                
                # Define medication lists
                diabetes_meds = [
                    'metformin', 'glipizide', 'glimepiride',
                    'metformin HCI', 'pioglitazone', 'lantus'
                ]
                
                cardiovascular_meds = [
                    'amlodipine', 'metoprolol', 'lisinopril',
                    'losartan', 'atorvastatin', 'pravastatin',
                    'benazepril', 'carvedilol', 'enalapril',
                    'fenofibrate', 'furosemide', 'hydrochlorothiazide',
                    'isosorbide mononitrate', 'lovastatin',
                    'rosuvastatin', 'simvastatin'
                ]
                
                # Process transactions
                def row_to_transaction(row):
                    return [med for med in row if pd.notna(med)]
                
                transactions = df.apply(row_to_transaction, axis=1).tolist()
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                
                # Generate frequent itemsets and rules
                frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
                rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
                rules = rules.sort_values('support', ascending=False)
                
                # Store everything in session state
                st.session_state.df = df
                st.session_state.melted_df = melted_df
                st.session_state.unique_meds = unique_meds
                st.session_state.diabetes_meds = diabetes_meds
                st.session_state.cardiovascular_meds = cardiovascular_meds
                st.session_state.df_encoded = df_encoded
                st.session_state.rules = rules
                st.session_state.frequent_itemsets = frequent_itemsets
                st.session_state.initialized = True
                
                return True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    return True

# Function to show code with toggle
def show_code_and_results(code, results_func, section_key):
    show_code = st.toggle("Show Code", key=f"toggle_{section_key}")
    if show_code:
        st.code(code, language="python")
    results_func()

# Initialize data at startup
initialized = initialize_data()

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Introduction",
     "I. Research Question",
     "II. Market Basket Justification",
     "III. Data Preparation and Analysis",
     "IV. Data Summary and Implications",
     "V. Attachments"]
)

# Main content
# Brandon Gillins 
# Student Number: 000400953 | Email: bgillin@my.wgu.edu | Date: 08/23/2024
if section == "Introduction":
    st.markdown("""
    # D212 Task 3
    #### WGU Western Governors University 
    """)
    
    st.markdown("""
    # Table of Contents

    ### I. Research Question
    - A1. Proposed Question
    - A2. Goal of the Data Analysis

    ### II. Data Analysis
    - B1. Explain Market Basket Analysis
    - B2. Example of a Transaction in the Data Set
    - B3. Assumption of Market Basket Analysis

    ### III. Data Transformation
    - C1. Data Transformation
    - C2. Generate Frequent Itemsets
    - C3. Generate Association Rules
    - C4. Top Three Relevant Rules from Apriori

    ### IV. Data Analysis       
    - D1. Summary of the Significance of Support, Lift, and Confidence
    - D2. Practical Significance of Findings
    - D3. Recommended Course of Action

    ### V. Attachments
    - F1. Recording
    - F2. Sources
    """)

elif section == "I. Research Question":
    st.markdown("""
    # Part I: Research Question
    ## A1. Proposed question:
    What proportion of our diabetic patients are also prescribed cardiovascular medications, and what are the most common cardiovascular drugs prescribed to these patients?

    ## A2. Goal of the data analysis:
    The goal of this analysis is to quantify the prevalence of cardiovascular medication use among diabetic patients and identify the most common cardiovascular drugs prescribed to this population, in order to improve comprehensive care strategies and optimize medication management for patients with both diabetes and cardiovascular risks.
    """)

elif section == "II. Market Basket Justification":
    st.markdown("""
    # Part II: Market Basket Justification
    ## B1. Explain market basket analyzes:
    Market basket analysis will analyze the prescription data set by identifying frequent itemsets (combinations of medications) and generating association rules. It will look for patterns in co-prescriptions, specifically focusing on the relationship between diabetes medications and cardiovascular drugs.

    We will accomplish this by focusing on our populaiton that is precribed diabetic medications. 

    1. Identify which medications are frequently prescribed together.
    2. Generate rules that show the likelihood of one medication being prescribed given the presence of another.

    Final outcomes:
    - Identification of the most common cardiovascular drugs prescribed to diabetic patients.
    - Discovery of unexpected or interesting medication combinations that might warrant further investigation.
    """)

    def show_initial_data():
        if st.session_state.initialized:
            st.write("### Initial Data Sample")
            st.dataframe(st.session_state.df.head(10))
            st.write("### Unique Medications")
            st.write(sorted(st.session_state.unique_meds))
            st.write("### Medication Categories")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Diabetes Medications:**")
                st.write(st.session_state.diabetes_meds)
            with col2:
                st.write("**Cardiovascular Medications:**")
                st.write(st.session_state.cardiovascular_meds)

    show_code_and_results(
        """
        import pandas as pd
        
        # Read the CSV file
        df = pd.read_csv('medical_market_basket.csv')
        
        # Drop all rows where Presc01 is null
        df = df.dropna(subset=['Presc01'])
        
        # Melt the dataframe to create a single column of medications
        melted_df = pd.melt(df, value_name='Medication')
        
        # Get unique medications and sort them
        unique_meds = sorted(melted_df['Medication'].dropna().unique())
        
        # Define medication lists
        diabetes_meds = [
            'metformin', 'glipizide', 'glimepiride',
            'metformin HCI', 'pioglitazone', 'lantus'
        ]
        
        cardiovascular_meds = [
            'amlodipine', 'metoprolol', 'lisinopril',
            'losartan', 'atorvastatin', 'pravastatin',
            'benazepril', 'carvedilol', 'enalapril',
            'fenofibrate', 'furosemide', 'hydrochlorothiazide',
            'isosorbide mononitrate', 'lovastatin',
            'rosuvastatin', 'simvastatin'
        ]
        """,
        show_initial_data,
        "initial_data"
    )

    st.markdown("""
    ## B2. Example of a transaction in the data set:
    A transaction in this data set represents a patient's prescription history, with each row corresponding to an individual patient and columns representing different medications prescribed to that patient.
    
    Here's an example of a transaction from our dataset that includes a diabetic medication:
    """)

    def show_transaction_example():
        if st.session_state.initialized:
            st.write(st.session_state.df.iloc[0])

    show_code_and_results(
        "# Example transaction\ndf.iloc[0]",
        show_transaction_example,
        "transaction_example"
    )

    st.markdown("""
    In this transaction, we can see that the patient has been prescribed glipizide, which is a medication used to treat diabetes. This example illustrates how each patient's prescription history is represented as a single transaction in the data set, with multiple medications potentially prescribed to the same patient. The presence of glipizide indicates that this patient is likely being treated for diabetes, while the other medications suggest treatment for various other conditions, potentially including cardiovascular issues with medications such as amlodipine or pravastatin.

    ## B3. Assumption of market basket analysis:
    A key assumption of market basket analysis is the support-confidence framework. In the context of our study on diabetic patients and cardiovascular medications, we can explained this as follows:

    The support-confidence framework assumes that the strength of associations between diabetes and cardiovascular medications can be meaningfully measured by two metrics: support and confidence.

    1. Support: This measures how frequently a medication set in the dataset.
    2. Confidence: This measures the likelihood of a cardiovascular medication being prescribed given that a diabetes medication is prescribed.

    - Quantify the prevalence of cardiovascular medication use among diabetic patients (support).
    - Identify the most common cardiovascular drugs prescribed to diabetic patients (support and confidence).
    - Assess the strength of associations between specific diabetes and cardiovascular medications (confidence).
    """)
elif section == "III. Data Preparation and Analysis":
    st.markdown("""
    # Part III: Data Preparation and Analysis
    ## C1. Data Transformation

    Step 1: Import Libraries
    - TransactionEncoder is used to convert data into the proper structure for market basket analysis.

    Step 2: Create a function to create transactions
    - We define a function called row_to_transaction that goes through each row and creates a list of non-null medications.

    Step 3: Encode Transactions
    - We encode the transactions using One-Hot Encoding, which converts all transactions into a binary array.

    Step 4: Save Encoded File
    - The encoded data is saved to a file named "cleaned_medical_market_basket.csv".
    """)

    def show_transformed_data():
        if st.session_state.initialized:
            st.write("### Encoded Data Sample (First 10 rows)")
            st.dataframe(st.session_state.df_encoded.head(10))
            st.write(f"Shape of encoded data: {st.session_state.df_encoded.shape}")

    show_code_and_results(
        """
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import apriori, association_rules

        # Function to convert a row to a list of non-null medications
        def row_to_transaction(row):
            return [med for med in row if pd.notna(med)]

        # Apply the function to each row to create the transactions
        transactions = df.apply(row_to_transaction, axis=1).tolist()

        # Use TransactionEncoder to convert the list of transactions into a one-hot encoded DataFrame
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        """,
        show_transformed_data,
        "transformed_data"
    )

    st.markdown("""
    ## C2. Generate Frequent Itemsets
    The Apriori algorithm is applied to find frequent itemsets with a minimum support of 0.005. This is because if we selected 1% for support we would expect to find at least 75 patients with similar patterns given there are 7501 patients. So .005 is looking at results with 35 or more patients.
    """)

    def show_frequent_itemsets():
        if st.session_state.initialized:
            st.write("### Frequent Itemsets")
            st.dataframe(st.session_state.frequent_itemsets.head(10))

    show_code_and_results(
        """
        # Generate frequent itemsets using Apriori algorithm
        frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
        """,
        show_frequent_itemsets,
        "frequent_itemsets"
    )

    st.markdown("""
    ## C3. Generate Association Rules
    Association rules are created from the frequent itemsets, with a minimum confidence threshold of 0.5. That means that if Medication A is precribed that Medication B is also precribed 50% of the time.
    """)

    def show_rules():
        if st.session_state.initialized:
            st.write("### Top 10 Association Rules")
            st.dataframe(st.session_state.rules.head(10))
            st.write("### Summary Statistics")
            st.dataframe(st.session_state.rules[['support', 'confidence', 'lift']].describe())

    show_code_and_results(
        """
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
        rules = rules.sort_values('support', ascending=False)
        """,
        show_rules,
        "rules"
    )

    st.markdown("""
    ## C4. Top three relevant rules from Apriori:

    1. **Top Support Rule**: (lisinopril, atorvastatin) -> (abilify)
       - Support: 0.011065
       - Confidence: 0.503030
       - Lift: 2.110308

    2. **Top Confidence Rule**: (metoprolol, metformin) -> (abilify)
       - Support: 0.005066
       - Confidence: 0.633333
       - Lift: 2.656954

    3. **Top Lift Rule**: (amlodipine, lisinopril) -> (carvedilol)
       - Support: 0.005999
       - Confidence: 0.523256
       - Lift: 3.005315
    """)

# Continuing from the previous part...

elif section == "IV. Data Summary and Implications":
    st.markdown("""
    # Part IV: Data Summary and Implications
    ## D1. Summary of the Significance of Support, Lift, and Confidence

    ### Support
    - Significance: Shows how often medications appear together in prescriptions.
    - Interpretation: 0.5733% of all prescriptions include metoprolol, glipizide, and carvedilol together. While lower than the top overall rule, this represents a significant number of patients given the specificity of combining diabetes and heart medications.

    ### Confidence
    - Significance: Indicates how likely one medication is prescribed when others are present.
    - Interpretation: When a patient is prescribed both metoprolol and glipizide, there's a 50.59% chance they will also be prescribed carvedilol. This suggests a strong association between these medications in treating patients with both diabetes and cardiovascular issues.

    ### Lift
    - Significance: Shows how much more likely medications are prescribed together compared to by chance.
    - Top Overall: (amlodipine, lisinopril) -> (carvedilol), Lift = 3.005315
    - Diabetes-Cardio Focus: (metoprolol, glipizide) -> (carvedilol), Lift = 2.905531
    - Interpretation: Patients taking both metoprolol and glipizide are 2.91 times more likely to also be prescribed carvedilol than the average patient in our dataset. This high lift value, close to the top overall rule, indicates a strong positive association between these medications.
    """)

    def show_metric_visualizations():
        if st.session_state.initialized:
            # Create visualization tabs
            tab1, tab2 = st.tabs(["Metric Distributions", "Support vs Confidence"])
            
            with tab1:
                fig = px.box(st.session_state.rules, 
                           y=["support", "confidence", "lift"],
                           title="Distribution of Support, Confidence, and Lift")
                st.plotly_chart(fig)
            
            with tab2:
                fig = px.scatter(st.session_state.rules, 
                               x="support", y="confidence",
                               color="lift",
                               title="Support vs Confidence (colored by Lift)",
                               labels={"support": "Support",
                                     "confidence": "Confidence",
                                     "lift": "Lift"})
                st.plotly_chart(fig)

    show_code_and_results(
        """
        # Visualize metric distributions
        fig1 = px.box(rules, y=["support", "confidence", "lift"],
                     title="Distribution of Support, Confidence, and Lift")

        # Create scatter plot
        fig2 = px.scatter(rules, x="support", y="confidence",
                         color="lift", title="Support vs Confidence")
        """,
        show_metric_visualizations,
        "metric_visualizations"
    )

    st.markdown("""
    ### Relationship with Research Question:

    Rule: (metoprolol, glipizide) -> (carvedilol)

    1. Support (0.5733%): This represents a significant number of patients given the specific combination of diabetes and heart medications.

    2. Confidence (50.59%): When metoprolol (heart) and glipizide (diabetes) are prescribed, there's about a 50% chance carvedilol (heart) is also prescribed. This suggests a strong link between diabetes and cardiovascular treatments.

    3. Lift (2.905531): Patients on metoprolol and glipizide are nearly 3 times more likely to also receive carvedilol. This high lift, close to the top overall rule, indicates a strong association between these medications.

    ## D2. Practical Significance of Findings

    Our analysis of the rule (metoprolol, glipizide) -> (carvedilol) reveals two key insights with significant practical implications:

    1. Risk Stratification for Cost Forecasting
       - High lift value (2.905531) indicates increased likelihood of multiple cardiovascular medications for diabetic patients.
       - Practical Use: Enhance risk scoring models to better predict patient health and future medical costs in value-based care systems.
       - Benefits: Improved resource allocation, targeted preventive care, and more accurate financial planning.

    2. Complex Medication Regimens and Healthcare Utilization
       - Combination of diabetes and multiple cardiovascular medications suggests complex treatment needs.
       - Implications: May increase frequency of doctor visits and healthcare utilization.
       - Opportunities: Implement targeted medication management programs and enhance patient education to potentially reduce complications and additional doctor visits.

    These insights can help healthcare organizations optimize their value-based care strategies, while putting the patient first and potentially improving patient outcomes while managing costs more effectively.
    """)

    st.markdown("""
    ## D3. Recommended Course of Action

    Our analysis reveals a strong connection between diabetes and cardiovascular medications, leading us to recommend a Medication Adherence Program. This program will include:

    1. Patient Education
       - Medication use and adherence training
       - Understanding medication interactions
       - Recognition of side effects

    2. Personalized Support
       - One-on-one counseling with pharmacists
       - Regular nurse educator sessions
       - Tailored adherence strategies

    3. Technology Integration
       - Mobile app for medication tracking
       - Text reminders for medication schedules
       - Digital educational resources

    4. Regular Monitoring
       - Progress tracking
       - Adherence assessments
       - Outcome measurements

    5. Healthcare Provider Coordination
       - Regular updates to primary care physicians
       - Integration with existing care plans
       - Collaborative care approach
    """)

elif section == "V. Attachments":
    st.markdown("""
    # Part V: Attachments
    ## F1. Recording:

    Link: https://wgu.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=afe1e1bc-9d0c-432a-9a00-b1dc001c498d

    ## F2. Sources 
    Datatab. (n.d.). Market basket analysis [Association analysis]. Retrieved August 27, 2024, from https://datatab.net/tutorial/market-basket-analysis

    mlxtend. (n.d.). apriori: Frequent itemsets via the Apriori algorithm. Retrieved August 29, 2024, from https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/
    """)

# Add footer with data loading status and additional information
st.sidebar.markdown("---")
if st.session_state.initialized:
    st.sidebar.success("✅ Data loaded and processed")
    st.sidebar.info(f"Total transactions: {len(st.session_state.df):,}")
    st.sidebar.info(f"Unique medications: {len(st.session_state.unique_meds):,}")
    st.sidebar.info(f"Total rules generated: {len(st.session_state.rules):,}")
else:
    st.sidebar.error("❌ Data not loaded")
    st.sidebar.warning("Please check if the data file is available in the correct location.")
