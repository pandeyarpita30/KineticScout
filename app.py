import streamlit as st
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

st.set_page_config(page_title="NovoDyn KineticScout", page_icon="🧬")

st.title("🧬 NovoDyn KineticScout")
st.write("AI-powered drug-target residence time predictions")

model = joblib.load('model.pkl')

uploaded_file = st.file_uploader("Upload CSV with SMILES column", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {len(df)} compounds")
    
    if st.button("🔬 Predict Kinetics"):
        results = []
        for _, row in df.iterrows():
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                features = [[
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Lipinski.NumHDonors(mol),
                    Lipinski.NumHAcceptors(mol),
                    Lipinski.NumRotatableBonds(mol),
                    Descriptors.TPSA(mol)
                ]]
                pred = model.predict(features)[0]
                results.append({
                    'Compound': row.get('Compound_ID', 'Unknown'),
                    'Prediction': pred
                })
        
        st.write("### Results")
        st.dataframe(pd.DataFrame(results))

st.markdown("---")
st.markdown("*Powered by NovoDyn | AI for Drug Discovery*")
