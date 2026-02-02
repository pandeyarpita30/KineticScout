
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
import warnings
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="KineticScout",
    page_icon="🎯",
    layout="wide"
)

# ============================================
# LOAD MODELS
# ============================================
@st.cache_resource
def load_models():
    models = {
        'HSP90': {
            'model': joblib.load('model_HSP90.pkl'),
            'scaler': joblib.load('scaler_HSP90.pkl'),
            'r2': 0.807
        },
        'AXL': {
            'model': joblib.load('model_AXL.pkl'),
            'scaler': joblib.load('scaler_AXL.pkl'),
            'r2': 0.347
        },
        'EGFR': {
            'model': joblib.load('model_EGFR.pkl'),
            'scaler': joblib.load('scaler_EGFR.pkl'),
            'r2': 0.392
        }
    }
    return models

@st.cache_resource
def load_descriptors():
    return joblib.load('descriptor_names.pkl')

# ============================================
# HELPER FUNCTIONS
# ============================================
def calculate_descriptors(smiles, desc_names):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
        descriptors = calc.CalcDescriptors(mol)
        desc_array = np.array(descriptors).reshape(1, -1)
        desc_df = pd.DataFrame(desc_array, columns=desc_names)
        desc_df = desc_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        return desc_df.values
    except:
        return None

def predict_all_targets(smiles, models, desc_names):
    X = calculate_descriptors(smiles, desc_names)
    if X is None:
        return None
    
    results = {}
    for target, data in models.items():
        X_scaled = data['scaler'].transform(X)
        log_koff = data['model'].predict(X_scaled)[0]
        koff = 10 ** log_koff
        rt = 1 / koff
        results[target] = {'rt': rt, 'r2': data['r2']}
    
    return results

def format_time(seconds):
    if seconds >= 86400:
        return f"{seconds/86400:.1f} days"
    elif seconds >= 3600:
        return f"{seconds/3600:.1f} hrs"
    elif seconds >= 60:
        return f"{seconds/60:.1f} min"
    else:
        return f"{seconds:.1f} s"

def get_category(seconds):
    if seconds >= 3600:
        return "Long"
    elif seconds >= 60:
        return "Medium"
    else:
        return "Short"

def get_best_target(results):
    if results is None:
        return "N/A", 0
    best = max(results.items(), key=lambda x: x[1]['rt'])
    return best[0], best[1]['r2']

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.title("🎯 KineticScout")
    st.markdown("### Multi-Target Drug Residence Time Prediction")
    st.markdown("---")
    
    # Load resources
    models = load_models()
    desc_names = load_descriptors()
    
    # Two tabs
    tab1, tab2 = st.tabs(["📁 Batch Upload (CSV)", "✏️ Single Compound"])
    
    # ============================================
    # TAB 1: BATCH UPLOAD
    # ============================================
    with tab1:
        st.markdown("#### Upload a CSV file with SMILES")
        
        # Template download
        template_df = pd.DataFrame({
            'Compound_ID': ['Imatinib', 'Gefitinib', 'Erlotinib'],
            'SMILES': [
                'Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1',
                'COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1',
                'COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC'
            ],
            'Original_Target': ['BCR-ABL', 'EGFR', 'EGFR'],
            'MW': [493.6, 446.9, 393.4]
        })
        
        st.download_button(
            "📥 Download Template CSV",
            template_df.to_csv(index=False),
            "kineticscout_template.csv",
            "text/csv"
        )
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], label_visibility="collapsed")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Find SMILES column
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break
            
            if smiles_col is None:
                st.error("❌ No SMILES column found!")
            else:
                st.success(f"✅ Loaded {len(df)} compounds")
                
                if st.button("🚀 Predict All Targets", type="primary", use_container_width=True):
                    results_list = []
                    progress = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        smiles = row[smiles_col]
                        compound_id = row.get('Compound_ID', row.get('compound_id', f'Cpd_{idx+1}'))
                        
                        preds = predict_all_targets(smiles, models, desc_names)
                        
                        if preds:
                            best_target, best_conf = get_best_target(preds)
                            best_rt = preds[best_target]['rt']
                            
                            results_list.append({
                                'Compound': compound_id,
                                'HSP90 τ': format_time(preds['HSP90']['rt']),
                                'AXL τ': format_time(preds['AXL']['rt']),
                                'EGFR τ': format_time(preds['EGFR']['rt']),
                                'Best Target': best_target,
                                'Category': get_category(best_rt),
                                'Confidence': f"{int(best_conf * 100)}%"
                            })
                        else:
                            results_list.append({
                                'Compound': compound_id,
                                'HSP90 τ': 'Error',
                                'AXL τ': 'Error',
                                'EGFR τ': 'Error',
                                'Best Target': 'N/A',
                                'Category': 'N/A',
                                'Confidence': 'N/A'
                            })
                        
                        progress.progress((idx + 1) / len(df))
                    
                    # Results
                    st.markdown("---")
                    st.markdown("### 📊 Results")
                    
                    results_df = pd.DataFrame(results_list)
                    
                    # Summary
                    st.markdown("#### Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total", len(results_df))
                    col2.metric("Long τ", len(results_df[results_df['Category'] == 'Long']))
                    col3.metric("Medium τ", len(results_df[results_df['Category'] == 'Medium']))
                    col4.metric("Short τ", len(results_df[results_df['Category'] == 'Short']))
                    
                    st.markdown("---")
                    
                    # Display table
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download
                    st.download_button(
                        "📥 Download Results",
                        results_df.to_csv(index=False),
                        "kineticscout_results.csv",
                        "text/csv",
                        use_container_width=True
                    )
    
    # ============================================
    # TAB 2: SINGLE COMPOUND
    # ============================================
    with tab2:
        st.markdown("#### Enter a SMILES string")
        
        smiles_input = st.text_input("SMILES", placeholder="e.g., Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1")
        
        # Examples
        with st.expander("📝 Example Compounds"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Imatinib:**")
                st.code("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", language=None)
            with col2:
                st.markdown("**Gefitinib:**")
                st.code("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", language=None)
        
        if st.button("🔮 Predict", type="primary"):
            if not smiles_input:
                st.error("Please enter a SMILES")
            else:
                preds = predict_all_targets(smiles_input, models, desc_names)
                
                if preds is None:
                    st.error("❌ Invalid SMILES")
                else:
                    st.markdown("---")
                    
                    # Molecule image
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        mol = Chem.MolFromSmiles(smiles_input)
                        img = Draw.MolToImage(mol, size=(300, 300))
                        st.image(img, caption="Structure")
                    
                    with col2:
                        st.markdown("### Predicted Residence Times")
                        
                        best, best_conf = get_best_target(preds)
                        best_rt = preds[best]['rt']
                        
                        for target in ['HSP90', 'AXL', 'EGFR']:
                            rt = preds[target]['rt']
                            conf = int(preds[target]['r2'] * 100)
                            
                            if target == best:
                                st.success(f"**{target}**: {format_time(rt)} | Confidence: {conf}% ⬆️ Best")
                            else:
                                st.info(f"**{target}**: {format_time(rt)} | Confidence: {conf}%")
                        
                        st.markdown("---")
                        st.markdown(f"**Category:** {get_category(best_rt)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>KineticScout v1.0 | NovoDyn Therapeutics</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
