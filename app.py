
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
            'r2': 0.426
        },
        'EGFR': {
            'model': joblib.load('model_EGFR.pkl'),
            'scaler': joblib.load('scaler_EGFR.pkl'),
            'r2': 0.410
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
    """Predict koff for all 3 targets"""
    X = calculate_descriptors(smiles, desc_names)
    if X is None:
        return None
    
    results = {}
    for target, data in models.items():
        X_scaled = data['scaler'].transform(X)
        log_koff = data['model'].predict(X_scaled)[0]
        koff = 10 ** log_koff
        rt = 1 / koff
        results[target] = {'koff': koff, 'rt': rt, 'r2': data['r2']}
    
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

def get_best_target(results):
    """Return target with longest residence time"""
    if results is None:
        return "N/A"
    best = max(results.items(), key=lambda x: x[1]['rt'])
    return best[0]

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.title("🎯 KineticScout")
    st.markdown("### Multi-Target Drug Residence Time Prediction")
    st.markdown("*Predict binding kinetics for **HSP90**, **AXL**, and **EGFR** simultaneously*")
    
    # Model info bar
    col1, col2, col3 = st.columns(3)
    col1.info("**HSP90** (R² = 0.81)")
    col2.info("**AXL** (R² = 0.43)")
    col3.info("**EGFR** (R² = 0.41)")
    
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
            ]
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
                            results_list.append({
                                'Compound': compound_id,
                                'SMILES': smiles,
                                'HSP90 τ': format_time(preds['HSP90']['rt']),
                                'HSP90 (s)': round(preds['HSP90']['rt'], 2),
                                'AXL τ': format_time(preds['AXL']['rt']),
                                'AXL (s)': round(preds['AXL']['rt'], 2),
                                'EGFR τ': format_time(preds['EGFR']['rt']),
                                'EGFR (s)': round(preds['EGFR']['rt'], 2),
                                'Best Target': get_best_target(preds)
                            })
                        else:
                            results_list.append({
                                'Compound': compound_id,
                                'SMILES': smiles,
                                'HSP90 τ': 'Error',
                                'HSP90 (s)': None,
                                'AXL τ': 'Error',
                                'AXL (s)': None,
                                'EGFR τ': 'Error',
                                'EGFR (s)': None,
                                'Best Target': 'N/A'
                            })
                        
                        progress.progress((idx + 1) / len(df))
                    
                    # Results
                    st.markdown("---")
                    st.markdown("### 📊 Results")
                    
                    results_df = pd.DataFrame(results_list)
                    
                    # Summary
                    st.markdown("#### Summary: Best Target Distribution")
                    best_counts = results_df['Best Target'].value_counts()
                    cols = st.columns(4)
                    cols[0].metric("Total", len(results_df))
                    for i, target in enumerate(['HSP90', 'AXL', 'EGFR']):
                        count = best_counts.get(target, 0)
                        cols[i+1].metric(f"Best for {target}", count)
                    
                    st.markdown("---")
                    
                    # Display table
                    display_df = results_df[['Compound', 'HSP90 τ', 'AXL τ', 'EGFR τ', 'Best Target']]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
                    
                    # Download
                    st.download_button(
                        "📥 Download Full Results",
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
                        
                        # Find best
                        best = get_best_target(preds)
                        
                        for target in ['HSP90', 'AXL', 'EGFR']:
                            rt = preds[target]['rt']
                            r2 = preds[target]['r2']
                            
                            if target == best:
                                st.success(f"**{target}** (R²={r2:.2f}): **{format_time(rt)}** ⬆️ Best")
                            else:
                                st.info(f"**{target}** (R²={r2:.2f}): {format_time(rt)}")
                        
                        st.markdown("---")
                        st.markdown(f"**Best Target:** {best}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>KineticScout v1.0 | NovoDyn Therapeutics</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
