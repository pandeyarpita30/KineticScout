
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
    layout="centered"
)

# ============================================
# LOAD MODELS (cached)
# ============================================
@st.cache_resource
def load_models():
    models = {
        'HSP90': {
            'model': joblib.load('model_HSP90.pkl'),
            'scaler': joblib.load('scaler_HSP90.pkl'),
            'r2': 0.807,
            'type': 'Random Forest',
            'description': 'Heat Shock Protein 90 - Cancer target'
        },
        'AXL': {
            'model': joblib.load('model_AXL.pkl'),
            'scaler': joblib.load('scaler_AXL.pkl'),
            'r2': 0.426,
            'type': 'SVR',
            'description': 'Receptor Tyrosine Kinase - Oncology target'
        },
        'EGFR': {
            'model': joblib.load('model_EGFR.pkl'),
            'scaler': joblib.load('scaler_EGFR.pkl'),
            'r2': 0.410,
            'type': 'SVR',
            'description': 'Epidermal Growth Factor Receptor - NSCLC target'
        }
    }
    return models

@st.cache_resource
def load_descriptors():
    return joblib.load('descriptor_names.pkl')

# ============================================
# DESCRIPTOR CALCULATION
# ============================================
def calculate_descriptors(smiles, desc_names):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "Invalid SMILES"
    
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    descriptors = calc.CalcDescriptors(mol)
    
    # Clean descriptors
    desc_array = np.array(descriptors).reshape(1, -1)
    desc_df = pd.DataFrame(desc_array, columns=desc_names)
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    
    if desc_df.isna().any().any():
        # Fill NaN with 0 (or could return error)
        desc_df = desc_df.fillna(0)
    
    return desc_df.values, mol

# ============================================
# PREDICTION FUNCTION
# ============================================
def predict_koff(smiles, target, models, desc_names):
    # Calculate descriptors
    X, mol = calculate_descriptors(smiles, desc_names)
    
    if X is None:
        return None, None, None, mol
    
    # Get model and scaler
    model = models[target]['model']
    scaler = models[target]['scaler']
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    log_koff_pred = model.predict(X_scaled)[0]
    
    # Convert from log10
    koff_pred = 10 ** log_koff_pred
    
    # Calculate residence time (τ = 1/koff)
    residence_time = 1 / koff_pred
    
    return koff_pred, residence_time, log_koff_pred, mol

# ============================================
# FORMAT RESIDENCE TIME
# ============================================
def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds/3600:.1f} hours"
    else:
        return f"{seconds/86400:.1f} days"

# ============================================
# MAIN APP
# ============================================
def main():
    # Header
    st.title("🎯 KineticScout")
    st.markdown("### Predict Drug-Target Binding Kinetics")
    st.markdown("---")
    
    # Load resources
    models = load_models()
    desc_names = load_descriptors()
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    **KineticScout** predicts how long a drug stays bound to its target.
    
    - **koff**: Dissociation rate (1/s)
    - **Residence Time**: 1/koff
    
    Longer residence time = longer drug effect
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model Performance")
    for name, data in models.items():
        st.sidebar.markdown(f"**{name}**: R² = {data['r2']:.3f}")
    
    # Main input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        smiles_input = st.text_input(
            "Enter SMILES",
            placeholder="e.g., CC(=O)Nc1ccc(O)cc1",
            help="Enter a valid SMILES string for your compound"
        )
    
    with col2:
        target = st.selectbox(
            "Select Target",
            options=list(models.keys()),
            format_func=lambda x: f"{x} (R²={models[x]['r2']:.2f})"
        )
    
    # Example molecules
    with st.expander("📝 Example SMILES"):
        examples = {
            "Imatinib": "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
            "Gefitinib": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
            "Erlotinib": "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC",
            "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
        }
        for name, smi in examples.items():
            if st.button(f"Use {name}", key=name):
                st.session_state.smiles = smi
                st.rerun()
    
    # Check for session state
    if 'smiles' in st.session_state:
        smiles_input = st.session_state.smiles
        del st.session_state.smiles
    
    # Predict button
    if st.button("🔮 Predict", type="primary", use_container_width=True):
        if not smiles_input:
            st.error("Please enter a SMILES string")
        else:
            with st.spinner("Calculating..."):
                koff, rt, log_koff, mol = predict_koff(
                    smiles_input, target, models, desc_names
                )
            
            if mol is None:
                st.error("❌ Invalid SMILES. Please check your input.")
            else:
                st.markdown("---")
                st.markdown("## Results")
                
                # Show molecule
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Molecule Structure**")
                    img = Draw.MolToImage(mol, size=(250, 250))
                    st.image(img)
                
                with col2:
                    st.markdown(f"**Target:** {target}")
                    st.markdown(f"*{models[target]['description']}*")
                    st.markdown("")
                    
                    # Results metrics
                    m1, m2 = st.columns(2)
                    m1.metric("koff (1/s)", f"{koff:.2e}")
                    m2.metric("Residence Time", format_time(rt))
                    
                    # Interpretation
                    st.markdown("---")
                    if rt > 3600:  # > 1 hour
                        st.success("✅ **Long residence time** - Favorable for sustained target engagement")
                    elif rt > 60:  # > 1 minute
                        st.info("ℹ️ **Moderate residence time** - Typical for many drugs")
                    else:
                        st.warning("⚠️ **Short residence time** - May require frequent dosing")
                
                # Technical details
                with st.expander("📊 Technical Details"):
                    st.markdown(f"""
                    | Parameter | Value |
                    |-----------|-------|
                    | log₁₀(koff) | {log_koff:.3f} |
                    | koff | {koff:.4e} 1/s |
                    | Residence Time (τ) | {rt:.2f} s |
                    | Model | {models[target]['type']} |
                    | Model R² | {models[target]['r2']:.3f} |
                    """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>KineticScout v1.0 | NovoDyn Therapeutics</p>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
