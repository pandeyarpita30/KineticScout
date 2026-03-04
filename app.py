import streamlit as st
import joblib
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
from rdkit.ML.Descriptors import MoleculeDescriptors
from datetime import datetime
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
# DATABASE CONNECTION
# ============================================
def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=st.secrets["database"]["host"],
            port=st.secrets["database"]["port"],
            database=st.secrets["database"]["database"],
            user=st.secrets["database"]["user"],
            password=st.secrets["database"]["password"]
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# ============================================
# DATABASE FUNCTIONS
# ============================================
def save_compound(conn, smiles, compound_name, molecular_weight, original_target):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO compounds (smiles, compound_name, molecular_weight, original_target)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (smiles) DO UPDATE SET compound_name = EXCLUDED.compound_name
            RETURNING compound_id
        """, (smiles, compound_name, molecular_weight, original_target))
        conn.commit()
        return cursor.fetchone()[0]
    except Exception as e:
        conn.rollback()
        return None
    finally:
        cursor.close()

def save_prediction(conn, compound_id, hsp90_tau, axl_tau, egfr_tau, best_target, category, confidence):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO predictions (compound_id, hsp90_tau_seconds, axl_tau_seconds, egfr_tau_seconds, best_target, category, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING prediction_id
        """, (compound_id, hsp90_tau, axl_tau, egfr_tau, best_target, category, confidence))
        conn.commit()
        return cursor.fetchone()[0]
    except Exception as e:
        conn.rollback()
        return None
    finally:
        cursor.close()

def get_prediction_history(conn, limit=50):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT c.compound_name, c.smiles, p.hsp90_tau_seconds, p.axl_tau_seconds, 
                   p.egfr_tau_seconds, p.best_target, p.category, p.confidence, p.predicted_at
            FROM predictions p
            JOIN compounds c ON p.compound_id = c.compound_id
            ORDER BY p.predicted_at DESC
            LIMIT %s
        """, (limit,))
        return cursor.fetchall()
    except Exception as e:
        return []
    finally:
        cursor.close()

def get_stats(conn):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT compound_id) as total_compounds,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN category = 'Long' THEN 1 ELSE 0 END) as long_count,
                SUM(CASE WHEN category = 'Medium' THEN 1 ELSE 0 END) as medium_count,
                SUM(CASE WHEN category = 'Short' THEN 1 ELSE 0 END) as short_count
            FROM predictions
        """)
        return cursor.fetchone()
    except Exception as e:
        return None
    finally:
        cursor.close()

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

def get_molecular_weight(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Descriptors.MolWt(mol)
        return None
    except:
        return None

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
    
    # Database connection
    conn = get_db_connection()
    
    # Show stats if connected
    if conn:
        stats = get_stats(conn)
        if stats and stats['total_predictions'] > 0:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Compounds", stats['total_compounds'])
            col2.metric("Total Predictions", stats['total_predictions'])
            col3.metric("Long Residence", stats['long_count'])
            col4.metric("Short Residence", stats['short_count'])
            st.markdown("---")
    
    # Three tabs
    tab1, tab2, tab3 = st.tabs(["📁 Batch Upload", "✏️ Single Compound", "📜 History"])
    
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
            "Download Template CSV",
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
                st.error("No SMILES column found!")
            else:
                st.success(f"Loaded {len(df)} compounds")
                
                # Save to database option
                save_to_db = st.checkbox("Save predictions to database", value=True)
                
                if st.button("Predict All Targets", type="primary", use_container_width=True):
                    results_list = []
                    progress = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        smiles = row[smiles_col]
                        compound_id = row.get('Compound_ID', row.get('compound_id', f'Cpd_{idx+1}'))
                        original_target = row.get('Original_Target', row.get('original_target', ''))
                        
                        preds = predict_all_targets(smiles, models, desc_names)
                        
                        if preds:
                            best_target, best_conf = get_best_target(preds)
                            best_rt = preds[best_target]['rt']
                            category = get_category(best_rt)
                            confidence = int(best_conf * 100)
                            
                            # Save to database if connected and checkbox is checked
                            if conn and save_to_db:
                                mw = get_molecular_weight(smiles)
                                db_compound_id = save_compound(conn, smiles, str(compound_id), mw, str(original_target))
                                if db_compound_id:
                                    save_prediction(conn, db_compound_id, 
                                                   preds['HSP90']['rt'], 
                                                   preds['AXL']['rt'], 
                                                   preds['EGFR']['rt'], 
                                                   best_target, category, confidence)
                            
                            results_list.append({
                                'Compound': compound_id,
                                'HSP90 τ': format_time(preds['HSP90']['rt']),
                                'AXL τ': format_time(preds['AXL']['rt']),
                                'EGFR τ': format_time(preds['EGFR']['rt']),
                                'Best Target': best_target,
                                'Category': category,
                                'Confidence': f"{confidence}%"
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
                    st.markdown("### Results")
                    
                    results_df = pd.DataFrame(results_list)
                    
                    # Summary
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total", len(results_df))
                    col2.metric("Long", len(results_df[results_df['Category'] == 'Long']))
                    col3.metric("Medium", len(results_df[results_df['Category'] == 'Medium']))
                    col4.metric("Short", len(results_df[results_df['Category'] == 'Short']))
                    
                    st.markdown("---")
                    
                    # Display table
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    if save_to_db and conn:
                        st.success("Predictions saved to database!")
                    
                    # Download
                    st.download_button(
                        "Download Results",
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
        compound_name = st.text_input("Compound Name (optional)", placeholder="e.g., Imatinib")
        
        # Save to database option
        save_single = st.checkbox("Save to database", value=True, key="save_single")
        
        # Examples
        with st.expander("Example Compounds"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Imatinib:**")
                st.code("Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", language=None)
            with col2:
                st.markdown("**Gefitinib:**")
                st.code("COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", language=None)
        
        if st.button("Predict", type="primary"):
            if not smiles_input:
                st.error("Please enter a SMILES")
            else:
                preds = predict_all_targets(smiles_input, models, desc_names)
                
                if preds is None:
                    st.error("Invalid SMILES")
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
                        category = get_category(best_rt)
                        confidence = int(best_conf * 100)
                        
                        for target in ['HSP90', 'AXL', 'EGFR']:
                            rt = preds[target]['rt']
                            conf = int(preds[target]['r2'] * 100)
                            
                            if target == best:
                                st.success(f"**{target}**: {format_time(rt)} | Confidence: {conf}% - Best")
                            else:
                                st.info(f"**{target}**: {format_time(rt)} | Confidence: {conf}%")
                        
                        st.markdown("---")
                        st.markdown(f"**Category:** {category}")
                        
                        # Save to database
                        if conn and save_single:
                            mw = get_molecular_weight(smiles_input)
                            name = compound_name if compound_name else f"Compound_{datetime.now().strftime('%H%M%S')}"
                            db_compound_id = save_compound(conn, smiles_input, name, mw, "")
                            if db_compound_id:
                                save_prediction(conn, db_compound_id, 
                                               preds['HSP90']['rt'], 
                                               preds['AXL']['rt'], 
                                               preds['EGFR']['rt'], 
                                               best, category, confidence)
                                st.success("Saved to database!")
    
    # ============================================
    # TAB 3: HISTORY
    # ============================================
    with tab3:
        st.markdown("#### Prediction History")
        
        if conn:
            history = get_prediction_history(conn)
            
            if history:
                history_df = pd.DataFrame(history)
                
                # Format columns
                history_df['HSP90 τ'] = history_df['hsp90_tau_seconds'].apply(format_time)
                history_df['AXL τ'] = history_df['axl_tau_seconds'].apply(format_time)
                history_df['EGFR τ'] = history_df['egfr_tau_seconds'].apply(format_time)
                history_df['Predicted'] = pd.to_datetime(history_df['predicted_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Display
                display_df = history_df[['compound_name', 'HSP90 τ', 'AXL τ', 'EGFR τ', 'best_target', 'category', 'confidence', 'Predicted']]
                display_df.columns = ['Compound', 'HSP90 τ', 'AXL τ', 'EGFR τ', 'Best Target', 'Category', 'Confidence', 'Predicted']
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Download history
                st.download_button(
                    "Download History",
                    display_df.to_csv(index=False),
                    "kineticscout_history.csv",
                    "text/csv"
                )
            else:
                st.info("No predictions yet. Upload compounds to get started!")
        else:
            st.warning("Database not connected. History unavailable.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>KineticScout v2.0 | NovoDyn Therapeutics</p>",
        unsafe_allow_html=True
    )
    
    # Close database connection
    if conn:
        conn.close()

if __name__ == "__main__":
    main()
