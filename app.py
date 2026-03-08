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
# SESSION STATE
# ============================================
if 'current_campaign' not in st.session_state:
    st.session_state.current_campaign = None
if 'view' not in st.session_state:
    st.session_state.view = 'list'

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
        return None

# ============================================
# DATABASE FUNCTIONS - COMPOUNDS & PREDICTIONS
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

def save_prediction(conn, compound_id, campaign_id, hsp90_tau, axl_tau, egfr_tau, best_target, category, confidence):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO predictions (compound_id, campaign_id, hsp90_tau_seconds, axl_tau_seconds, egfr_tau_seconds, best_target, category, confidence)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING prediction_id
        """, (compound_id, campaign_id, hsp90_tau, axl_tau, egfr_tau, best_target, category, confidence))
        conn.commit()
        return cursor.fetchone()[0]
    except Exception as e:
        conn.rollback()
        return None
    finally:
        cursor.close()

def add_compound_to_campaign(conn, campaign_id, compound_id, pipeline_stage='Predicted'):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO campaign_compounds (campaign_id, compound_id, pipeline_stage)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            RETURNING campaign_compound_id
        """, (campaign_id, compound_id, pipeline_stage))
        conn.commit()
        result = cursor.fetchone()
        return result[0] if result else None
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
# DATABASE FUNCTIONS - CAMPAIGNS
# ============================================
def create_campaign(conn, campaign_name, target_protein, description, status='Active'):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO campaigns (campaign_name, target_protein, description, status)
            VALUES (%s, %s, %s, %s)
            RETURNING campaign_id
        """, (campaign_name, target_protein, description, status))
        conn.commit()
        return cursor.fetchone()[0]
    except Exception as e:
        conn.rollback()
        return None
    finally:
        cursor.close()

def get_all_campaigns(conn):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT c.*, 
                   COUNT(DISTINCT cc.compound_id) as compound_count,
                   COUNT(DISTINCT p.prediction_id) as prediction_count
            FROM campaigns c
            LEFT JOIN campaign_compounds cc ON c.campaign_id = cc.campaign_id
            LEFT JOIN predictions p ON c.campaign_id = p.campaign_id
            GROUP BY c.campaign_id
            ORDER BY c.created_at DESC
        """)
        return cursor.fetchall()
    except Exception as e:
        return []
    finally:
        cursor.close()

def get_campaign(conn, campaign_id):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT * FROM campaigns WHERE campaign_id = %s
        """, (campaign_id,))
        return cursor.fetchone()
    except Exception as e:
        return None
    finally:
        cursor.close()

def get_campaign_compounds(conn, campaign_id):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT c.*, cc.pipeline_stage, cc.priority, cc.notes, cc.added_at,
                   p.hsp90_tau_seconds, p.axl_tau_seconds, p.egfr_tau_seconds,
                   p.best_target, p.category, p.confidence, p.predicted_at
            FROM campaign_compounds cc
            JOIN compounds c ON cc.compound_id = c.compound_id
            LEFT JOIN predictions p ON c.compound_id = p.compound_id AND p.campaign_id = cc.campaign_id
            WHERE cc.campaign_id = %s
            ORDER BY cc.added_at DESC
        """, (campaign_id,))
        return cursor.fetchall()
    except Exception as e:
        return []
    finally:
        cursor.close()

def delete_campaign(conn, campaign_id):
    cursor = conn.cursor()
    try:
        # Delete related records first
        cursor.execute("DELETE FROM predictions WHERE campaign_id = %s", (campaign_id,))
        cursor.execute("DELETE FROM campaign_compounds WHERE campaign_id = %s", (campaign_id,))
        cursor.execute("DELETE FROM campaigns WHERE campaign_id = %s", (campaign_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
    finally:
        cursor.close()

def update_campaign_status(conn, campaign_id, status):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE campaigns SET status = %s, updated_at = CURRENT_TIMESTAMP
            WHERE campaign_id = %s
        """, (status, campaign_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
    finally:
        cursor.close()

def update_compound_stage(conn, campaign_id, compound_id, new_stage):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE campaign_compounds 
            SET pipeline_stage = %s, stage_updated_at = CURRENT_TIMESTAMP
            WHERE campaign_id = %s AND compound_id = %s
        """, (new_stage, campaign_id, compound_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
    finally:
        cursor.close()

def update_compound_notes(conn, campaign_id, compound_id, notes):
    cursor = conn.cursor()
    try:
        cursor.execute("""
            UPDATE campaign_compounds 
            SET notes = %s
            WHERE campaign_id = %s AND compound_id = %s
        """, (notes, campaign_id, compound_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        return False
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
    if seconds is None:
        return "N/A"
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
            return float(Descriptors.MolWt(mol))
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
        if stats and stats['total_predictions'] and stats['total_predictions'] > 0:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Compounds", stats['total_compounds'])
            col2.metric("Total Predictions", stats['total_predictions'])
            col3.metric("Long Residence", stats['long_count'])
            col4.metric("Short Residence", stats['short_count'])
            st.markdown("---")
    
    # Four tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📁 Campaigns", "🔬 Quick Predict", "📊 Batch Upload", "📜 History"])
    
    # ============================================
    # TAB 1: CAMPAIGNS
    # ============================================
    with tab1:
        if st.session_state.current_campaign is None:
            # Show campaign list
            show_campaign_list(conn, models, desc_names)
        else:
            # Show campaign detail
            show_campaign_detail(conn, models, desc_names)
    
    # ============================================
    # TAB 2: QUICK PREDICT (Single Compound)
    # ============================================
    with tab2:
        st.markdown("#### Enter a SMILES string")
        
        smiles_input = st.text_input("SMILES", placeholder="e.g., Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1")
        compound_name = st.text_input("Compound Name (optional)", placeholder="e.g., Imatinib")
        
        # Save to database option
        save_single = st.checkbox("Save to database", value=True, key="save_single")
        
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
                        category = get_category(best_rt)
                        confidence = int(best_conf * 100)
                        
                        for target in ['HSP90', 'AXL', 'EGFR']:
                            rt = preds[target]['rt']
                            conf = int(preds[target]['r2'] * 100)
                            
                            if target == best:
                                st.success(f"**{target}**: {format_time(rt)} | Confidence: {conf}% ⬆️ Best")
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
                                pred_result = save_prediction(conn, db_compound_id, None,
                                               float(preds['HSP90']['rt']), 
                                               float(preds['AXL']['rt']), 
                                               float(preds['EGFR']['rt']), 
                                               best, category, confidence)
                                if pred_result:
                                    st.success("✅ Saved to database!")
    
    # ============================================
    # TAB 3: BATCH UPLOAD
    # ============================================
    with tab3:
        st.markdown("#### Upload a CSV file with SMILES")
        
        # Campaign selection for batch upload
        if conn:
            campaigns = get_all_campaigns(conn)
            campaign_options = ["No Campaign (Quick Predict)"] + [f"{c['campaign_name']} (ID: {c['campaign_id']})" for c in campaigns]
            selected_campaign = st.selectbox("Add to Campaign (optional)", campaign_options)
            
            if selected_campaign != "No Campaign (Quick Predict)":
                campaign_id = int(selected_campaign.split("ID: ")[1].rstrip(")"))
            else:
                campaign_id = None
        else:
            campaign_id = None
        
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
                
                # Save to database option
                save_to_db = st.checkbox("Save predictions to database", value=True)
                
                if st.button("🚀 Predict All Targets", type="primary"):
                    results_list = []
                    progress = st.progress(0)
                    
                    for idx, row in df.iterrows():
                        smiles = row[smiles_col]
                        compound_id_name = row.get('Compound_ID', row.get('compound_id', f'Cpd_{idx+1}'))
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
                                db_compound_id = save_compound(conn, smiles, str(compound_id_name), mw, str(original_target))
                                if db_compound_id:
                                    save_prediction(conn, db_compound_id, campaign_id,
                                                   float(preds['HSP90']['rt']), 
                                                   float(preds['AXL']['rt']), 
                                                   float(preds['EGFR']['rt']), 
                                                   best_target, category, confidence)
                                    # Add to campaign if selected
                                    if campaign_id:
                                        add_compound_to_campaign(conn, campaign_id, db_compound_id)
                            
                            results_list.append({
                                'Compound': compound_id_name,
                                'HSP90 τ': format_time(preds['HSP90']['rt']),
                                'AXL τ': format_time(preds['AXL']['rt']),
                                'EGFR τ': format_time(preds['EGFR']['rt']),
                                'Best Target': best_target,
                                'Category': category,
                                'Confidence': f"{confidence}%"
                            })
                        else:
                            results_list.append({
                                'Compound': compound_id_name,
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
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total", len(results_df))
                    col2.metric("Long τ", len(results_df[results_df['Category'] == 'Long']))
                    col3.metric("Medium τ", len(results_df[results_df['Category'] == 'Medium']))
                    col4.metric("Short τ", len(results_df[results_df['Category'] == 'Short']))
                    
                    st.markdown("---")
                    
                    # Display table
                    st.dataframe(results_df, hide_index=True)
                    
                    if save_to_db and conn:
                        st.success("✅ Predictions saved to database!")
                        if campaign_id:
                            st.success(f"✅ Compounds added to campaign!")
                    
                    # Download
                    st.download_button(
                        "📥 Download Results",
                        results_df.to_csv(index=False),
                        "kineticscout_results.csv",
                        "text/csv"
                    )
    
    # ============================================
    # TAB 4: HISTORY
    # ============================================
    with tab4:
        st.markdown("#### 📜 Prediction History")
        
        if conn:
            history = get_prediction_history(conn)
            
            if history:
                history_df = pd.DataFrame(history)
                
                # Format columns
                history_df['HSP90 τ'] = history_df['hsp90_tau_seconds'].apply(format_time)
                history_df['AXL τ'] = history_df['axl_tau_seconds'].apply(format_time)
                history_df['EGFR τ'] = history_df['egfr_tau_seconds'].apply(format_time)
                history_df['Predicted'] = pd.to_datetime(history_df['predicted_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Predictions", len(history_df))
                col2.metric("Long Residence", len(history_df[history_df['category'] == 'Long']))
                col3.metric("Short Residence", len(history_df[history_df['category'] == 'Short']))
                
                st.markdown("---")
                
                # Display
                display_df = history_df[['compound_name', 'HSP90 τ', 'AXL τ', 'EGFR τ', 'best_target', 'category', 'confidence', 'Predicted']]
                display_df.columns = ['Compound', 'HSP90 τ', 'AXL τ', 'EGFR τ', 'Best Target', 'Category', 'Confidence', 'Predicted']
                
                st.dataframe(display_df, hide_index=True)
                
                # Download history
                st.download_button(
                    "📥 Download History",
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

# ============================================
# CAMPAIGN LIST VIEW
# ============================================
def show_campaign_list(conn, models, desc_names):
    st.markdown("#### 📁 My Campaigns")
    
    # Create new campaign section
    with st.expander("➕ Create New Campaign", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            new_name = st.text_input("Campaign Name", placeholder="e.g., EGFR Inhibitors Q1 2024")
        with col2:
            new_target = st.selectbox("Target Protein", ["Multi-target", "HSP90", "AXL", "EGFR"])
        
        new_description = st.text_area("Description (optional)", placeholder="Brief description of this campaign...")
        
        if st.button("Create Campaign", type="primary"):
            if not new_name:
                st.error("Please enter a campaign name")
            else:
                campaign_id = create_campaign(conn, new_name, new_target, new_description)
                if campaign_id:
                    st.success(f"✅ Campaign '{new_name}' created!")
                    st.rerun()
                else:
                    st.error("Failed to create campaign")
    
    st.markdown("---")
    
    # List campaigns
    if conn:
        campaigns = get_all_campaigns(conn)
        
        if campaigns:
            for campaign in campaigns:
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{campaign['campaign_name']}**")
                        st.caption(f"Target: {campaign['target_protein']} | {campaign['compound_count']} compounds | {campaign['prediction_count']} predictions")
                    
                    with col2:
                        status_color = "🟢" if campaign['status'] == 'Active' else "🟡" if campaign['status'] == 'Paused' else "⚫"
                        st.markdown(f"{status_color} {campaign['status']}")
                    
                    with col3:
                        if st.button("Open", key=f"open_{campaign['campaign_id']}"):
                            st.session_state.current_campaign = campaign['campaign_id']
                            st.rerun()
                    
                    with col4:
                        if st.button("🗑️", key=f"delete_{campaign['campaign_id']}"):
                            if delete_campaign(conn, campaign['campaign_id']):
                                st.success("Campaign deleted!")
                                st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("No campaigns yet. Create one to get started!")
    else:
        st.warning("Database not connected")

# ============================================
# CAMPAIGN DETAIL VIEW
# ============================================
def show_campaign_detail(conn, models, desc_names):
    campaign_id = st.session_state.current_campaign
    campaign = get_campaign(conn, campaign_id)
    
    if not campaign:
        st.error("Campaign not found")
        st.session_state.current_campaign = None
        return
    
    # Back button
    if st.button("← Back to Campaigns"):
        st.session_state.current_campaign = None
        st.rerun()
    
    # Campaign header
    st.markdown(f"## {campaign['campaign_name']}")
    
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Target:** {campaign['target_protein']}")
    col2.markdown(f"**Status:** {campaign['status']}")
    col3.markdown(f"**Created:** {campaign['created_at'].strftime('%Y-%m-%d')}")
    
    if campaign['description']:
        st.markdown(f"*{campaign['description']}*")
    
    st.markdown("---")
    
    # Status update
    col1, col2 = st.columns([3, 1])
    with col2:
        new_status = st.selectbox("Update Status", ["Active", "Paused", "Completed"], 
                                   index=["Active", "Paused", "Completed"].index(campaign['status']))
        if new_status != campaign['status']:
            if update_campaign_status(conn, campaign_id, new_status):
                st.success(f"Status updated to {new_status}")
                st.rerun()
    
    st.markdown("---")
    
    # Add compounds section
    with st.expander("➕ Add Compounds to Campaign"):
        st.markdown("**Upload CSV with SMILES**")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'], key=f"upload_{campaign_id}")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break
            
            if smiles_col is None:
                st.error("No SMILES column found!")
            else:
                st.success(f"Found {len(df)} compounds")
                
                if st.button("Add & Predict", type="primary", key=f"predict_{campaign_id}"):
                    progress = st.progress(0)
                    added_count = 0
                    
                    for idx, row in df.iterrows():
                        smiles = row[smiles_col]
                        compound_name = row.get('Compound_ID', row.get('compound_id', f'Cpd_{idx+1}'))
                        original_target = row.get('Original_Target', row.get('original_target', ''))
                        
                        preds = predict_all_targets(smiles, models, desc_names)
                        
                        if preds:
                            best_target, best_conf = get_best_target(preds)
                            best_rt = preds[best_target]['rt']
                            category = get_category(best_rt)
                            confidence = int(best_conf * 100)
                            
                            mw = get_molecular_weight(smiles)
                            db_compound_id = save_compound(conn, smiles, str(compound_name), mw, str(original_target))
                            
                            if db_compound_id:
                                save_prediction(conn, db_compound_id, campaign_id,
                                               float(preds['HSP90']['rt']),
                                               float(preds['AXL']['rt']),
                                               float(preds['EGFR']['rt']),
                                               best_target, category, confidence)
                                add_compound_to_campaign(conn, campaign_id, db_compound_id)
                                added_count += 1
                        
                        progress.progress((idx + 1) / len(df))
                    
                    st.success(f"✅ Added {added_count} compounds to campaign!")
                    st.rerun()
    
    st.markdown("---")
    
    # Show compounds in campaign
    st.markdown("### Compounds in Campaign")
    
    compounds = get_campaign_compounds(conn, campaign_id)
    
    if compounds:
        # Summary by pipeline stage
        stages = ['Predicted', 'To Synthesize', 'In Synthesis', 'Testing', 'Advanced', 'Deprioritized']
        stage_counts = {stage: 0 for stage in stages}
        for c in compounds:
            stage = c['pipeline_stage'] if c['pipeline_stage'] else 'Predicted'
            if stage in stage_counts:
                stage_counts[stage] += 1
        
        cols = st.columns(len(stages))
        for i, stage in enumerate(stages):
            cols[i].metric(stage, stage_counts[stage])
        
        st.markdown("---")
        
        # Compound table
        compound_data = []
        for c in compounds:
            compound_data.append({
                'Compound': c['compound_name'],
                'HSP90 τ': format_time(c['hsp90_tau_seconds']),
                'AXL τ': format_time(c['axl_tau_seconds']),
                'EGFR τ': format_time(c['egfr_tau_seconds']),
                'Best Target': c['best_target'] if c['best_target'] else 'N/A',
                'Category': c['category'] if c['category'] else 'N/A',
                'Stage': c['pipeline_stage'] if c['pipeline_stage'] else 'Predicted'
            })
        
        compound_df = pd.DataFrame(compound_data)
        st.dataframe(compound_df, hide_index=True)
        
        # Download
        st.download_button(
            "📥 Download Campaign Data",
            compound_df.to_csv(index=False),
            f"campaign_{campaign_id}_compounds.csv",
            "text/csv"
        )
    else:
        st.info("No compounds in this campaign yet. Add some above!")

if __name__ == "__main__":
    main()
