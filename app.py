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
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'table'
if 'selected_compound' not in st.session_state:
    st.session_state.selected_compound = None

# Pipeline stages
PIPELINE_STAGES = ['Predicted', 'To Synthesize', 'In Synthesis', 'Testing', 'Advanced', 'Deprioritized']

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
# DATABASE FUNCTIONS - DASHBOARD
# ============================================
def get_dashboard_stats(conn):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT 
                (SELECT COUNT(*) FROM campaigns) as total_campaigns,
                (SELECT COUNT(*) FROM campaigns WHERE status = 'Active') as active_campaigns,
                (SELECT COUNT(*) FROM compounds) as total_compounds,
                (SELECT COUNT(*) FROM predictions) as total_predictions,
                (SELECT COUNT(*) FROM predictions WHERE category = 'Long') as long_count,
                (SELECT COUNT(*) FROM predictions WHERE category = 'Medium') as medium_count,
                (SELECT COUNT(*) FROM predictions WHERE category = 'Short') as short_count
        """)
        return cursor.fetchone()
    except Exception as e:
        return None
    finally:
        cursor.close()

def get_pipeline_stats(conn):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT pipeline_stage, COUNT(*) as count
            FROM campaign_compounds
            GROUP BY pipeline_stage
            ORDER BY 
                CASE pipeline_stage
                    WHEN 'Predicted' THEN 1
                    WHEN 'To Synthesize' THEN 2
                    WHEN 'In Synthesis' THEN 3
                    WHEN 'Testing' THEN 4
                    WHEN 'Advanced' THEN 5
                    WHEN 'Deprioritized' THEN 6
                END
        """)
        return cursor.fetchall()
    except Exception as e:
        return []
    finally:
        cursor.close()

def get_recent_predictions(conn, limit=10):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT c.compound_name, p.best_target, p.category, p.predicted_at,
                   cam.campaign_name
            FROM predictions p
            JOIN compounds c ON p.compound_id = c.compound_id
            LEFT JOIN campaigns cam ON p.campaign_id = cam.campaign_id
            ORDER BY p.predicted_at DESC
            LIMIT %s
        """, (limit,))
        return cursor.fetchall()
    except Exception as e:
        return []
    finally:
        cursor.close()

def get_target_distribution(conn):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT best_target, COUNT(*) as count
            FROM predictions
            WHERE best_target IS NOT NULL
            GROUP BY best_target
            ORDER BY count DESC
        """)
        return cursor.fetchall()
    except Exception as e:
        return []
    finally:
        cursor.close()

def get_campaign_summary(conn):
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cursor.execute("""
            SELECT c.campaign_name, c.target_protein, c.status,
                   COUNT(DISTINCT cc.compound_id) as compound_count,
                   SUM(CASE WHEN cc.pipeline_stage = 'Advanced' THEN 1 ELSE 0 END) as advanced_count
            FROM campaigns c
            LEFT JOIN campaign_compounds cc ON c.campaign_id = cc.campaign_id
            GROUP BY c.campaign_id, c.campaign_name, c.target_protein, c.status
            ORDER BY c.created_at DESC
        """)
        return cursor.fetchall()
    except Exception as e:
        return []
    finally:
        cursor.close()

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

def get_stage_color(stage):
    colors = {
        'Predicted': '🔵',
        'To Synthesize': '🟡',
        'In Synthesis': '🟠',
        'Testing': '🟣',
        'Advanced': '🟢',
        'Deprioritized': '⚫'
    }
    return colors.get(stage, '⚪')

def time_ago(dt):
    if dt is None:
        return "Unknown"
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} days ago"
    elif diff.seconds >= 3600:
        return f"{diff.seconds // 3600} hours ago"
    elif diff.seconds >= 60:
        return f"{diff.seconds // 60} minutes ago"
    else:
        return "Just now"

# ============================================
# MAIN APP
# ============================================
def main():
    st.title("🎯 KineticScout")
    st.markdown("### Multi-Target Drug Residence Time Prediction")
    st.markdown("---")
    
    models = load_models()
    desc_names = load_descriptors()
    conn = get_db_connection()
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Dashboard", "📁 Campaigns", "🔬 Quick Predict", "📊 Batch Upload", "📜 History"])
    
    with tab1:
        show_dashboard(conn)
    
    with tab2:
        if st.session_state.current_campaign is None:
            show_campaign_list(conn, models, desc_names)
        else:
            show_campaign_detail(conn, models, desc_names)
    
    with tab3:
        st.markdown("#### Enter a SMILES string")
        
        smiles_input = st.text_input("SMILES", placeholder="e.g., Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1")
        compound_name = st.text_input("Compound Name (optional)", placeholder="e.g., Imatinib")
        
        save_single = st.checkbox("Save to database", value=True, key="save_single")
        
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
                    st.error("Invalid SMILES")
                else:
                    st.markdown("---")
                    
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
                                st.success(f"**{target}**: {format_time(rt)} | Confidence: {conf}% | Best")
                            else:
                                st.info(f"**{target}**: {format_time(rt)} | Confidence: {conf}%")
                        
                        st.markdown("---")
                        st.markdown(f"**Category:** {category}")
                        
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
                                    st.success("Saved to database!")
    
    with tab4:
        st.markdown("#### Upload a CSV file with SMILES")
        
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
            
            smiles_col = None
            for col in df.columns:
                if 'smiles' in col.lower():
                    smiles_col = col
                    break
            
            if smiles_col is None:
                st.error("No SMILES column found!")
            else:
                st.success(f"Loaded {len(df)} compounds")
                
                save_to_db = st.checkbox("Save predictions to database", value=True)
                
                if st.button("Predict All Targets", type="primary"):
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
                            
                            if conn and save_to_db:
                                mw = get_molecular_weight(smiles)
                                db_compound_id = save_compound(conn, smiles, str(compound_id_name), mw, str(original_target))
                                if db_compound_id:
                                    save_prediction(conn, db_compound_id, campaign_id,
                                                   float(preds['HSP90']['rt']), 
                                                   float(preds['AXL']['rt']), 
                                                   float(preds['EGFR']['rt']), 
                                                   best_target, category, confidence)
                                    if campaign_id:
                                        add_compound_to_campaign(conn, campaign_id, db_compound_id)
                            
                            results_list.append({
                                'Compound': compound_id_name,
                                'HSP90': format_time(preds['HSP90']['rt']),
                                'AXL': format_time(preds['AXL']['rt']),
                                'EGFR': format_time(preds['EGFR']['rt']),
                                'Best Target': best_target,
                                'Category': category,
                                'Confidence': f"{confidence}%"
                            })
                        else:
                            results_list.append({
                                'Compound': compound_id_name,
                                'HSP90': 'Error',
                                'AXL': 'Error',
                                'EGFR': 'Error',
                                'Best Target': 'N/A',
                                'Category': 'N/A',
                                'Confidence': 'N/A'
                            })
                        
                        progress.progress((idx + 1) / len(df))
                    
                    st.markdown("---")
                    st.markdown("### Results")
                    
                    results_df = pd.DataFrame(results_list)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total", len(results_df))
                    col2.metric("Long", len(results_df[results_df['Category'] == 'Long']))
                    col3.metric("Medium", len(results_df[results_df['Category'] == 'Medium']))
                    col4.metric("Short", len(results_df[results_df['Category'] == 'Short']))
                    
                    st.markdown("---")
                    st.dataframe(results_df, hide_index=True)
                    
                    if save_to_db and conn:
                        st.success("Predictions saved to database!")
                        if campaign_id:
                            st.success("Compounds added to campaign!")
                    
                    st.download_button(
                        "Download Results",
                        results_df.to_csv(index=False),
                        "kineticscout_results.csv",
                        "text/csv"
                    )
    
    with tab5:
        st.markdown("#### Prediction History")
        
        if conn:
            history = get_prediction_history(conn)
            
            if history:
                history_df = pd.DataFrame(history)
                
                history_df['HSP90'] = history_df['hsp90_tau_seconds'].apply(format_time)
                history_df['AXL'] = history_df['axl_tau_seconds'].apply(format_time)
                history_df['EGFR'] = history_df['egfr_tau_seconds'].apply(format_time)
                history_df['Predicted'] = pd.to_datetime(history_df['predicted_at']).dt.strftime('%Y-%m-%d %H:%M')
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Predictions", len(history_df))
                col2.metric("Long Residence", len(history_df[history_df['category'] == 'Long']))
                col3.metric("Short Residence", len(history_df[history_df['category'] == 'Short']))
                
                st.markdown("---")
                
                display_df = history_df[['compound_name', 'HSP90', 'AXL', 'EGFR', 'best_target', 'category', 'confidence', 'Predicted']]
                display_df.columns = ['Compound', 'HSP90', 'AXL', 'EGFR', 'Best Target', 'Category', 'Confidence', 'Predicted']
                
                st.dataframe(display_df, hide_index=True)
                
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
    
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: gray;'>KineticScout v2.0 | NovoDyn Therapeutics</p>",
        unsafe_allow_html=True
    )
    
    if conn:
        conn.close()

# ============================================
# DASHBOARD VIEW
# ============================================
def show_dashboard(conn):
    st.markdown("#### Dashboard")
    
    if not conn:
        st.warning("Database not connected")
        return
    
    stats = get_dashboard_stats(conn)
    pipeline_stats = get_pipeline_stats(conn)
    recent = get_recent_predictions(conn, limit=5)
    target_dist = get_target_distribution(conn)
    campaign_summary = get_campaign_summary(conn)
    
    st.markdown("### Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    if stats:
        col1.metric("Active Campaigns", stats['active_campaigns'])
        col2.metric("Total Compounds", stats['total_compounds'])
        col3.metric("Total Predictions", stats['total_predictions'])
        col4.metric("Total Campaigns", stats['total_campaigns'])
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Pipeline Overview")
        if pipeline_stats:
            for stage_data in pipeline_stats:
                stage = stage_data['pipeline_stage']
                count = stage_data['count']
                color = get_stage_color(stage)
                st.markdown(f"{color} **{stage}**: {count} compounds")
        else:
            st.info("No compounds in pipeline yet")
    
    with col2:
        st.markdown("### Category Breakdown")
        if stats:
            total = (stats['long_count'] or 0) + (stats['medium_count'] or 0) + (stats['short_count'] or 0)
            if total > 0:
                long_pct = 100 * (stats['long_count'] or 0) // total
                med_pct = 100 * (stats['medium_count'] or 0) // total
                short_pct = 100 * (stats['short_count'] or 0) // total
                st.markdown(f"🟢 **Long** (>1 hr): {stats['long_count'] or 0} ({long_pct}%)")
                st.markdown(f"🟡 **Medium** (1 to 60 min): {stats['medium_count'] or 0} ({med_pct}%)")
                st.markdown(f"🔴 **Short** (<1 min): {stats['short_count'] or 0} ({short_pct}%)")
            else:
                st.info("No predictions yet")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Best Target Distribution")
        if target_dist:
            for target_data in target_dist:
                target = target_data['best_target']
                count = target_data['count']
                st.markdown(f"**{target}**: {count} compounds")
        else:
            st.info("No predictions yet")
    
    with col2:
        st.markdown("### Campaign Summary")
        if campaign_summary:
            for camp in campaign_summary:
                status_color = "🟢" if camp['status'] == 'Active' else "🟡" if camp['status'] == 'Paused' else "⚫"
                st.markdown(f"{status_color} **{camp['campaign_name']}**: {camp['compound_count']} compounds, {camp['advanced_count'] or 0} advanced")
        else:
            st.info("No campaigns yet")
    
    st.markdown("---")
    
    st.markdown("### Recent Predictions")
    if recent:
        for pred in recent:
            time_str = time_ago(pred['predicted_at'])
            campaign_str = f" in {pred['campaign_name']}" if pred['campaign_name'] else ""
            category_color = "🟢" if pred['category'] == 'Long' else "🟡" if pred['category'] == 'Medium' else "🔴"
            st.markdown(f"{category_color} **{pred['compound_name']}** | {pred['best_target']} ({pred['category']}){campaign_str} | {time_str}")
    else:
        st.info("No recent predictions")

# ============================================
# CAMPAIGN LIST VIEW
# ============================================
def show_campaign_list(conn, models, desc_names):
    st.markdown("#### My Campaigns")
    
    with st.expander("Create New Campaign", expanded=False):
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
                    st.success(f"Campaign '{new_name}' created!")
                    st.rerun()
                else:
                    st.error("Failed to create campaign")
    
    st.markdown("---")
    
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
                        if st.button("Delete", key=f"delete_{campaign['campaign_id']}"):
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
    
    if st.button("Back to Campaigns"):
        st.session_state.current_campaign = None
        st.session_state.view_mode = 'table'
        st.rerun()
    
    st.markdown(f"## {campaign['campaign_name']}")
    
    col1, col2, col3 = st.columns(3)
    col1.markdown(f"**Target:** {campaign['target_protein']}")
    col2.markdown(f"**Status:** {campaign['status']}")
    col3.markdown(f"**Created:** {campaign['created_at'].strftime('%Y-%m-%d')}")
    
    if campaign['description']:
        st.markdown(f"*{campaign['description']}*")
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col2:
        new_status = st.selectbox("Update Status", ["Active", "Paused", "Completed"], 
                                   index=["Active", "Paused", "Completed"].index(campaign['status']),
                                   key="status_update")
        if new_status != campaign['status']:
            if update_campaign_status(conn, campaign_id, new_status):
                st.success(f"Status updated to {new_status}")
                st.rerun()
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Table View", type="primary" if st.session_state.view_mode == 'table' else "secondary"):
            st.session_state.view_mode = 'table'
            st.rerun()
    with col2:
        if st.button("Pipeline View", type="primary" if st.session_state.view_mode == 'pipeline' else "secondary"):
            st.session_state.view_mode = 'pipeline'
            st.rerun()
    
    st.markdown("---")
    
    with st.expander("Add Compounds to Campaign"):
        uploaded_file = st.file_uploader("Upload CSV with SMILES", type=['csv'], key=f"upload_{campaign_id}")
        
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
                
                if st.button("Add and Predict", type="primary", key=f"predict_{campaign_id}"):
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
                    
                    st.success(f"Added {added_count} compounds to campaign!")
                    st.rerun()
    
    st.markdown("---")
    
    if st.session_state.view_mode == 'table':
        show_table_view(conn, campaign_id)
    else:
        show_pipeline_view(conn, campaign_id)

# ============================================
# TABLE VIEW
# ============================================
def show_table_view(conn, campaign_id):
    st.markdown("### Compounds Table")
    
    compounds = get_campaign_compounds(conn, campaign_id)
    
    if compounds:
        stage_counts = {stage: 0 for stage in PIPELINE_STAGES}
        for c in compounds:
            stage = c['pipeline_stage'] if c['pipeline_stage'] else 'Predicted'
            if stage in stage_counts:
                stage_counts[stage] += 1
        
        cols = st.columns(len(PIPELINE_STAGES))
        for i, stage in enumerate(PIPELINE_STAGES):
            cols[i].metric(stage, stage_counts[stage])
        
        st.markdown("---")
        
        for compound in compounds:
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 2])
                
                with col1:
                    st.markdown(f"**{compound['compound_name']}**")
                    st.caption(f"Best: {compound['best_target']} | {compound['category']}")
                
                with col2:
                    st.markdown(f"HSP90: {format_time(compound['hsp90_tau_seconds'])}")
                
                with col3:
                    st.markdown(f"AXL: {format_time(compound['axl_tau_seconds'])}")
                
                with col4:
                    st.markdown(f"EGFR: {format_time(compound['egfr_tau_seconds'])}")
                
                with col5:
                    current_stage = compound['pipeline_stage'] if compound['pipeline_stage'] else 'Predicted'
                    new_stage = st.selectbox(
                        "Stage",
                        PIPELINE_STAGES,
                        index=PIPELINE_STAGES.index(current_stage),
                        key=f"stage_{compound['compound_id']}",
                        label_visibility="collapsed"
                    )
                    if new_stage != current_stage:
                        if update_compound_stage(conn, campaign_id, compound['compound_id'], new_stage):
                            st.rerun()
                
                with st.expander(f"Notes for {compound['compound_name']}", expanded=False):
                    current_notes = compound['notes'] if compound['notes'] else ""
                    new_notes = st.text_area("Notes", value=current_notes, key=f"notes_{compound['compound_id']}")
                    if new_notes != current_notes:
                        if st.button("Save Notes", key=f"save_notes_{compound['compound_id']}"):
                            if update_compound_notes(conn, campaign_id, compound['compound_id'], new_notes):
                                st.success("Notes saved!")
                                st.rerun()
                
                st.markdown("---")
        
        compound_data = []
        for c in compounds:
            compound_data.append({
                'Compound': c['compound_name'],
                'HSP90': format_time(c['hsp90_tau_seconds']),
                'AXL': format_time(c['axl_tau_seconds']),
                'EGFR': format_time(c['egfr_tau_seconds']),
                'Best Target': c['best_target'] if c['best_target'] else 'N/A',
                'Category': c['category'] if c['category'] else 'N/A',
                'Stage': c['pipeline_stage'] if c['pipeline_stage'] else 'Predicted',
                'Notes': c['notes'] if c['notes'] else ''
            })
        
        compound_df = pd.DataFrame(compound_data)
        st.download_button(
            "Download Campaign Data",
            compound_df.to_csv(index=False),
            f"campaign_{campaign_id}_compounds.csv",
            "text/csv"
        )
    else:
        st.info("No compounds in this campaign yet. Add some above!")

# ============================================
# PIPELINE VIEW (KANBAN)
# ============================================
def show_pipeline_view(conn, campaign_id):
    st.markdown("### Pipeline View")
    
    compounds = get_campaign_compounds(conn, campaign_id)
    
    if not compounds:
        st.info("No compounds in this campaign yet. Add some above!")
        return
    
    stage_compounds = {stage: [] for stage in PIPELINE_STAGES}
    for c in compounds:
        stage = c['pipeline_stage'] if c['pipeline_stage'] else 'Predicted'
        if stage in stage_compounds:
            stage_compounds[stage].append(c)
    
    cols = st.columns(len(PIPELINE_STAGES))
    
    for i, stage in enumerate(PIPELINE_STAGES):
        with cols[i]:
            st.markdown(f"**{get_stage_color(stage)} {stage}**")
            st.markdown(f"*({len(stage_compounds[stage])} compounds)*")
            st.markdown("---")
            
            for compound in stage_compounds[stage]:
                with st.container():
                    name = compound['compound_name']
                    if len(name) > 15:
                        name = name[:15] + "..."
                    st.markdown(f"**{name}**")
                    
                    category_color = "🟢" if compound['category'] == 'Long' else "🟡" if compound['category'] == 'Medium' else "🔴"
                    best_time = compound['hsp90_tau_seconds'] if compound['best_target'] == 'HSP90' else compound['axl_tau_seconds'] if compound['best_target'] == 'AXL' else compound['egfr_tau_seconds']
                    st.caption(f"{category_color} {compound['best_target']} | {format_time(best_time)}")
                    
                    button_cols = st.columns(2)
                    
                    if i > 0:
                        with button_cols[0]:
                            if st.button("<", key=f"left_{compound['compound_id']}_{stage}"):
                                new_stage = PIPELINE_STAGES[i - 1]
                                if update_compound_stage(conn, campaign_id, compound['compound_id'], new_stage):
                                    st.rerun()
                    
                    if i < len(PIPELINE_STAGES) - 1:
                        with button_cols[1]:
                            if st.button(">", key=f"right_{compound['compound_id']}_{stage}"):
                                new_stage = PIPELINE_STAGES[i + 1]
                                if update_compound_stage(conn, campaign_id, compound['compound_id'], new_stage):
                                    st.rerun()
                    
                    st.markdown("---")
    
    st.markdown("### Summary")
    col1, col2, col3 = st.columns(3)
    
    total = len(compounds)
    in_progress = len(stage_compounds['To Synthesize']) + len(stage_compounds['In Synthesis']) + len(stage_compounds['Testing'])
    advanced = len(stage_compounds['Advanced'])
    
    col1.metric("Total Compounds", total)
    col2.metric("In Progress", in_progress)
    col3.metric("Advanced", advanced)

if __name__ == "__main__":
    main()
