# 🎯 KineticScout

## Multi-Target Drug Residence Time Prediction Platform

**KineticScout** is an AI-powered tool that predicts how long a drug molecule stays bound to its target protein. This metric, called **residence time (τ)**, is crucial for drug efficacy and dosing.

🔗 **Live App:** [https://novodyn-kineticscout.streamlit.app](https://novodyn-kineticscout.streamlit.app)

---

## 📌 What is Residence Time?

| Term | Definition | Why It Matters |
|------|------------|----------------|
| **koff** | Dissociation rate (1/s) | How fast drug leaves the target |
| **Residence Time (τ)** | 1/koff (seconds) | How long drug stays bound |

**Simple analogy:** Think of a drug as a key in a lock. Residence time measures how long the key stays in the lock before falling out.

| Residence Time | Drug Effect | Clinical Impact |
|----------------|-------------|-----------------|
| **Long** (hours) | Sustained action | Once-daily dosing possible |
| **Medium** (minutes) | Moderate action | Multiple daily doses |
| **Short** (seconds) | Brief action | Frequent dosing needed |

---

## 🎯 Supported Targets

KineticScout predicts residence time for **3 protein targets**:

| Target | Full Name | Disease Area | Model Accuracy (R²) |
|--------|-----------|--------------|---------------------|
| **HSP90** | Heat Shock Protein 90 | Cancer | 0.807 (81%) |
| **AXL** | AXL Receptor Tyrosine Kinase | Cancer, Fibrosis | 0.347 (35%) |
| **EGFR** | Epidermal Growth Factor Receptor | Lung Cancer (NSCLC) | 0.392 (39%) |

---

## 🚀 Features

### 1. Batch Prediction (CSV Upload)
- Upload multiple compounds at once
- Get predictions for all 3 targets simultaneously
- Download results as CSV

### 2. Single Compound Prediction
- Enter SMILES string manually
- View molecular structure
- Get instant predictions

### 3. Results Include
- Predicted residence time (τ) for each target
- Best target identification
- Category classification (Long/Medium/Short)
- Confidence score based on model accuracy

---

## 📊 How to Use

### Option 1: Batch Upload
1. Prepare a CSV file with columns:
   - `Compound_ID` (name or identifier)
   - `SMILES` (molecular structure)
2. Upload to the app
3. Click "Predict All Targets"
4. Download results

### Option 2: Single Compound
1. Go to "Single Compound" tab
2. Enter SMILES string
3. Click "Predict"
4. View results with molecular structure

---

## 📁 Input Format

### Required CSV Columns:

| Column | Required | Description | Example |
|--------|----------|-------------|---------|
| `Compound_ID` | Optional | Name or ID | Imatinib |
| `SMILES` | **Yes** | Molecular structure | Cc1ccc(NC(=O)...)n1 |
| `Original_Target` | Optional | Known target | BCR-ABL |
| `MW` | Optional | Molecular weight | 493.6 |

### Example Input:
```csv
Compound_ID,SMILES,Original_Target,MW
Imatinib,Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1,BCR-ABL,493.6
Gefitinib,COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1,EGFR,446.9
```

---

## 📤 Output Format

| Column | Description | Example |
|--------|-------------|---------|
| `Compound` | Compound identifier | Imatinib |
| `HSP90 τ` | Predicted residence time for HSP90 | 10.7 min |
| `AXL τ` | Predicted residence time for AXL | 2.7 s |
| `EGFR τ` | Predicted residence time for EGFR | 58.3 s |
| `Best Target` | Target with longest residence time | HSP90 |
| `Category` | Long (>1hr) / Medium (1min-1hr) / Short (<1min) | Medium |
| `Confidence` | Model accuracy for best target | 80% |

### Example Output:
```csv
Compound,HSP90 τ,AXL τ,EGFR τ,Best Target,Category,Confidence
Imatinib,10.7 min,2.7 s,58.3 s,HSP90,Medium,80%
Gefitinib,1.4 min,2.6 s,1.4 min,HSP90,Medium,80%
```

---

## 🔬 Technical Details

### Machine Learning Models

| Target | Algorithm | Training Data | Features |
|--------|-----------|---------------|----------|
| HSP90 | Random Forest (500 trees) | 160 compounds (K4DD) | 210 RDKit descriptors |
| AXL | Random Forest (500 trees) | 123 compounds (Kinome Survey) | 210 RDKit descriptors |
| EGFR | Support Vector Regression | 132 compounds (Kinome Survey) | 210 RDKit descriptors |

### Molecular Descriptors
We use **210 RDKit molecular descriptors** including:
- Molecular weight, LogP, TPSA
- Number of H-bond donors/acceptors
- Rotatable bonds
- Ring counts
- Topological indices
- And more...

### Training Data Sources
1. **K4DD Database** - Kinetics for Drug Discovery (Nature Chemical Biology, 2018)
2. **Kinome Kinetics Survey** - Binding kinetics of 270 inhibitors across 40 kinases (JACS, 2018)

---

## 📈 Model Performance

### Interpretation of R² Values:

| R² Score | Interpretation | Our Models |
|----------|----------------|------------|
| > 0.8 | Excellent | HSP90 ✅ |
| 0.6 - 0.8 | Good | - |
| 0.4 - 0.6 | Moderate | - |
| 0.3 - 0.4 | Usable with caution | AXL, EGFR ⚠️ |
| < 0.3 | Poor | - |

**Note:** HSP90 predictions are highly reliable (R² = 0.81). AXL and EGFR predictions should be used as rough estimates.

---

## ⚠️ Limitations

1. **Training Data Size**
   - HSP90: 160 compounds (good coverage)
   - AXL/EGFR: ~120 compounds each (limited coverage)

2. **Chemical Space**
   - Best for kinase inhibitor-like molecules
   - May not generalize to very different compound classes

3. **Predictions Are Estimates**
   - Use for prioritization, not final decisions
   - Always validate experimentally

4. **No Uncertainty for EGFR**
   - SVR model doesn't provide prediction intervals
   - HSP90 and AXL use Random Forest (ensemble-based)

---

## 🛠️ Local Installation

If you want to run locally:
```bash
# Clone repository
git clone https://github.com/pandeyarpita30/kineticscout.git
cd kineticscout

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### Requirements:
- Python 3.9+
- streamlit
- pandas
- numpy
- scikit-learn
- rdkit
- joblib

---

## 📂 Repository Structure
```
kineticscout/
├── app.py                 # Main Streamlit application
├── model_HSP90.pkl        # HSP90 Random Forest model
├── model_AXL.pkl          # AXL Random Forest model
├── model_EGFR.pkl         # EGFR SVR model
├── scaler_HSP90.pkl       # Feature scaler for HSP90
├── scaler_AXL.pkl         # Feature scaler for AXL
├── scaler_EGFR.pkl        # Feature scaler for EGFR
├── descriptor_names.pkl   # List of 210 RDKit descriptors
├── requirements.txt       # Python dependencies
├── packages.txt           # System dependencies (for Streamlit Cloud)
└── README.md              # This file
```

---

## 🔮 Future Improvements

- [ ] Add more protein targets (JAK2, KIT, MET)
- [ ] Improve AXL/EGFR model accuracy with more data
- [ ] Add molecular visualization features
- [ ] Implement ADMET property predictions
- [ ] Add batch comparison across compounds

---

## 📚 References

1. Georgi V, et al. (2018). "Binding Kinetics Survey of the Drugged Kinome." *J. Am. Chem. Soc.* 140(46):15774-15782.

2. Köster H, et al. (2018). "Kinetics for Drug Discovery: an industry-driven effort to target drug residence time." *Nature Chemical Biology* 14:763-775.

3. Copeland RA (2016). "The drug-target residence time model: a 10-year retrospective." *Nat Rev Drug Discov.* 15(2):87-95.

---

## 👨‍💻 About

**KineticScout** is developed by **NovoDyn Therapeutics** as part of an AI-driven drug discovery initiative.

**Mission:** Accelerate drug discovery by predicting binding kinetics early in the development pipeline.

---

## 📧 Contact

For questions, feedback, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/pandeyarpita30/kineticscout/issues)

---

---

*Built with ❤️ using Streamlit, RDKit, and scikit-learn*

