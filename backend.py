import numpy as np
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel
from fastapi import FastAPI
import torch.nn as nn
import pennylane as qml
from dataclasses import dataclass, field
from typing import Dict, List
from architecteParallel import ParallelQuantumConfig, ParallelHybridQuantumClassifier
from dataclasses import dataclass
from architecteSimple import  QuantumConfig ,HybridQuantumClassifier
import joblib
scaler = joblib.load("scalerOneCircuit.pkl")
from sklearn.preprocessing import normalize


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou ["http://127.0.0.1:5500"] si tu veux sécurisé
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===== LOAD CODEBERT =====
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
codebert_model = AutoModel.from_pretrained("microsoft/codebert-base")
codebert_model.to(DEVICE)
codebert_model.eval()
codebert_model.requires_grad_(False)

# ===== INPUT FORMAT =====
class CodeRequest(BaseModel):
    code: str
    model_id: int

# =====================================================================
# 2. FONCTION D'EXTRACTION (À appeler pour chaque code envoyé par l'utilisateur)
# =====================================================================
def extract_features(code: str):
    """
    Extrait les caractéristiques du code source en suivant le pipeline exact de l'expert :
    CodeBERT -> L2 Norm -> Nettoyage -> Scaler -> L2 Norm finale.
    """
    # Sécurité type d'entrée
    if not isinstance(code, str):
        code = str(code)

    # 1. Extraction via CodeBERT
    inputs = tokenizer(
        [code],
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt"
    )
    
    # Déplacement sur le GPU/CPU
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        # Extraction du vecteur CLS (position 0)
        outputs = codebert_model(**inputs)
        features = outputs.last_hidden_state[:, 0, :]
        feat_np = features.cpu().numpy()

        # 2. Première Normalisation L2 (protection division par zéro avec 1e-10)
        # Indispensable avant d'attaquer le scaler de l'expert
        norms = np.linalg.norm(feat_np, axis=1, keepdims=True)
        feat_np = feat_np / (norms + 1e-10)

        # 3. Nettoyage TOTAL (Stabilité pour le circuit quantique)
        # Remplace les NaN ou Inf par 0.0 pour éviter de casser les portes quantiques
        feat_np = np.nan_to_num(feat_np, nan=0.0, posinf=0.0, neginf=0.0)

        # 4. Scaler (chargé via joblib) et Re-normalisation L2 finale
        # Utilise le scaler entraîné sur les données normalisées
        feat_np = scaler.transform(feat_np)
        
        # Normalisation finale pour que les données soient dans la plage de phase du circuit
        feat_np = normalize(feat_np, norm='l2', axis=1)

        # 5. Conversion finale en Tensor PyTorch
        final_features = torch.tensor(feat_np, dtype=torch.float32).to(DEVICE)

    return final_features
#============= charger modèles ===========
config1_1 = QuantumConfig(
    encoding_type='iqp',
    n_qubits=6,
    n_layers=2,
    ansatz_type='strongly_entangling',
    ansatz_params={
        'rotation': 'RY',
        'iqp_repeats': 2,
        'rotation_blocks': ['RY', 'RZ'],
        'entanglement': 'linear',
        'post_hidden_dims': [4, 2],
        'config_name': 'Config1_1'
    },
    measurement_type='expval_z',
    measurement_params={
        'n_pairs': 3,
    },
    use_dim_reduction=True,
    reduced_dim=6,
    dim_reduction_layers=[512, 128, 32],
    batch_size=256,
    learning_rate=0.001,
    epochs=80,
    dropout_pre=0.2,
    dropout_post=0.3,
    use_batch_norm=True,
    data_reuploading=False,
    reupload_layers=1
)

config1_2 = QuantumConfig(
    encoding_type='iqp',
    n_qubits=6,
    n_layers=2,
    ansatz_type='strongly_entangling',
    ansatz_params={
        'rotation': 'RY',
        'iqp_repeats': 2,
        'rotation_blocks': ['RY', 'RZ'],
        'entanglement': 'linear',
        'post_hidden_dims': [3],
        'config_name': 'config1_2'
    },
    measurement_type='expval_z',
    measurement_params={
        'n_pairs': 3,
    },
    use_dim_reduction=True,
    reduced_dim=6,
    dim_reduction_layers=[512, 128, 32],
    batch_size=256,
    learning_rate=0.001,
    epochs=80,
    dropout_pre=0.2,
    dropout_post=0.3,
    use_batch_norm=True,
    data_reuploading=False,
    reupload_layers=1
)
config1 = ParallelQuantumConfig(
    split_strategy='fixed_parts',
    num_parts=12,
    features_per_part=64,
    total_processed_dim=768,
    encoding_type='amplitude',
    n_qubits=6,
    n_layers=2,
    ansatz_type='strongly_entangling',
    ansatz_params={
        'post_hidden_dims': [64, 32],
        'config_name': 'config1'
    },
    measurement_type='expval_z',
    use_dim_reduction=True,
    dim_reduction_layers=[256, 128],
    batch_size=256,
    learning_rate=0.001,
    epochs=80,
    dropout_rate=0.3,
    aggregation='concat'
)
n_qubits = 5
features_per_part = 2 ** n_qubits
total_features = 768
num_parts = total_features 
config2 = ParallelQuantumConfig(
    split_strategy='fixed_parts',
    num_parts=num_parts,
    features_per_part=features_per_part,
    total_processed_dim=total_features,
    encoding_type='amplitude',
    n_qubits=n_qubits,
    n_layers=2,
    ansatz_type='strongly_entangling',
    ansatz_params={
        'post_hidden_dims': [64, 32],
        'config_name': 'config2'
    },
    measurement_type='expval_z',
    use_dim_reduction=True,
    dim_reduction_layers=[256, 128],
    batch_size=256,
    learning_rate=0.001,
    epochs=80,
    dropout_rate=0.3,
    aggregation='concat'
)

model1 = None
model2 = None
model3= None
#model4=None
# dictionnaire
models = {}
@app.on_event("startup")
def load_models():
    global model1, model2, Model3 ,Model4

    print("Loading models...")

    # model 1
    model1 = HybridQuantumClassifier(input_dim=768, config=config1_1)
    model1.load_state_dict(torch.load("config1_1.pth", map_location=DEVICE), strict=True)
    model1.to(DEVICE)
    model1.eval()
    print("Model1 loaded ✔")
    #model2
    model2 = HybridQuantumClassifier(input_dim=768, config=config1_2)
    model2.load_state_dict(torch.load("config1_2.pth", map_location=DEVICE), strict=True)
    model2.to(DEVICE)
    model2.eval()
    print("Model2 loaded ✔")
    #model3
    model3 = ParallelHybridQuantumClassifier(input_dim=768, config=config1)
    model3.load_state_dict(torch.load("config1 (1).pth", map_location=DEVICE), strict=True)
    model3.to(DEVICE)
    model3.eval()
    print("Model3 loaded ✔")
    #model4
    #model4 = ParallelHybridQuantumClassifier(input_dim=768, config=config2)
    #model4.load_state_dict(torch.load("config2.pth", map_location=DEVICE), strict=True)
    #model4.to(DEVICE)
    #model4.eval()
    #print("Model4 loaded ✔")

    #print(" Backend ready")
    models[1] = model1
    models[2] = model2
    models[3] = model3
    #models[4] = model4
    

    print("Model1 addet to dictionnaire  ")
    print("Model2 addet to dictionnaire ")
    print("Model3 addet to dictionnaire  ")
    #print("Model4 addet to dictionnaire  ")

codebert_model.to(DEVICE)

#=====================================================================================================================================================================#

history_db = []
# celle qui le frontend a besoin  
@app.post("/predict")
def predict(data: CodeRequest):

    print(" [1] Request received")

    try:
        #  Vérifier le modèle
        model = models.get(data.model_id)
        print(" [2] Model ID:", data.model_id)

        if model is None:
            print(" [ERROR] Model not found")
            return {
                "message": "Invalid model id",
                "confidence": 0,
                "type": "error"
            }

        print(" [3] Model selected successfully")

        #  Vérifier code vide
        if not data.code.strip():
            print(" [WARNING] Empty code")
            return {
                "message": "Empty code provided",
                "confidence": 0,
                "type": "warning"
            }

        print(" [4] Code is not empty")

        #  Feature extraction
        print(" [5] Starting feature extraction...")
        features = extract_features(data.code).to(DEVICE)
        print(" [6] Features extracted")

        #  DEBUG (TRÈS IMPORTANT)
        print(" INPUT MEAN:", features.mean().item())
        print(" INPUT STD:", features.std().item())

        #  Prediction
        print(" [7] Starting prediction...")
        with torch.no_grad():
            output = model(features)

        print(" [8] Raw output:", output)

        #  IMPORTANT : appliquer sigmoid UNE SEULE FOIS
        score = output.item()

        print(" [9] Probability:", score)

        #  Décision finale
        bug_prob = score
        clean_prob = 1 - score
#==========================seuil===========================================================================
        if data.model_id == 1: 
            Prediction = "BUG" if bug_prob > 0.6071 else "CLEAN"
        elif data.model_id == 2:
            Prediction = "BUG" if bug_prob >  0.6138 else "CLEAN"
        elif data.model_id ==3:
            Prediction = "BUG" if bug_prob > 0.61957 else "CLEAN"
        else:
            Prediction = "BUG" if bug_prob > 0.7 else "CLEAN"
        
        confidence_percent = round(max(bug_prob, clean_prob) * 100, 2)

        print(" [10] Sending response")
        history_db.append({
            "model": data.model_id,
            "code": data.code,
            "result": Prediction
})
        return {
            "message": Prediction,
            "confidence": confidence_percent,
            "type": "error" if Prediction == "BUG" else "success"
        }

    except Exception as e:
        print(" [EXCEPTION]", str(e))
        return {
            "message": f"Server error: {str(e)}",
            "confidence": 0,
            "type": "error"
        }
# ===== ROUTE : FILE INPUT =====
@app.post("/predict_file")
async def predict_file(file: UploadFile = File(...), model_id: int = 1):
    content = await file.read()
    code = content.decode("utf-8")

    data = CodeRequest(code=code, model_id=model_id)
    return predict(data)


#===========history====================================================================
@app.get("/history")
def get_history():
    return history_db

@app.delete("/history/{index}")
def delete_history(index: int):
    if 0 <= index < len(history_db):
        history_db.pop(index)
        return {"message": "Deleted"}
    return {"error": "Invalid index"}


#=================signup et login ==============================
#user → signup → backend save → redirect debug
#user → login → verify → redirect debug
#fake database   users_db = memory only
# disparaît si serveur restart
users_db = []
@app.post("/signup")
def signup(data: dict):
    for user in users_db:
        if user["email"] == data["email"]:
            return {"success": False, "message": "Email exists"}

    users_db.append(data)
    return {"success": True}

@app.post("/login")
def login(data: dict):
    for user in users_db:
        if user["email"] == data["email"] and user["password"] == data["password"]:
            return {"success": True}

    return {"success": False}

@app.get("/favicon.ico")
def favicon():
    return {}

@app.get("/")
def home():
    return {"message": "Quantum AI Backend is running "}