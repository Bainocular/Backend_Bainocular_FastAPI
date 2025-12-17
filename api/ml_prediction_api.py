from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import random
import os
import json

ml_prediction_router = APIRouter()

# Global variables for models (will be loaded on first use)
model = None
le = None
mlb = None


def _load_models():
    """Lazy load models on first use"""
    global model, le, mlb
    
    if model is None or le is None or mlb is None:
        try:
            # Try to find models in different possible locations
            base_paths = [
                os.path.join(os.getcwd(), "models"),
                os.path.join(os.getcwd(), "sla_models"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "sla_models"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sla_models"),
                os.path.expanduser("~/Downloads/SLA/models"),  # Downloads/SLA/models directory
                "/Users/sainathreddy/Downloads/SLA/models",  # Absolute path to Downloads/SLA/models
            ]
            
            model_path = None
            le_path = None
            mlb_path = None
            
            for base_path in base_paths:
                potential_model = os.path.join(base_path, "request_code_model.pkl")
                potential_le = os.path.join(base_path, "label_encoder.pkl")
                potential_mlb = os.path.join(base_path, "multi_label_binarizer.pkl")
                
                if os.path.exists(potential_model) and os.path.exists(potential_le) and os.path.exists(potential_mlb):
                    model_path = potential_model
                    le_path = potential_le
                    mlb_path = potential_mlb
                    break
            
            if not model_path:
                raise FileNotFoundError("Model files not found. Please ensure models are in the correct directory.")
            
            model = joblib.load(model_path)
            le = joblib.load(le_path)
            mlb = joblib.load(mlb_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")


def predict_codes(name_list):
    """Predict codes for a list of request names"""
    _load_models()
    encoded_names = le.transform(name_list).reshape(-1, 1)
    preds = model.predict(encoded_names)
    decoded = mlb.inverse_transform(preds)
    return {name: codes for name, codes in zip(name_list, decoded)}


@ml_prediction_router.get("/api/predict_incident")
async def predict_incident():
    """Predict incident codes from data1.csv in uploads_sla directory"""
    try:
        # Get the path to uploads_sla directory
        upload_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads_sla")
        file_path = os.path.join(upload_dir, "data1.csv")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File 'data1.csv' not found in uploads_sla directory")
        
        # Read file into DataFrame
        df = pd.read_csv(file_path)

        # Try to find sla_config.json in different locations
        # Get all possible base paths (same as model loading logic)
        base_paths = [
            os.path.join(os.getcwd(), "models"),
            os.path.join(os.getcwd(), "sla_models"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "sla_models"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "sla_models"),
            os.path.expanduser("~/Downloads/SLA/models"),  # Downloads/SLA/models directory
            "/Users/sainathreddy/Downloads/SLA/models",  # Absolute path to Downloads/SLA/models
        ]
        
        config_paths = []
        for base_path in base_paths:
            config_paths.append(os.path.join(base_path, "sla_config.json"))
        
        sla_config = None
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as fp:
                        sla_config = json.load(fp)
                    break
                except Exception as e:
                    continue
        
        if sla_config is None:
            raise HTTPException(status_code=404, detail=f"sla_config.json not found. Searched in: {', '.join(config_paths[:4])}")

        # Predict codes
        # Group by Request - ID and Request - Text Request, and aggregate Resource Assigned To - Name
        # This handles cases where the same request ID and text have multiple assigned resources
        grouped_df = df.groupby(["Request - ID", "Request - Text Request"])['Request - Resource Assigned To - Name'].agg(list).reset_index()
        df_filtered = grouped_df[grouped_df["Request - Text Request"].isin(sla_config['Text_Request'])]
        code_mapping = predict_codes(df_filtered["Request - Text Request"].tolist())
        
        res = []
        for _, record in df_filtered.iterrows():
            # Handle Resource Assigned To - Name (may be a list after aggregation)
            resource_name = record['Request - Resource Assigned To - Name']
            if isinstance(resource_name, list):
                resource_name = resource_name[0] if len(resource_name) > 0 else ''
            
            t = [record['Request - ID'], resource_name]
            if len(code_mapping[record['Request - Text Request']]) < 1:
                continue
            code = random.choice(code_mapping[record['Request - Text Request']])
            t.extend(code.split('_')[:-1] if len(code.split('_')) == 5 else code.split('_'))
            t.append(code)
            res.append(t)
        
        res_df = pd.DataFrame(res).iloc[:, :7]
        res_df.columns = ['Ticket ID', 'Name', "Department", "Sub Functional Area", "Location", "Site", 'Review']
        res_df = res_df[['Ticket ID', 'Department', 'Location', 'Site', 'Sub Functional Area', 'Name', 'Review']]

        print(res_df.head())
        # Return results as JSON
        return res_df.to_dict(orient="records")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@ml_prediction_router.post("/api/predict")
async def predict_review(description: str = Form(...)):
    """Predict review code for a single description"""
    try:
        out = predict_codes([description])

        res = {}
        res_val = ['Department', 'Location', 'Site', 'Sub Functional Area']

        if len(out[description]) == 0:
            raise HTTPException(status_code=400, detail="No prediction available for this description")

        for idx, part in enumerate(out[description][0].split('_')[:-1]):
            res[res_val[idx]] = part

        res['Review'] = out[description][0]
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting review: {str(e)}")

