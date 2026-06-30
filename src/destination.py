import polars as pl
import numpy as np
import pickle
import json
import time
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier #this is not very optimal, but for a prototype this will work for now. 
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

"""
DestinationNNPredictor: 
- Inputs: Zone, Hour of Week (Day of Week, Hour of Day)
- Outputs: A 262-length array of probabilities that a taxi would move from the input zone to a particular zone. 
- Architecture: 
--- A simple MLP with 2 hidden layers, with 128 and 64 neurons in each respective hidden layer. 
--- ReLU activations in between, Softmax activation in output layer. 
"""

class DestinationNNPredictor:
    def __init__(self, hidden_layer_sizes=(128, 64), max_iter=15, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.encoder = None
        self.model = None
        
        # Meta-information for mapping and one-hot encoding
        self.unique_pu = []
        self.unique_dow = []
        self.unique_hour = []
        self.unique_do = []
        
        # Mappings from DOLocationID to target index (0-261)
        self.do_to_idx = {}
        self.idx_to_do = {}

    def fit_encoder(self, df_full):
        """
        Pre-fit the OneHotEncoder with all unique values in the dataset to avoid shape mismatch.
        """
        print("Fitting encoder with all unique pickup locations, days of week, and request hours...")
        self.unique_pu = sorted(df_full["PULocationID"].unique().to_list())
        self.unique_dow = sorted(df_full["day_of_week"].unique().to_list())
        self.unique_hour = sorted(df_full["request_hour"].unique().to_list())
        self.unique_do = sorted(df_full["DOLocationID"].unique().to_list())
        
        # Set up mappings
        self.do_to_idx = {val: idx for idx, val in enumerate(self.unique_do)}
        self.idx_to_do = {idx: val for idx, val in enumerate(self.unique_do)}
        
        # Initialize and fit OneHotEncoder
        self.encoder = OneHotEncoder(
            categories=[self.unique_pu, self.unique_dow, self.unique_hour],
            sparse_output=False,
            handle_unknown='ignore'
        )
        # Fit on a single dummy row to initialize output shape
        self.encoder.fit([[self.unique_pu[0], self.unique_dow[0], self.unique_hour[0]]])
        print(f"Encoder fitted. Feature dimensions: {len(self.unique_pu)} (PULocationID) + "
              f"{len(self.unique_dow)} (day_of_week) + {len(self.unique_hour)} (request_hour) = "
              f"{len(self.unique_pu) + len(self.unique_dow) + len(self.unique_hour)}")

    def preprocess(self, df):
        """
        One-hot encode inputs and map target DOLocationIDs.
        """
        # Transform inputs
        X_inputs = df.select(["PULocationID", "day_of_week", "request_hour"]).to_numpy()
        X_encoded = self.encoder.transform(X_inputs)
        
        # Map target classes to indices (0 to 261)
        y = np.array([self.do_to_idx[val] for val in df["DOLocationID"]])
        return X_encoded, y

    def fit(self, df_train_val):
        """
        Trains the neural network using sklearn's MLPClassifier with partial_fit to support all 262 classes.
        """
        if self.encoder is None:
            self.fit_encoder(df_train_val)
            
        print("Preprocessing training and validation data...")
        X, y = self.preprocess(df_train_val)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.15, random_state=self.random_state
        )
        
        print(f"Training MLP Classifier (neural network) on {len(X_train)} samples...")
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation='relu',
            solver='adam',
            random_state=self.random_state
        )
        
        # Use partial_fit to ensure all 262 destination zones are available as output classes
        classes = np.arange(len(self.unique_do))
        batch_size = 1024
        start_time = time.time()
        
        for epoch in range(self.max_iter):
            # Shuffle training data at each epoch
            perm = np.random.RandomState(self.random_state + epoch).permutation(len(X_train))
            X_train_shuf = X_train[perm]
            y_train_shuf = y_train[perm]
            
            # Mini-batch training
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuf[i:i+batch_size]
                y_batch = y_train_shuf[i:i+batch_size]
                self.model.partial_fit(X_batch, y_batch, classes=classes)
            
            # Compute evaluation metrics for this epoch
            train_probs = self.model.predict_proba(X_train)
            val_probs = self.model.predict_proba(X_val)
            
            train_loss = log_loss(y_train, train_probs, labels=classes)
            val_loss = log_loss(y_val, val_probs, labels=classes)
            
            val_preds = np.argmax(val_probs, axis=1)
            val_acc = np.mean(val_preds == y_val)
            
            print(f"Epoch {epoch+1:02d}/{self.max_iter:02d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f}")
                  
        elapsed = time.time() - start_time
        print(f"Training finished in {elapsed:.2f} seconds.")

    def evaluate(self, df_test):
        """
        Evaluates the effectiveness of the neural network on a test dataset.
        Computes Average Log Loss, Top-1 Accuracy, Top-5 Accuracy, and Top-10 Accuracy.
        """
        print("\n--- Evaluating Neural Network ---")
        X_test, y_test = self.preprocess(df_test)
        
        # Get probability distributions
        probs = self.model.predict_proba(X_test) # shape: (n_samples, 262)
        
        # Calculate Log Loss (cross-entropy) using sklearn's log_loss
        classes = np.arange(len(self.unique_do))
        loss = log_loss(y_test, probs, labels=classes)
        
        # Top-k accuracies
        top1_correct = 0
        top5_correct = 0
        top10_correct = 0
        
        # Sort predictions per row in descending order of probability
        sorted_indices = np.argsort(-probs, axis=1)
        
        for i in range(len(y_test)):
            actual = y_test[i]
            top_preds = sorted_indices[i]
            
            if actual == top_preds[0]:
                top1_correct += 1
            if actual in top_preds[:5]:
                top5_correct += 1
            if actual in top_preds[:10]:
                top10_correct += 1
                
        n_samples = len(y_test)
        top1_acc = top1_correct / n_samples
        top5_acc = top5_correct / n_samples
        top10_acc = top10_correct / n_samples
        
        evaluation_results = {
            "log_loss": loss,
            "top1_accuracy": top1_acc,
            "top5_accuracy": top5_acc,
            "top10_accuracy": top10_acc
        }
        
        print(f"Average Cross-Entropy Loss: {loss:.4f}")
        print(f"Top-1 Accuracy:            {top1_acc:.4f} (baseline classification accuracy)")
        print(f"Top-5 Accuracy:            {top5_acc:.4f}")
        print(f"Top-10 Accuracy:           {top10_acc:.4f}")
        
        return evaluation_results

    def export_weights(self, base_path):
        """
        Exports the neural network's weights and biases.
        Saves as raw NumPy formats (.npz) and human-readable JSON (.json).
        """
        if self.model is None:
            raise ValueError("Model is not trained yet.")
            
        base_path = Path(base_path)
        npz_path = base_path.with_suffix('.npz')
        json_path = base_path.with_suffix('.json')
        
        coefs = self.model.coefs_
        intercepts = self.model.intercepts_
        
        # Save as npz (compressed weights and bias terms)
        npz_dict = {}
        for idx, (w, b) in enumerate(zip(coefs, intercepts)):
            npz_dict[f"W_{idx}"] = w
            npz_dict[f"b_{idx}"] = b
        
        npz_dict["unique_pu"] = np.array(self.unique_pu)
        npz_dict["unique_dow"] = np.array(self.unique_dow)
        npz_dict["unique_hour"] = np.array(self.unique_hour)
        npz_dict["unique_do"] = np.array(self.unique_do)
        
        np.savez_compressed(npz_path, **npz_dict)
        print(f"Exported weights in NPZ format to {npz_path}")
        
        # Save as JSON format
        json_dict = {
            "metadata": {
                "hidden_layer_sizes": list(self.hidden_layer_sizes),
                "activation": self.model.activation,
                "input_dim": coefs[0].shape[0],
                "output_dim": coefs[-1].shape[1]
            },
            "unique_pu": self.unique_pu,
            "unique_dow": self.unique_dow,
            "unique_hour": self.unique_hour,
            "unique_do": self.unique_do,
            "layers": []
        }
        
        for idx, (w, b) in enumerate(zip(coefs, intercepts)):
            json_dict["layers"].append({
                "layer_index": idx,
                "weights": w.tolist(),
                "biases": b.tolist()
            })
            
        with open(json_path, "w") as f:
            json.dump(json_dict, f)
        print(f"Exported weights in JSON format to {json_path}")

    def predict(self, pu_location_id, day_of_week, request_hour):
        """
        Perform inference for a single input.
        Returns a probability array of length 262 corresponding to unique DOLocationIDs.
        """
        if self.model is None or self.encoder is None:
            raise ValueError("Model is not trained/loaded yet.")
            
        # One-hot encode input
        input_data = np.array([[pu_location_id, day_of_week, request_hour]])
        encoded_input = self.encoder.transform(input_data)
        
        # Predict class probabilities
        probs = self.model.predict_proba(encoded_input)[0]
        return probs

    def save_model(self, file_path):
        """
        Saves the entire predictor object (model + encoder + mappings) via pickle.
        """
        with open(file_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved complete predictor model to {file_path}")

    @classmethod
    def load_model(cls, file_path):
        """
        Loads the entire predictor object.
        """
        with open(file_path, "rb") as f:
            obj = pickle.load(f)
        print(f"Loaded complete predictor model from {file_path}")
        return obj

def train_and_evaluate_workflow(data_path, export_base, model_pickle_path, sample_size=300000):
    print(f"Loading dataset from {data_path}...")
    df = pl.read_parquet(data_path)
    
    print(f"Dataset loaded. Total records: {len(df)}")
    
    # Shuffle and sample to make training time reasonable while keeping it representative
    if sample_size and sample_size < len(df):
        print(f"Sampling {sample_size} records for training & evaluation...")
        df_sampled = df.sample(n=sample_size, seed=42)
    else:
        df_sampled = df
        
    # Split into train+val (85%) and test (15%)
    # Using polars, we can split by index
    shuffled_indices = np.random.RandomState(42).permutation(len(df_sampled))
    split_idx = int(0.85 * len(df_sampled))
    
    train_indices = shuffled_indices[:split_idx]
    test_indices = shuffled_indices[split_idx:]
    
    df_train = df_sampled[train_indices]
    df_test = df_sampled[test_indices]
    
    print(f"Train/Val size: {len(df_train)}, Test size: {len(df_test)}")
    
    # Instantiate predictor
    predictor = DestinationNNPredictor(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)
    
    # Train encoder and model
    predictor.fit_encoder(df)  # Fit encoder on complete dataset to capture all classes
    predictor.fit(df_train)
    
    # Evaluate
    predictor.evaluate(df_test)
    
    # Export weights
    predictor.export_weights(export_base)
    
    # Save complete model
    predictor.save_model(model_pickle_path)
    
    # Verify inference
    test_pu = df_test["PULocationID"][0]
    test_dow = df_test["day_of_week"][0]
    test_hr = df_test["request_hour"][0]
    test_do = df_test["DOLocationID"][0]
    
    probs = predictor.predict(test_pu, test_dow, test_hr)
    max_idx = np.argmax(probs)
    pred_do = predictor.idx_to_do[max_idx]
    print("\n--- Inference Verification ---")
    print(f"Input: PULocationID={test_pu}, day_of_week={test_dow}, request_hour={test_hr}")
    print(f"Actual DOLocationID: {test_do}")
    print(f"Predicted DOLocationID: {pred_do} (probability: {probs[max_idx]:.4f})")
    print(f"Output probability vector length: {len(probs)}")

if __name__ == "__main__":
    # Setup paths relative to file location
    SCRIPT_DIR = Path(__file__).resolve().parent
    data_dir = SCRIPT_DIR.parent / "data"
    parquet_file = data_dir / "new.parquet"
    
    export_base_path = data_dir / "destination_model_weights"
    pickle_path = data_dir / "destination_predictor.pkl"
    
    train_and_evaluate_workflow(
        data_path=str(parquet_file),
        export_base=str(export_base_path),
        model_pickle_path=str(pickle_path),
        sample_size=300000
    )


"""

## Usage Example  
To load the trained predictor and perform inference in Python:

predictor = DestinationNNPredictor.load_model("data/destination_predictor.pkl") #load model weights

probs = predictor.predict(pu_location_id=260, day_of_week=6, request_hour=21) #can dynamically adjust inputs

print(len(probs))  # Outputs: 262
print(probs[:5])   # Outputs array of probabilities for the first 5 destination zones
print(max(probs))

"""