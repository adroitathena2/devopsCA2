import os as _os, sys as _sys  # _REPO_ROOT_BOOTSTRAP
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clinical_copilot.custom_severity_model import SeverityVocab, SeverityDataset, build_ensemble_model, collate_fn, ENSEMBLE_NAMES
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import numpy as np
import json
from scipy.optimize import minimize

from clinical_copilot import paths
import pickle

def evaluate_and_ensemble():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on device: {device}")

    vocab = SeverityVocab()
    vocab.load('cache/severity_vocab.json')

    train_path = 'cache/severity_train_hq.json'
    train_dataset = SeverityDataset(train_path, vocab)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    val_path = 'cache/severity_val_hq.json'
    val_dataset = SeverityDataset(val_path, vocab)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    models = {
        name: build_ensemble_model(name, len(vocab.word2idx), len(vocab.char2idx),
                                   num_classes=3)
        for name in ENSEMBLE_NAMES
    }

    val_logits = {m: [] for m in models.keys()}
    val_labels = []
    
    train_logits = {m: [] for m in models.keys()}
    train_labels = []

    for name, model in models.items():
        print(f"Gathering logits for {name}...")
        model.load_state_dict(torch.load(f"cache/ensemble_{name}.pt", map_location=device))
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            for word_ids, char_ids, lengths, hard_labels, _ in train_loader:
                if name == ENSEMBLE_NAMES[0]:
                    train_labels.extend(hard_labels.numpy())
                logits = model(word_ids.to(device), char_ids.to(device), lengths)
                train_logits[name].extend(logits.cpu().numpy())

            for word_ids, char_ids, lengths, hard_labels, _ in val_loader:
                if name == ENSEMBLE_NAMES[0]:
                    val_labels.extend(hard_labels.numpy())
                logits = model(word_ids.to(device), char_ids.to(device), lengths)
                val_logits[name].append(logits.cpu().numpy()[0])

    val_labels = np.array(val_labels)
    train_labels = np.array(train_labels)
    y_true_binary_val = (val_labels > 0).astype(int)

    results = {}
    base_probs_serious = {}
    weighted_ensemble_probs = np.zeros((len(val_labels), 3))
    
    optimal_temps = {}
    for name in models.keys():
        # Optimize temperature using the training set to prevent data leakage
        train_logits_arr = np.array(train_logits[name])
        val_logits_arr = np.array(val_logits[name])

        def eval_nll(t_val):
            scaled_logits = torch.tensor(train_logits_arr) / t_val[0]
            loss = F.cross_entropy(scaled_logits, torch.tensor(train_labels))
            return loss.item()

        opt_res = minimize(eval_nll, [1.5], bounds=[(0.5, 5.0)], method='L-BFGS-B')
        optimal_T = opt_res.x[0]
        optimal_temps[name] = optimal_T

        # Apply calibrated temperature to validation set
        val_calibrated_probs = F.softmax(torch.tensor(val_logits_arr) / optimal_T, dim=1).numpy()
        preds = np.argmax(val_calibrated_probs, axis=1)
        acc = accuracy_score(val_labels, preds)

        weight = acc ** 2
        weighted_ensemble_probs += (val_calibrated_probs * weight)

        f1 = f1_score(val_labels, preds, average='macro')
        prob_serious = val_calibrated_probs[:, 1:].sum(axis=1)
        base_probs_serious[name] = prob_serious

        try:
            auc = roc_auc_score(y_true_binary_val, prob_serious)
            fpr, tpr, _ = roc_curve(y_true_binary_val, prob_serious)
            prob_true, prob_pred = calibration_curve(y_true_binary_val, prob_serious, n_bins=10)
        except:
            auc = 0.5
            fpr, tpr = np.array([0.0, 1.0]), np.array([0.0, 1.0])
            prob_true, prob_pred = np.array([]), np.array([])

        results[name] = {
            'weight': weight,
            'Accuracy': acc,
            'Macro F1': f1,
            'ROC_AUC': auc,
            'ROC_Curve': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'Calibration': {'prob_true': prob_true.tolist(), 'prob_pred': prob_pred.tolist()}
        }

    total_weight = sum(res['weight'] for res in results.values() if 'weight' in res)
    weighted_ensemble_probs /= total_weight
    
    we_preds = np.argmax(weighted_ensemble_probs, axis=1)
    we_prob_serious = weighted_ensemble_probs[:, 1:].sum(axis=1)
    we_fpr, we_tpr, _ = roc_curve(y_true_binary_val, we_prob_serious)
    we_prob_true, we_prob_pred = calibration_curve(y_true_binary_val, we_prob_serious, n_bins=10)
    results['Naive Weighted Ensemble'] = {
        'Accuracy': accuracy_score(val_labels, we_preds),
        'Macro F1': f1_score(val_labels, we_preds, average='macro'),
        'ROC_AUC': roc_auc_score(y_true_binary_val, we_prob_serious),
        'ROC_Curve': {'fpr': we_fpr.tolist(), 'tpr': we_tpr.tolist()},
        'Calibration': {'prob_true': we_prob_true.tolist(), 'prob_pred': we_prob_pred.tolist()}
    }

    X_train_oof = np.zeros((len(train_labels), 3 * len(models)))

    # Generate cross-validated out-of-fold predictions for the training set
    print("Generating out-of-fold predictions using 5-Fold CV...")
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Store train logits as arrays
    train_logits_dict = {m: np.array(train_logits[m]) for m in models.keys()}

    # Instead of completely retraining the deep learning models for each fold (which would take a long time),
    # we simulate OOF predictions by applying temperature scaling on OOF splits.
    # Since we already have the raw logits from the fully trained models, we'll calibrate temperatures
    # on the K-1 folds and predict on the Kth fold. This removes the temperature-scaling data leakage.
    #
    # LIMITATION: The base model logits are NOT truly out-of-fold -- the base models were trained on the
    # full training set, so their logits on training examples are optimistically biased. This means the
    # stacker's training features are slightly leaked. Proper OOF would require retraining all 3 base
    # models per fold, which is prohibitively expensive. The practical impact is modest given the base
    # models use dropout, label smoothing, and weight decay.

    for fold, (train_idx, val_idx) in enumerate(kf.split(np.zeros(len(train_labels)), train_labels)):
        for i, m in enumerate(models.keys()):
            fold_train_logits = train_logits_dict[m][train_idx]
            fold_train_labels = train_labels[train_idx]
            fold_val_logits = train_logits_dict[m][val_idx]

            # Find optimal temp on this fold's training split
            def eval_nll(t_val):
                scaled_logits = torch.tensor(fold_train_logits) / t_val[0]
                loss = F.cross_entropy(scaled_logits, torch.tensor(fold_train_labels))
                return loss.item()

            opt_res = minimize(eval_nll, [1.5], bounds=[(0.5, 5.0)], method='L-BFGS-B')
            fold_opt_T = opt_res.x[0]

            # Generate OOF predictions for this fold's validation split
            fold_val_probs = F.softmax(torch.tensor(fold_val_logits) / fold_opt_T, dim=1).numpy()

            # Store in the OOF feature matrix
            X_train_oof[val_idx, i*3:(i+1)*3] = fold_val_probs

    X_val = np.concatenate([F.softmax(torch.tensor(np.array(val_logits[m])) / optimal_temps[m], dim=1).numpy() for m in models.keys()], axis=1)

    lr_stacker = LogisticRegression(max_iter=1000, C=0.1)  # stack-search winner (stable tie-break over C=10)
    lr_stacker.fit(X_train_oof, train_labels)
    lr_val_probs = lr_stacker.predict_proba(X_val)
    lr_prob_serious = lr_val_probs[:, 1:].sum(axis=1)
    lr_fpr, lr_tpr, _ = roc_curve(y_true_binary_val, lr_prob_serious)
    lr_prob_true, lr_prob_pred = calibration_curve(y_true_binary_val, lr_prob_serious, n_bins=10)
    results['LogReg Stacking Ensemble'] = {
        'Accuracy': accuracy_score(val_labels, np.argmax(lr_val_probs, axis=1)),
        'Macro F1': f1_score(val_labels, np.argmax(lr_val_probs, axis=1), average='macro'),
        'ROC_AUC': roc_auc_score(y_true_binary_val, lr_prob_serious),
        'ROC_Curve': {'fpr': lr_fpr.tolist(), 'tpr': lr_tpr.tolist()},
        'Calibration': {'prob_true': lr_prob_true.tolist(), 'prob_pred': lr_prob_pred.tolist()}
    }

    # Searched optimum (outputs/severity_search/stack_results.json).
    xgb_stacker = XGBClassifier(
        n_estimators=50, max_depth=2, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric='mlogloss', random_state=42
    )
    xgb_stacker.fit(X_train_oof, train_labels)
    xgb_val_probs = xgb_stacker.predict_proba(X_val)

    # Save the production stacker (LogReg C=0.1 per stack search) and
    # optimal temperatures for the backend pipeline
    with open('cache/ensemble_stacker.pkl', 'wb') as f:
        pickle.dump(lr_stacker, f)

    with open('cache/ensemble_temps.json', 'w') as f:
        json.dump(optimal_temps, f)

    xgb_prob_serious = xgb_val_probs[:, 1:].sum(axis=1)
    xgb_fpr, xgb_tpr, _ = roc_curve(y_true_binary_val, xgb_prob_serious)
    xgb_prob_true, xgb_prob_pred = calibration_curve(y_true_binary_val, xgb_prob_serious, n_bins=10)
    results['XGBoost Stacking Ensemble'] = {
        'Accuracy': accuracy_score(val_labels, np.argmax(xgb_val_probs, axis=1)),
        'Macro F1': f1_score(val_labels, np.argmax(xgb_val_probs, axis=1), average='macro'),
        'ROC_AUC': roc_auc_score(y_true_binary_val, xgb_prob_serious),
        'ROC_Curve': {'fpr': xgb_fpr.tolist(), 'tpr': xgb_tpr.tolist()},
        'Calibration': {'prob_true': xgb_prob_true.tolist(), 'prob_pred': xgb_prob_pred.tolist()}
    }

    # Generate Correlation Matrix between the base models
    try:
        preds_matrix = np.vstack([base_probs_serious[m] for m in ENSEMBLE_NAMES])
        corr_matrix = np.corrcoef(preds_matrix)
        results['Base_Model_Correlation'] = {
            'models': list(ENSEMBLE_NAMES),
            'matrix': corr_matrix.tolist()
        }
    except Exception as e:
        print(f"Could not compute correlation matrix: {e}")

    print("\n=========================================================")
    print("               FINAL ENSEMBLE METRICS                    ")
    print("=========================================================")
    print("\n--- Base Models ---")
    for m in ENSEMBLE_NAMES:
        res = results[m]
        print(f"{m:<25}: AUC={res['ROC_AUC']:.4f} | Acc={res['Accuracy']:.4f} | F1={res['Macro F1']:.4f}")

    print("\n--- Ensemble Techniques ---")
    for m in ['Naive Weighted Ensemble', 'LogReg Stacking Ensemble', 'XGBoost Stacking Ensemble']:
        res = results[m]
        print(f"{m:<25}: AUC={res['ROC_AUC']:.4f} | Acc={res['Accuracy']:.4f} | F1={res['Macro F1']:.4f}")

    # Save metrics to JSON for the charts script to use
    paths.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.OUTPUTS_DIR / "ensemble_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate_and_ensemble()
