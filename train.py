import mlflow
import mlflow.sklearn
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

class CreditModelTrainer:
    def __init__(self, experiment_name: str = "Credit Score Classification",
                 cv_splits: int = 5):
        self.experiment_name = experiment_name
        self.random_state = 42

        self.cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                                  random_state=42)
        self.scoring = {
            'f1_macro': make_scorer(f1_score, average='macro'),
            'recall_poor': make_scorer(recall_score, labels=[0],
                                       average='macro', zero_division=0),
            'precision_good': make_scorer(precision_score, labels=[2],
                                          average='macro', zero_division=0),
        }
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.set_experiment(self.experiment_name)

    def model_configs(self):
        return {
            'RandomForest': {
                'estimator': RandomForestClassifier(
                    n_estimators=150, max_depth=None, min_samples_split=5,
                    min_samples_leaf=2, max_features='sqrt',
                    class_weight='balanced', random_state=self.random_state),
                'params': {'n_estimators': 150, 'max_depth': None,
                           'min_samples_split': 5, 'min_samples_leaf': 2,
                           'max_features': 'sqrt'},
                'balancing': 'class_weight=balanced',
            },
            'XGBoost': {
                'estimator': XGBClassifier(
                    n_estimators=150, max_depth=7, learning_rate=0.1,
                    subsample=1.0, colsample_bytree=0.8,
                    eval_metric='mlogloss', random_state=self.random_state),
                'params': {'n_estimators': 150, 'max_depth': 7,
                           'learning_rate': 0.1, 'subsample': 1.0,
                           'colsample_bytree': 0.8},
                'balancing': 'sample_weight=balanced',
            },
            'CatBoost': {
                'estimator': CatBoostClassifier(
                    iterations=80, depth=8, learning_rate=0.1, l2_leaf_reg=3,
                    auto_class_weights='Balanced', random_state=self.random_state,
                    verbose=0),
                'params': {'iterations': 80, 'depth': 8,
                           'learning_rate': 0.1, 'l2_leaf_reg': 3},
                'balancing': 'auto_class_weights=Balanced',
            },
        }

    def run(self, x_train, y_train, transformer):
        print("--- Step 3: Training + Cross-Validation (3 models) ---")
        results = {}

        for name, cfg in self.model_configs().items():
            # clone transformer agar tiap model punya preprocessing terpisah
            pipe = Pipeline([
                ('preprocessing', clone(transformer)),
                ('classifier', cfg['estimator']),
            ])

            with mlflow.start_run(run_name=name) as run:
                mlflow.log_param('model', name)
                mlflow.log_param('balancing', cfg['balancing'])
                mlflow.log_param('cv_folds', self.cv.get_n_splits())
                for k, v in cfg['params'].items():
                    mlflow.log_param(k, v)

                cv_res = cross_validate(pipe, x_train, y_train, cv=self.cv,
                                        scoring=self.scoring, n_jobs=-1,
                                        error_score='raise')
                cv_f1 = float(cv_res['test_f1_macro'].mean())
                cv_rec = float(cv_res['test_recall_poor'].mean())
                cv_prec = float(cv_res['test_precision_good'].mean())

                mlflow.log_metric('cv_f1_macro', cv_f1)
                mlflow.log_metric('cv_recall_poor', cv_rec)
                mlflow.log_metric('cv_precision_good', cv_prec)

                if name == 'XGBoost':
                    sw = compute_sample_weight('balanced', y_train)
                    pipe.fit(x_train, y_train, classifier__sample_weight=sw)
                else:
                    pipe.fit(x_train, y_train)

                mlflow.sklearn.log_model(pipe, name="model")

                results[name] = {'run_id': run.info.run_id, 'cv_f1_macro': cv_f1,
                                 'cv_recall_poor': cv_rec, 'cv_precision_good': cv_prec}
                print(f"  ✅ [{name}] CV f1_macro={cv_f1:.4f} | "
                      f"logged to MLflow (run_id={run.info.run_id[:8]}…)")

        return results
