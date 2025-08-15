"""Unified advanced anomaly detector (src version) aligned with root package.

Adds extended algorithms, diagnostics, feature-dimension robustness.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

try:
    from statsmodels.tsa.seasonal import STL  # type: ignore
    STL_AVAILABLE = True
except Exception:
    STL_AVAILABLE = False

try:
    from prophet import Prophet  # type: ignore
    PROPHET_AVAILABLE = True
except Exception:
    try:
        from fbprophet import Prophet  # type: ignore
        PROPHET_AVAILABLE = True
    except Exception:
        PROPHET_AVAILABLE = False

from ..utils.logger import LoggerMixin, log_function_call
from ..core.config import TelecomConfig


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU for computations")
    return device


@dataclass
class AnomalyResult:
    timestamp: str
    site_id: str
    sector_id: Optional[str]
    kpi_name: str
    value: float
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    method: str
    severity: str
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super().__init__()
        self.device = get_device()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        z = self.encoder(x)
        return self.decoder(z)


class LSTMAutoEncoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int = 1, latent_dim: int = 32):
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.device = get_device()
        self.encoder = nn.LSTM(n_features, latent_dim, batch_first=True)
        self.decoder = nn.LSTM(latent_dim, latent_dim, batch_first=True)
        self.output_layer = nn.Linear(latent_dim, n_features)
        self.is_lstm = True
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        _, (h, _) = self.encoder(x)
        repeat_latent = torch.repeat_interleave(h.transpose(0, 1), repeats=self.seq_len, dim=1)
        dec_out, _ = self.decoder(repeat_latent)
        return self.output_layer(dec_out)


class KPISpecificDetector(LoggerMixin):
    KPI_ALGORITHM_MAP: Dict[str, str] = {
        "RSRP": "isolation_forest",
        "SINR": "local_outlier_factor",
        "DL_Throughput": "autoencoder",
        "UL_Throughput": "autoencoder",
        "Call_Drop_Rate": "ensemble_if_gmm",
        "RTT": "isolation_forest",
        "CPU_Utilization": "time_series_decomposition",
        "Active_Users": "seasonal_hybrid_esd",
        "Packet_Loss": "one_class_svm",
        "Handover_Success_Rate": "gaussian_mixture",
        "Handover_Success": "gaussian_mixture",
    }

    def __init__(self, kpi_name: str, config: TelecomConfig):
        self.kpi_name = kpi_name
        self.config = config
        self.device = get_device()
        self.algorithm = self.KPI_ALGORITHM_MAP.get(kpi_name, "isolation_forest")
        self.model: Any = None
        self.scaler = StandardScaler()
        self.threshold: Optional[float] = None
        self.is_fitted = False
        self.params: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.logger.info(f"Initialized {self.algorithm} detector for {self.kpi_name} on {self.device}")

    def _create_model(self):
        if self.algorithm == "isolation_forest":
            return IsolationForest(**self.config.model.isolation_forest_params)
        if self.algorithm == "one_class_svm":
            return OneClassSVM(**self.config.model.one_class_svm_params)
        if self.algorithm == "gaussian_mixture":
            return GaussianMixture(n_components=2, random_state=42)
        if self.algorithm == "local_outlier_factor":
            return LocalOutlierFactor(contamination=self.config.model.contamination_rate)
        if self.algorithm == "autoencoder":
            return None
        if self.algorithm in ["ensemble_if_gmm", "time_series_decomposition", "seasonal_hybrid_esd"]:
            return None
        raise ValueError(f"Unknown algorithm: {self.algorithm}")

    @log_function_call
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "KPISpecificDetector":
        self.feature_names = [f"f{i}" for i in range(X.shape[1])]
        if self.algorithm in ["time_series_decomposition", "seasonal_hybrid_esd"]:
            X_proc = X.astype(float)
        else:
            X_proc = self.scaler.fit_transform(X)

        if self.algorithm == "autoencoder":
            seq_len = int(getattr(self.config.model, "sequence_length", 7) or 7)
            if self.kpi_name in ["DL_Throughput", "UL_Throughput"] and seq_len > 1:
                self._fit_lstm_autoencoder(X_proc, seq_len)
            else:
                self._fit_autoencoder(X_proc)
            self._calculate_threshold(X_proc)
        elif self.algorithm == "ensemble_if_gmm":
            if_model = IsolationForest(**self.config.model.isolation_forest_params).fit(X_proc)
            gmm_model = GaussianMixture(n_components=2, random_state=42).fit(X_proc)
            self.model = {"if": if_model, "gmm": gmm_model}
            self.is_fitted = True
            scores = self._get_anomaly_scores(X_proc)
            self.threshold = float(np.percentile(scores, (1 - self.config.model.contamination_rate) * 100))
        elif self.algorithm == "time_series_decomposition":
            series = X_proc[:, 0]
            stl_res = None
            if STL_AVAILABLE and len(series) >= 14:
                try:
                    stl_res = STL(series, period=7, robust=True).fit()
                    resid = stl_res.resid
                    seasonal = stl_res.seasonal
                except Exception:
                    stl_res = None
            if stl_res is None:
                trend = pd.Series(series).rolling(window=min(7, max(3, len(series)//10)), min_periods=1, center=True).mean().to_numpy()
                resid = series - trend
                seasonal = np.zeros_like(series)
            model_dict: Dict[str, Any] = {
                "res_mean": float(np.nanmean(resid)),
                "res_std": float(np.nanstd(resid) + 1e-8),
                "seasonal_sample": seasonal[-5000:] if len(seasonal) > 5000 else seasonal,
                "use_stl": bool(stl_res is not None),
            }
            if PROPHET_AVAILABLE and len(series) >= 30:
                try:
                    dfp = pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=len(series), freq="D"), "y": series})
                    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
                    m.fit(dfp)
                    fc = m.predict(dfp)
                    resid_p = series - fc["yhat"].values
                    model_dict["prophet_res_mean"] = float(np.nanmean(resid_p))
                    model_dict["prophet_res_std"] = float(np.nanstd(resid_p) + 1e-8)
                    model_dict["prophet_used"] = True
                except Exception:
                    model_dict["prophet_used"] = False
            else:
                model_dict["prophet_used"] = False
            self.model = model_dict
            self.threshold = 3.0
            self.is_fitted = True
        elif self.algorithm == "seasonal_hybrid_esd":
            series = X_proc[:, 0]
            period = max(3, min(7, len(series)//12 or 7))
            seasonal = pd.Series(series).rolling(window=period, min_periods=1, center=True).median().to_numpy()
            resid = series - seasonal
            res_mean = float(np.nanmean(resid))
            res_std = float(np.nanstd(resid) + 1e-8)
            z_scores = np.abs((resid - res_mean) / (res_std + 1e-8))
            self.model = {"seasonal_sample": seasonal[-5000:] if len(seasonal) > 5000 else seasonal, "res_mean": res_mean, "res_std": res_std, "period": period}
            try:
                self.threshold = float(np.percentile(z_scores, (1 - self.config.model.contamination_rate) * 100))
            except Exception:
                self.threshold = 3.0
            self.is_fitted = True
        else:
            self.model = self._create_model()
            if self.model is not None:
                self.model.fit(X_proc)
                self.is_fitted = True
                self._calculate_threshold(X_proc)
        self.is_fitted = True
        return self

    def _fit_autoencoder(self, X: np.ndarray):
        input_dim = X.shape[1]
        enc_dim = min(self.config.model.autoencoder_params["encoding_dim"], max(4, input_dim // 2))
        self.model = AutoEncoder(input_dim, enc_dim)
        crit = nn.MSELoss()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.model.autoencoder_params["learning_rate"])
        X_t = torch.FloatTensor(X).to(self.model.device)
        epochs = self.config.model.autoencoder_params["epochs"]
        self.model.train()
        for ep in range(epochs):
            opt.zero_grad(); rec = self.model(X_t); loss = crit(rec, X_t); loss.backward(); opt.step()
        self.is_fitted = True

    def _fit_lstm_autoencoder(self, X: np.ndarray, seq_len: int = 7):
        values = X[:, 0]
        if len(values) <= seq_len:
            self._fit_autoencoder(X); return
        windows = np.array([values[i:i+seq_len] for i in range(len(values) - seq_len + 1)])
        windows = windows.reshape(-1, seq_len, 1)
        model = LSTMAutoEncoder(seq_len=seq_len, n_features=1, latent_dim=min(64, max(8, seq_len * 2)))
        crit = nn.MSELoss(); opt = torch.optim.Adam(model.parameters(), lr=self.config.model.autoencoder_params["learning_rate"])
        X_t = torch.FloatTensor(windows).to(model.device)
        model.train()
        for ep in range(self.config.model.autoencoder_params["epochs"]):
            opt.zero_grad(); rec = model(X_t); loss = crit(rec, X_t); loss.backward(); opt.step()
        self.model = model; self.is_fitted = True

    def _calculate_threshold(self, X: np.ndarray):
        scores = self._get_anomaly_scores(X)
        if self.algorithm in ["isolation_forest", "one_class_svm"]:
            self.threshold = float(np.percentile(scores, (1 - self.config.model.contamination_rate) * 100))
        else:
            self.threshold = float(np.mean(scores) + 2 * np.std(scores))

    def _get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        if self.algorithm == "autoencoder":
            if hasattr(self.model, "is_lstm") and getattr(self.model, "is_lstm"):
                seq_len = self.model.seq_len; vals = X[:, 0]
                if len(vals) <= seq_len:
                    vals = np.pad(vals, (seq_len - len(vals) + 1, 0), mode="edge")
                wins = np.array([vals[i:i+seq_len] for i in range(len(vals) - seq_len + 1)]).reshape(-1, seq_len, 1)
                self.model.eval();
                with torch.no_grad():
                    t = torch.FloatTensor(wins).to(self.model.device); rec = self.model(t); w_scores = torch.mean((t - rec) ** 2, dim=(1,2)).cpu().numpy()
                scores = np.zeros(len(vals));
                for i, s in enumerate(w_scores): scores[i + seq_len - 1] = s
                return scores[-len(X):]
            else:
                self.model.eval();
                with torch.no_grad():
                    t = torch.FloatTensor(X).to(self.model.device); rec = self.model(t); return torch.mean((t - rec) ** 2, dim=1).cpu().numpy()
        if self.algorithm == "isolation_forest": return -self.model.score_samples(X)
        if self.algorithm == "one_class_svm": return -self.model.score_samples(X)
        if self.algorithm == "gaussian_mixture": return -self.model.score_samples(X)
        if self.algorithm == "local_outlier_factor": return -self.model.negative_outlier_factor_
        if self.algorithm == "ensemble_if_gmm":
            if_scores = -self.model["if"].score_samples(X); gmm_scores = -self.model["gmm"].score_samples(X)
            def _n(s): r = np.max(s) - np.min(s) + 1e-8; return (s - np.min(s))/r
            return 0.5*_n(if_scores)+0.5*_n(gmm_scores)
        if self.algorithm == "time_series_decomposition":
            series = X[:,0]; trend = pd.Series(series).rolling(window=7, min_periods=1, center=True).mean().to_numpy(); resid = series - trend
            mean_r = self.model.get("prophet_res_mean", self.model["res_mean"]); std_r = self.model.get("prophet_res_std", self.model["res_std"])
            return np.abs(resid - mean_r)/(std_r + 1e-8)
        if self.algorithm == "seasonal_hybrid_esd":
            series = X[:,0]; period = self.model["period"]; seasonal = pd.Series(series).rolling(window=period, min_periods=1, center=True).median().to_numpy(); resid = series - seasonal
            return np.abs(resid - self.model["res_mean"])/(self.model["res_std"] + 1e-8)
        raise ValueError(f"Unknown algorithm: {self.algorithm}")

    def get_diagnostics(self, X_raw: np.ndarray) -> Dict[str, Any]:
        if not self.is_fitted: return {"error": "not_fitted"}
        try:
            if self.algorithm in ["time_series_decomposition", "seasonal_hybrid_esd"]:
                X_proc = X_raw.astype(float)
            else:
                expected_dim = int(getattr(self.scaler, 'mean_', np.array([0])).shape[0])
                X_in = X_raw
                if X_in.shape[1] != expected_dim and expected_dim > 0:
                    if X_in.shape[1] == 1:
                        X_in = np.repeat(X_in, expected_dim, axis=1)
                    elif X_in.shape[1] > expected_dim:
                        X_in = X_in[:, :expected_dim]
                    else:
                        X_in = np.hstack([X_in, np.zeros((X_in.shape[0], expected_dim - X_in.shape[1]))])
                X_proc = self.scaler.transform(X_in)
            scores = self._get_anomaly_scores(X_proc)
            thr = self.threshold if self.threshold is not None else np.inf
            return {
                "algorithm": self.algorithm,
                "threshold": float(self.threshold) if self.threshold is not None else None,
                "score_min": float(np.min(scores)),
                "score_median": float(np.median(scores)),
                "score_mean": float(np.mean(scores)),
                "score_max": float(np.max(scores)),
                "num_flagged": int(np.sum(scores > thr)),
                "total": int(len(scores)),
            }
        except Exception as e:
            return {"error": str(e)}

    @log_function_call
    def predict(self, X: np.ndarray) -> List[AnomalyResult]:
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        X_in = X
        if self.algorithm not in ["time_series_decomposition", "seasonal_hybrid_esd"]:
            try:
                expected_dim = int(getattr(self.scaler, 'mean_', np.array([0])).shape[0])
                if X_in.shape[1] != expected_dim:
                    if X_in.shape[1] == 1 and expected_dim > 1:
                        X_in = np.repeat(X_in, expected_dim, axis=1)
                    elif X_in.shape[1] > expected_dim:
                        X_in = X_in[:, :expected_dim]
                    else:
                        X_in = np.hstack([X_in, np.zeros((X_in.shape[0], expected_dim - X_in.shape[1]))])
            except Exception:
                pass
        if self.algorithm in ["time_series_decomposition", "seasonal_hybrid_esd"]:
            X_proc = X_in.astype(float)
        else:
            try:
                X_proc = self.scaler.transform(X_in)
            except Exception:
                self.scaler.fit(X_in); X_proc = self.scaler.transform(X_in)
        scores = self._get_anomaly_scores(X_proc)
        is_anom = scores > self.threshold
        out: List[AnomalyResult] = []
        for i,(sc,an) in enumerate(zip(scores, is_anom)):
            conf = float(min(abs(sc - self.threshold)/(self.threshold + 1e-8),1.0))
            if an:
                sev = "high" if sc > self.threshold*2 else ("medium" if sc > self.threshold*1.5 else "low")
            else:
                sev = "normal"
            out.append(AnomalyResult(timestamp=f"sample_{i}", site_id="unknown", sector_id=None, kpi_name=self.kpi_name, value=float(X[i,0]) if X.shape[1] else 0.0, is_anomaly=bool(an), anomaly_score=float(sc), confidence=conf, method=self.algorithm, severity=sev, threshold=float(self.threshold)))
        return out

    def save_model(self, filepath: str):
        if not self.is_fitted: raise ValueError("Cannot save unfitted model")
        data: Dict[str, Any] = {"kpi_name": self.kpi_name, "algorithm": self.algorithm, "threshold": self.threshold, "scaler": self.scaler, "params": self.params, "feature_names": self.feature_names}
        if self.algorithm == "autoencoder":
            if hasattr(self.model, "is_lstm") and getattr(self.model, "is_lstm"):
                data.update({"model_state": self.model.state_dict(), "lstm": True, "seq_len": int(self.model.seq_len), "latent_dim": int(self.model.latent_dim)})
            else:
                data.update({"model_state": self.model.state_dict(), "input_dim": int(self.model.encoder[0].in_features), "encoding_dim": int(self.model.encoder[2].out_features)})
        else:
            data["model"] = self.model
        with open(filepath, "wb") as f: pickle.dump(data, f)

    def load_model(self, filepath: str):
        with open(filepath, "rb") as f: data = pickle.load(f)
        self.kpi_name = data["kpi_name"]; self.algorithm = data["algorithm"]; self.threshold = data["threshold"]; self.scaler = data["scaler"]; self.params = data.get("params", {}); self.feature_names = data.get("feature_names", [])
        if self.algorithm == "autoencoder":
            state = data.get("model_state")
            if data.get("lstm"):
                model = LSTMAutoEncoder(seq_len=int(data.get("seq_len",7)), n_features=1, latent_dim=int(data.get("latent_dim",32)))
                model.load_state_dict(state); model.to(self.device); model.eval(); self.model = model
            else:
                in_dim = int(data.get("input_dim")); enc_dim = int(data.get("encoding_dim")); model = AutoEncoder(in_dim, enc_dim); model.load_state_dict(state); model.to(self.device); model.eval(); self.model = model
        else:
            self.model = data["model"]
        self.is_fitted = True


class KPIAnomalyDetector(LoggerMixin):
    def __init__(self, config: TelecomConfig):
        self.config = config
        self.device = get_device()
        self.detectors: Dict[str, KPISpecificDetector] = {}
        self.is_fitted = False

    @log_function_call
    def fit(self, data: pd.DataFrame) -> "KPIAnomalyDetector":
        available = [k for k in self.config.data.kpi_columns if k in data.columns]
        for kpi in available:
            feat_cols = [c for c in data.columns if c == kpi or (kpi in c and data[c].dtype in ["int64","float64"])] or [kpi]
            X = data[feat_cols].values
            det = KPISpecificDetector(kpi, self.config).fit(X)
            self.detectors[kpi] = det
        self.is_fitted = True
        return self

    @log_function_call
    def detect_anomalies(self, data: pd.DataFrame, kpi_name: Optional[str] = None, site_id: Optional[str] = None, date_range: Optional[Tuple[str, str]] = None) -> List[AnomalyResult]:
        if not self.is_fitted: raise ValueError("Detectors not fitted yet")
        df = data.copy()
        if site_id and "Site_ID" in df.columns: df = df[df["Site_ID"] == site_id]
        if date_range and "Date" in df.columns:
            start,end = date_range; df = df[(df["Date"]>=start) & (df["Date"]<=end)]
        target = [kpi_name] if kpi_name else list(self.detectors.keys())
        target = [k for k in target if k in df.columns]
        all_results: List[AnomalyResult] = []
        for kpi in target:
            det = self.detectors[kpi]
            feat_cols = [c for c in df.columns if c == kpi or (kpi in c and df[c].dtype in ["int64","float64"])] or [kpi]
            X = df[feat_cols].fillna(0.0).values
            if len(X)==0: continue
            res_list = det.predict(X)
            for i,res in enumerate(res_list):
                row = df.iloc[i]
                res.timestamp = str(row.get("Date", f"row_{i}")); res.site_id = str(row.get("Site_ID","unknown")); res.sector_id = str(row.get("Sector_ID", None)) if "Sector_ID" in row else None; res.value = float(row[kpi])
            all_results.extend(res_list)
        all_results.sort(key=lambda r: r.anomaly_score, reverse=True)
        return all_results

    def get_model_summary(self) -> Dict[str, Any]:
        if not self.is_fitted: return {"error": "Models not fitted yet"}
        return {k: {"algorithm": d.algorithm, "threshold": float(d.threshold), "is_fitted": d.is_fitted} for k,d in self.detectors.items()}

    def save_all_models(self, models_dir: Optional[str] = None):
        if not self.is_fitted: raise ValueError("Cannot save unfitted models")
        models_dir = Path(models_dir or self.config.models_dir); models_dir.mkdir(parents=True, exist_ok=True)
        for k,det in self.detectors.items(): det.save_model(str(models_dir / f"{k}_detector.pkl"))
        with open(models_dir / "detectors_config.json", "w") as f: json.dump(self.get_model_summary(), f, indent=2)

    def load_all_models(self, models_dir: Optional[str] = None):
        models_dir = Path(models_dir or self.config.models_dir); loaded = 0
        for k in self.config.data.kpi_columns:
            fp = models_dir / f"{k}_detector.pkl"
            if fp.exists():
                try:
                    det = KPISpecificDetector(k, self.config); det.load_model(str(fp)); self.detectors[k] = det; loaded += 1
                except Exception:
                    pass
        self.is_fitted = loaded > 0
