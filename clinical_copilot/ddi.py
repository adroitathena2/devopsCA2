"""
Drug-drug interaction detection using the DrugBank database, Gemini, and custom severity models.
"""

import difflib
import json
import logging
import os
import pickle
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clinical_copilot import severity
from clinical_copilot.config import ConfigLoader

try:
    import torch
    from clinical_copilot.custom_severity_model import SeverityVocab, build_ensemble_model, ENSEMBLE_NAMES
    CUSTOM_MODEL_AVAILABLE = True
except ImportError:
    CUSTOM_MODEL_AVAILABLE = False

logger = logging.getLogger(__name__)


def _extract_retry_delay_seconds(error_text: str, default_seconds: float = 2.0) -> float:
    """Extract retry delay from Gemini error text when present."""
    if not error_text:
        return default_seconds

    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]([0-9]+(?:\.[0-9]+)?)s['\"]", error_text)
    if match:
        return max(float(match.group(1)), 0.2)

    match = re.search(r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s", error_text, flags=re.IGNORECASE)
    if match:
        return max(float(match.group(1)), 0.2)

    return default_seconds


def _is_gemini_quota_exhausted(error_text: str) -> bool:
    if not error_text:
        return False
    upper = error_text.upper()
    return "RESOURCE_EXHAUSTED" in upper or "QUOTA EXCEEDED" in upper


def _is_gemini_zero_quota(error_text: str) -> bool:
    if not error_text:
        return False
    return "limit: 0" in error_text.lower()


class InteractionDetector:
    """Detects drug-drug interactions using DrugBank database."""
    
    def __init__(self, drugbank_json: Dict[str, Any] = None,
                 prebuilt_drug_index: Optional[Dict[str, Any]] = None,
                 config: Optional[ConfigLoader] = None):
        """
        Initialize interaction detector.
        
        Args:
            drugbank_json: Loaded DrugBank JSON data (ignored if prebuilt_drug_index given)
            prebuilt_drug_index: Optional pre-built drug index dict (from cache)
        """
        self.drugbank_json = drugbank_json or {}
        self.config = config
        if prebuilt_drug_index is not None:
            logger.info("Using pre-built drug index from cache.")
            self.drug_index = prebuilt_drug_index
        else:
            self.drug_index = self._build_drug_index()

        # Initialize Gemini client (optional)
        self.gemini_client = None
        self.gemini_enabled = False
        self.gemini_model_severity = "gemini-3.8-flash"
        self.gemini_model_reasoning = "gemini-3.8-flash"
        self.gemini_temperature_severity = 0.0
        self.gemini_top_p_severity = 0.9
        self.gemini_temperature_reasoning = 0.2
        self.gemini_top_p_reasoning = 0.9
        self.gemini_thinking_severity = "medium"
        self.gemini_thinking_reasoning = "medium"
        self.gemini_reasoning_enabled = True
        self.gemini_min_request_interval_seconds = 1.0
        self.gemini_max_retries = 3
        self.gemini_backoff_base_seconds = 1.0
        self.gemini_backoff_max_seconds = 30.0
        self.gemini_last_request_epoch = 0.0
        self.gemini_retry_after_epoch = 0.0
        self.gemini_session_quota_blocked = False
        # Per-context API usage: {context: {calls, errors, input_tokens,
        # output_tokens, latency_seconds}}. Powers cost reporting.
        self.gemini_usage = {}
        self.gemini_price_input_per_1m = 0.0
        self.gemini_price_output_per_1m = 0.0

        self.custom_model = None
        self.custom_vocab = None
        self.custom_model_device = None
        # Codebook policy gates (physician-confirmed; tuned on
        # severity_val_hq, never test).
        self.SEVERE_MIN_PROB = 0.50
        self.SEVERE_MARGIN = 0.05
        # Severity gate ablation flag: 'learned' (production default;
        # Severe-vs-rest LogisticRegression gate, outputs/learned_gate/),
        # 'regex' (legacy deterministic rule), or 'none' (raw stacker
        # argmax). Env override SEVERITY_GATE wins over config when set.
        self.SEVERITY_GATE = os.environ.get("SEVERITY_GATE", "").lower() or None
        self._learned_gate = None
        self._learned_gate_threshold = None
        self._learned_gate_failed = False

        custom_config = self.config.config.get('models', {}).get('custom_severity', {}) if self.config else {}
        if self.SEVERITY_GATE is None:
            self.SEVERITY_GATE = str(custom_config.get('severity_gate', 'learned')).lower()
        if self.SEVERITY_GATE not in ('none', 'regex', 'learned'):
            logger.warning(f"Unknown SEVERITY_GATE={self.SEVERITY_GATE!r}; falling back to 'regex'.")
            self.SEVERITY_GATE = 'regex'
        if custom_config.get('enabled', False) and CUSTOM_MODEL_AVAILABLE:
            self._load_custom_severity_model(custom_config)

        self._load_gemini_config()
        try:
            from google import genai
            api_key = os.environ.get("GEMINI_API_KEY")
            if api_key and self.gemini_enabled:
                self.gemini_client = genai.Client(api_key=api_key)
                logger.info("Gemini API client initialized for DDI severity and reasoning")
        except Exception as e:
            logger.warning(f"Gemini API client initialization failed: {e}")

    def _load_gemini_config(self) -> None:
        """Load Gemini behavior from config and optional tuning file."""
        if not self.config:
            return

        self.gemini_enabled = bool(self.config.get('gemini', 'enabled', default=False))
        self.gemini_model_severity = self.config.get(
            'gemini', 'severity_model', default=self.gemini_model_severity
        )
        self.gemini_model_reasoning = self.config.get(
            'gemini', 'reasoning_model', default=self.gemini_model_reasoning
        )
        self.gemini_temperature_severity = float(self.config.get(
            'gemini', 'severity_temperature', default=self.gemini_temperature_severity
        ))
        self.gemini_top_p_severity = float(self.config.get(
            'gemini', 'severity_top_p', default=self.gemini_top_p_severity
        ))
        self.gemini_temperature_reasoning = float(self.config.get(
            'gemini', 'reasoning_temperature', default=self.gemini_temperature_reasoning
        ))
        self.gemini_top_p_reasoning = float(self.config.get(
            'gemini', 'reasoning_top_p', default=self.gemini_top_p_reasoning
        ))
        self.gemini_thinking_severity = str(self.config.get(
            'gemini', 'severity_thinking_level', default='minimal'
        ))
        self.gemini_thinking_reasoning = str(self.config.get(
            'gemini', 'reasoning_thinking_level', default='medium'
        ))
        self.gemini_reasoning_enabled = bool(self.config.get(
            'gemini', 'reasoning_enabled', default=True
        ))
        self.gemini_min_request_interval_seconds = float(self.config.get(
            'gemini', 'min_request_interval_seconds', default=self.gemini_min_request_interval_seconds
        ))
        self.gemini_max_retries = int(self.config.get(
            'gemini', 'max_retries', default=self.gemini_max_retries
        ))
        self.gemini_backoff_base_seconds = float(self.config.get(
            'gemini', 'backoff_base_seconds', default=self.gemini_backoff_base_seconds
        ))
        self.gemini_backoff_max_seconds = float(self.config.get(
            'gemini', 'backoff_max_seconds', default=self.gemini_backoff_max_seconds
        ))
        self.gemini_price_input_per_1m = float(self.config.get(
            'gemini', 'pricing_input_per_1m_usd', default=0.25
        ))
        self.gemini_price_output_per_1m = float(self.config.get(
            'gemini', 'pricing_output_per_1m_usd', default=1.50
        ))

        tuning_config_path = self.config.get('gemini', 'tuning_config_path', default='')
        if tuning_config_path and Path(tuning_config_path).exists():
            try:
                with open(tuning_config_path, 'r', encoding='utf-8') as f:
                    tuned = json.load(f)
                self.update_gemini_config(tuned)
                logger.info(f"Loaded tuned Gemini config from {tuning_config_path}")
            except Exception as e:
                logger.warning(f"Failed to load Gemini tuning config from {tuning_config_path}: {e}")

    def update_gemini_config(self, params: Dict[str, Any]) -> None:
        """Update Gemini runtime parameters (used by tuning/evaluation scripts)."""
        self.gemini_enabled = bool(params.get('enabled', self.gemini_enabled))
        self.gemini_model_severity = params.get('severity_model', self.gemini_model_severity)
        self.gemini_model_reasoning = params.get('reasoning_model', self.gemini_model_reasoning)
        self.gemini_temperature_severity = float(
            params.get('severity_temperature', self.gemini_temperature_severity)
        )
        self.gemini_top_p_severity = float(
            params.get('severity_top_p', self.gemini_top_p_severity)
        )
        self.gemini_temperature_reasoning = float(
            params.get('reasoning_temperature', self.gemini_temperature_reasoning)
        )
        self.gemini_top_p_reasoning = float(
            params.get('reasoning_top_p', self.gemini_top_p_reasoning)
        )

    def _sleep_for_burst_control(self) -> None:
        """Ensure a minimum interval between Gemini calls to avoid bursts."""
        now = time.time()

        # Honor API-directed cooldown first.
        if now < self.gemini_retry_after_epoch:
            time.sleep(self.gemini_retry_after_epoch - now)
            now = time.time()

        elapsed = now - self.gemini_last_request_epoch
        min_interval = max(self.gemini_min_request_interval_seconds, 0.0)
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    def _compute_backoff_delay(self, attempt_index: int) -> float:
        """Exponential backoff with jitter to desynchronize retries."""
        base = max(self.gemini_backoff_base_seconds, 0.1)
        cap = max(self.gemini_backoff_max_seconds, base)
        delay = min(cap, base * (2 ** attempt_index))
        jitter = random.uniform(0.0, delay * 0.25)
        return delay + jitter

    def _call_gemini_with_retry(self, model: str, contents: Any, config: Dict[str, Any], context: str) -> Optional[str]:
        """Call Gemini with pacing + exponential backoff retry to reduce request bursts."""
        if not self.gemini_client or not self.gemini_enabled or self.gemini_session_quota_blocked:
            return None

        retries = max(self.gemini_max_retries, 0)
        for attempt in range(retries + 1):
            self._sleep_for_burst_control()
            t0 = time.time()
            try:
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                self.gemini_last_request_epoch = time.time()
                self._record_gemini_usage(context, response, time.time() - t0, error=False)
                return (response.text or '').strip() if response else None
            except Exception as e:
                self.gemini_last_request_epoch = time.time()
                error_text = str(e)

                if _is_gemini_quota_exhausted(error_text):
                    retry_delay = _extract_retry_delay_seconds(
                        error_text,
                        default_seconds=self._compute_backoff_delay(attempt),
                    )
                    self.gemini_retry_after_epoch = max(
                        self.gemini_retry_after_epoch,
                        time.time() + retry_delay,
                    )

                    if _is_gemini_zero_quota(error_text):
                        self.gemini_session_quota_blocked = True
                        logger.warning(
                            "Gemini %s disabled for this app session because project quota is 0. "
                            "Enable quota/billing and restart app.",
                            context,
                        )
                        return None

                    if attempt < retries:
                        logger.warning(
                            "Gemini %s rate-limited. Retrying (%d/%d) in %.1fs.",
                            context,
                            attempt + 1,
                            retries,
                            retry_delay,
                        )
                        continue

                    logger.warning(
                        "Gemini %s failed after retries due to quota/rate limit: %s",
                        context,
                        e,
                    )
                    self._record_gemini_usage(context, None, 0.0, error=True)
                    return None

                if attempt < retries:
                    delay = self._compute_backoff_delay(attempt)
                    logger.warning(
                        "Gemini %s transient error. Retrying (%d/%d) in %.1fs. Error: %s",
                        context,
                        attempt + 1,
                        retries,
                        delay,
                        e,
                    )
                    time.sleep(delay)
                    continue

                logger.warning(f"Gemini {context} failed: {e}")
                self._record_gemini_usage(context, None, 0.0, error=True)
                return None

        self._record_gemini_usage(context, None, 0.0, error=True)
        return None

    def _record_gemini_usage(self, context: str, response: Any, latency_seconds: float, error: bool = False) -> None:
        """Accumulate per-context API usage (calls, tokens, latency)."""
        entry = self.gemini_usage.setdefault(context, {
            "calls": 0, "errors": 0, "input_tokens": 0,
            "output_tokens": 0, "thinking_tokens": 0,
            "latency_seconds": 0.0,
        })
        if error:
            entry["errors"] += 1
            return
        entry["calls"] += 1
        entry["latency_seconds"] += latency_seconds
        try:
            usage = getattr(response, "usage_metadata", None)
            entry["input_tokens"] += int(getattr(usage, "prompt_token_count", 0) or 0)
            # Thinking tokens bill at output rates: candidates + thoughts.
            entry["output_tokens"] += int(getattr(usage, "candidates_token_count", 0) or 0)
            entry["thinking_tokens"] += int(getattr(usage, "thoughts_token_count", 0) or 0)
        except Exception:
            pass

    def get_gemini_stats(self) -> Dict[str, Any]:
        """Per-context and total Gemini usage plus estimated cost in USD."""
        contexts = {}
        total_calls = total_errors = total_in = total_out = total_think = 0
        total_latency = 0.0
        for context, u in self.gemini_usage.items():
            calls = u["calls"]
            avg_latency = u["latency_seconds"] / calls if calls else 0.0
            contexts[context] = {
                "calls": calls,
                "errors": u["errors"],
                "input_tokens": u["input_tokens"],
                "output_tokens": u["output_tokens"],
                "thinking_tokens": u.get("thinking_tokens", 0),
                "latency_seconds_total": round(u["latency_seconds"], 2),
                "latency_seconds_avg": round(avg_latency, 2),
            }
            total_calls += calls
            total_errors += u["errors"]
            total_in += u["input_tokens"]
            total_out += u["output_tokens"]
            total_think += u.get("thinking_tokens", 0)
            total_latency += u["latency_seconds"]
        est_cost = total_in / 1e6 * self.gemini_price_input_per_1m + (total_out + total_think) / 1e6 * self.gemini_price_output_per_1m
        return {
            "models": {
                "severity": self.gemini_model_severity,
                "reasoning": self.gemini_model_reasoning,
            },
            "pricing_per_1m_usd": {
                "input": self.gemini_price_input_per_1m,
                "output": self.gemini_price_output_per_1m,
            },
            "contexts": contexts,
            "total_calls": total_calls,
            "total_errors": total_errors,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_thinking_tokens": total_think,
            "total_latency_seconds": round(total_latency, 2),
            "est_cost_usd": round(est_cost, 6),
        }

    @staticmethod
    def _normalize_gemini_severity(text: str) -> Optional[str]:
        """Map Gemini output into severity classes or return None."""
        return severity.normalize(text, strict=True)


    def _load_custom_severity_model(self, custom_config: dict) -> None:
        try:
            vocab_path = custom_config.get('vocab_path', 'cache/severity_vocab.json')
            if Path(vocab_path).exists():
                logger.info("Loading Custom Ensemble Models...")
                self.custom_vocab = SeverityVocab()
                self.custom_vocab.load(vocab_path)
                self.custom_model_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                self.ensemble_models = []
                self.ensemble_names = []
                for member in ENSEMBLE_NAMES:
                    try:
                        m = build_ensemble_model(member, len(self.custom_vocab.word2idx), len(self.custom_vocab.char2idx), num_classes=3).to(self.custom_model_device)
                        m.load_state_dict(torch.load(f'cache/ensemble_{member}.pt', map_location=self.custom_model_device))
                        m.eval()
                        self.ensemble_models.append(m)
                        self.ensemble_names.append(member)
                    except Exception as e:
                        logger.warning(f"Could not load {member}: {e}")

                # Load Stacker and Temperatures
                try:
                    stacker_path = 'cache/ensemble_stacker.pkl'
                    if not Path(stacker_path).exists():
                        stacker_path = 'cache/ensemble_xgb_stacker.pkl'  # legacy name
                    if Path(stacker_path).exists():
                        with open(stacker_path, 'rb') as f:
                            self.stacker = pickle.load(f)
                        logger.info(f"Stacker loaded successfully from {stacker_path}.")
                    else:
                        self.stacker = None

                    temps_path = 'cache/ensemble_temps.json'
                    if Path(temps_path).exists():
                        with open(temps_path, 'r') as f:
                            self.optimal_temps = json.load(f)
                    else:
                        self.optimal_temps = {m: 1.5 for m in ENSEMBLE_NAMES}
                except Exception as e:
                    logger.warning(f"Could not load stacker/temps: {e}")
                    self.stacker = None
                    self.optimal_temps = {m: 1.5 for m in ENSEMBLE_NAMES}

                self.custom_model = True # Flag to indicate it's active
                logger.info(f"Custom Ensemble loaded successfully with {len(self.ensemble_models)} models.")
            else:
                logger.warning("Custom severity vocab not found despite being enabled in config.")
        except Exception as e:
            logger.error(f"Failed to load custom severity ensemble: {e}")
            self.custom_model = None

    def _custom_classify_severity(self, description: str) -> tuple:
        if not getattr(self, 'ensemble_models', None) or not self.custom_vocab:
            return "Moderate", 0.75, np.array([0.25, 0.50, 0.25])
        try:
            word_ids = self.custom_vocab.encode_words(description)
            char_ids = self.custom_vocab.encode_chars(description)
            if not word_ids:
                return "Moderate", 0.75, np.array([0.25, 0.50, 0.25])
            w_tensor = torch.tensor([word_ids], dtype=torch.long).to(self.custom_model_device)
            lengths = torch.tensor([len(word_ids)], dtype=torch.long)
            max_word_len = max(len(c) for c in char_ids) if char_ids else 1
            c_tensor = torch.zeros((1, len(word_ids), max_word_len), dtype=torch.long).to(self.custom_model_device)
            for j, c_seq in enumerate(char_ids):
                c_tensor[0, j, :len(c_seq)] = torch.tensor(c_seq, dtype=torch.long)

            with torch.no_grad():
                batch_features = []
                model_names = list(getattr(self, "ensemble_names", ENSEMBLE_NAMES))
                for i, model in enumerate(self.ensemble_models):
                    m_name = model_names[i] if i < len(model_names) else 'Unknown'
                    t = self.optimal_temps.get(m_name, 1.5)
                    logits = model(w_tensor, c_tensor, lengths)
                    probs = torch.softmax(logits / t, dim=-1)[0].cpu().numpy()
                    batch_features.append(probs)

                concat_probs = np.concatenate(batch_features).reshape(1, -1)

                if hasattr(self, 'stacker') and self.stacker is not None:
                    ensemble_probs = self.stacker.predict_proba(concat_probs)[0]
                else:
                    # Fallback
                    ensemble_probs = np.mean(batch_features, axis=0)

                pred_idx = np.argmax(ensemble_probs)
                pred_label = severity.SEVERITY_LABELS[pred_idx]
                risk_score = float(ensemble_probs[1] + ensemble_probs[2])

            return pred_label, risk_score, np.array(ensemble_probs, dtype=float)
        except Exception as e:
            logger.error(f"Error during custom severity inference: {e}")
            return "Moderate", 0.75, np.array([0.25, 0.50, 0.25])

    @staticmethod
    def _is_genuinely_mild(description: str) -> bool:
        """Codebook policy (physician-confirmed, see severity_audit/):
        True when the description itself is genuinely mild — GI irritation
        only, minor-drug efficacy loss, excretion shift with NO serum
        consequence stated, or decreased sedation/hypertension. Anything
        with a stated serum level change or systemic consequence is not
        Mild."""
        import re as _re
        t = description or ""
        if _re.search(r"gastrointestinal irritation", t, _re.I):
            return True
        if _re.search(r"may decrease the (sedative|hypertensive)", t, _re.I):
            return True
        if _re.search(r"hypertension can be decreased", t, _re.I):
            return True
        if _re.search(
            r"therapeutic efficacy of (castor oil|sennosides|tadalafil|domperidone|"
            r"dantron|beraprost|neosaxitoxin|magnesium)\b", t, _re.I,
        ):
            return True
        if _re.search(r"excretion", t, _re.I) and not _re.search(
            r"serum level|serum concentration", t, _re.I
        ):
            return True
        return False

    def _learned_gate_proba(self, description: str, custom_probs) -> Optional[float]:
        """P(Severe) from the learned Severe-vs-rest gate, or None.

        Loads outputs/learned_gate/gate.pkl lazily; on any failure returns
        None so the caller falls back to the regex gate. The Mild path never
        reaches here (deterministic allowlist only — no learned Mild weights).
        """
        import numpy as _np
        if self._learned_gate_failed:
            return None
        if self._learned_gate is None:
            try:
                from clinical_copilot.learned_gate import extract_gate_features, load_gate
                model, threshold, _meta = load_gate()
                self._learned_gate = (model, extract_gate_features)
                self._learned_gate_threshold = threshold
                logger.info(
                    f"Learned severity gate loaded (threshold={threshold}).")
            except Exception as e:
                logger.warning(f"Learned gate unavailable ({e}); using regex gate.")
                self._learned_gate_failed = True
                return None
        try:
            model, extract = self._learned_gate
            x = extract(description, _np.asarray(custom_probs, dtype=float)).reshape(1, -1)
            return float(model.predict_proba(x)[0, 1])
        except Exception as e:
            logger.warning(f"Learned gate inference failed ({e}); using regex gate.")
            self._learned_gate_failed = True
            return None

    def _apply_severity_gate(self, custom_severity: str, custom_risk: float,
                             custom_probs, description: str) -> tuple:
        """Apply the Mild/Severe policy to the frozen stacker output.

        Modes (self.SEVERITY_GATE, ablation):
          'learned' — production default: Mild path is the deterministic
                       allowlist; all non-Mild rows decided by the
                       Severe-vs-rest gate (demote weak Severe calls,
                       upgrade strong Moderate calls).
          'regex'   — legacy deterministic Mild allowlist +
                       SEVERE_MIN_PROB/SEVERE_MARGIN demotion rule.
                       Kept for ablation and as fallback.
          'none'    — raw stacker argmax (Mild still collapses to Moderate
                      unless genuinely mild, matching legacy behavior).
        Returns (severity_categorical, risk_score).
        """
        if custom_severity == "Mild":
            if self._is_genuinely_mild(description):
                return "Mild", float(custom_risk)
            return "Moderate", max(float(custom_risk), severity.risk_score("Moderate"))
        if self.SEVERITY_GATE == "learned":
            p_gate = self._learned_gate_proba(description, custom_probs)
            if p_gate is None:
                p_gate = 1.0 if custom_severity == "Severe" else 0.0
                gate_thr = 0.5
            else:
                gate_thr = self._learned_gate_threshold
            if p_gate >= gate_thr:
                return "Severe", float(custom_risk)
            return "Moderate", float(custom_risk)
        if custom_severity == "Severe":
            p_mod = float(custom_probs[1])
            p_sev = float(custom_probs[2])
            if self.SEVERITY_GATE == "none":
                return "Severe", float(custom_risk)
            if p_sev < self.SEVERE_MIN_PROB or (p_sev - p_mod) < self.SEVERE_MARGIN:
                return "Moderate", float(custom_risk)
            return "Severe", float(custom_risk)
        return custom_severity, float(custom_risk)

    def _extract_severity_from_description(self, description: str) -> str:
        if not description:
            return "Moderate"
        desc_lower = description.lower()
        severe_keywords = [
            'contraindicated', 'life-threatening', 'fatal', 'death', 'deaths',
            'severe bleeding', 'serious adverse', 'do not use', 'avoid use',
            'serious harm', 'toxicity', 'toxic', 'cardiac arrest', 'respiratory failure',
            'anaphylaxis', 'severe hypotension', 'coma', 'seizures',
            'major', 'significant', 'requires immediate', 'avoid combination',
            'serious', 'substantially', 'marked increase'
        ]
        moderate_keywords = [
            'closely monitor', 'reduces efficacy', 'increases risk',
            'adjust dose', 'dose adjustment', 'important',
            'may increase', 'may decrease', 'caution'
        ]
        mild_keywords = [
            'minor', 'possible interaction',
            'may occur', 'clinical significance unknown', 'limited data',
            'unlikely', 'small increase', 'minimal', 'slight'
        ]
        for kw in severe_keywords:
            if kw in desc_lower: return "Severe"
        for kw in moderate_keywords:
            if kw in desc_lower: return "Moderate"
        for kw in mild_keywords:
            if kw in desc_lower: return "Mild"
        return "Moderate"

    def _get_gemini_severity(self, drug_a: str, drug_b: str, description: str) -> str:
        """
        Get severity classification from Gemini API.

        Args:
            drug_a: First drug name
            drug_b: Second drug name
            description: Interaction description

        Returns:
            Severity category: "Mild", "Moderate", or "Severe"
        """
        if not self.gemini_client:
            return None

        try:
            prompt = f"""You are a clinical pharmacologist grading drug-drug interaction severity.
Decide the clinical consequence of THIS interaction from its description.

Drug A: {drug_a}
Drug B: {drug_b}
Interaction Description: {description}

OUTPUT (mandatory):
1) Output EXACTLY one label from this closed set: Mild, Moderate, Severe
2) Do not output any extra words, punctuation, explanation, markdown, or newline text
3) If genuinely torn between two levels, choose the higher (safety-first)

Severity codebook - judge WHAT is harmed and how soon, not trigger words alone:
- Severe (risks outweigh any benefit, do not combine): death; contraindicated combos; QTc prolongation with torsades-capable drugs, torsades, ventricular arrhythmia, cardiac arrest; anaphylaxis; bleeding/hemorrhage as an event (including GI bleeding); respiratory depression/failure; coma; seizure; serotonin syndrome; rhabdomyolysis/myopathy with myoglobinuria; renal failure; nephro/cardiotoxicity, neurotoxicity, hepatotoxicity, ototoxicity, pulmonary toxicity, or liver damage as events; methemoglobinemia; neuromuscular blockade (paralysis/apnea risk); thrombosis/thromboembolism; angioedema with airway risk.
- Moderate (avoid unless necessary, monitor or adjust dose, possible long-term harm): altered serum exposure or excretion shifts STATING higher/lower serum levels; reduced efficacy of critical therapy; electrolyte disturbances; increased bleeding risk without a severe-bleed event; CNS depression; QT change without torsades evidence; lone tachycardia/bradycardia; renal/hepatic stress short of organ toxicity; vague systemic changes (e.g. "adverse effects can be increased").
- Mild (avoid only if an alternative exists, no long-term effects; use sparingly): gastrointestinal irritation ONLY; efficacy loss of a minor/non-critical drug (laxative, supplement, imaging agent); excretion shift with NO serum consequence stated; decreased sedation or decreased hypertension. Vague systemic exposure changes are Moderate, not Mild."""

            config = {
                "temperature": self.gemini_temperature_severity,
                "top_p": self.gemini_top_p_severity,
                "thinking_config": {"thinking_level": self.gemini_thinking_severity},
            }
            response_text = self._call_gemini_with_retry(
                model=self.gemini_model_severity,
                contents=prompt,
                config=config,
                context="severity_classification"
            )

            if response_text:
                severity = self._normalize_gemini_severity(response_text)
                if severity is not None:
                    return severity
                logger.warning(f"Invalid Gemini severity response: {response_text}")
                return None
        except Exception as e:
            logger.warning(f"Gemini severity classification failed: {e}")
            return None

    def _get_gemini_reasoning(self, drug_a: str, drug_b: str, description: str) -> str:
        """
        Get clinical reasoning from Gemini API.

        Args:
            drug_a: First drug name
            drug_b: Second drug name
            description: Interaction description

        Returns:
            Clinical reasoning text (None when reasoning is disabled in config)
        """
        if not self.gemini_client or not self.gemini_reasoning_enabled:
            return None

        try:
            prompt = f"""Rewrite and simplify this drug-drug interaction for a layman.

Interaction Description: {description}

Limit the answer to 2 concise sentences."""

            config = {
                "temperature": self.gemini_temperature_reasoning,
                "top_p": self.gemini_top_p_reasoning,
                "thinking_config": {"thinking_level": self.gemini_thinking_reasoning},
            }
            response_text = self._call_gemini_with_retry(
                model=self.gemini_model_reasoning,
                contents=prompt,
                config=config,
                context="reasoning_generation"
            )

            if response_text:
                reasoning = response_text.strip()
                # Enforce reasoning-only output by removing explicit severity labels if present.
                reasoning = re.sub(r'\b(Mild|Severe|VERY\s+SEVERE)\b', 'interaction risk', reasoning, flags=re.IGNORECASE)
                return reasoning
        except Exception as e:
            logger.warning(f"Gemini reasoning generation failed: {e}")
            return None

    def _build_drug_index(self) -> Dict[str, Dict[str, Any]]:
        """
        Build index of drugs from DrugBank JSON for fast lookup.
        
        Returns:
            Dictionary mapping drug names (lowercase) to drug entries
        """
        index = {}
        
        drugbank = self.drugbank_json.get('{http://www.drugbank.ca}drugbank', {})
        drugs = drugbank.get('{http://www.drugbank.ca}drug', [])
        
        for drug in drugs:
            name = drug.get('{http://www.drugbank.ca}name', '').lower()
            
            minimal_drug = {}
            interactions_raw = drug.get('{http://www.drugbank.ca}drug-interactions')
            if interactions_raw:
                i_list = []
                if isinstance(interactions_raw, dict):
                    i_list = interactions_raw.get('{http://www.drugbank.ca}drug-interaction', [])
                elif isinstance(interactions_raw, list):
                    i_list = interactions_raw
                    
                if isinstance(i_list, dict):
                    i_list = [i_list]
                    
                minimal_interactions = []
                for inter in i_list:
                    minimal_interactions.append({
                        '{http://www.drugbank.ca}name': inter.get('{http://www.drugbank.ca}name', ''),
                        '{http://www.drugbank.ca}description': inter.get('{http://www.drugbank.ca}description', ''),
                        '{http://www.drugbank.ca}drugbank-id': inter.get('{http://www.drugbank.ca}drugbank-id', '')
                    })
                
                if minimal_interactions:
                    minimal_drug['{http://www.drugbank.ca}drug-interactions'] = {
                        '{http://www.drugbank.ca}drug-interaction': minimal_interactions
                    }

            if name:
                index[name] = minimal_drug
            
            db_ids = drug.get('{http://www.drugbank.ca}drugbank-id', [])
            if isinstance(db_ids, str):
                db_ids = [db_ids]
            
            for db_id in db_ids:
                index[db_id.lower()] = minimal_drug
        
        # Free original JSON data to reclaim memory
        self.drugbank_json.clear()
        
        return index
    
    def check_interactions(self, all_generics: List[str]) -> List[Dict[str, Any]]:
        """
        Check for drug-drug interactions among prescribed medications.
        
        Args:
            all_generics: List of generic drug names
            
        Returns:
            List of detected interactions with details
        """
        interactions = []
        
        for i, drug1 in enumerate(all_generics):
            for drug2 in all_generics[i+1:]:
                interaction = self._find_interaction(drug1, drug2)
                if not interaction:
                    # DrugBank interaction lists are per-drug and asymmetric:
                    # a pair may only be listed under drug2's entry (or under
                    # a name variant matching drug1 better from that side).
                    interaction = self._find_interaction(drug2, drug1)
                if interaction:
                    interactions.append(interaction)
        
        return interactions
    
    def _find_interaction(self, drug1, drug2) -> Optional[Dict[str, Any]]:
        if isinstance(drug1, dict): drug1 = drug1.get('name', str(drug1))
        if isinstance(drug2, dict): drug2 = drug2.get('name', str(drug2))
        """
        Find interaction between two drugs.
        
        Args:
            drug1: First generic drug name
            drug2: Second generic drug name
            
        Returns:
            Interaction details or None
        """
        drug1_entry, match_score1 = self._fuzzy_drug_lookup(drug1)

        if not drug1_entry:
            return None
        
        # The DrugBank JSON nests interactions as:
        #   {ns}drug-interactions -> dict with key {ns}drug-interaction -> list of interactions
        # OR the entire value may already be a list (depending on serialisation).
        interactions_raw = drug1_entry.get('{http://www.drugbank.ca}drug-interactions', {})
        
        if isinstance(interactions_raw, dict):
            interactions_list = interactions_raw.get(
                '{http://www.drugbank.ca}drug-interaction', []
            )
        elif isinstance(interactions_raw, list):
            interactions_list = interactions_raw
        else:
            interactions_list = []
        
        # Normalise single-interaction entries (dict instead of list)
        if isinstance(interactions_list, dict):
            interactions_list = [interactions_list]
        
        if not interactions_list:
            return None
        
        for interaction in interactions_list:
            interaction_name = interaction.get('{http://www.drugbank.ca}name', '').lower()

            match_score2 = difflib.SequenceMatcher(None, drug2.lower(), interaction_name).ratio()
            if match_score2 >= 0.85:
                description = interaction.get('{http://www.drugbank.ca}description', 'No description available')
                drugbank_id = interaction.get('{http://www.drugbank.ca}drugbank-id', 'Unknown')

                # Extract severity from database description
                if self.custom_model:
                    database_severity = self._extract_severity_from_description(description)
                    custom_severity, custom_risk, custom_probs = self._custom_classify_severity(description)

                    # Ensemble-only: existing XGB stacker is the severity signal.
                    # database_severity is kept for audit only, never fused.
                    # CODEBOOK POLICY (physician-confirmed standardized
                    # criteria, see severity_audit/): permanent decision
                    # rules, not temporary fixes.
                    # 1. Mild (low-risk bucket) maps to Moderate, EXCEPT when
                    #    the description itself is genuinely mild per the
                    #    codebook (GI irritation only, minor-drug efficacy
                    #    loss, excretion shift with NO serum consequence
                    #    stated, decreased sedation/hypertension). Those stay
                    #    Mild so the Mild class remains predictable.
                    # 2. Severe needs a clear margin (SEVERE_MIN_PROB /
                    #    SEVERE_MARGIN attributes below), else Moderate.
                    # 3. SEVERITY_GATE ablation ('none' | 'regex' | 'learned'):
                    #    see _apply_severity_gate. The Mild path is identical
                    #    in every mode (deterministic allowlist, no learned
                    #    Mild weights).
                    severity_categorical, risk_score = self._apply_severity_gate(
                        custom_severity, custom_risk, custom_probs, description)

                    gemini_severity = "N/A (Custom Model)"
                    reasoning = description
                else:
                    database_severity = self._extract_severity_from_description(description)

                    # Get a second-opinion severity and separate reasoning from Gemini
                    gemini_severity = self._get_gemini_severity(drug1, drug2, description)
                    reasoning = self._get_gemini_reasoning(drug1, drug2, description)

                    # Determine final severity using safety-first approach
                    severity_categorical = self._determine_final_severity(
                        database_severity, gemini_severity
                    )

                return {
                    'drug_a': drug1.title(),
                    'drug_b': drug2.title(),
                    'severity_categorical': severity_categorical,
                    'database_severity': database_severity,
                    'gemini_severity': gemini_severity or "N/A",
                    'reasoning': reasoning or description,
                    'description': description,
                    'drugbank_id': drugbank_id,
                    'detection_confidence': min(match_score1, match_score2),
                    'serious_risk_score': risk_score if self.custom_model else None
                }

        return None

    def _determine_final_severity(self, database_severity: str, gemini_severity: str = None) -> str:
        """
        Determine final severity using safety-first approach.

        Args:
            database_severity: Severity from database keyword extraction
            gemini_severity: Severity from Gemini API (optional)

        Returns:
            Final severity category
        """
        database_severity = severity.normalize(database_severity)
        if not gemini_severity or gemini_severity == "N/A":
            return database_severity

        gemini_severity = severity.normalize(gemini_severity)

        if severity.rank(gemini_severity) > severity.rank(database_severity):
            logger.info(f"Severity mismatch: DB={database_severity}, Gemini={gemini_severity}. Using Gemini (more severe).")
            return gemini_severity

        if severity.rank(database_severity) > severity.rank(gemini_severity):
            logger.info(f"Severity mismatch: DB={database_severity}, Gemini={gemini_severity}. Using DB (more severe).")
            return database_severity

        logger.debug(f"Severity agree: DB={database_severity}, Gemini={gemini_severity}.")
        return database_severity
    
    def _fuzzy_drug_lookup(self, drug_name: str) -> Tuple[Optional[Dict[str, Any]], float]:
        """
        Fuzzy lookup of drug in index.

        Args:
            drug_name: Generic drug name

        Returns:
            Tuple of (drug entry or None, match score 0.0-1.0)
        """
        drug_lower = drug_name.lower()

        if drug_lower in self.drug_index:
            return self.drug_index[drug_lower], 1.0

        matches = difflib.get_close_matches(drug_lower, self.drug_index.keys(), n=1, cutoff=0.85)
        if matches:
            score = difflib.SequenceMatcher(None, drug_lower, matches[0]).ratio()
            return self.drug_index[matches[0]], score

        return None, 0.0
    
    def _is_similar_drug(self, drug1: str, drug2: str, threshold: float = 0.85) -> bool:
        """
        Check if two drug names are similar.
        
        Args:
            drug1: First drug name
            drug2: Second drug name
            threshold: Similarity threshold
            
        Returns:
            True if similar, False otherwise
        """
        return difflib.SequenceMatcher(None, drug1.lower(), drug2.lower()).ratio() >= threshold
