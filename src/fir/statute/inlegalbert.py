"""InLegalBERT multi-label statute-identification head.

`law-ai/InLegalBERT` is BERT-base further-pretrained on Indian legal text, so it
starts with the vocabulary of the domain (section numbers, "the accused", "PW-1")
that a general checkpoint has to learn from scratch.

Two implementation notes:

* **No HuggingFace `Trainer` and no `datasets`.** `datasets` pulls in pyarrow,
  which Smart App Control blocks on the Windows host. A plain torch loop keeps
  that dependency at zero and is short enough to read.

* **512-token budget on ~1,800-token documents.** ILSI documents are court
  fact-statements and BERT's positional embeddings stop at 512. Two truncations
  are available (config `truncation`): `head` keeps the opening; `head_tail`
  keeps the opening and the closing, where judgments name the sections invoked
  (Sun et al. 2019). The truncation ablation in PROGRESS.md finding 8 measured
  this ceiling at ~8.6% relative on TF-IDF -- real, but not the main gap.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fir.config import ARTIFACT_DIR, pipeline
from fir.data.ilsi import FactInstance
from fir.statute.labels import LabelSpace

DEFAULT_DIR = ARTIFACT_DIR / "models" / "inlegalbert_statute"


def pick_device(prefer: str = "auto") -> str:
    import torch

    if prefer != "auto":
        return prefer
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(slots=True)
class InLegalBertStatuteClassifier:
    """Multi-label sequence classifier over ILSI fact text."""

    label_space: LabelSpace
    model_name: str = "law-ai/InLegalBERT"
    max_seq_len: int = 512
    truncation: str = "head"        # "head" | "head_tail"  (config: inlegalbert.truncation)
    tail_fraction: float = 0.25     # share of the budget given to the tail when head_tail
    device: str = "auto"
    threshold: float = 0.5
    train_seconds: float = 0.0
    history: list[dict] = field(default_factory=list)

    model: object = None
    tokenizer: object = None

    name: str = "InLegalBERT"

    # -- setup --------------------------------------------------------------
    def _ensure_loaded(self, num_labels: int | None = None) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=num_labels or len(self.label_space),
                problem_type="multi_label_classification",
            )
            self.model.to(pick_device(self.device))

    def _encode(self, texts: list[str], chunk: int = 2000):
        """Tokenise to (input_ids, attention_mask) tensors, in chunks.

        Chunking is not an optimisation, it is a survival requirement on this
        box: a single tokenizer call over 42,835 documents builds ~22M Python
        ints as nested lists before anything becomes a tensor, and the WSL VM
        reboots itself when a process's Python heap grows past ~2 GB
        (PROGRESS.md finding 6e -- numpy/torch block allocations do not trigger
        it, small-object churn does). Each chunk is converted to tensors and the
        lists freed before the next.
        """
        import torch

        if len(texts) > chunk:
            ids, masks = [], []
            for i in range(0, len(texts), chunk):
                a, b = self._encode(texts[i : i + chunk], chunk=chunk)
                ids.append(a)
                masks.append(b)
            return torch.cat(ids), torch.cat(masks)

        if self.truncation == "head":
            enc = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            return enc["input_ids"], enc["attention_mask"]

        # head+tail: keep the opening (parties, charge framing) and the closing
        # (findings, sections invoked) of each judgment, drop the procedural
        # middle. Sun et al. 2019 found this the best fixed-budget truncation for
        # long-document classification. Budget: max_seq_len - 2 for [CLS]/[SEP].
        budget = self.max_seq_len - 2
        n_tail = int(budget * self.tail_fraction)
        n_head = budget - n_tail
        cls, sep, pad = (
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        )
        raw = self.tokenizer(texts, add_special_tokens=False, truncation=False)["input_ids"]
        ids_out, mask_out = [], []
        for toks in raw:
            if len(toks) > budget:
                # guard n_tail == 0: toks[-0:] is the whole list, not an empty one
                toks = toks[:n_head] + (toks[-n_tail:] if n_tail else [])
            seq = [cls] + toks + [sep]
            pad_n = self.max_seq_len - len(seq)
            ids_out.append(seq + [pad] * pad_n)
            mask_out.append([1] * len(seq) + [0] * pad_n)
        return torch.tensor(ids_out, dtype=torch.long), torch.tensor(mask_out, dtype=torch.long)

    # -- training -----------------------------------------------------------
    def fit(
        self,
        train: list[FactInstance],
        dev: list[FactInstance] | None = None,
        progress: bool = True,
    ) -> "InLegalBertStatuteClassifier":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        cfg = pipeline()["statute_id"]["inlegalbert"]
        self.truncation = str(cfg.get("truncation", self.truncation))
        self.tail_fraction = float(cfg.get("tail_fraction", self.tail_fraction))
        device = pick_device(self.device)
        self.device = device
        self._ensure_loaded()
        t0 = time.perf_counter()
        if self.truncation != "head":
            print(f"    truncation: {self.truncation} (tail {self.tail_fraction:.0%})", flush=True)

        ids, mask = self._encode([i.text for i in train])
        y = torch.from_numpy(self.label_space.encode(train))
        loader = DataLoader(
            TensorDataset(ids, mask, y),
            batch_size=cfg["batch_size"],
            shuffle=True,
            drop_last=False,
        )

        accum = max(1, int(cfg.get("grad_accum", 1)))
        epochs = int(cfg["epochs"])
        steps = (len(loader) // accum) * epochs

        # Optional discriminative LR: the classifier head is randomly initialised
        # and has to learn a 100-way mapping from scratch, while the encoder only
        # needs nudging. One shared LR of 2e-5 is a compromise that starves the
        # head -- finding 8 hypothesis #4. `head_lr_multiplier` (config, default
        # 1.0 = off) gives the head its own, larger rate.
        base_lr = float(cfg["lr"])
        head_mult = float(cfg.get("head_lr_multiplier", 1.0))
        head_params = [p for n, p in self.model.named_parameters() if n.startswith("classifier")]
        body_params = [p for n, p in self.model.named_parameters() if not n.startswith("classifier")]
        groups = [{"params": body_params, "lr": base_lr}]
        if head_params:
            groups.append({"params": head_params, "lr": base_lr * head_mult})
        optim = torch.optim.AdamW(groups)
        if head_mult != 1.0:
            print(f"    head LR x{head_mult:g} = {base_lr * head_mult:.1e}", flush=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=[g["lr"] for g in groups],
            total_steps=max(steps, 1),
            pct_start=float(cfg.get("warmup_ratio", 0.1)),
            anneal_strategy="linear",
        )
        # fp16 keeps batch-8 x 512 tokens inside the 8 GB card.
        use_amp = device == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # Positive-class weighting. ILSI averages 3.67 positive labels out of
        # 100, so plain BCE is minimised by predicting all-zeros: the first run
        # without this scored macro-F1 0.018, i.e. it learned two head labels
        # and gave up on the tail. pos_weight = neg/pos per label restores the
        # gradient signal on rare sections, and mirrors the class_weight=
        # "balanced" the TF-IDF baseline already gets -- without it the two
        # models are not comparable.
        loss_fn = torch.nn.BCEWithLogitsLoss()
        if cfg.get("pos_weight", True):
            pos = y.sum(dim=0)
            neg = y.shape[0] - pos
            # clamp: a label with 1-2 examples would otherwise get a weight in
            # the thousands and destabilise training
            pw = torch.clamp(neg / torch.clamp(pos, min=1.0), max=float(cfg.get("pos_weight_max", 50.0)))
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
            print(
                f"    pos_weight: median {pw.median():.1f}, max {pw.max():.1f} "
                f"({int((pos == 0).sum())} labels unseen in train)",
                flush=True,
            )

        self.model.train()
        for epoch in range(epochs):
            running, n_batches = 0.0, 0
            optim.zero_grad(set_to_none=True)
            for step, (b_ids, b_mask, b_y) in enumerate(loader):
                b_ids = b_ids.to(device, non_blocking=True)
                b_mask = b_mask.to(device, non_blocking=True)
                b_y = b_y.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = self.model(input_ids=b_ids, attention_mask=b_mask).logits
                    loss = loss_fn(logits.float(), b_y) / accum

                scaler.scale(loss).backward()
                if (step + 1) % accum == 0:
                    scaler.unscale_(optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    if sched.last_epoch < sched.total_steps - 1:
                        sched.step()

                # detach explicitly: float() on a grad-tracking tensor warns
                running += loss.detach().item() * accum
                n_batches += 1
                if progress and step % 50 == 0:
                    print(
                        f"  epoch {epoch + 1}/{epochs}  "
                        f"step {step + 1}/{len(loader)}  "
                        f"loss {running / max(n_batches, 1):.4f}",
                        flush=True,
                    )

            entry = {"epoch": epoch + 1, "train_loss": running / max(n_batches, 1)}
            if dev:
                from harness.metrics import tune_threshold

                probs = self.predict_proba([d.text for d in dev])
                thr, sc = tune_threshold(self.label_space.encode(dev), probs)
                self.threshold = thr
                entry.update(dev_micro_f1=sc.micro_f1, dev_macro_f1=sc.macro_f1,
                             threshold=thr)
                print(
                    f"  epoch {epoch + 1}: dev micro-F1={sc.micro_f1:.4f} "
                    f"macro-F1={sc.macro_f1:.4f} @thr={thr:.2f}",
                    flush=True,
                )
                self.model.train()
            self.history.append(entry)

        self.train_seconds = time.perf_counter() - t0
        return self

    # -- inference ----------------------------------------------------------
    def predict_proba(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        import torch

        self._ensure_loaded()
        device = pick_device(self.device)
        self.model.eval()
        out: list[np.ndarray] = []
        use_amp = device == "cuda"

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                ids, mask = self._encode(texts[i : i + batch_size])
                ids, mask = ids.to(device), mask.to(device)
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = self.model(input_ids=ids, attention_mask=mask).logits
                out.append(torch.sigmoid(logits.float()).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def predict_sections(
        self, text: str, threshold: float | None = None
    ) -> list[str]:
        probs = self.predict_proba([text])[0]
        return self.label_space.decode(probs, threshold or self.threshold)

    def top_k(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        return self.label_space.top_k(self.predict_proba([text])[0], k=k)

    # -- persistence --------------------------------------------------------
    def save(self, path: str | Path = DEFAULT_DIR) -> Path:
        import json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "statute_head.json").write_text(
            json.dumps(
                {
                    "labels": self.label_space.labels,
                    "threshold": self.threshold,
                    "max_seq_len": self.max_seq_len,
                    "truncation": self.truncation,
                    "tail_fraction": self.tail_fraction,
                    "base_model": self.model_name,
                    "train_seconds": self.train_seconds,
                    "history": self.history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, path: str | Path = DEFAULT_DIR) -> "InLegalBertStatuteClassifier":
        import json

        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(path)
        meta_path = path / "statute_head.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"no trained InLegalBERT head at {path}. "
                "Run: python -m harness.run_statute_baseline"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        obj = cls(
            label_space=LabelSpace(labels=meta["labels"]),
            model_name=meta.get("base_model", "law-ai/InLegalBERT"),
            max_seq_len=meta.get("max_seq_len", 512),
            truncation=meta.get("truncation", "head"),
            tail_fraction=meta.get("tail_fraction", 0.25),
            threshold=meta.get("threshold", 0.5),
            train_seconds=meta.get("train_seconds", 0.0),
            history=meta.get("history", []),
        )
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.model.to(pick_device(obj.device))
        return obj
