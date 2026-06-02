"""Synthetic interaction note generation for the X Education lead dataset.

The tabular dataset has no meaningful free-text column, so we synthesize 1–3
sentence sales-rep notes that a downstream sentiment / intent classifier can
train on. The notes follow four attitude classes:

    positive_engagement | objection | disengaged | neutral

**Leakage discipline (this is the whole point of the module).**

1.  Attitude is assigned from *behavioral features that are observable at
    scoring time* (TotalVisits, Total Time on Site, Page Views per Visit,
    Last Activity) plus independent Gaussian noise. It is NEVER derived from
    `Converted`. The shared upstream cause produces a realistic but
    non-deterministic correlation between attitude and conversion.

2.  The LLM prompt only sees the assigned attitude and a neutral feature
    context (lead source, specialization, occupation, activity, time bucket).
    `Converted`, `Tags`, `Lead Quality`, `Last Notable Activity` are
    explicitly withheld — these either are the label or are filled in after
    the outcome is known (temporal leakage).

3.  The system prompt forbids outcome-revealing phrases ("closed", "signed",
    "müşteri oldu", "satın aldı", "vazgeçti", "churn", …) so the note reads
    as a pre-decision interaction, not a post-mortem.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from tqdm import tqdm  # type: ignore[import-untyped]

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from langchain_openai import AzureChatOpenAI

logger = logging.getLogger(__name__)


class AttitudeClass(StrEnum):
    POSITIVE_ENGAGEMENT = "positive_engagement"
    OBJECTION = "objection"
    DISENGAGED = "disengaged"
    NEUTRAL = "neutral"


class LanguageMode(StrEnum):
    TR = "tr"
    EN = "en"
    MIX = "mix"


DEFAULT_LANG_MIX: Mapping[LanguageMode, float] = {
    LanguageMode.TR: 0.50,
    LanguageMode.EN: 0.20,
    LanguageMode.MIX: 0.30,
}

# Behavioral feature → activity bonus contributing to the latent engagement
# score. Values are deliberately modest so the bonus does not dominate the
# continuous features.
_ACTIVITY_BONUS: Mapping[str, float] = {
    "SMS Sent": 1.5,
    "Had a Phone Conversation": 1.5,
    "Email Link Clicked": 0.3,
    "Email Opened": 0.3,
    "Page Visited on Website": 0.2,
    "Olark Chat Conversation": 0.0,
    "Form Submitted on Website": 0.6,
    "Approached upfront": 0.6,
    "Converted to Lead": 0.0,  # neutral — this is a status flip, not behavior
    "Email Bounced": -1.0,
    "Unsubscribed": -1.0,
    "Unreachable": -1.0,
    "Email Marked Spam": -1.0,
}

_LEAKAGE_SUSPECT_COLUMNS: tuple[str, ...] = (
    "Tags",
    "Lead Quality",
    "Last Notable Activity",
)


def _clip_normalize(series: pd.Series, upper_quantile: float = 0.95) -> np.ndarray:
    """Clip the upper tail and scale to [0, 1]. NaNs become 0."""

    arr = series.astype(float).to_numpy()
    arr = np.nan_to_num(arr, nan=0.0)
    upper = float(np.quantile(arr, upper_quantile))
    if upper <= 0:
        return np.zeros_like(arr)
    clipped: np.ndarray = np.clip(arr / upper, 0.0, 1.0)
    return clipped


def assign_attitudes(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    engagement_noise_sigma: float = 0.30,
) -> pd.Series:
    """Assign one attitude class per lead, derived from observable behavior.

    The function deliberately ignores ``Converted``. attitude × Converted
    correlation arises only from the shared upstream behavioral signal plus
    noise — there is no deterministic link.

    Weights were tuned empirically against the X Education dataset to land
    Cramér's V (attitude × Converted) in the 0.25–0.30 band. The bulk of the
    raw signal comes from ``Total Time Spent on Website`` (corr ≈ 0.36 with
    Converted) and ``Last Activity`` ("SMS Sent" / phone calls vs. bounces);
    ``TotalVisits`` and ``Page Views Per Visit`` carry almost no marginal
    signal on this dataset, so they receive only a small weight.

    Args:
        df: Lead dataframe. Must carry
            ``TotalVisits``, ``Total Time Spent on Website``,
            ``Page Views Per Visit``, ``Last Activity``.
        seed: Numpy RNG seed for reproducibility.
        engagement_noise_sigma: Standard deviation of the additive Gaussian
            noise on the latent engagement score.

    Returns:
        ``pd.Series`` of :class:`AttitudeClass` values aligned to ``df.index``.
    """

    rng = np.random.default_rng(seed)

    norm_time = _clip_normalize(df["Total Time Spent on Website"])
    norm_visits = _clip_normalize(df["TotalVisits"])
    norm_pages = _clip_normalize(df["Page Views Per Visit"])
    activity_bonus = df["Last Activity"].map(_ACTIVITY_BONUS).fillna(0.0).astype(float).to_numpy()

    engagement = (
        0.50 * norm_time
        + 0.10 * norm_visits
        + 0.05 * norm_pages
        + 0.35 * activity_bonus
        + rng.normal(0.0, engagement_noise_sigma, size=len(df))
    )
    # Friction is independent of engagement and of Converted by construction —
    # it decides whether engaged leads come across positive vs. objecting, and
    # whether disengaged leads look indifferent vs. silent.
    friction = rng.normal(0.0, 1.0, size=len(df))

    engaged = engagement >= float(np.median(engagement))

    attitudes = np.empty(len(df), dtype=object)
    # Engaged branch: positive_engagement vs objection.
    attitudes[engaged & (friction >= -0.5)] = AttitudeClass.POSITIVE_ENGAGEMENT.value
    attitudes[engaged & (friction < -0.5)] = AttitudeClass.OBJECTION.value
    # Disengaged branch: disengaged vs neutral.
    attitudes[(~engaged) & (friction < 0.25)] = AttitudeClass.DISENGAGED.value
    attitudes[(~engaged) & (friction >= 0.25)] = AttitudeClass.NEUTRAL.value

    return pd.Series(attitudes, index=df.index, name="attitude", dtype="string")


def _time_bucket(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "unknown"
    if seconds < 60:
        return "very_low"
    if seconds < 300:
        return "low"
    if seconds < 900:
        return "medium"
    return "high"


def build_neutral_context(row: pd.Series) -> dict[str, str]:
    """Build the small, non-leaky context dict that goes into the prompt."""

    def _clean(value: object) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "Belirtilmemiş"
        text = str(value).strip()
        if not text or text.lower() == "select":
            return "Belirtilmemiş"
        return text

    return {
        "lead_source": _clean(row.get("Lead Source")),
        "specialization": _clean(row.get("Specialization")),
        "occupation": _clean(row.get("What is your current occupation")),
        "last_activity": _clean(row.get("Last Activity")),
        "time_bucket": _time_bucket(row.get("Total Time Spent on Website")),
    }


# ----- Prompt construction --------------------------------------------------

_ATTITUDE_TONE_HINTS: Mapping[AttitudeClass, str] = {
    AttitudeClass.POSITIVE_ENGAGEMENT: (
        "Lead ilgili, sorular soruyor, takip/demo istiyor. Sıcak ama gerçekçi."
    ),
    AttitudeClass.OBJECTION: (
        "Lead itiraz/şüphe gösteriyor (fiyat, süre, içerik, zamanlama). "
        "Diyalog açık ama temkinli."
    ),
    AttitudeClass.DISENGAGED: (
        "Lead kısa, mesafeli, geri dönüş yapmıyor veya ilgilenmiyor gibi. "
        "Sonuç DEĞİL, mevcut etkileşimin soğukluğu yansıtılır."
    ),
    AttitudeClass.NEUTRAL: ("Bilgilendirici, ne olumlu ne olumsuz. Standart bir takip notu."),
}

_LANGUAGE_INSTRUCTION: Mapping[LanguageMode, str] = {
    LanguageMode.TR: "Notu TAMAMEN Türkçe yaz.",
    LanguageMode.EN: "Write the note entirely in English.",
    LanguageMode.MIX: (
        "Notu Türkçe-İngilizce karışık (code-switching) yaz; CRM jargonu "
        "doğal olduğu yerde İngilizce kalsın "
        "('demo', 'pricing', 'follow-up', 'pipeline' vb.), gövde Türkçe olabilir."
    ),
}

SYSTEM_PROMPT = """Sen bir B2B/edu satış temsilcisinin CRM notlarını yazıyorsun.
Görev: SADECE 1–3 cümlelik gerçekçi bir etkileşim notu üret (e-posta özeti veya
arama notu tonunda).

KESİN KURALLAR:
1. Verilen tutum sınıfının İSMİNİ ya da eş anlamlısını ASLA yazma
   (örn. 'positive_engagement', 'olumlu yaklaşım', 'objection', 'itiraz var',
   'disengaged', 'mesafeli' gibi etiket ifadeleri kullanma). Tutum YALNIZ
   içerikten/tondan anlaşılmalı.
2. Sonucu ifşa eden ifadeleri YASAK kullan:
   Türkçe: 'sözleşme imzalandı', 'müşteri oldu', 'satın aldı', 'kaydoldu',
   'abone oldu', 'iptal etti', 'vazgeçti', 'kazanıldı', 'kaybedildi'.
   İngilizce: 'closed won', 'closed lost', 'signed up', 'purchased',
   'subscribed', 'churned', 'cancelled', 'converted', 'won the deal'.
   Not KARAR ÖNCESİ etkileşimi yansıtmalı, karar SONRASI sonucu değil.
3. Lead adı/şirket adı uydurma; "Lead", "müşteri adayı", "the lead" gibi genel
   referanslar veya isimsiz aktif cümleler kullan.
4. Doğal, kısa CRM tonu. Tek paragraf. Madde işareti veya başlık kullanma.
5. Açıklama, başlık veya parantez içi not EKLEME — sadece notun kendisi.
"""

USER_PROMPT_TEMPLATE = """Tutum sınıfı (yalnız sana yönelik tonlama ipucu, NOT İÇİNDE
asla geçmeyecek): {attitude}
Tonlama ipucu: {tone_hint}

Etkileşim bağlamı (nötr, sonuç sızdırmaz):
- Lead kaynağı: {lead_source}
- Uzmanlık alanı: {specialization}
- Meslek: {occupation}
- Son aktivite: {last_activity}
- Site ziyaret süresi (kategori): {time_bucket}

Dil yönergesi: {language_instruction}

Şimdi sadece notu yaz."""


def build_prompt_messages(
    attitude: AttitudeClass,
    context: Mapping[str, str],
    language: LanguageMode,
) -> list[dict[str, str]]:
    """Build the chat messages payload for the LLM call."""

    user_msg = USER_PROMPT_TEMPLATE.format(
        attitude=attitude.value,
        tone_hint=_ATTITUDE_TONE_HINTS[attitude],
        lead_source=context["lead_source"],
        specialization=context["specialization"],
        occupation=context["occupation"],
        last_activity=context["last_activity"],
        time_bucket=context["time_bucket"],
        language_instruction=_LANGUAGE_INSTRUCTION[language],
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def generate_note(
    llm: AzureChatOpenAI,
    attitude: AttitudeClass,
    context: Mapping[str, str],
    language: LanguageMode,
) -> str:
    """Call the LLM and return a single note string. Raises on failure."""

    messages = build_prompt_messages(attitude, context, language)
    response = llm.invoke(messages)
    content = response.content
    if isinstance(content, list):  # langchain may return content blocks
        text = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    else:
        text = str(content)
    return text.strip()


# ----- Generation loop ------------------------------------------------------


def _pick_languages(
    n: int,
    mix: Mapping[LanguageMode, float],
    rng: random.Random,
) -> list[LanguageMode]:
    modes = list(mix.keys())
    weights = [mix[m] for m in modes]
    return rng.choices(modes, weights=weights, k=n)


def run_generation(
    df: pd.DataFrame,
    llm: AzureChatOpenAI,
    *,
    out_path: Path,
    seed: int = 42,
    n_samples: int | None = None,
    lang_mix: Mapping[LanguageMode, float] | None = None,
    progress: bool = True,
    concurrency: int = 1,
) -> pd.DataFrame:
    """Generate notes for ``df`` and persist them as Parquet + CSV.

    Args:
        df: Lead dataframe (must include ``Prospect ID`` and the columns used
            by :func:`assign_attitudes`).
        llm: Configured :class:`langchain_openai.AzureChatOpenAI` instance.
        out_path: Output **without** suffix; both ``.parquet`` and ``.csv``
            will be written next to each other.
        seed: RNG seed (affects attitude assignment + language sampling).
        n_samples: If given and < len(df), a stratified-by-attitude subset is
            generated. ``None`` or 0 ⇒ generate for every lead.
        lang_mix: Override the default TR/EN/Mix distribution.
        progress: Toggle tqdm.

    Returns:
        The generated dataframe (also written to disk).
    """

    if lang_mix is None:
        lang_mix = DEFAULT_LANG_MIX

    df = df.copy()
    df["__attitude"] = assign_attitudes(df, seed=seed)

    if n_samples and n_samples > 0 and n_samples < len(df):
        # Stratified subsample so all four attitude classes stay represented.
        sampled_frames: list[pd.DataFrame] = []
        for _, group in df.groupby("__attitude", group_keys=False):
            take = max(1, round(n_samples * len(group) / len(df)))
            sampled_frames.append(group.sample(n=take, random_state=seed))
        subset = pd.concat(sampled_frames, ignore_index=True)
    else:
        subset = df.reset_index(drop=True)

    py_rng = random.Random(seed)
    languages = _pick_languages(len(subset), lang_mix, py_rng)

    def _one(idx: int) -> dict[str, str]:
        row = subset.iloc[idx]
        attitude = AttitudeClass(row["__attitude"])
        language = languages[idx]
        ctx = build_neutral_context(row)
        try:
            text = generate_note(llm, attitude, ctx, language)
        except Exception as exc:
            logger.warning("LLM call failed for lead %s: %s", row.get("Prospect ID"), exc)
            text = ""
        return {
            "idx": str(idx),  # keep original ordering when we re-sort
            "lead_id": str(row["Prospect ID"]),
            "attitude": attitude.value,
            "language": language.value,
            "text": text,
        }

    results: list[dict[str, str]] = []
    if concurrency <= 1:
        iterator = range(len(subset))
        wrapped: Any = tqdm(iterator, total=len(subset), desc="LLM notes") if progress else iterator
        for i in wrapped:
            results.append(_one(i))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(_one, i) for i in range(len(subset))]
            pbar = tqdm(total=len(subset), desc="LLM notes") if progress else None
            for fut in as_completed(futures):
                results.append(fut.result())
                if pbar is not None:
                    pbar.update(1)
            if pbar is not None:
                pbar.close()
        results.sort(key=lambda r: int(r["idx"]))

    successes = sum(1 for r in results if r["text"])
    out_df = pd.DataFrame.from_records(
        [{k: v for k, v in r.items() if k != "idx"} for r in results]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path.with_suffix(".parquet")
    csv_path = out_path.with_suffix(".csv")
    out_df.to_parquet(parquet_path, index=False)
    out_df.to_csv(csv_path, index=False)

    failure_rate = 1 - successes / max(1, len(subset))
    logger.info(
        "wrote %d notes to %s (failure rate %.2f%%)",
        len(out_df),
        parquet_path,
        failure_rate * 100,
    )
    return out_df


__all__ = [
    "DEFAULT_LANG_MIX",
    "_LEAKAGE_SUSPECT_COLUMNS",
    "AttitudeClass",
    "LanguageMode",
    "assign_attitudes",
    "build_neutral_context",
    "build_prompt_messages",
    "generate_note",
    "run_generation",
]
