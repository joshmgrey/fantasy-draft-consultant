"""The analysis service's own database.

This schema is owned solely by the analysis service. Nothing in the core app
imports it, and it shares no tables or models with the core database. It is an
audit log of completed analyses (and a foundation for future caching /
idempotency).

A write failure here never fails an analysis — see api._record.
"""

from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AnalysisRecord(db.Model):
    __tablename__ = "analysis_record"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=_utcnow)
    requested_by = db.Column(db.String(64), nullable=False, index=True)  # actor token 'sub'
    request_id = db.Column(db.String(64), nullable=True)
    player_name = db.Column(db.String(64), nullable=False)
    model = db.Column(db.String(64), nullable=True)
    risk_score = db.Column(db.Integer, nullable=True)
    verdict = db.Column(db.String(8), nullable=True)
    reason = db.Column(db.Text, nullable=True)
    latency_ms = db.Column(db.Integer, nullable=True)
    injection_flagged = db.Column(db.Boolean, nullable=False, default=False)

    @property
    def public_id(self) -> str:
        return f"an_{self.id}"

    def to_history_item(self) -> dict:
        return {
            "analysis_id": self.public_id,
            "created_at": self.created_at.isoformat(),
            "player": self.player_name,
            "risk_score": self.risk_score,
            "verdict": self.verdict,
            "reason": self.reason,
            "model": self.model,
            "injection_flagged": self.injection_flagged,
        }
