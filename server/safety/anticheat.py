import logging
from ..schemas import Action

logger = logging.getLogger(__name__)

class AntiCheat:
    """Anti-cheat system for action validation."""

    def __init__(self):
        self.recent_submissions = []

    def validate_action(self, action: Action) -> bool:
        """Validate action against anti-cheat rules.

        Args:
            action: Action to validate.

        Returns:
            True if valid.
        """
        # Deduplication
        action_str = str(action.model_dump())
        if action_str in self.recent_submissions:
            self.log_rejection("Duplicate submission")
            return False
        self.recent_submissions.append(action_str)
        if len(self.recent_submissions) > 10:
            self.recent_submissions.pop(0)
        # Format validators
        if not action.garment_ids:
            self.log_rejection("No garments selected")
            return False
        return True

    def log_rejection(self, reason: str):
        """Log rejection reason.

        Args:
            reason: Reason for rejection.
        """
        logger.warning(f"Action rejected: {reason}")