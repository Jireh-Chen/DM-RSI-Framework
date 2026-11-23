"""dpolicy: Decision Policy RSI for a trained decision model.

A decision policy turns the model's scores into decisions: calibration -> threshold rule ->
optional abstain band -> optional lightweight second-stage model. It never touches the model
weights and never reads training curves. See ../README.md and ../SKILL.md.
"""

__version__ = "1.2.0"   # Skill 1.2 (history snapshot, see ../../../VERSIONS.md)
