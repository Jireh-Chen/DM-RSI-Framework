"""dpolicy: Decision Policy RSI for a trained decision model.

A decision policy turns the model's scores into decisions: calibration -> threshold rule ->
optional abstain band -> optional lightweight second-stage model. It never touches the model
weights and never reads training curves. See ../README.md and ../SKILL.md.
"""

__version__ = "4.0.0"   # Skill v4 (current); older versions in ../history/, see ../VERSIONS.md
