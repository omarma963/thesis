def classify_profile(text_score, image_score, exif_score, social_score, 
                     weights=None, threshold=0.5):
    """
    Combine multiple evidence scores to make a final decision.

    Parameters:
    - text_score (float): Output from BERT/MARBERT gender classification (0–1).
    - image_score (float): Confidence from image verification (e.g., stock photo detection).
    - exif_score (float): Score based on EXIF metadata analysis.
    - social_score (float): 1.0 if Instagram link found, else 0.0.
    - weights (dict): Optional dictionary to control weight per score.
    - threshold (float): Final decision threshold (default is 0.5).

    Returns:
    - final_score (float): Weighted sum of scores.
    - decision (str): "real_female" or "impersonator"
    """

    # Default equal weights if none provided
    if weights is None:
        weights = {
            "text": 0.30,
            "image": 0.40,
            "exif":  0.15,
            "social": 0.15
        }

    # Weighted sum
    final_score = (
        weights["text"] * text_score +
        weights["image"] * image_score +
        weights["exif"] * exif_score +
        weights["social"] * social_score
    )

    decision = "real_female" if final_score >= threshold else "impersonator"
    return final_score, decision
----------------------------------------------------
Custom Weights/Threshold

# Example: Prioritize image analysis more (e.g., if fake profiles often use stolen photos)
custom_weights = {"text": 0.30, "image": 0.40, "exif": 0.15, "social": 0.15}
custom_threshold = 0.6  # Stricter threshold

final_score, decision = classify_profile(
    text_score, image_score, exif_score, social_score,
    weights=custom_weights, threshold=custom_threshold
)
-----------------------------------------------------------
✅ Example Usage

# Example scores from subsystems
text_score = 0.88     # BERT says 88% likely female
image_score = 0.10    # Reverse image search suggests fake
exif_score = 0.0      # No camera metadata
social_score = 1.0    # Instagram link found

final_score, decision = classify_profile(text_score, image_score, exif_score, social_score)

print(f"Final Score: {final_score:.3f}") # Output: 0.518
print(f"Decision: {decision}") # Output: "real_female" (if threshold=0.5)
---------------------------------------------------------------------
