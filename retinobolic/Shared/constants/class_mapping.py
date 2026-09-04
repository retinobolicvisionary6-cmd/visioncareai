"""
Shared Class Mapping Constants

WARNING: 
Currently, the Vinayak DR model is trained as a 4-class model:
0 = No DR
1 = Mild DR
2 = Moderate DR
3 = Severe/PDR

However, SIH26038 requires alignment with the International Clinical DR scale (0-4):
0 = No DR
1 = Mild NPDR
2 = Moderate NPDR
3 = Severe NPDR
4 = PDR

Action Required: The model needs revision before final SIH compliance.
For now, the integration supports the 4-class mapping.
"""

CURRENT_MODEL_CLASSES = 5

MODEL_CLASS_MAPPING = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}

TARGET_SIH_CLASS_MAPPING = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "PDR"
}
