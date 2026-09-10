"""
Author: Camila Cristiano-Romero
Contact: ccristiano@bcamath.org / ccristiano001@ikasle.ehu.eus
License: MIT License
"""

def predict_Lz_Binary(output):
    return (output > 0).long()

def predict_logits(output):
    return output.argmax(dim=1)

def predict_fidelity(output):
    return output.argmax(dim=1)

