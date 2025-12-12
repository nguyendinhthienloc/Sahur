"""
POS tagging tools.
"""
from typing import Dict
import numpy as np

def extract_dependency_depths(doc) -> Dict[str, float]:
    """
    Extract dependency tree depth percentiles.
    
    Returns 10th, 50th (median), 90th percentile of all token depths.
    """
    depths = []
    for token in doc:
        depth = 0
        current = token
        while current.head != current:
            depth += 1
            current = current.head
        depths.append(depth)
    
    if not depths:
        return {'depth_p10': 0, 'depth_p50': 0, 'depth_p90': 0}
    
    return {
        'depth_p10': np.percentile(depths, 10),
        'depth_p50': np.percentile(depths, 50),
        'depth_p90': np.percentile(depths, 90)
    }
