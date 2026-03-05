"""
Pipeline stages
"""

# Use lazy imports to avoid loading dependencies when not needed
def __getattr__(name):
    if name == 'embed_stage':
        from .embed import embed_stage
        return embed_stage
    elif name == 'evaluate_stage':
        from .evaluate import evaluate_stage
        return evaluate_stage
    elif name == 'llm_judge_stage':
        from .llm_judge import llm_judge_stage
        return llm_judge_stage
    elif name == 'visualize_stage':
        from .visualize import visualize_stage
        return visualize_stage
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'embed_stage',
    'evaluate_stage',
    'llm_judge_stage',
    'visualize_stage'
]
