import sys
import importlib
from pathlib import Path
from functools import wraps

def _get_swapped_wrapper(sibling_path_str: str, module_name: str, target_func_name: str):
    sibling_path = Path(sibling_path_str).resolve()
    
    # Backup current 'src'
    backup = {}
    for k in list(sys.modules.keys()):
        if k == 'src' or k.startswith('src.'):
            backup[k] = sys.modules.pop(k)
            
    sys.path.insert(0, str(sibling_path))
    
    try:
        mod = importlib.import_module(module_name)
        target_func = getattr(mod, target_func_name)
        
        # Capture the sibling's 'src' modules
        sibling_mods = {}
        for k in list(sys.modules.keys()):
            if k == 'src' or k.startswith('src.'):
                sibling_mods[k] = sys.modules.pop(k)
    finally:
        sys.path.pop(0)
        
        # Restore original 'src'
        for k, v in backup.items():
            sys.modules[k] = v

    @wraps(target_func)
    def wrapper(*args, **kwargs):
        # Backup main 'src'
        main_backup = {}
        for k in list(sys.modules.keys()):
            if k == 'src' or k.startswith('src.'):
                main_backup[k] = sys.modules.pop(k)
                
        # Inject sibling's 'src'
        for k, v in sibling_mods.items():
            sys.modules[k] = v
            
        sys.path.insert(0, str(sibling_path))
        try:
            return target_func(*args, **kwargs)
        finally:
            sys.path.pop(0)
            
            # Save any newly imported sibling 'src' modules
            for k in list(sys.modules.keys()):
                if k == 'src' or k.startswith('src.'):
                    sibling_mods[k] = sys.modules.pop(k)
                    
            # Restore main 'src'
            for k, v in main_backup.items():
                sys.modules[k] = v
                
    return wrapper

calculate_uncertainty = _get_swapped_wrapper(Path(__file__).resolve().parent.parent.parent.parent / 'Anuj_Uncertainty', 'src.uncertainty', 'calculate_uncertainty')

__all__ = ['calculate_uncertainty']

