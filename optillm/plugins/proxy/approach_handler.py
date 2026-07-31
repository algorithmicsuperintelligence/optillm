"""
Dynamic handler for approaches and plugins - no hardcoding.
"""
import importlib
import importlib.util
import logging
import inspect
from typing import Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class ApproachHandler:
    """Dynamically handles both approaches and plugins"""
    
    def __init__(self):
        self._approaches_cache = {}
        self._plugins_cache = {}
        self._discovered = False
    
    def handle(self, name: str, system_prompt: str, initial_query: str, 
               client, model: str, request_config: dict = None) -> Optional[Tuple[str, int]]:
        """
        Try to handle the given name as an approach or plugin.
        Returns None if not found, otherwise returns (response, tokens)
        """
        # Lazy discovery
        if not self._discovered:
            self._discover_handlers()
            self._discovered = True
        
        # Check if it's an approach
        if name in self._approaches_cache:
            logger.info(f"Routing approach '{name}' through proxy")
            handler = self._approaches_cache[name]
            return self._execute_handler(
                handler, system_prompt, initial_query, client, model, request_config
            )
        
        # Check if it's a plugin
        if name in self._plugins_cache:
            logger.info(f"Routing plugin '{name}' through proxy")
            handler = self._plugins_cache[name]
            return self._execute_handler(
                handler, system_prompt, initial_query, client, model, request_config
            )
        
        logger.debug(f"'{name}' not recognized as approach or plugin")
        return None
    
    def _discover_handlers(self):
        """Discover available approaches and plugins dynamically"""
        
        # Discover approaches
        self._discover_approaches()
        
        # Discover plugins
        self._discover_plugins()
        
        logger.info(f"Discovered {len(self._approaches_cache)} approaches, "
                   f"{len(self._plugins_cache)} plugins")
    
    def _discover_approaches(self):
        """Discover built-in approaches from optillm package"""
        approach_modules = {
            'mcts': ('optillm.mcts', 'chat_with_mcts'),
            'bon': ('optillm.bon', 'best_of_n_sampling'),
            'moa': ('optillm.moa', 'mixture_of_agents'),
            'rto': ('optillm.rto', 'round_trip_optimization'),
            'self_consistency': ('optillm.self_consistency', 'advanced_self_consistency_approach'),
            'pvg': ('optillm.pvg', 'inference_time_pv_game'),
            'z3': ('optillm.z3_solver', None),  # Special case
            'rstar': ('optillm.rstar', None),  # Special case
            'cot_reflection': ('optillm.cot_reflection', 'cot_reflection'),
            'plansearch': ('optillm.plansearch', 'plansearch'),
            'leap': ('optillm.leap', 'leap'),
            're2': ('optillm.reread', 're2_approach'),
            'cepo': ('optillm.cepo.cepo', 'cepo'),  # CEPO approach
        }
        
        for name, (module_path, func_name) in approach_modules.items():
            try:
                module = importlib.import_module(module_path)
                
                if name == 'z3':
                    # Special handling for Z3
                    solver_class = getattr(module, 'Z3SymPySolverSystem')
                    self._approaches_cache[name] = lambda s, q, c, m, **kw: \
                        solver_class(s, c, m).process_query(q)
                elif name == 'rstar':
                    # Special handling for RStar
                    rstar_class = getattr(module, 'RStar')
                    self._approaches_cache[name] = lambda s, q, c, m, **kw: \
                        rstar_class(s, c, m, **kw).solve(q)
                elif name == 'cepo':
                    # Special handling for CEPO which needs special config
                    cepo_func = getattr(module, func_name)
                    # We'll pass empty CepoConfig for now - it can be enhanced later
                    self._approaches_cache[name] = cepo_func
                else:
                    if func_name:
                        self._approaches_cache[name] = getattr(module, func_name)
                    
            except (ImportError, AttributeError) as e:
                logger.debug(f"Could not load approach '{name}': {e}")
    
    def _discover_plugins(self):
        """Discover available plugins dynamically"""
        try:
            import optillm
            import glob
            
            # Get plugin directories
            package_dir = Path(optillm.__file__).parent / 'plugins'
            
            # Find all Python files in plugins directory
            plugin_files = []
            if package_dir.exists():
                plugin_files.extend(glob.glob(str(package_dir / '*.py')))
            
            for plugin_file in plugin_files:
                if '__pycache__' in plugin_file or '__init__' in plugin_file:
                    continue
                    
                try:
                    # Extract module name
                    module_name = Path(plugin_file).stem
                    
                    # Skip self
                    if module_name == 'proxy_plugin':
                        continue
                    
                    # Import module
                    spec = importlib.util.spec_from_file_location(module_name, plugin_file)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        
                        # Check if it has required attributes
                        if hasattr(module, 'SLUG') and hasattr(module, 'run'):
                            slug = getattr(module, 'SLUG')
                            run_func = getattr(module, 'run')
                            self._plugins_cache[slug] = run_func
                            
                except Exception as e:
                    logger.debug(f"Could not load plugin from {plugin_file}: {e}")
                    
        except Exception as e:
            logger.debug(f"Error discovering plugins: {e}")
    
    def _execute_handler(self, handler, system_prompt: str, initial_query: str,
                        client, model: str, request_config: dict = None) -> Tuple[str, int]:
        """Execute a handler function with proper signature detection"""
        try:
            # Check function signature
            sig = inspect.signature(handler)
            params = sig.parameters
            
            # Build arguments based on signature
            args = [system_prompt, initial_query, client, model]
            kwargs = {}
            
            # Check if handler accepts request_config
            if 'request_config' in params:
                kwargs['request_config'] = request_config
            
            # Some handlers may accept additional kwargs
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
                # Only add safe kwargs that won't conflict
                if request_config:
                    # Filter out parameters that might conflict
                    safe_kwargs = {k: v for k, v in request_config.items() 
                                 if k not in ['model', 'messages', 'system_prompt', 'initial_query']}
                    kwargs.update(safe_kwargs)
            
            # Execute handler
            return handler(*args, **kwargs)
            
        except Exception as e:
            logger.error(f"Error executing handler: {e}")
            raise