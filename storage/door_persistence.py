"""Door configuration persistence for production use."""

import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import threading

class DoorPersistence:
    """Handles saving and loading door configurations."""
    
    def __init__(self, storage_dir: str = "storage/doors"):
        """Initialize door persistence."""
        self.storage_dir = storage_dir
        self.config_file = os.path.join(storage_dir, "door_config.json")
        self.state_file = os.path.join(storage_dir, "door_states.json")
        self.history_file = os.path.join(storage_dir, "door_history.json")
        
        # Create storage directory
        os.makedirs(storage_dir, exist_ok=True)
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Load existing configurations
        self.door_configs = self.load_configs()
        self.door_states = self.load_states()
        
    def load_configs(self) -> Dict[str, Any]:
        """Load saved door configurations."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading door configs: {e}")
        return {}
        
    def save_configs(self, configs: Dict[str, Any]) -> bool:
        """Save door configurations to disk."""
        with self.lock:
            try:
                # Add metadata
                save_data = {
                    'version': '1.0',
                    'updated': datetime.now().isoformat(),
                    'doors': configs
                }
                
                # Write to temp file first (atomic operation)
                temp_file = self.config_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                
                # Move temp file to actual file
                os.replace(temp_file, self.config_file)
                
                self.door_configs = configs
                return True
                
            except Exception as e:
                print(f"Error saving door configs: {e}")
                return False
                
    def load_states(self) -> Dict[str, Any]:
        """Load saved door states."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading door states: {e}")
        return {}
        
    def save_state(self, door_id: str, state: Dict[str, Any]) -> bool:
        """Save individual door state."""
        with self.lock:
            try:
                # Load current states
                states = self.load_states()
                
                # Update state
                states[door_id] = {
                    **state,
                    'last_updated': datetime.now().isoformat()
                }
                
                # Save to disk
                temp_file = self.state_file + '.tmp'
                with open(temp_file, 'w') as f:
                    json.dump(states, f, indent=2)
                
                os.replace(temp_file, self.state_file)
                
                self.door_states = states
                return True
                
            except Exception as e:
                print(f"Error saving door state: {e}")
                return False
                
    def add_door(self, door_id: str, bbox: tuple, confidence: float, 
                 zone: Optional[str] = None) -> bool:
        """Add or update a door configuration."""
        config = {
            'id': door_id,
            'bbox': list(bbox),  # Convert tuple to list for JSON
            'confidence': confidence,
            'zone': zone,
            'created': datetime.now().isoformat(),
            'active': True
        }
        
        configs = self.door_configs.copy()
        configs[door_id] = config
        
        return self.save_configs(configs)
        
    def remove_door(self, door_id: str) -> bool:
        """Remove a door configuration."""
        configs = self.door_configs.copy()
        if door_id in configs:
            del configs[door_id]
            return self.save_configs(configs)
        return False
        
    def get_door_config(self, door_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific door."""
        return self.door_configs.get(door_id)
        
    def get_all_doors(self) -> Dict[str, Any]:
        """Get all door configurations."""
        if 'doors' in self.door_configs:
            return self.door_configs['doors']
        return self.door_configs
        
    def update_door_state(self, door_id: str, is_open: bool, confidence: float) -> bool:
        """Update door open/closed state."""
        state = {
            'is_open': is_open,
            'state': 'open' if is_open else 'closed',
            'confidence': confidence
        }
        
        # Save to history
        self.add_to_history(door_id, state)
        
        # Save current state
        return self.save_state(door_id, state)
        
    def add_to_history(self, door_id: str, state: Dict[str, Any]):
        """Add door state change to history."""
        with self.lock:
            try:
                # Load history
                history = []
                if os.path.exists(self.history_file):
                    try:
                        with open(self.history_file, 'r') as f:
                            history = json.load(f)
                    except:
                        history = []
                
                # Add new entry
                entry = {
                    'door_id': door_id,
                    'timestamp': datetime.now().isoformat(),
                    **state
                }
                history.append(entry)
                
                # Keep only last 1000 entries
                history = history[-1000:]
                
                # Save history
                with open(self.history_file, 'w') as f:
                    json.dump(history, f, indent=2)
                    
            except Exception as e:
                print(f"Error saving door history: {e}")
                
    def get_door_history(self, door_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get history for a specific door."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    history = json.load(f)
                    
                # Filter by door_id
                door_history = [h for h in history if h.get('door_id') == door_id]
                return door_history[-limit:]
                
            except Exception as e:
                print(f"Error loading door history: {e}")
                
        return []
        
    def clear_inactive_doors(self):
        """Remove doors marked as inactive."""
        configs = self.door_configs.copy()
        active_configs = {k: v for k, v in configs.items() 
                         if v.get('active', True)}
        
        if len(active_configs) != len(configs):
            self.save_configs(active_configs)