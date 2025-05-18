import os
import json
import shutil
from datetime import datetime
import hashlib

class ModelVersioning:
    def __init__(self, models_dir='data/models'):
        self.models_dir = models_dir
        self.ensure_models_dir()
        
    def ensure_models_dir(self):
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
    def save_model_version(self, model_path, metadata=None):
        """Save a new version of the model with metadata"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Calculate model hash
        with open(model_path, 'rb') as f:
            model_hash = hashlib.sha256(f.read()).hexdigest()
            
        # Create version directory
        version_dir = os.path.join(self.models_dir, f'version_{timestamp}')
        os.makedirs(version_dir)
        
        # Copy model file
        new_model_path = os.path.join(version_dir, 'model.joblib')
        shutil.copy2(model_path, new_model_path)
        
        # Save metadata
        metadata = metadata or {}
        metadata.update({
            'timestamp': timestamp,
            'model_hash': model_hash,
            'original_path': model_path
        })
        
        metadata_path = os.path.join(version_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        return {
            'version': timestamp,
            'model_path': new_model_path,
            'metadata_path': metadata_path
        }
        
    def get_model_version(self, version):
        """Get a specific model version"""
        version_dir = os.path.join(self.models_dir, f'version_{version}')
        if not os.path.exists(version_dir):
            return None
            
        model_path = os.path.join(version_dir, 'model.joblib')
        metadata_path = os.path.join(version_dir, 'metadata.json')
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        return {
            'version': version,
            'model_path': model_path,
            'metadata': metadata
        }
        
    def list_versions(self):
        """List all model versions"""
        versions = []
        for item in os.listdir(self.models_dir):
            if item.startswith('version_'):
                version = item.replace('version_', '')
                version_info = self.get_model_version(version)
                if version_info:
                    versions.append(version_info)
        return sorted(versions, key=lambda x: x['version'], reverse=True)
        
    def get_latest_version(self):
        """Get the latest model version"""
        versions = self.list_versions()
        return versions[0] if versions else None
        
    def compare_versions(self, version1, version2):
        """Compare two model versions"""
        v1_info = self.get_model_version(version1)
        v2_info = self.get_model_version(version2)
        
        if not v1_info or not v2_info:
            return None
            
        return {
            'version1': v1_info,
            'version2': v2_info,
            'time_diff': (datetime.strptime(v2_info['version'], '%Y%m%d_%H%M%S') - 
                         datetime.strptime(v1_info['version'], '%Y%m%d_%H%M%S')).total_seconds()
        } 