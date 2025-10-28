"""
Run management for storing checkpoints and artifacts.
"""

import os
import json
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class RunMetadata:
    """Metadata for a training run."""
    run_id: str
    start_time: str
    end_time: Optional[str] = None
    status: str = "running"  # "running", "completed", "failed"
    config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class RunManager:
    """Manager for training runs and checkpoints."""
    
    def __init__(self, runs_dir: str):
        """
        Initialize the run manager.
        
        Args:
            runs_dir: Directory to store runs
        """
        self.runs_dir = runs_dir
        os.makedirs(runs_dir, exist_ok=True)
        
        # Load existing runs
        self.runs = self._load_runs()
        
        logger.info(f"Initialized run manager with {len(self.runs)} runs")
    
    def _load_runs(self) -> Dict[str, RunMetadata]:
        """Load existing runs from disk."""
        runs_file = os.path.join(self.runs_dir, "runs.json")
        if not os.path.exists(runs_file):
            return {}
        
        with open(runs_file, 'r') as f:
            data = json.load(f)
        
        runs = {}
        for run_id, run_data in data.items():
            runs[run_id] = RunMetadata(**run_data)
        
        return runs
    
    def _save_runs(self):
        """Save runs to disk."""
        runs_file = os.path.join(self.runs_dir, "runs.json")
        with open(runs_file, 'w') as f:
            data = {run_id: asdict(run) for run_id, run in self.runs.items()}
            json.dump(data, f, indent=2)
    
    def create_run(
        self,
        run_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ) -> str:
        """
        Create a new training run.
        
        Args:
            run_id: Optional custom run ID
            config: Configuration for the run
            notes: Optional notes about the run
            
        Returns:
            The run ID
        """
        if run_id is None:
            run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if run_id in self.runs:
            raise ValueError(f"Run {run_id} already exists")
        
        # Create run directory
        run_dir = os.path.join(self.runs_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Create metadata
        metadata = RunMetadata(
            run_id=run_id,
            start_time=datetime.now().isoformat(),
            config=config,
            notes=notes
        )
        
        self.runs[run_id] = metadata
        self._save_runs()
        
        logger.info(f"Created run {run_id}")
        return run_id
    
    def get_run(self, run_id: str) -> Optional[RunMetadata]:
        """Get run metadata by ID."""
        return self.runs.get(run_id)
    
    def list_runs(self) -> List[str]:
        """List all run IDs."""
        return list(self.runs.keys())
    
    def update_run(
        self,
        run_id: str,
        status: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None
    ):
        """
        Update run metadata.
        
        Args:
            run_id: Run ID to update
            status: New status
            metrics: New metrics
            notes: New notes
        """
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        run = self.runs[run_id]
        
        if status is not None:
            run.status = status
        
        if metrics is not None:
            run.metrics = metrics
        
        if notes is not None:
            run.notes = notes
        
        if status == "completed":
            run.end_time = datetime.now().isoformat()
        
        self._save_runs()
        logger.info(f"Updated run {run_id}")
    
    def get_run_dir(self, run_id: str) -> str:
        """Get the directory for a run."""
        return os.path.join(self.runs_dir, run_id)
    
    def save_checkpoint(
        self,
        run_id: str,
        checkpoint_name: str,
        model_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a model checkpoint.
        
        Args:
            run_id: Run ID
            checkpoint_name: Name for the checkpoint
            model_path: Path to the model to save
            metadata: Optional metadata for the checkpoint
            
        Returns:
            Path to the saved checkpoint
        """
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        run_dir = self.get_run_dir(run_id)
        checkpoint_dir = os.path.join(run_dir, "checkpoints", checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Copy model files
        if os.path.isdir(model_path):
            shutil.copytree(model_path, checkpoint_dir, dirs_exist_ok=True)
        else:
            shutil.copy2(model_path, checkpoint_dir)
        
        # Save checkpoint metadata
        checkpoint_metadata = {
            'checkpoint_name': checkpoint_name,
            'model_path': model_path,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(os.path.join(checkpoint_dir, "metadata.json"), 'w') as f:
            json.dump(checkpoint_metadata, f, indent=2)
        
        logger.info(f"Saved checkpoint {checkpoint_name} for run {run_id}")
        return checkpoint_dir
    
    def load_checkpoint(
        self,
        run_id: str,
        checkpoint_name: str
    ) -> str:
        """
        Load a checkpoint path.
        
        Args:
            run_id: Run ID
            checkpoint_name: Checkpoint name
            
        Returns:
            Path to the checkpoint
        """
        run_dir = self.get_run_dir(run_id)
        checkpoint_dir = os.path.join(run_dir, "checkpoints", checkpoint_name)
        
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint {checkpoint_name} not found for run {run_id}")
        
        return checkpoint_dir
    
    def list_checkpoints(self, run_id: str) -> List[str]:
        """List all checkpoints for a run."""
        run_dir = self.get_run_dir(run_id)
        checkpoints_dir = os.path.join(run_dir, "checkpoints")
        
        if not os.path.exists(checkpoints_dir):
            return []
        
        return os.listdir(checkpoints_dir)
    
    def save_artifacts(
        self,
        run_id: str,
        artifacts: Dict[str, str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Save artifacts for a run.
        
        Args:
            run_id: Run ID
            artifacts: Dictionary mapping artifact names to file paths
            metadata: Optional metadata for the artifacts
        """
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        run_dir = self.get_run_dir(run_id)
        artifacts_dir = os.path.join(run_dir, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)
        
        # Copy artifacts
        for artifact_name, source_path in artifacts.items():
            dest_path = os.path.join(artifacts_dir, artifact_name)
            if os.path.isdir(source_path):
                shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copy2(source_path, dest_path)
        
        # Save artifacts metadata
        artifacts_metadata = {
            'artifacts': artifacts,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        with open(os.path.join(artifacts_dir, "metadata.json"), 'w') as f:
            json.dump(artifacts_metadata, f, indent=2)
        
        logger.info(f"Saved artifacts for run {run_id}")
    
    def get_artifacts(self, run_id: str) -> List[str]:
        """Get list of artifacts for a run."""
        run_dir = self.get_run_dir(run_id)
        artifacts_dir = os.path.join(run_dir, "artifacts")
        
        if not os.path.exists(artifacts_dir):
            return []
        
        return os.listdir(artifacts_dir)
    
    def save_metrics(
        self,
        run_id: str,
        metrics: Dict[str, Any],
        step: Optional[int] = None
    ):
        """
        Save metrics for a run.
        
        Args:
            run_id: Run ID
            metrics: Dictionary of metrics
            step: Optional step number
        """
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        run_dir = self.get_run_dir(run_id)
        metrics_file = os.path.join(run_dir, "metrics.json")
        
        # Load existing metrics
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {}
        
        # Add new metrics
        if step is not None:
            all_metrics[f"step_{step}"] = metrics
        else:
            all_metrics["latest"] = metrics
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        # Update run metadata
        self.update_run(run_id, metrics=all_metrics)
        
        logger.info(f"Saved metrics for run {run_id}")
    
    def get_metrics(self, run_id: str) -> Dict[str, Any]:
        """Get metrics for a run."""
        run_dir = self.get_run_dir(run_id)
        metrics_file = os.path.join(run_dir, "metrics.json")
        
        if not os.path.exists(metrics_file):
            return {}
        
        with open(metrics_file, 'r') as f:
            return json.load(f)
    
    def delete_run(self, run_id: str):
        """Delete a run and all its data."""
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        # Delete run directory
        run_dir = self.get_run_dir(run_id)
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        
        # Remove from runs
        del self.runs[run_id]
        self._save_runs()
        
        logger.info(f"Deleted run {run_id}")
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get a summary of a run."""
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")
        
        run = self.runs[run_id]
        run_dir = self.get_run_dir(run_id)
        
        summary = {
            'run_id': run_id,
            'status': run.status,
            'start_time': run.start_time,
            'end_time': run.end_time,
            'config': run.config,
            'notes': run.notes,
            'checkpoints': self.list_checkpoints(run_id),
            'artifacts': self.get_artifacts(run_id),
            'metrics': self.get_metrics(run_id)
        }
        
        return summary
    
    def list_runs_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all runs."""
        summaries = []
        for run_id in self.runs:
            try:
                summary = self.get_run_summary(run_id)
                summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to get summary for run {run_id}: {e}")
        
        return summaries
