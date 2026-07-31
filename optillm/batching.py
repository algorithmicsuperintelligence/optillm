#!/usr/bin/env python3
"""
Request Batching Module for OptILLM

This module implements automatic request batching to improve throughput
while maintaining OpenAI API compatibility. Requests are queued and
processed together when possible.

Key Features:
- Transparent batching behind OpenAI API
- Fail-fast error handling (no silent fallbacks)
- Per-model batching queues
- Configurable batch size and wait time
"""

import threading
import queue
import time
import logging
from typing import Dict, List, Any, Optional
from concurrent.futures import Future
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class BatchRequest:
    """Container for a single request in a batch"""
    request_data: Dict[str, Any]
    future: Future
    timestamp: float
    model: str
    approach: Optional[str] = None

class BatchingError(Exception):
    """Raised when batch processing fails"""
    pass

class RequestBatcher:
    """
    Automatic request batching for OptILLM
    
    Collects incoming requests into batches and processes them together
    for improved throughput. Maintains separate queues per model type
    to avoid incompatible mixing.
    """
    
    def __init__(self, 
                 max_batch_size: int = 4,
                 max_wait_ms: int = 50,
                 enable_logging: bool = True):
        """
        Initialize the request batcher
        
        Args:
            max_batch_size: Maximum number of requests per batch
            max_wait_ms: Maximum time to wait for batch formation (milliseconds)
            enable_logging: Whether to log batching operations
        """
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self.enable_logging = enable_logging
        
        # Separate queues for different models/approaches
        self.queues: Dict[str, queue.Queue] = {}
        self.batch_threads: Dict[str, threading.Thread] = {}
        
        # Stats for monitoring
        self.stats = {
            'total_requests': 0,
            'total_batches': 0,
            'avg_batch_size': 0.0,
            'total_wait_time': 0.0
        }
        
        self._shutdown = False
        
        if self.enable_logging:
            logger.info(f"RequestBatcher initialized: max_batch_size={max_batch_size}, "
                       f"max_wait_ms={max_wait_ms}")
    
    def _get_request_key(self, request_data: Dict[str, Any]) -> str:
        """
        Generate key to group compatible requests
        
        Args:
            request_data: The request data dictionary
            
        Returns:
            String key for grouping compatible requests
        """
        model = request_data.get('model', 'default')
        approach = request_data.get('optillm_approach', 'none')
        
        # Stream requests cannot be batched
        if request_data.get('stream', False):
            raise BatchingError("Streaming requests cannot be batched")
        
        return f"{model}:{approach}"
    
    def _validate_batch_compatibility(self, requests: List[BatchRequest]) -> None:
        """
        Validate that all requests in batch are compatible
        
        Args:
            requests: List of batch requests
            
        Raises:
            BatchingError: If requests are not compatible
        """
        if not requests:
            return
        
        # Check model consistency
        models = set(req.model for req in requests)
        if len(models) > 1:
            raise BatchingError(f"Cannot batch different models: {models}")
        
        # Check approach consistency
        approaches = set(req.approach for req in requests)
        if len(approaches) > 1:
            raise BatchingError(f"Cannot batch different optillm approaches: {approaches}")
        
        # Check for streaming (should be caught earlier, but double-check)
        streaming = any(req.request_data.get('stream', False) for req in requests)
        if streaming:
            raise BatchingError("Cannot batch streaming requests")
    
    def _create_batch_processor(self, queue_key: str) -> None:
        """
        Create and start a batch processor thread for a specific queue
        
        Args:
            queue_key: The key identifying the queue/model type
        """
        def batch_processor():
            """Background thread that forms and processes batches"""
            if self.enable_logging:
                logger.debug(f"Batch processor started for {queue_key}")
            
            while not self._shutdown:
                try:
                    batch = []
                    queue_obj = self.queues[queue_key]
                    deadline = time.time() + self.max_wait_seconds
                    
                    # Collect requests until batch is full or timeout
                    while len(batch) < self.max_batch_size and time.time() < deadline:
                        timeout = max(0.001, deadline - time.time())  # Minimum 1ms timeout
                        try:
                            request = queue_obj.get(timeout=timeout)
                            batch.append(request)
                            
                            if self.enable_logging and len(batch) == 1:
                                logger.debug(f"Started batch formation for {queue_key}")
                                
                        except queue.Empty:
                            break
                    
                    if batch:
                        if self.enable_logging:
                            wait_time = time.time() - batch[0].timestamp
                            logger.info(f"Processing batch of {len(batch)} requests for {queue_key} "
                                       f"(waited {wait_time*1000:.1f}ms)")
                        
                        # Update stats
                        self.stats['total_batches'] += 1
                        self.stats['total_requests'] += len(batch)
                        self.stats['avg_batch_size'] = self.stats['total_requests'] / self.stats['total_batches']
                        self.stats['total_wait_time'] += sum(time.time() - req.timestamp for req in batch)
                        
                        self._process_batch(batch)
                        
                except Exception as e:
                    logger.error(f"Error in batch processor for {queue_key}: {e}")
                    # Continue processing other batches
                    
            if self.enable_logging:
                logger.debug(f"Batch processor stopped for {queue_key}")
        
        thread = threading.Thread(target=batch_processor, daemon=True)
        thread.start()
        self.batch_threads[queue_key] = thread
    
    def _process_batch(self, batch: List[BatchRequest]) -> None:
        """
        Process a batch of requests
        
        Args:
            batch: List of batch requests to process
        """
        try:
            # Validate batch compatibility
            self._validate_batch_compatibility(batch)
            
            if not hasattr(self, '_processor_func'):
                raise BatchingError("No batch processor function set")
            
            # Extract request data
            request_data_list = [req.request_data for req in batch]
            
            # Process the batch
            responses = self._processor_func(request_data_list)
            
            # Validate response count
            if len(responses) != len(batch):
                raise BatchingError(f"Processor returned {len(responses)} responses for {len(batch)} requests")
            
            # Set results
            for req, response in zip(batch, responses):
                req.future.set_result(response)
                
        except Exception as e:
            # Fail all requests in batch with the same error
            error_msg = f"Batch processing failed: {str(e)}"
            logger.error(error_msg)
            
            for req in batch:
                req.future.set_exception(BatchingError(error_msg))
    
    def set_processor(self, processor_func):
        """
        Set the batch processing function
        
        Args:
            processor_func: Function that takes list of request data and returns list of responses
        """
        self._processor_func = processor_func
    
    def add_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a request to be batched
        
        Args:
            request_data: The request data dictionary
            
        Returns:
            The response from batch processing
            
        Raises:
            BatchingError: If request cannot be processed
        """
        try:
            # Generate key for request grouping
            queue_key = self._get_request_key(request_data)
            
            # Create queue and processor if needed
            if queue_key not in self.queues:
                self.queues[queue_key] = queue.Queue()
                self._create_batch_processor(queue_key)
            
            # Create batch request
            future = Future()
            batch_request = BatchRequest(
                request_data=request_data,
                future=future,
                timestamp=time.time(),
                model=request_data.get('model', 'default'),
                approach=request_data.get('optillm_approach')
            )
            
            # Add to appropriate queue
            self.queues[queue_key].put(batch_request)
            
            if self.enable_logging:
                logger.debug(f"Added request to batch queue {queue_key}")
            
            # Wait for result
            return future.result()
            
        except Exception as e:
            raise BatchingError(f"Failed to process request: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        return self.stats.copy()
    
    def shutdown(self):
        """Shutdown the batcher and all background threads"""
        self._shutdown = True
        if self.enable_logging:
            logger.info("RequestBatcher shutting down...")
        
        # Wait for threads to finish
        for thread in self.batch_threads.values():
            thread.join(timeout=1.0)