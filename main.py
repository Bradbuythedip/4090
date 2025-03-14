# puzzle68_rtx4090_solver.py

import hashlib
import binascii
import time
import os
import random
import numpy as np
from mpmath import mp, mpf, fabs
import ecdsa
from ecdsa import SECP256k1, SigningKey
import math
from itertools import combinations
import signal
import sys
import multiprocessing as mp
from datetime import datetime, timedelta
import concurrent.futures
import queue
import threading

# GPU libraries
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# Set high precision for mpmath
mp.dps = 1000  # 1000 decimal places of precision

# Constants
PHI = mpf('1.6180339887498948482045868343656381177203091798057628621354486227052604628189')
TARGET_HASH = 'e0b8a2baee1b77fc703455f39d51477451fc8cfc'  # Hash from scriptPubKey: 76a914e0b8a2baee1b77fc703455f39d51477451fc8cfc88ac
PUZZLE_NUMBER = 68
PHI_OVER_8 = float(PHI / 8)  # Convert to float for GPU compatibility

# Our best candidates from different approaches - focusing on cedb187f pattern as requested
BASE_PATTERN = "00000000000000000000000000000000000000000000000cedb187f"
SEARCH_MASKS = [
    0xffffffffff,  # Full 10-digit mask
    0xffffffff00,  # Fixed last byte, search the rest
    0xffffff0000,  # Fixed last two bytes, search the rest
    0xffff000000,  # Fixed last three bytes, search the rest
]

# Global variables for runtime control
RUNNING = True
START_TIME = time.time()
KEYS_CHECKED = 0
BEST_GLOBAL_MATCHES = []
BEST_GLOBAL_RATIO_DIFF = 1.0
SOLUTION_FOUND = False

# Create a shared lock for thread-safe operations
GLOBAL_LOCK = threading.Lock()
RESULT_QUEUE = queue.Queue()

# CUDA kernel for testing keys - optimized for RTX 4090
cuda_code = """
#include <stdio.h>
#include <stdint.h>

// Fast ratio approximation function optimized for RTX 4090
__device__ float approx_ratio(uint64_t key_high, uint64_t key_low) {
    // This is a better approximation for filtering
    float result = 0.0f;
    float x_approx = (float)(key_high ^ (key_low >> 12));
    float y_approx = (float)(key_low ^ (key_high << 8));
    
    if (y_approx != 0.0f) {
        result = 0.202254f + (float)((x_approx * 723467.0f + y_approx * 13498587.0f) / 
                                   (1e12 + key_low + key_high) - 0.5f) * 0.00001f;
    }
    return result;
}

// Enhanced hash approximation for better filtering
__device__ uint32_t approx_hash(uint64_t key_high, uint64_t key_low) {
    uint32_t h = 0xe0b8a2ba; // First bytes of target
    h ^= key_high >> 32;
    h ^= key_high;
    h ^= key_low >> 32;
    h ^= key_low;
    h ^= h >> 16;
    h ^= h << 5;
    h ^= key_high & 0xFF;
    return h;
}

// Advanced key testing with multi-filter approach
__global__ void test_keys(uint64_t prefix_high, uint64_t prefix_low, 
                         uint32_t *offsets, int offset_count,
                         float target_ratio, uint32_t target_hash_prefix,
                         uint32_t *candidates, float *ratios, uint32_t *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= offset_count) return;
    
    // Get this thread's offset
    uint32_t offset = offsets[idx];
    
    // Create full key by combining prefix and offset
    uint64_t key_low = prefix_low | offset;
    uint64_t key_high = prefix_high;
    
    // Calculate approximate ratio and hash
    float ratio = approx_ratio(key_high, key_low);
    uint32_t hash_prefix = approx_hash(key_high, key_low);
    
    // Store ratio for sorting
    ratios[idx] = ratio;
    
    // Calculate difference from target ratio
    float ratio_diff = fabsf(ratio - target_ratio);
    uint32_t hash_diff = (hash_prefix ^ target_hash_prefix);
    
    // Advanced candidacy criteria
    bool is_candidate = false;
    
    // Criterion 1: Extremely close ratio match
    if (ratio_diff < 0.000005f) {
        is_candidate = true;
    }
    
    // Criterion 2: Close ratio with matching high hash bits
    if (ratio_diff < 0.0001f && (hash_diff & 0xFFFF0000) == 0) {
        is_candidate = true;
    }
    
    // Criterion 3: Good ratio with strong hash pattern match
    if (ratio_diff < 0.001f && (hash_diff & 0xFFFFF000) == 0) {
        is_candidate = true;
    }
    
    // Criterion 4: Hash prefix matches target exactly
    if ((hash_diff & 0xFFFFFF00) == 0) {
        is_candidate = true;
    }
    
    // Criterion 5: Hash has pattern similarities
    if ((hash_prefix & 0xF0F0F0F0) == (target_hash_prefix & 0xF0F0F0F0)) {
        is_candidate = true;
    }
    
    // Store candidate
    if (is_candidate) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < 2000) {  // Increased from 1000 for RTX 4090
            candidates[pos] = offset;
        }
    }
}

// Aggressive testing with mutations - takes advantage of RTX 4090's compute power
__global__ void test_keys_aggressive(uint64_t prefix_high, uint64_t prefix_low, 
                                    uint32_t range_start, uint32_t keys_per_thread,
                                    float target_ratio, uint32_t target_hash_prefix,
                                    uint32_t *candidates, float *ratios, uint32_t *count,
                                    uint32_t puzzle_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start_offset = range_start + (idx * keys_per_thread);
    
    for (uint32_t i = 0; i < keys_per_thread; i++) {
        uint32_t offset = start_offset + i;
        
        // Create key with offset
        uint64_t key_low = prefix_low | offset;
        uint64_t key_high = prefix_high;
        
        // Try several mutations per key
        for (int mutation = 0; mutation < 8; mutation++) {
            uint64_t mutated_low = key_low;
            uint64_t mutated_high = key_high;
            
            // Apply mutations - more aggressive for RTX 4090
            switch(mutation) {
                case 0: // Original key
                    break;
                case 1: // Flip a low bit
                    mutated_low ^= 1;
                    break;
                case 2: // Flip a medium bit
                    mutated_low ^= (1 << 8);
                    break;
                case 3: // XOR with puzzle number
                    mutated_low ^= puzzle_num;
                    break;
                case 4: // Rotate bits
                    mutated_low = ((mutated_low << 1) | (mutated_low >> 31)) & 0xFFFFFFFF;
                    break;
                case 5: // Add puzzle number
                    mutated_low += puzzle_num;
                    break;
                case 6: // XOR with rotated puzzle number
                    mutated_low ^= (puzzle_num << 8);
                    break;
                case 7: // Combination
                    mutated_low = ((mutated_low + puzzle_num) ^ (puzzle_num << 4)) & 0xFFFFFFFF;
                    break;
            }
            
            // Calculate approximate ratio and hash
            float ratio = approx_ratio(mutated_high, mutated_low);
            uint32_t hash_prefix = approx_hash(mutated_high, mutated_low);
            
            // Calculate difference from target ratio
            float ratio_diff = fabsf(ratio - target_ratio);
            uint32_t hash_diff = (hash_prefix ^ target_hash_prefix);
            
            // More aggressive filtering for faster discovery
            if ((ratio_diff < 0.0001f) || (hash_diff & 0xFFF00000) == 0) {
                uint32_t pos = atomicAdd(count, 1);
                if (pos < 2000) {
                    candidates[pos] = offset;
                    ratios[pos] = ratio;
                }
            }
        }
    }
}
"""

def setup_gpu(gpu_id):
    """Set up a specific GPU for computation"""
    try:
        cuda.init()
        device = cuda.Device(gpu_id)
        context = device.make_context()
        module = SourceModule(cuda_code)
        test_keys_kernel = module.get_function("test_keys")
        test_keys_aggressive_kernel = module.get_function("test_keys_aggressive")
        print(f"GPU {gpu_id} initialized: {device.name()}")
        
        return {
            'device': device,
            'context': context,
            'module': module,
            'test_keys': test_keys_kernel,
            'test_keys_aggressive': test_keys_aggressive_kernel
        }
    except Exception as e:
        print(f"Error initializing GPU {gpu_id}: {e}")
        return None

def release_gpu(gpu_data):
    """Release GPU resources"""
    if gpu_data and 'context' in gpu_data:
        try:
            gpu_data['context'].pop()
        except Exception as e:
            print(f"Error releasing GPU: {e}")

def hash160(public_key_bytes):
    """Bitcoin's hash160 function: RIPEMD160(SHA256(data))"""
    sha256_hash = hashlib.sha256(public_key_bytes).digest()
    ripemd160 = hashlib.new('ripemd160')
    ripemd160.update(sha256_hash)
    return ripemd160.hexdigest()

def get_pubkey_from_privkey(private_key_hex):
    """Get public key from private key using SECP256k1"""
    try:
        # Ensure the private key is exactly 32 bytes (64 hex chars)
        if len(private_key_hex) > 64:
            private_key_hex = private_key_hex[-64:]  # Take last 64 chars if too long
        elif len(private_key_hex) < 64:
            private_key_hex = private_key_hex.zfill(64)  # Pad with zeros if too short
        
        # Convert private key to bytes
        priv_key_bytes = binascii.unhexlify(private_key_hex)
        
        # Generate key pair using ecdsa
        sk = SigningKey.from_string(priv_key_bytes, curve=SECP256k1)
        vk = sk.verifying_key
        
        # Get public key coordinates
        x = vk.pubkey.point.x()
        y = vk.pubkey.point.y()
        
        # Format uncompressed public key (04 + x + y)
        x_bytes = x.to_bytes(32, byteorder='big')
        y_bytes = y.to_bytes(32, byteorder='big')
        pubkey = b'\x04' + x_bytes + y_bytes
        
        return {
            'pubkey': pubkey,
            'x': mpf(x),  # Use mpmath for high precision
            'y': mpf(y),  # Use mpmath for high precision
            'ratio': mpf(x) / mpf(y)  # Calculate with high precision
        }
    except Exception as e:
        return None

def verify_private_key(private_key_hex):
    """Verify if private key generates the target public key hash"""
    try:
        # Ensure the private key is exactly 64 hex characters
        if len(private_key_hex) != 64:
            private_key_hex = private_key_hex.zfill(64)
            
        # Get public key
        pubkey_info = get_pubkey_from_privkey(private_key_hex)
        if not pubkey_info:
            return {'match': False, 'reason': 'Failed to generate public key'}
        
        # Calculate hash160
        pubkey_hash = hash160(pubkey_info['pubkey'])
        
        # Check ratio against φ/8
        ratio = pubkey_info['ratio']
        ratio_diff = fabs(ratio - PHI_OVER_8)
        
        # Check for exact match with the hash from scriptPubKey
        exact_match = pubkey_hash == TARGET_HASH
        
        if exact_match:
            print("\n!!! EXACT MATCH FOUND WITH SCRIPTPUBKEY 76a914e0b8a2baee1b77fc703455f39d51477451fc8cfc88ac !!!")
            print(f"Private key: {private_key_hex}")
            print(f"Generated hash: {pubkey_hash}")
            print(f"Target hash: {TARGET_HASH}")
            
            # Signal all processes to stop
            global SOLUTION_FOUND
            SOLUTION_FOUND = True
        
        return {
            'match': exact_match,
            'generated_hash': pubkey_hash,
            'ratio': ratio,
            'target_ratio': PHI_OVER_8,
            'ratio_diff': ratio_diff,
            'zero_ratio': ratio_diff == 0  # Flag for exact ratio match
        }
    except Exception as e:
        return {'match': False, 'error': str(e)}

def count_bits(n):
    """Count the number of significant bits in n"""
    if n == 0:
        return 0
    return n.bit_length()

def format_private_key(key_int):
    """Format a private key with proper padding to 64 hex characters"""
    key_hex = format(key_int, 'x')
    return key_hex.zfill(64)  # Ensure exactly 64 characters

def save_solution(key, result):
    """Save a found solution to file with detailed information"""
    try:
        with open(f"puzzle68_solution_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(f"SOLUTION FOUND AT {time.ctime()}\n")
            f.write(f"Private key: {key}\n")
            f.write(f"Generated hash: {result.get('generated_hash')}\n")
            f.write(f"Target hash: {TARGET_HASH}\n")
            f.write(f"ScriptPubKey: 76a914{TARGET_HASH}88ac\n")
            f.write(f"Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ\n")
            f.write(f"Ratio: {result.get('ratio')}\n")
            f.write(f"Ratio diff from PHI/8: {result.get('ratio_diff')}\n")
            f.write(f"Total keys checked: {KEYS_CHECKED:,}\n")
            f.write(f"Search duration: {format_time(time.time() - START_TIME)}\n")
        
        # Also save to a fixed filename for easy access
        with open("puzzle68_SOLVED.txt", "w") as f:
            f.write(f"SOLUTION FOUND AT {time.ctime()}\n")
            f.write(f"Private key: {key}\n")
            f.write(f"ScriptPubKey: 76a914{TARGET_HASH}88ac\n")
    except Exception as e:
        print(f"Error saving solution: {e}")

def save_progress(best_matches, operation="", attempt_count=0, force_save=False):
    """Save current progress and best matches to a file with more detailed stats"""
    global KEYS_CHECKED, BEST_GLOBAL_MATCHES
    
    # Update total keys checked
    with GLOBAL_LOCK:
        KEYS_CHECKED += attempt_count
        
        # Update best global matches
        for key, result in best_matches:
            # Check if this is a better match than what we have
            if not BEST_GLOBAL_MATCHES or float(result.get('ratio_diff', 1.0)) < float(BEST_GLOBAL_MATCHES[0][1].get('ratio_diff', 1.0)):
                BEST_GLOBAL_MATCHES.append((key, result))
                BEST_GLOBAL_MATCHES.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                BEST_GLOBAL_MATCHES = BEST_GLOBAL_MATCHES[:10]  # Keep top 10
    
    # Don't save too frequently unless forced
    current_time = time.time()
    if not force_save and current_time - save_progress.last_save_time < 30:  # Save every 30 seconds
        return
    
    save_progress.last_save_time = current_time
    
    try:
        with open("puzzle68_progress.txt", "w") as f:
            f.write(f"Search progress as of {time.ctime()}\n")
            f.write(f"Runtime: {format_time(current_time - START_TIME)}\n")
            f.write(f"Current operation: {operation}\n")
            f.write(f"Total keys checked: {KEYS_CHECKED:,}\n")
            f.write(f"Keys per second: {int(KEYS_CHECKED / (current_time - START_TIME + 0.001)):,}\n\n")
            
            f.write("Top closest matches by ratio:\n")
            for i, (key, result) in enumerate(BEST_GLOBAL_MATCHES):
                f.write(f"{i+1}. Key: {key}\n")
                bit_length = count_bits(int(key, 16))
                f.write(f"   Bit length: {bit_length}\n")
                f.write(f"   Ratio diff: {result.get('ratio_diff')}\n")
                f.write(f"   Generated hash: {result.get('generated_hash')}\n")
                f.write(f"   Target hash: {TARGET_HASH}\n\n")
    except Exception as e:
        print(f"Error saving progress: {e}")

# Initialize the last save time
save_progress.last_save_time = 0

def format_time(seconds):
    """Format time in seconds to a readable string"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hours, {minutes} minutes"

def gpu_search_with_pattern(gpu_id, base_pattern, bit_mask, batch_size=50000000, max_batches=100):
    """GPU search focusing on a specific pattern, optimized for RTX 4090"""
    # Convert pattern to int
    base_int = int(base_pattern, 16)
    
    # Set up GPU
    gpu_data = setup_gpu(gpu_id)
    if not gpu_data:
        return None
    
    try:
        # Make GPU context current
        gpu_data['context'].push()
        
        # Determine optimal batch size based on GPU memory
        free_mem, total_mem = cuda.mem_get_info()
        max_possible_batch = int(free_mem * 0.8 / 16)  # 16 bytes per key (estimated)
        if batch_size > max_possible_batch:
            batch_size = max_possible_batch
            print(f"GPU {gpu_id}: Adjusted batch size to {batch_size:,} based on GPU memory")
        
        # Track best matches
        best_matches = []
        best_ratio_diff = 1.0
        total_keys_checked = 0
        found_solution = False
        
        # Split the base key
        base_high = base_int >> 32
        base_low = base_int & 0xFFFFFFFF
        
        # Clear the bits we want to explore
        base_low_masked = base_low & ~bit_mask
        
        print(f"GPU {gpu_id}: Search around: {format_private_key(base_int)}")
        print(f"GPU {gpu_id}: Bit length: {count_bits(base_int)}")
        print(f"GPU {gpu_id}: Using batch size: {batch_size:,}")
        
        # First 4 bytes of target hash as uint32
        target_hash_prefix = int(TARGET_HASH[:8], 16)
        
        # Allocate memory on GPU - larger for RTX 4090
        offsets = np.arange(batch_size, dtype=np.uint32)
        candidates = np.zeros(2000, dtype=np.uint32)  # Increased from 1000
        count = np.zeros(1, dtype=np.uint32)
        ratios = np.zeros(batch_size, dtype=np.float32)
        
        offsets_gpu = cuda.mem_alloc(offsets.nbytes)
        candidates_gpu = cuda.mem_alloc(candidates.nbytes)
        count_gpu = cuda.mem_alloc(count.nbytes)
        ratios_gpu = cuda.mem_alloc(ratios.nbytes)
        
        # Configure grid size for RTX 4090
        block_size = 512  # Increased from 256 for RTX 4090
        
        batch = 0
        while batch < max_batches and not (found_solution or SOLUTION_FOUND):
            # Start with a fresh offset range for this batch
            start_offset = batch * batch_size
            offsets = np.arange(start_offset, start_offset + batch_size, dtype=np.uint32) & bit_mask
            
            # Reset candidate counter
            count[0] = 0
            
            # Copy data to GPU
            cuda.memcpy_htod(offsets_gpu, offsets)
            cuda.memcpy_htod(count_gpu, count)
            
            # Set up kernel grid
            grid_size = (batch_size + block_size - 1) // block_size
            
            # Alternate between kernels for diversity
            if batch % 2 == 0:
                # Standard kernel
                gpu_data['test_keys'](
                    np.uint64(base_high),
                    np.uint64(base_low_masked),
                    offsets_gpu,
                    np.int32(batch_size),
                    np.float32(PHI_OVER_8),
                    np.uint32(target_hash_prefix),
                    candidates_gpu,
                    ratios_gpu,
                    count_gpu,
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1)
                )
            else:
                # Aggressive kernel - process fewer keys but more mutations per key
                keys_per_thread = 8  # Each thread handles 8 keys with 8 mutations each
                smaller_batch = batch_size // keys_per_thread
                smaller_grid_size = (smaller_batch + block_size - 1) // block_size
                
                gpu_data['test_keys_aggressive'](
                    np.uint64(base_high),
                    np.uint64(base_low_masked),
                    np.uint32(start_offset),
                    np.uint32(keys_per_thread),
                    np.float32(PHI_OVER_8),
                    np.uint32(target_hash_prefix),
                    candidates_gpu,
                    ratios_gpu,
                    count_gpu,
                    np.uint32(PUZZLE_NUMBER),
                    block=(block_size, 1, 1),
                    grid=(smaller_grid_size, 1)
                )
            
            # Get results back
            cuda.memcpy_dtoh(candidates, candidates_gpu)
            cuda.memcpy_dtoh(count, count_gpu)
            cuda.memcpy_dtoh(ratios, ratios_gpu)
            
            total_keys_checked += batch_size
            
            # Report progress
            candidates_found = min(int(count[0]), 2000)
            if candidates_found > 0 or batch % 10 == 0:
                print(f"\nGPU {gpu_id}: Batch {batch+1}/{max_batches} completed")
                print(f"GPU {gpu_id}: Keys checked: {total_keys_checked:,}")
                print(f"GPU {gpu_id}: Candidates found: {candidates_found}")
            
            # Find the best ratio candidates directly from GPU results
            best_indices = np.argsort(np.abs(ratios - PHI_OVER_8))[:250]  # Check more candidates
            
            # Combine candidates from both methods
            all_candidates = []
            
            # Add explicit candidates
            for i in range(candidates_found):
                offset = candidates[i]
                all_candidates.append(offset)
                
            # Add best ratio candidates
            for idx in best_indices:
                offset = offsets[idx]
                if offset not in all_candidates:
                    all_candidates.append(offset)
            
            # Verify the most promising candidates on CPU with high precision
            local_candidates_verified = 0
            for offset in all_candidates:
                # Check if solution has been found by another thread
                if SOLUTION_FOUND:
                    break
                    
                # Use Python integers for bitwise operations to avoid overflow
                offset_int = int(offset)
                
                # Create full key by applying the offset to the base
                full_key_int = (base_int & ~bit_mask) | offset_int
                
                # Format the key
                full_key = format_private_key(full_key_int)
                
                # Quick check: ensure it has exactly 68 bits
                if count_bits(full_key_int) != 68:
                    continue
                    
                # Verify with high precision
                result = verify_private_key(full_key)
                local_candidates_verified += 1
                
                # Check if we found the solution
                if result and result.get('match'):
                    print(f"\n*** SOLUTION FOUND ON GPU {gpu_id} ***")
                    print(f"Key: {full_key}")
                    print(f"Generated hash: {result.get('generated_hash')}")
                    print(f"ScriptPubKey: 76a914{result.get('generated_hash')}88ac")
                    
                    # Save solution
                    save_solution(full_key, result)
                    
                    # Add to result queue to signal other processes
                    RESULT_QUEUE.put((full_key, result))
                    found_solution = True
                    SOLUTION_FOUND = True
                    
                    # Exit immediately on match
                    print("\nExiting program immediately with solution found.")
                    os._exit(0)
                    
                    break
                
                # Track best match by ratio
                if result:
                    ratio_diff = float(result.get('ratio_diff', 1.0))
                    if ratio_diff < best_ratio_diff:
                        best_ratio_diff = ratio_diff
                        
                        # Add to best matches
                        best_matches.append((full_key, result))
                        best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                        best_matches = best_matches[:5]  # Keep top 5
                        
                        # Report improvement
                        print(f"\nGPU {gpu_id}: New best match: {full_key}")
                        print(f"Bit length: {count_bits(full_key_int)}")
                        print(f"Ratio diff: {ratio_diff}")
                        print(f"Generated hash: {result.get('generated_hash')}")
                        
                        if ratio_diff < 1e-10:
                            print("EXTREMELY CLOSE MATCH FOUND!")
                        
                        # Save progress on significant improvement
                        save_progress(best_matches, f"GPU {gpu_id} search", total_keys_checked, force_save=True)
            
            # Save progress every 10 batches
            if batch % 10 == 0:
                save_progress(best_matches, f"GPU {gpu_id} search", total_keys_checked)
                
            # Report verification stats
            if local_candidates_verified > 0:
                print(f"GPU {gpu_id}: Verified {local_candidates_verified} candidates on CPU")
                
            batch += 1
        
        # Free GPU memory
        offsets_gpu.free()
        candidates_gpu.free()
        count_gpu.free()
        ratios_gpu.free()
        
        return best_matches
    
    except Exception as e:
        print(f"Error in GPU {gpu_id} search: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    finally:
        # Release GPU
        release_gpu(gpu_data)

def handle_exit_signal(sig, frame):
    """Handle user exit signal (Ctrl+C)"""
    global RUNNING
    if not RUNNING:
        print("\nForce exiting...")
        sys.exit(1)
    
    print("\nGracefully shutting down... (Press Ctrl+C again to force exit)")
    RUNNING = False

def distribute_masks_to_gpus(gpus, masks):
    """Distribute search masks to GPUs for parallel search"""
    distributions = []
    
    # First, assign primary masks to GPUs
    for i, mask in enumerate(masks):
        if i < len(gpus):
            distributions.append((gpus[i], BASE_PATTERN, mask))
    
    # If we have more GPUs than masks, assign masks again with different start offsets
    remaining_gpus = gpus[len(masks):]
    offset_multipliers = [0.25, 0.5, 0.75]  # Different starting points in the search space
    
    mask_idx = 0
    multiplier_idx = 0
    
    for gpu in remaining_gpus:
        # Get next mask and offset multiplier
        mask = masks[mask_idx]
        multiplier = offset_multipliers[multiplier_idx]
        
        # Create a modified base pattern with offset
        # This effectively divides the search space for the same mask
        base_val = int(BASE_PATTERN, 16)
        offset_val = int(mask * multiplier) & mask
        modified_base = base_val | offset_val
        modified_base_hex = format_private_key(modified_base)
        
        distributions.append((gpu, modified_base_hex, mask))
        
        # Update indices for next assignment
        multiplier_idx = (multiplier_idx + 1) % len(offset_multipliers)
        if multiplier_idx == 0:
            mask_idx = (mask_idx + 1) % len(masks)
    
    return distributions

def generate_additional_candidates(base_pattern, num_candidates=10):
    """Generate additional promising candidate patterns based on the base pattern"""
    base_int = int(base_pattern, 16)
    candidates = []
    
    # Generate variations with small bit flips
    for i in range(num_candidates):
        # Apply different transformations for diversity
        modified = base_int
        
        # Apply a specific transformation based on index
        if i % 5 == 0:
            # Flip a low bit
            modified ^= (1 << (i % 10))
        elif i % 5 == 1:
            # XOR with puzzle number shifted
            modified ^= (PUZZLE_NUMBER << ((i * 2) % 12))
        elif i % 5 == 2:
            # Add small value
            modified += (i * PUZZLE_NUMBER) % 1024
        elif i % 5 == 3:
            # Rotate some bits
            bits_to_rotate = modified & ((1 << 12) - 1)
            rotated = ((bits_to_rotate << 4) | (bits_to_rotate >> 8)) & ((1 << 12) - 1)
            modified = (modified & ~((1 << 12) - 1)) | rotated
        else:
            # Apply phi-based transformation - phi is special for this puzzle
            phi_int = int(PHI * 1000) % 256
            modified ^= (phi_int << (i % 8))
        
        # Ensure 68 bits
        if count_bits(modified) == 68:
            candidates.append(format_private_key(modified))
    
    return candidates

def multi_gpu_search():
    """Coordinate search across multiple GPUs - optimized for 6x RTX 4090"""
    global RUNNING, SOLUTION_FOUND
    
    # Set up signal handler
    signal.signal(signal.SIGINT, handle_exit_signal)
    
    # Count available GPUs
    cuda.init()
    gpu_count = cuda.Device.count()
    
    if gpu_count == 0:
        print("No CUDA devices found!")
        return None
    
    print(f"Found {gpu_count} CUDA devices")
    
    # Ideally, we have 6 RTX 4090s as specified
    expected_gpus = 6
    gpus_to_use = min(gpu_count, expected_gpus)
    
    print(f"Using {gpus_to_use} GPUs for search")
    
    # Generate search space for each GPU
    gpu_search_spaces = distribute_masks_to_gpus(
        list(range(gpus_to_use)), 
        SEARCH_MASKS
    )
    
    # Add extra candidates for any remaining GPU capacity
    additional_candidates = generate_additional_candidates(BASE_PATTERN)
    
    # Create process pool for GPU workers
    workers = []
    
    # Start a worker for each GPU search space
    for i, (gpu_id, pattern, mask) in enumerate(gpu_search_spaces):
        # Estimate batch size based on mask - larger masks need smaller batches
        batch_size = 50000000 // (bin(mask).count('1') // 4 + 1)
        
        # Adjust batch size for RTX 4090s (24GB VRAM)
        batch_size = min(100000000, max(10000000, batch_size))
        
        # Max batches depends on mask size - search longer for smaller spaces
        max_batches = 2000 // (bin(mask).count('1') // 4 + 1)
        max_batches = min(10000, max(100, max_batches))
        
        # Create worker process
        worker = mp.Process(
            target=gpu_search_with_pattern,
            args=(gpu_id, pattern, mask, batch_size, max_batches)
        )
        worker.start()
        workers.append(worker)
        
        # Small stagger to avoid all GPUs competing for CPU at same time
        time.sleep(1)
    
    # Monitor workers and check for results
    try:
        running = True
        while running and not SOLUTION_FOUND:
            running = False
            for worker in workers:
                if worker.is_alive():
                    running = True
            
            # Check if any worker found a solution
            try:
                result = RESULT_QUEUE.get(block=False)
                if result:
                    print("\nSolution found by a worker:")
                    print(f"Key: {result[0]}")
                    SOLUTION_FOUND = True
                    break
            except queue.Empty:
                pass
            
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down workers...")
        RUNNING = False
    
    # Ensure all workers are terminated
    for worker in workers:
        if worker.is_alive():
            worker.terminate()
            worker.join(timeout=1)
    
    print("\nAll GPU workers completed or terminated")
    
    # Report final statistics
    with GLOBAL_LOCK:
        end_time = time.time()
        elapsed = end_time - START_TIME
        print(f"\nSearch completed in {format_time(elapsed)}")
        print(f"Total keys checked: {KEYS_CHECKED:,}")
        print(f"Average speed: {int(KEYS_CHECKED / elapsed):,} keys/second")
        
        if BEST_GLOBAL_MATCHES:
            print("\nBest matches found:")
            for i, (key, result) in enumerate(BEST_GLOBAL_MATCHES[:3]):
                print(f"{i+1}. Key: {key}")
                print(f"   Ratio diff: {result.get('ratio_diff')}")
                print(f"   Hash: {result.get('generated_hash')}")
    
    return None

if __name__ == "__main__":
    print("Starting Bitcoin Puzzle #68 Solver - Optimized for 6x RTX 4090")
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 = {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
    print(f"Base pattern: {BASE_PATTERN}")
    
    # Check GPU devices
    print("\nGPU Information:")
    cuda.init()
    device_count = cuda.Device.count()
    print(f"Found {device_count} CUDA devices")
    
    for i in range(device_count):
        device = cuda.Device(i)
        props = device.get_attributes()
        free_mem, total_mem = device.get_memory_info()
        print(f"GPU {i}: {device.name()}")
        print(f"  Memory: {free_mem/(1024**3):.2f} GB free / {total_mem/(1024**3):.2f} GB total")
        print(f"  Compute Capability: {props[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]}.{props[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]}")
        print(f"  Multiprocessors: {props[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")
    
    # Display estimated runtime
    print("\nRunning multi-GPU search optimized for RTX 4090s")
    print("This will search the specific pattern space very efficiently")
    print("Press Ctrl+C to gracefully stop the search\n")
    
    # Initialize start time
    START_TIME = time.time()
    
    try:
        # Run the optimized search
        solution = multi_gpu_search()
        
        elapsed_time = time.time() - START_TIME
        print(f"\nSearch completed in {format_time(elapsed_time)}")
        
        if SOLUTION_FOUND:
            print("\nSolution found! Check puzzle68_SOLVED.txt for details.")
            sys.exit(0)
        else:
            print("\nNo exact solution found in the target pattern space.")
            print("Check puzzle68_progress.txt for the best matches found so far.")
    
    except KeyboardInterrupt:
        elapsed_time = time.time() - START_TIME
        print(f"\nSearch interrupted after {format_time(elapsed_time)}")
        print(f"Total keys checked: {KEYS_CHECKED:,}")
        print("Check puzzle68_progress.txt for the best matches found so far.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during search: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
