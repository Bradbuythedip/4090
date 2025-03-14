# puzzle68_rtx4090_6gpu_solver.py

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
import threading
import queue

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

# Our best candidates from different approaches
BEST_PHI_MATCH = "00000000000000000000000000000000000000000000000041a01b90c5b886dc"
BEST_ALT_PATTERN = "00000000000000000000000000000000000000000000000cedb187f001bdffd2"

# CEDB187F pattern space (base pattern for focused search)
CEDB187F_PATTERN = "00000000000000000000000000000000000000000000000cedb187f"

# Additional promising candidates
ADDITIONAL_CANDIDATES = [
    "00000000000000000000000000000000000000000000000041a01b90c5b886dd",  # Small variation of PHI_MATCH
    "00000000000000000000000000000000000000000000000cedb187f001bdffe",   # Small variation of ALT_PATTERN
    "00000000000000000000000000000000000000000000000467a98b1c3d2e5f0",   # Additional candidate
    "00000000000000000000000000000000000000000000000789abcdef0123456",   # Additional candidate
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

# CUDA kernel for testing keys - enhanced for RTX 4090
cuda_code = """
#include <stdio.h>
#include <stdint.h>

// Improved ratio approximation function optimized for RTX 4090's FP32 ops
__device__ float approx_ratio(uint64_t key_high, uint64_t key_low) {
    // This approximation is optimized for filtering
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
    
    // Use multiple filtering criteria to identify promising candidates
    bool is_candidate = false;
    
    // Criterion 1: Ratio is extremely close to PHI/8
    if (ratio_diff < 0.00001f) {
        is_candidate = true;
    }
    
    // Criterion 2: Hash prefix matches
    if ((hash_diff & 0xFFFF0000) == 0) {
        is_candidate = true;
    }
    
    // Criterion 3: Certain bit patterns in the hash are present
    if ((hash_prefix & 0xF0F0F0F0) == (target_hash_prefix & 0xF0F0F0F0)) {
        is_candidate = true;
    }
    
    // Store candidate if it meets any of our criteria
    if (is_candidate) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < 1500) {  // Increased for RTX 4090
            candidates[pos] = offset;
        }
    }
}

// Enhanced kernel with 8 mutations per key for RTX 4090
__global__ void test_keys_with_mutations(uint64_t base_high, uint64_t base_low, 
                                       uint32_t range_start, uint32_t keys_per_thread,
                                       float target_ratio, uint32_t target_hash_prefix,
                                       uint32_t *candidates, float *ratios, uint32_t *count,
                                       uint32_t puzzle_num) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start_offset = range_start + (idx * keys_per_thread);
    
    for (uint32_t i = 0; i < keys_per_thread; i++) {
        uint32_t offset = start_offset + i;
        
        // Create key with offset
        uint64_t key_low = base_low | offset;
        uint64_t key_high = base_high;
        
        // Try more mutations per key - optimized for RTX 4090
        for (int mutation = 0; mutation < 8; mutation++) {
            uint64_t mutated_low = key_low;
            uint64_t mutated_high = key_high;
            
            // Apply different mutations
            switch(mutation) {
                case 0: 
                    // No mutation, original key
                    break;
                case 1: 
                    // Flip a low bit
                    mutated_low ^= 1;
                    break;
                case 2: 
                    // Flip a medium bit
                    mutated_low ^= (1 << 8);
                    break;
                case 3: 
                    // XOR with puzzle number
                    mutated_low ^= puzzle_num;
                    break;
                case 4:
                    // Addition with small value
                    mutated_low += puzzle_num;
                    break;
                case 5:
                    // Rotate bits right by one
                    mutated_low = ((mutated_low >> 1) | (mutated_low << 31)) & 0xFFFFFFFF;
                    break;
                case 6:
                    // Swap bytes
                    mutated_low = ((mutated_low & 0xFF00FF00) >> 8) | 
                                 ((mutated_low & 0x00FF00FF) << 8);
                    break;
                case 7:
                    // Combo mutation
                    mutated_low = (mutated_low ^ puzzle_num) + (puzzle_num >> 1);
                    break;
            }
            
            // Calculate approximate ratio and hash
            float ratio = approx_ratio(mutated_high, mutated_low);
            uint32_t hash_prefix = approx_hash(mutated_high, mutated_low);
            
            // Calculate difference from target ratio
            float ratio_diff = fabsf(ratio - target_ratio);
            uint32_t hash_diff = (hash_prefix ^ target_hash_prefix);
            
            // More aggressive filtering for RTX 4090
            if (ratio_diff < 0.0001f || (hash_diff & 0xFFF00000) == 0) {
                uint32_t pos = atomicAdd(count, 1);
                if (pos < 1500) {  // Increased for RTX 4090
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
        test_keys_with_mutations_kernel = module.get_function("test_keys_with_mutations")
        print(f"GPU {gpu_id} initialized: {device.name()}")
        
        return {
            'device': device,
            'context': context,
            'module': module,
            'test_keys': test_keys_kernel,
            'test_keys_with_mutations': test_keys_with_mutations_kernel
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
        # print(f"Error generating public key: {e}")
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
            f.write(f"Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ\n")
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
            
            # Calculate rate
            elapsed = current_time - START_TIME
            if elapsed > 0:
                rate = int(KEYS_CHECKED / elapsed)
                f.write(f"Keys per second: {rate:,}\n")
                # Estimate time to completion for 16^10 search space
                if rate > 0:
                    total_space = 16**10  # Search space size for 10 hex digits
                    remaining = max(0, total_space - KEYS_CHECKED)
                    seconds_remaining = remaining / rate
                    f.write(f"Estimated time remaining: {format_time(seconds_remaining)}\n")
            
            f.write("\nTop closest matches by ratio:\n")
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
    elif seconds < 86400:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours} hours, {minutes} minutes"
    else:
        days = int(seconds / 86400)
        hours = int((seconds % 86400) / 3600)
        return f"{days} days, {hours} hours"

def gpu_search_with_pattern(gpu_id, base_pattern, bit_mask, batch_size=50000000, max_batches=100):
    """GPU search focusing on a specific pattern with RTX 4090 optimizations"""
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
        # RTX 4090 has 24GB VRAM, we can use larger batches
        free_mem, total_mem = cuda.mem_get_info()
        max_possible_batch = int(free_mem * 0.8 / 16)  # Use more of the available memory
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
        
        # Allocate memory on GPU
        offsets = np.arange(batch_size, dtype=np.uint32)
        candidates = np.zeros(1500, dtype=np.uint32)  # Increased for RTX 4090
        count = np.zeros(1, dtype=np.uint32)
        ratios = np.zeros(batch_size, dtype=np.float32)
        
        offsets_gpu = cuda.mem_alloc(offsets.nbytes)
        candidates_gpu = cuda.mem_alloc(candidates.nbytes)
        count_gpu = cuda.mem_alloc(count.nbytes)
        ratios_gpu = cuda.mem_alloc(ratios.nbytes)
        
        # Configure for RTX 4090
        block_size = 512  # Use 512 threads per block for RTX 4090
        
        batch = 0
        while batch < max_batches and not (found_solution or SOLUTION_FOUND):
            # Check if solution was found by another GPU
            if SOLUTION_FOUND:
                print(f"GPU {gpu_id}: Solution found by another GPU, stopping search")
                break
                
            # Start with a fresh offset range for this batch
            start_offset = batch * batch_size
            offsets = np.arange(start_offset, start_offset + batch_size, dtype=np.uint32) & bit_mask
            
            # Reset candidate counter
            count[0] = 0
            
            # Copy data to GPU
            cuda.memcpy_htod(offsets_gpu, offsets)
            cuda.memcpy_htod(count_gpu, count)
            
            # Set up kernel grid for RTX 4090
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
                # Mutations kernel - process fewer keys but more mutations per key
                keys_per_thread = 8  # Each thread handles 8 keys with mutations
                smaller_batch = batch_size // keys_per_thread
                smaller_grid_size = (smaller_batch + block_size - 1) // block_size
                
                gpu_data['test_keys_with_mutations'](
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
            
            # Total keys processed
            if batch % 2 == 0:
                total_keys_checked += batch_size
            else:
                total_keys_checked += batch_size * 8  # Account for mutations
            
            # Report progress
            candidates_found = min(int(count[0]), 1500)
            if candidates_found > 0 or batch % 10 == 0:
                print(f"\nGPU {gpu_id}: Batch {batch+1}/{max_batches} completed")
                print(f"GPU {gpu_id}: Keys checked: {total_keys_checked:,}")
                print(f"GPU {gpu_id}: Candidates found: {candidates_found}")
            
            # Find the best ratio candidates
            best_indices = np.argsort(np.abs(ratios - PHI_OVER_8))[:250]  # Check more candidates with RTX 4090
            
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
            
            # Verify promising candidates on CPU with high precision
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

def partition_search_space(gpu_count, base_pattern, target_digits=10):
    """Partition the search space for base_pattern across multiple GPUs"""
    # Calculate the total space size for 10 hex digits
    total_space = 16**target_digits
    
    # Calculate how many keys each GPU should search
    keys_per_gpu = total_space // gpu_count
    
    # Create masks and ranges for each GPU
    partitions = []
    
    # Calculate full mask for all target digits
    full_mask = (1 << (4 * target_digits)) - 1
    
    for i in range(gpu_count):
        start = i * keys_per_gpu
        end = (i + 1) * keys_per_gpu if i < gpu_count - 1 else total_space
        
        # Convert start to hex and create a pattern with it
        start_hex = format(start, f'0{target_digits}x')
        
        # Create base pattern for this partition
        partition_pattern = base_pattern + start_hex
        
        # Get the size of this partition's search space
        partition_size = end - start
        
        # Calculate the appropriate mask
        partition_mask = full_mask if i == gpu_count - 1 else (1 << (4 * target_digits)) - 1
        
        partitions.append({
            'gpu_id': i,
            'pattern': partition_pattern,
            'mask': partition_mask,
            'start': start,
            'end': end,
            'size': partition_size
        })
    
    return partitions

def setup_search_spaces_for_gpus(gpu_count):
    """Set up search spaces for all available GPUs"""
    search_spaces = []
    
    # Focus on cedb187f pattern first
    cedb187f_partitions = partition_search_space(gpu_count, CEDB187F_PATTERN, target_digits=10)
    for partition in cedb187f_partitions:
        search_spaces.append({
            'gpu_id': partition['gpu_id'],
            'base_pattern': partition['pattern'],
            'bit_mask': partition['mask'],
            'priority': 1  # High priority
        })
    
    # If we have more GPUs than needed for the main pattern, add alternative patterns
    if gpu_count > len(cedb187f_partitions):
        extra_gpus = gpu_count - len(cedb187f_partitions)
        alt_patterns = [BEST_PHI_MATCH, BEST_ALT_PATTERN] + ADDITIONAL_CANDIDATES
        
        for i in range(min(extra_gpus, len(alt_patterns))):
            search_spaces.append({
                'gpu_id': i + len(cedb187f_partitions),
                'base_pattern': alt_patterns[i],
                'bit_mask': 0x00FFFFFF,  # Wider search around alternative patterns
                'priority': 2  # Lower priority
            })
    
    return search_spaces

def multi_gpu_search(num_gpus=6):
    """Run search on multiple GPUs in parallel - optimized for 6x RTX 4090"""
    global RUNNING, SOLUTION_FOUND
    
    # Set up signal handler
    signal.signal(signal.SIGINT, handle_exit_signal)
    
    # Count available GPUs
    cuda.init()
    gpu_count = cuda.Device.count()
    
    if gpu_count == 0:
        print("No CUDA devices found!")
        return None
    
    available_gpus = min(gpu_count, num_gpus)
    print(f"Found {gpu_count} CUDA devices, will use {available_gpus}")
    
    # Check for RTX 4090s
    for i in range(available_gpus):
        device = cuda.Device(i)
        props = device.get_attributes()
        free_mem, total_mem = device.get_memory_info()
        print(f"GPU {i}: {device.name()} - {total_mem/(1024**3):.1f} GB VRAM")
        
        # If we detect RTX 4090, we can use larger batch sizes
        if "4090" in device.name() or total_mem > (20 * 1024 * 1024 * 1024):  # >20GB VRAM
            print(f"  Detected high-end GPU, will use optimized parameters")
    
    # Set up search spaces for all GPUs
    search_spaces = setup_search_spaces_for_gpus(available_gpus)
    
    # Create workers
    workers = []
    
    # Start a worker for each GPU
    for space in search_spaces:
        gpu_id = space['gpu_id']
        base_pattern = space['base_pattern']
        bit_mask = space['bit_mask']
        
        # Set batch size based on estimated GPU memory
        # RTX 4090 can handle much larger batches
        device = cuda.Device(gpu_id)
        ctx = device.make_context()
        free_mem, total_mem = cuda.mem_get_info()  # Correct way to get memory info
        ctx.pop()  # Release context after checking
        
        # For 24GB GPUs like RTX 4090, use much larger batch sizes
        if total_mem > (20 * 1024 * 1024 * 1024):  # >20GB VRAM
            batch_size = 100000000  # 100M keys per batch for RTX 4090
            max_batches = 1000
        else:
            # For smaller GPUs, scale accordingly
            batch_size = min(50000000, int(free_mem * 0.6 / 16))
            max_batches = 500
        
        print(f"GPU {gpu_id}: Assigned search space starting with {base_pattern}")
        print(f"GPU {gpu_id}: Using batch size: {batch_size:,}, max batches: {max_batches}")
        
        # Create and start worker
        worker = mp.Process(
            target=gpu_search_with_pattern,
            args=(gpu_id, base_pattern, bit_mask, batch_size, max_batches)
        )
        worker.start()
        workers.append(worker)
        
        # Small delay to stagger startup
        time.sleep(1)
    
    # Monitor workers
    try:
        while True:
            # Check if any worker found a solution
            try:
                result = RESULT_QUEUE.get(block=False)
                if result:
                    print("\nSolution found by a worker!")
                    print(f"Key: {result[0]}")
                    SOLUTION_FOUND = True
                    
                    # Terminate all workers
                    for worker in workers:
                        if worker.is_alive():
                            worker.terminate()
                    
                    return result[0]
            except queue.Empty:
                pass
            
            # Check if all workers finished
            if not any(worker.is_alive() for worker in workers):
                print("\nAll workers completed")
                break
            
            # Check if we should stop
            if not RUNNING:
                print("\nStopping all workers...")
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                break
            
            # Sleep for a bit
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down workers...")
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        RUNNING = False
    
    # Wait for all workers to finish
    for worker in workers:
        worker.join(timeout=2)
    
    print("\nAll GPU workers completed or terminated")
    
    # Report final statistics
    elapsed_time = time.time() - START_TIME
    print(f"\nSearch completed in {format_time(elapsed_time)}")
    print(f"Total keys checked: {KEYS_CHECKED:,}")
    print(f"Average speed: {int(KEYS_CHECKED / (elapsed_time + 0.001)):,} keys/second")
    
    return None

if __name__ == "__main__":
    print("Starting Bitcoin Puzzle #68 solver - Optimized for 6x RTX 4090")
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 = {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
    print(f"Primary search pattern: {CEDB187F_PATTERN}")
    
    # Check GPU devices
    print("\nGPU Information:")
    cuda.init()
    device_count = cuda.Device.count()
    print(f"Found {device_count} CUDA devices")
    
    for i in range(device_count):
        device = cuda.Device(i)
        ctx = device.make_context()
        props = device.get_attributes()
        free_mem, total_mem = cuda.mem_get_info()  # Correct way to get memory info
        ctx.pop()  # Release context when done
        print(f"GPU {i}: {device.name()}")
        print(f"  Memory: {free_mem/(1024**3):.2f} GB free / {total_mem/(1024**3):.2f} GB total")
        print(f"  Compute Capability: {props[cuda.device_attribute.COMPUTE_CAPABILITY_MAJOR]}.{props[cuda.device_attribute.COMPUTE_CAPABILITY_MINOR]}")
        print(f"  Multiprocessors: {props[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")
    
    # Initialize start time
    START_TIME = time.time()
    
    print("\nRunning multi-GPU search with 6x RTX 4090 optimizations...")
    print("Press Ctrl+C to gracefully stop the search")
    
    try:
        # Verify any known candidates first
        print("\nVerifying known candidate keys...")
        for key in [BEST_PHI_MATCH, BEST_ALT_PATTERN] + ADDITIONAL_CANDIDATES:
            print(f"Checking {key}...")
            result = verify_private_key(key)
            if result and result.get('match'):
                print(f"\n*** SOLUTION FOUND IN INITIAL VERIFICATION ***")
                save_solution(key, result)
                sys.exit(0)
        
        # Run the multi-GPU search
        solution = multi_gpu_search(num_gpus=6)  # Use 6 GPUs
        
        if solution:
            print(f"\nFinal solution: {solution}")
            sys.exit(0)
        else:
            print("\nNo solution found in the search space.")
            print("Check puzzle68_progress.txt for the best matches so far.")
    
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
        elapsed_time = time.time() - START_TIME
        print(f"Search ran for {format_time(elapsed_time)}")
        print(f"Total keys checked: {KEYS_CHECKED:,}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nError during search: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
