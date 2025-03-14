# puzzle68_exhaustive_solver.py

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

# Additional promising candidates (randomly generated examples - should be replaced with actual good candidates)
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

# Create a shared lock for thread-safe operations
import threading
GLOBAL_LOCK = threading.Lock()

# CUDA kernel for testing keys - improved with more sophisticated filtering
cuda_code = """
#include <stdio.h>
#include <stdint.h>

// Improved ratio approximation function
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
    uint32_t hash_diff = (hash_prefix ^ target_hash_prefix) & 0xFFFF0000; // Check top bits
    
    // Use multiple filtering criteria to identify promising candidates
    bool is_candidate = false;
    
    // Criterion 1: Ratio is extremely close to PHI/8
    if (ratio_diff < 0.00001f) {
        is_candidate = true;
    }
    
    // Criterion 2: Hash prefix matches
    if (hash_diff == 0) {
        is_candidate = true;
    }
    
    // Criterion 3: Certain bit patterns in the hash are present
    if ((hash_prefix & 0xF0F0F0F0) == (target_hash_prefix & 0xF0F0F0F0)) {
        is_candidate = true;
    }
    
    // Store candidate if it meets any of our criteria
    if (is_candidate) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < 1000) {  // Limit to 1000 candidates
            candidates[pos] = offset;
        }
    }
}

// Additional kernel for range search with mutations
__global__ void test_keys_with_mutations(uint64_t base_high, uint64_t base_low, 
                                       uint32_t range_start, uint32_t keys_per_thread,
                                       float target_ratio, uint32_t target_hash_prefix,
                                       uint32_t *candidates, float *ratios, uint32_t *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start_offset = range_start + (idx * keys_per_thread);
    
    for (uint32_t i = 0; i < keys_per_thread; i++) {
        uint32_t offset = start_offset + i;
        
        // Create key with offset
        uint64_t key_low = base_low | offset;
        uint64_t key_high = base_high;
        
        // Try some mutations
        for (int mutation = 0; mutation < 4; mutation++) {
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
                    mutated_low ^= 68;
                    break;
            }
            
            // Calculate approximate ratio and hash
            float ratio = approx_ratio(mutated_high, mutated_low);
            uint32_t hash_prefix = approx_hash(mutated_high, mutated_low);
            
            // Calculate difference from target ratio
            float ratio_diff = fabsf(ratio - target_ratio);
            uint32_t hash_diff = (hash_prefix ^ target_hash_prefix);
            
            // Check if it's a candidate worth verifying on CPU
            if (ratio_diff < 0.0001f || (hash_diff & 0xFFFF0000) == 0) {
                uint32_t pos = atomicAdd(count, 1);
                if (pos < 1000) {  // Limit to 1000 candidates
                    candidates[pos] = offset;
                    ratios[pos] = ratio;
                }
            }
        }
    }
}
"""

# Compile CUDA kernel
mod = SourceModule(cuda_code)
test_keys_kernel = mod.get_function("test_keys")
test_keys_with_mutations_kernel = mod.get_function("test_keys_with_mutations")

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
    if not force_save and current_time - save_progress.last_save_time < 60:  # Save at most once per minute
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

def try_multi_bit_manipulations(key, max_flips=5, max_positions=32):
    """Try flipping multiple bits at once in the key - enhanced for exhaustive search"""
    key_int = int(key, 16)
    print(f"Trying multi-bit manipulations of {key}")
    print(f"Max bits to flip: {max_flips}, Max positions: {max_positions}")
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Positions to try bit flips (focus on both lower and higher bits)
    positions = list(range(max_positions))
    
    # Try flipping 1 to max_flips bits at a time
    for num_flips in range(1, max_flips + 1):
        print(f"Trying {num_flips}-bit flips...")
        # Generate combinations of num_flips positions
        for combo in combinations(positions, num_flips):
            # Start with original key
            modified_key = key_int
            
            # Flip bits at all positions in the combo
            for pos in combo:
                modified_key ^= (1 << pos)
            
            # Ensure it's still 68 bits
            if count_bits(modified_key) != 68:
                continue
            
            # Format and test
            modified_key_hex = format_private_key(modified_key)
            result = verify_private_key(modified_key_hex)
            count += 1
            
            if count % 10000 == 0:
                print(f"Tested {count:,} multi-bit variations...")
                # Save progress periodically
                save_progress(best_matches, f"Multi-bit manipulation ({num_flips} bits)", count)
            
            # Check if we found the solution
            if result and result.get('match'):
                print(f"\n*** SOLUTION FOUND ***")
                print(f"Key: {modified_key_hex}")
                print(f"Generated hash: {result.get('generated_hash')}")
                print(f"ScriptPubKey: 76a914{result.get('generated_hash')}88ac")
                
                # Save solution
                save_solution(modified_key_hex, result)
                
                # Exit immediately on match
                print("\nExiting program immediately with solution found.")
                os._exit(0)  # Force immediate exit
                
                return modified_key_hex
            
            # Track best match by ratio
            if result:
                ratio_diff = float(result.get('ratio_diff', 1.0))
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    
                    # Add to best matches
                    best_matches.append((modified_key_hex, result))
                    best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                    best_matches = best_matches[:5]  # Keep top 5
                    
                    # Report improvement
                    print(f"\nNew best match: {modified_key_hex}")
                    print(f"Bit length: {count_bits(modified_key)}")
                    print(f"Ratio diff: {ratio_diff}")
                    print(f"Generated hash: {result.get('generated_hash')}")
                    
                    # Save progress on significant improvement
                    save_progress(best_matches, f"Multi-bit manipulation ({num_flips} bits)", count, force_save=True)
    
    print(f"Completed multi-bit manipulations, tested {count:,} variations")
    return None

def precise_search_around_key(key, range_size=100000, step_size=1, max_keys=10000000):
    """Perform a precise search around a key with very small steps - enhanced for exhaustive search"""
    key_int = int(key, 16)
    print(f"Starting precise search around {key}")
    print(f"Range: +/- {range_size}, Step size: {step_size}, Max keys: {max_keys:,}")
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Search within range of the key
    for offset in range(-range_size, range_size + 1, step_size):
        if offset == 0:  # Skip the original key
            continue
            
        # Calculate modified key
        modified_key = key_int + offset
        
        # Ensure it's still 68 bits
        if count_bits(modified_key) != 68:
            continue
        
        # Format and test
        modified_key_hex = format_private_key(modified_key)
        result = verify_private_key(modified_key_hex)
        count += 1
        
        if count % 10000 == 0:
            print(f"Tested {count:,} keys in precise search...")
            # Save progress periodically
            save_progress(best_matches, "Precise search", count)
        
        # Check if we found the solution
        if result and result.get('match'):
            print(f"\n*** SOLUTION FOUND ***")
            print(f"Key: {modified_key_hex}")
            print(f"Generated hash: {result.get('generated_hash')}")
            
            # Save solution
            save_solution(modified_key_hex, result)
            
            return modified_key_hex
        
        # Track best match by ratio
        if result:
            ratio_diff = float(result.get('ratio_diff', 1.0))
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                
                # Add to best matches
                best_matches.append((modified_key_hex, result))
                best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                best_matches = best_matches[:5]  # Keep top 5
                
                # Report improvement
                print(f"\nNew best match: {modified_key_hex}")
                print(f"Bit length: {count_bits(modified_key)}")
                print(f"Ratio diff: {ratio_diff}")
                print(f"Generated hash: {result.get('generated_hash')}")
                
                # Save progress on significant improvement
                save_progress(best_matches, "Precise search", count, force_save=True)
        
        # Check if we've tested enough keys
        if count >= max_keys:
            print(f"Reached max keys limit of {max_keys:,}")
            break
        
        # Check if we should continue running
        if not RUNNING:
            print("Search interrupted")
            break
    
    print(f"Completed precise search, tested {count:,} keys")
    return None

def gpu_search_with_dual_focus(base_int, batch_size=10000000, max_batches=100, bit_mask=0x0000FFFF):
    """GPU search with focus on both ratio and hash pattern - enhanced for exhaustive search"""
    # Convert base_int to Python int to avoid overflow
    base_int = int(base_int)
    
    # Determine optimal batch size based on GPU memory
    free_mem, total_mem = cuda.mem_get_info()
    max_possible_batch = int(free_mem * 0.7 / 16)  # 16 bytes per key (estimated)
    if batch_size > max_possible_batch:
        batch_size = max_possible_batch
        print(f"Adjusted batch size to {batch_size:,} based on GPU memory")
    
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
    
    print(f"GPU search around: {format_private_key(base_int)}")
    print(f"Bit length: {count_bits(base_int)}")
    print(f"Using batch size: {batch_size:,}")
    
    # First 4 bytes of target hash as uint32
    target_hash_prefix = int(TARGET_HASH[:8], 16)
    
    # Allocate memory on GPU
    offsets = np.arange(batch_size, dtype=np.uint32)
    candidates = np.zeros(1000, dtype=np.uint32)
    count = np.zeros(1, dtype=np.uint32)
    ratios = np.zeros(batch_size, dtype=np.float32)
    
    offsets_gpu = cuda.mem_alloc(offsets.nbytes)
    candidates_gpu = cuda.mem_alloc(candidates.nbytes)
    count_gpu = cuda.mem_alloc(count.nbytes)
    ratios_gpu = cuda.mem_alloc(ratios.nbytes)
    
    for batch in range(max_batches):
        if found_solution or not RUNNING:
            break
            
        # Start with a fresh offset range for this batch
        start_offset = batch * batch_size
        offsets = np.arange(start_offset, start_offset + batch_size, dtype=np.uint32) & bit_mask
        
        # Reset candidate counter
        count[0] = 0
        
        # Copy data to GPU
        cuda.memcpy_htod(offsets_gpu, offsets)
        cuda.memcpy_htod(count_gpu, count)
        
        # Set up kernel grid
        block_size = 256
        grid_size = (batch_size + block_size - 1) // block_size
        
        # Launch kernel with hash prefix checking
        test_keys_kernel(
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
        
        # Get results back
        cuda.memcpy_dtoh(candidates, candidates_gpu)
        cuda.memcpy_dtoh(count, count_gpu)
        cuda.memcpy_dtoh(ratios, ratios_gpu)
        
        total_keys_checked += batch_size
        
        # Report progress
        print(f"\nBatch {batch+1}/{max_batches} completed")
        print(f"Keys checked: {total_keys_checked:,}")
        
        # Get the number of candidates found
        candidates_found = min(int(count[0]), 1000)
        print(f"Candidates found: {candidates_found}")
        
        # Find the best ratios directly from the GPU results
        best_indices = np.argsort(np.abs(ratios - PHI_OVER_8))[:200]  # Check top 200 instead of 100
        
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
        print(f"Verifying {len(all_candidates)} candidates on CPU...")
        
        for offset in all_candidates:
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
            
            # Check if we found the solution
            if result and result.get('match'):
                print(f"\n*** SOLUTION FOUND ***")
                print(f"Key: {full_key}")
                print(f"Generated hash: {result.get('generated_hash')}")
                print(f"Target hash: {TARGET_HASH}")
                
                # Save solution
                save_solution(full_key, result)
                
                found_solution = True
                best_matches.append((full_key, result))
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
                    print(f"\nNew best match: {full_key}")
                    print(f"Bit length: {count_bits(full_key_int)}")
                    print(f"Ratio diff: {ratio_diff}")
                    print(f"Generated hash: {result.get('generated_hash')}")
                    
                    if ratio_diff < 1e-10:
                        print("EXTREMELY CLOSE MATCH FOUND!")
                    
                    # Save progress on significant improvement
                    save_progress(best_matches, f"GPU dual search", total_keys_checked, force_save=True)
        
        # Save progress after each batch
        save_progress(best_matches, f"GPU dual search", total_keys_checked)
    
    # Free GPU memory
    offsets_gpu.free()
    candidates_gpu.free()
    count_gpu.free()
    ratios_gpu.free()
    
    if found_solution and best_matches:
        return best_matches[0][0]
    elif best_matches:
        return None  # Continue searching
    else:
        return None

def random_bit_permutation_search(key, iterations=1000000):
    """Try random bit permutations of a key"""
    key_int = int(key, 16)
    print(f"Starting random bit permutation search from {key}")
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Define bit operations to try
    operations = [
        lambda k: k ^ (1 << random.randint(0, 40)),  # Flip random bit
        lambda k: k ^ (random.randint(1, 255)),     # XOR with small random value
        lambda k: (k & ~0xFF) | random.randint(0, 255),  # Randomize lowest byte
        lambda k: k + random.randint(-1000, 1000),  # Small addition/subtraction
        lambda k: ((k << 1) & ((1 << 68) - 1)) | (k >> 67),  # Rotate bits
        lambda k: k ^ (PUZZLE_NUMBER << random.randint(0, 16))  # XOR with puzzle number
    ]
    
    for i in range(iterations):
        if not RUNNING:
            break
            
        # Choose random operation
        op = random.choice(operations)
        modified_key = op(key_int)
        
        # Ensure it's still 68 bits
        if count_bits(modified_key) != 68:
            continue
            
        # Format and test
        modified_key_hex = format_private_key(modified_key)
        result = verify_private_key(modified_key_hex)
        count += 1
        
        if count % 10000 == 0:
            print(f"Tested {count:,} random bit permutations...")
            save_progress(best_matches, "Random bit permutations", count)
        
        # Check if we found the solution
        if result and result.get('match'):
            print(f"\n*** SOLUTION FOUND ***")
            print(f"Key: {modified_key_hex}")
            print(f"Generated hash: {result.get('generated_hash')}")
            
            # Save solution
            save_solution(modified_key_hex, result)
            
            return modified_key_hex
        
        # Track best match by ratio
        if result:
            ratio_diff = float(result.get('ratio_diff', 1.0))
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                
                # Add to best matches
                best_matches.append((modified_key_hex, result))
                best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                best_matches = best_matches[:5]  # Keep top 5
                
                # Report improvement
                print(f"\nNew best match: {modified_key_hex}")
                print(f"Bit length: {count_bits(modified_key)}")
                print(f"Ratio diff: {ratio_diff}")
                print(f"Generated hash: {result.get('generated_hash')}")
                
                # Save progress
                save_progress(best_matches, "Random bit permutations", count, force_save=True)
    
    print(f"Completed random bit permutations, tested {count:,} variations")
    return None

def try_mathematical_transformations(key):
    """Try various mathematical transformations of the key"""
    key_int = int(key, 16)
    print(f"Trying mathematical transformations of {key}")
    
    best_matches = []
    best_ratio_diff = 1.0
    count = 0
    
    # Multipliers based on various mathematical constants
    multipliers = [
        int(PHI * 2**32) & ((1 << 68) - 1),  # PHI scaled to 68 bits
        int(math.e * 2**32) & ((1 << 68) - 1),  # e scaled to 68 bits
        int(math.pi * 2**32) & ((1 << 68) - 1),  # pi scaled to 68 bits
        int(math.sqrt(2) * 2**32) & ((1 << 68) - 1),  # sqrt(2) scaled to 68 bits
        PUZZLE_NUMBER,  # Puzzle number
        PUZZLE_NUMBER**2,  # Puzzle number squared
        int(PUZZLE_NUMBER * PHI),  # Puzzle number * PHI
    ]
    
    transformations = [
        # Addition with constants
        lambda k, m: (k + m) & ((1 << 68) - 1),
        # Subtraction with constants
        lambda k, m: (k - m) & ((1 << 68) - 1),
        # XOR with constants
        lambda k, m: k ^ m,
        # Multiply low bits with constant
        lambda k, m: (k & ~0xFFFF) | ((k & 0xFFFF) * (m & 0xFF)) & 0xFFFF,
        # Bit rotations
        lambda k, m: ((k << (m % 68)) | (k >> (68 - (m % 68)))) & ((1 << 68) - 1),
    ]
    
    for transform in transformations:
        for multiplier in multipliers:
            modified_key = transform(key_int, multiplier)
            
            # Ensure it's still 68 bits
            if count_bits(modified_key) != 68:
                continue
                
            # Format and test
            modified_key_hex = format_private_key(modified_key)
            result = verify_private_key(modified_key_hex)
            count += 1
            
            # Check if we found the solution
            if result and result.get('match'):
                print(f"\n*** SOLUTION FOUND ***")
                print(f"Key: {modified_key_hex}")
                print(f"Generated hash: {result.get('generated_hash')}")
                
                # Save solution
                save_solution(modified_key_hex, result)
                
                return modified_key_hex
            
            # Track best match by ratio
            if result:
                ratio_diff = float(result.get('ratio_diff', 1.0))
                if ratio_diff < best_ratio_diff:
                    best_ratio_diff = ratio_diff
                    
                    # Add to best matches
                    best_matches.append((modified_key_hex, result))
                    best_matches.sort(key=lambda x: float(x[1].get('ratio_diff', 1.0)))
                    best_matches = best_matches[:5]  # Keep top 5
                    
                    # Report improvement
                    print(f"\nNew best match: {modified_key_hex}")
                    print(f"Bit length: {count_bits(modified_key)}")
                    print(f"Ratio diff: {ratio_diff}")
                    print(f"Generated hash: {result.get('generated_hash')}")
                    
                    # Save progress
                    save_progress(best_matches, "Mathematical transformations", count, force_save=True)
    
    print(f"Completed mathematical transformations, tested {count:,} variations")
    save_progress(best_matches, "Mathematical transformations", count)
    return None

def handle_exit_signal(sig, frame):
    """Handle user exit signal (Ctrl+C)"""
    global RUNNING
    if not RUNNING:
        print("\nForce exiting...")
        sys.exit(1)
    
    print("\nGracefully shutting down... (Press Ctrl+C again to force exit)")
    RUNNING = False

def exhaustive_search():
    """Run a comprehensive exhaustive search using all strategies in a continuous loop"""
    global RUNNING
    signal.signal(signal.SIGINT, handle_exit_signal)  # Register Ctrl+C handler
    
    # First, verify if any of our initial candidates already match
    print("\nVerifying initial candidate keys...")
    for key in [BEST_PHI_MATCH, BEST_ALT_PATTERN] + ADDITIONAL_CANDIDATES:
        print(f"Checking {key}...")
        result = verify_private_key(key)
        if result and result.get('match'):
            print(f"\n*** SOLUTION FOUND IN INITIAL VERIFICATION ***")
            save_solution(key, result)
            return key
    
    solution = None
    search_round = 1
    
    # Create a list of all search strategies with configurable parameters
    search_strategies = [
        # GPU searches
        lambda: gpu_search_with_dual_focus(int(BEST_PHI_MATCH, 16), max_batches=20, bit_mask=0x000FFFFF),
        lambda: gpu_search_with_dual_focus(int(BEST_ALT_PATTERN, 16), max_batches=20, bit_mask=0x000FFFFF),
        
        # Precise searches (gradually increasing range)
        lambda: precise_search_around_key(BEST_PHI_MATCH, range_size=100000 * search_round, step_size=1, max_keys=10000000),
        lambda: precise_search_around_key(BEST_ALT_PATTERN, range_size=100000 * search_round, step_size=1, max_keys=10000000),
        
        # Bit manipulations (gradually increasing complexity)
        lambda: try_multi_bit_manipulations(BEST_PHI_MATCH, max_flips=min(3 + search_round//2, 6), max_positions=min(20 + search_round*4, 40)),
        lambda: try_multi_bit_manipulations(BEST_ALT_PATTERN, max_flips=min(3 + search_round//2, 6), max_positions=min(20 + search_round*4, 40)),
        
        # Random bit permutations
        lambda: random_bit_permutation_search(BEST_PHI_MATCH, iterations=1000000),
        lambda: random_bit_permutation_search(BEST_ALT_PATTERN, iterations=1000000),
        
        # Mathematical transformations
        lambda: try_mathematical_transformations(BEST_PHI_MATCH),
        lambda: try_mathematical_transformations(BEST_ALT_PATTERN),
    ]
    
    # Optional: Add additional candidate keys from our expanded list
    for candidate in ADDITIONAL_CANDIDATES:
        search_strategies.append(lambda key=candidate: gpu_search_with_dual_focus(int(key, 16), max_batches=10, bit_mask=0x000FFFFF))
        search_strategies.append(lambda key=candidate: precise_search_around_key(key, range_size=50000, step_size=1, max_keys=5000000))
        search_strategies.append(lambda key=candidate: try_multi_bit_manipulations(key, max_flips=3, max_positions=20))
    
    # Search forever until solution is found or interrupted
    while RUNNING:
        print(f"\n=== SEARCH ROUND {search_round} ===")
        print(f"Runtime so far: {format_time(time.time() - START_TIME)}")
        print(f"Total keys checked: {KEYS_CHECKED:,}")

        # Run each strategy in turn
        for i, strategy in enumerate(search_strategies):
            if not RUNNING:
                break
                
            print(f"\n--- Strategy {i+1}/{len(search_strategies)} in round {search_round} ---")
            
            # Run the strategy
            solution = strategy()
            
            # If solution found, exit
            if solution:
                print("\nEXIT: Solution found!")
                return solution
        
        # Increment round counter
        search_round += 1
        
        # After each complete round, save progress and report
        save_progress([], "Completed full search round", 0, force_save=True)
        
        # After each complete round, generate a random promising key to add variety
        # Use the best matches we've found so far as seeds
        if BEST_GLOBAL_MATCHES:
            new_seed = BEST_GLOBAL_MATCHES[0][0]  # Use the best match found so far
            new_candidate = format_private_key(int(new_seed, 16) ^ random.randint(1, 1000))
            search_strategies.append(lambda key=new_candidate: gpu_search_with_dual_focus(int(key, 16), max_batches=5, bit_mask=0x000FFFFF))
            print(f"Added new search candidate: {new_candidate}")
    
    # If we got here, we were interrupted or solution was found
    if solution:
        return solution
    else:
        print("\nSearch interrupted. No solution found yet.")
        return None

if __name__ == "__main__":
    print("Starting exhaustive Bitcoin Puzzle #68 solver")
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 = {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
    print(f"Best phi/8 match: {BEST_PHI_MATCH}")
    print(f"Alternative pattern: {BEST_ALT_PATTERN}")
    
    # Check GPU devices
    print("\nGPU Information:")
    device = cuda.Device(0)
    print(f"Using GPU: {device.name()}")
    free_mem, total_mem = cuda.mem_get_info()
    print(f"GPU Memory: {free_mem/(1024**3):.2f} GB free / {total_mem/(1024**3):.2f} GB total")
    
    # Display estimated runtime
    print("\nRunning exhaustive search. This will continue until a solution is found.")
    print("Press Ctrl+C to gracefully stop the search.\n")
    
    # Initialize start time
    START_TIME = time.time()
    
    try:
        # Run the exhaustive search
        solution = exhaustive_search()
        
        elapsed_time = time.time() - START_TIME
        print(f"\nSearch completed in {format_time(elapsed_time)}")
        
        if solution:
            print(f"\nFINAL SOLUTION: {solution}")
            final_result = verify_private_key(solution)
            print(f"Hash match: {final_result.get('match')}")
            print(f"Ratio difference: {float(final_result.get('ratio_diff'))}")
            print(f"Generated hash: {final_result.get('generated_hash')}")
            print(f"Target hash: {TARGET_HASH}")
            print(f"ScriptPubKey: 76a914{TARGET_HASH}88ac")
            
            print("\nSaving solution to file...")
            save_solution(solution, final_result)
            
            # Immediately exit when solution is found
            print("\nSolution found! Exiting program.")
            sys.exit(0)
        else:
            print("\nNo exact solution found through these strategies.")
            print("Check puzzle68_progress.txt for the best matches found so far.")
    
    except KeyboardInterrupt:
        elapsed_time = time.time() - START_TIME
        print(f"\nSearch interrupted after {format_time(elapsed_time)}")
        print(f"Total keys checked: {KEYS_CHECKED:,}")
        print("Check puzzle68_progress.txt for the best matches found so far.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during search: {str(e)}")
        sys.exit(1)
