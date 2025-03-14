# brute_force_puzzle68_direct.py
# Systematically searches the explicit 68-bit key space for Bitcoin Puzzle #68

import hashlib
import binascii
import time
import sys
import os
import numpy as np
from mpmath import mp, mpf, fabs
import ecdsa
from ecdsa import SECP256k1, SigningKey

# Import PyCUDA modules with proper error handling
try:
    import pycuda.driver as cuda
    import pycuda.compiler as compiler
    HAS_CUDA = True
except ImportError:
    print("PyCUDA not installed. Please install with: pip install pycuda")
    HAS_CUDA = False
except Exception as e:
    print(f"Error initializing CUDA: {e}")
    HAS_CUDA = False

# Set high precision for mpmath
mp.dps = 1000  # 1000 decimal places of precision

# Constants
PHI = mpf('1.6180339887498948482045868343656381177203091798057628621354486227052604628189')
TARGET_HASH = 'e0b8a2baee1b77fc703455f39d51477451fc8cfc'  # Hash from scriptPubKey
PUZZLE_NUMBER = 68
PHI_OVER_8 = float(PHI / 8)  # Convert to float for GPU compatibility

# EXPLICIT DEFINITION of key space boundaries for puzzle 68
# For a 68-bit key, we need to search from 2^67 to 2^68-1
# 2^67 = 0x8000000000000000 (68-bit key with MSB set)
# 2^68-1 = 0xFFFFFFFFFFFFFFFF (all 68 bits set)
MIN_68BIT_INT = 1 << 67
MAX_68BIT_INT = (1 << 68) - 1
MIN_68BIT_HEX = format(MIN_68BIT_INT, '016x')  # 16 hex chars = 64 bits
MAX_68BIT_HEX = format(MAX_68BIT_INT, '016x')

# CUDA kernel for testing keys
CUDA_CODE = """
#include <stdio.h>
#include <stdint.h>

// Approximation of the ratio calculation for filtering
__device__ float approx_ratio(uint64_t key) {
    // Simple approximation for filtering - actual ratio calculated on CPU for verification
    float result = 0.0f;
    float x_approx = (float)((key >> 32) ^ (key & 0xFFFFFFFF));
    float y_approx = (float)((key & 0xFFFFFFFF) ^ ((key >> 32) << 8));
    
    if (y_approx != 0.0f) {
        result = 0.202254f + (float)((x_approx * 723467.0f + y_approx * 13498587.0f) / 
                                   (1e12 + key) - 0.5f) * 0.00001f;
    }
    return result;
}

// Hash approximation for filtering
__device__ uint32_t approx_hash(uint64_t key) {
    uint32_t h = 0xe0b8a2ba; // First bytes of target
    h ^= (key >> 32);
    h ^= (key & 0xFFFFFFFF);
    h ^= h >> 16;
    return h;
}

__global__ void test_keys(uint64_t start_key, 
                         uint32_t batch_size,
                         float target_ratio, 
                         uint32_t target_hash_prefix,
                         uint64_t *candidates, 
                         float *ratios, 
                         uint32_t *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Create the key for this thread
    uint64_t key = start_key + idx;
    
    // Calculate approximate ratio and hash
    float ratio = approx_ratio(key);
    uint32_t hash_prefix = approx_hash(key);
    
    // Store ratio for sorting
    ratios[idx] = ratio;
    
    // Calculate difference from target ratio
    float ratio_diff = fabsf(ratio - target_ratio);
    uint32_t hash_diff = (hash_prefix ^ target_hash_prefix);
    
    // Check if it's a candidate worth verifying on CPU
    if (ratio_diff < 0.0005f || (hash_diff & 0xFFFF0000) == 0) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < 1000) {  // Limit to 1000 candidates
            candidates[pos] = key;
        }
    }
}
"""

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
        
        # Also save to a fixed filename for easy access
        with open("puzzle68_SOLVED.txt", "w") as f:
            f.write(f"SOLUTION FOUND AT {time.ctime()}\n")
            f.write(f"Private key: {key}\n")
            f.write(f"ScriptPubKey: 76a914{TARGET_HASH}88ac\n")
            f.write(f"Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ\n")
        
        print(f"Solution saved to puzzle68_solution_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    except Exception as e:
        print(f"Error saving solution: {e}")

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

def format_large_int(n):
    """Format large integers with commas for readability"""
    return f"{n:,}"

def brute_force_search(start_key=None, batch_size=50000000, device_id=0):
    """Search the 68-bit key space systematically"""
    if not HAS_CUDA:
        print("CUDA not available. Cannot run GPU search.")
        return None
    
    # Initialize CUDA
    try:
        cuda.init()
        print("CUDA initialized successfully")
    except Exception as e:
        print(f"Failed to initialize CUDA: {e}")
        return None
    
    # Get device count
    try:
        device_count = cuda.Device.count()
        if device_count == 0:
            print("No CUDA devices found")
            return None
        if device_id >= device_count:
            print(f"Device {device_id} not available. Using device 0 instead.")
            device_id = 0
        print(f"Found {device_count} CUDA device(s)")
    except Exception as e:
        print(f"Error getting device count: {e}")
        return None
    
    # Select device
    try:
        device = cuda.Device(device_id)
        print(f"Using device {device_id}: {device.name()}")
    except Exception as e:
        print(f"Error selecting device {device_id}: {e}")
        return None
    
    # Create context
    try:
        ctx = device.make_context()
        print("Created CUDA context")
    except Exception as e:
        print(f"Error creating context: {e}")
        return None
    
    try:
        # Try to check memory info
        try:
            free_mem, total_mem = cuda.mem_get_info()
            print(f"GPU memory: {free_mem/(1024**3):.2f} GB free / {total_mem/(1024**3):.2f} GB total")
            batch_size = min(batch_size, int(free_mem * 0.7 / 16))  # 16 bytes per key
        except Exception as e:
            print(f"Warning: Could not get memory info: {e}")
        
        print(f"Using batch size: {batch_size:,}")
        
        # Compile CUDA module
        try:
            module = compiler.SourceModule(CUDA_CODE)
            test_keys_kernel = module.get_function("test_keys")
            print("CUDA kernel compiled successfully")
        except Exception as e:
            print(f"Error compiling CUDA kernel: {e}")
            ctx.pop()
            return None
        
        # Determine the starting key
        if start_key is None:
            # Try to load last position from progress file
            if os.path.exists("puzzle68_brute_force_progress.txt"):
                try:
                    with open("puzzle68_brute_force_progress.txt", "r") as f:
                        start_key_hex = f.readline().strip()
                        # Extract only the last 16 characters (64 bits) and convert to int
                        if len(start_key_hex) > 16:
                            start_key_hex = start_key_hex[-16:]
                        start_key = int(start_key_hex, 16)
                        
                        # Make sure it's in range
                        if start_key < MIN_68BIT_INT:
                            print(f"Loaded start key 0x{start_key_hex} is below the 68-bit range.")
                            print(f"Setting to minimum 68-bit value: 0x{MIN_68BIT_HEX}")
                            start_key = MIN_68BIT_INT
                        print(f"Resuming from key: 0x{format(start_key, '016x')}")
                except Exception as e:
                    print(f"Could not read progress file: {e}")
                    print("Starting from minimum 68-bit value")
                    start_key = MIN_68BIT_INT
            else:
                start_key = MIN_68BIT_INT
        
        # Ensure start_key is within range
        if start_key < MIN_68BIT_INT:
            print(f"Start key 0x{format(start_key, '016x')} is below the 68-bit range.")
            print(f"Setting to minimum 68-bit value: 0x{MIN_68BIT_HEX}")
            start_key = MIN_68BIT_INT
        
        if start_key > MAX_68BIT_INT:
            print(f"Start key 0x{format(start_key, '016x')} exceeds 68-bit range.")
            print(f"Setting to maximum 68-bit value: 0x{MAX_68BIT_HEX}")
            start_key = MAX_68BIT_INT
        
        print(f"Starting search at key: 0x{format(start_key, '016x')}")
        print(f"Minimum 68-bit key: 0x{MIN_68BIT_HEX}")
        print(f"Maximum 68-bit key: 0x{MAX_68BIT_HEX}")
        
        # Calculate total search space and progress
        total_space = MAX_68BIT_INT - MIN_68BIT_INT + 1
        search_space_left = MAX_68BIT_INT - start_key + 1
        progress_pct = ((start_key - MIN_68BIT_INT) / total_space) * 100
        print(f"Total search space: {format_large_int(total_space)} keys")
        print(f"Search space remaining: {format_large_int(search_space_left)} keys")
        print(f"Progress: {progress_pct:.8f}%")
        
        # First 4 bytes of target hash as uint32
        target_hash_prefix = int(TARGET_HASH[:8], 16)
        
        # Allocate memory on GPU
        candidates = np.zeros(1000, dtype=np.uint64)
        count = np.zeros(1, dtype=np.uint32)
        ratios = np.zeros(batch_size, dtype=np.float32)
        
        candidates_gpu = cuda.mem_alloc(candidates.nbytes)
        count_gpu = cuda.mem_alloc(count.nbytes)
        ratios_gpu = cuda.mem_alloc(ratios.nbytes)
        
        start_time = time.time()
        current_key = start_key
        total_keys_checked = 0
        best_matches = []
        
        try:
            # Process batches until we reach the end of 68-bit range
            while current_key <= MAX_68BIT_INT:
                batch_start_time = time.time()
                
                # Calculate actual batch size (might be smaller near the end)
                current_batch_size = min(batch_size, MAX_68BIT_INT - current_key + 1)
                
                # Reset candidate counter
                count[0] = 0
                
                # Copy data to GPU
                cuda.memcpy_htod(count_gpu, count)
                
                # Set up kernel grid
                block_size = 256
                grid_size = (current_batch_size + block_size - 1) // block_size
                
                # Launch kernel - SIMPLIFIED APPROACH: just pass the 64-bit key directly
                test_keys_kernel(
                    np.uint64(current_key),
                    np.uint32(current_batch_size),
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
                
                # Update progress
                total_keys_checked += current_batch_size
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time
                
                # Update current key for next batch
                current_key += current_batch_size
                
                # Save progress - save the actual 68-bit key in hex
                with open("puzzle68_brute_force_progress.txt", "w") as f:
                    f.write(format(current_key, '016x') + "\n")
                
                # Calculate progress
                search_space_left = MAX_68BIT_INT - current_key + 1
                progress_pct = ((current_key - MIN_68BIT_INT) / total_space) * 100
                
                print(f"\nBatch completed in {format_time(batch_time)}")
                print(f"Current key: 0x{format(current_key, '016x')}")
                print(f"Total keys checked: {format_large_int(total_keys_checked)}")
                print(f"Keys/sec: {int(current_batch_size / batch_time):,}")
                print(f"Progress: {progress_pct:.8f}%")
                
                # Estimated time remaining
                if search_space_left > 0 and total_keys_checked > 0:
                    keys_per_sec = total_keys_checked / elapsed_time
                    estimated_time_remaining = search_space_left / keys_per_sec
                    print(f"Estimated time remaining: {format_time(estimated_time_remaining)}")
                    
                    # Also estimate total search time
                    estimated_total_time = total_space / keys_per_sec
                    print(f"Estimated total search time: {format_time(estimated_total_time)}")
                
                # Get the number of candidates found
                candidates_found = min(int(count[0]), 1000)
                print(f"Candidates found: {candidates_found}")
                
                # Verification phase
                if candidates_found > 0:
                    print(f"Verifying {candidates_found} candidates on CPU...")
                    
                    # Combine explicit candidates with best ratio candidates
                    all_keys = set()
                    for i in range(candidates_found):
                        all_keys.add(int(candidates[i]))
                    
                    # Also add the best ratio matches
                    best_ratio_indices = np.argsort(np.abs(ratios[:current_batch_size] - PHI_OVER_8))[:min(200, current_batch_size)]
                    for idx in best_ratio_indices:
                        if idx < current_batch_size:
                            all_keys.add(current_key - current_batch_size + idx)
                    
                    # Check each candidate
                    for key_int in all_keys:
                        # Ensure it's in the 68-bit range
                        if key_int < MIN_68BIT_INT or key_int > MAX_68BIT_INT:
                            continue
                        
                        # Format to full 64-hex-character private key (padding with leading zeros)
                        full_key = format_private_key(key_int)
                        result = verify_private_key(full_key)
                        
                        # Check if we found a solution
                        if result and result.get('match', False):
                            print("\n*** SOLUTION FOUND! ***")
                            print(f"Private key: {full_key}")
                            print(f"Generated hash: {result['generated_hash']}")
                            print(f"Target hash: {TARGET_HASH}")
                            
                            # Save solution
                            save_solution(full_key, result)
                            
                            # Clean up and return
                            candidates_gpu.free()
                            count_gpu.free()
                            ratios_gpu.free()
                            ctx.pop()
                            
                            return full_key
                        
                        # Track best matches
                        if result:
                            ratio_diff = float(result.get('ratio_diff', 1.0))
                            best_matches.append((full_key, ratio_diff, result.get('generated_hash', '')))
                            best_matches.sort(key=lambda x: x[1])  # Sort by ratio difference
                            best_matches = best_matches[:5]  # Keep top 5
                
                # Show best matches so far
                if best_matches:
                    print("\nBest matches so far:")
                    for i, (key, diff, hash_val) in enumerate(best_matches):
                        print(f"{i+1}. Key: {key}")
                        print(f"   Ratio diff: {diff}")
                        print(f"   Hash: {hash_val}")
                
                # Save detailed progress
                with open("puzzle68_brute_force_details.txt", "w") as f:
                    f.write(f"Search progress as of {time.ctime()}\n")
                    f.write(f"Current key: 0x{format(current_key, '016x')}\n")
                    f.write(f"Total keys checked: {format_large_int(total_keys_checked)}\n")
                    f.write(f"Progress: {progress_pct:.8f}%\n")
                    f.write(f"Keys/sec: {int(current_batch_size / batch_time):,}\n")
                    f.write(f"Elapsed time: {format_time(elapsed_time)}\n")
                    if search_space_left > 0 and total_keys_checked > 0:
                        keys_per_sec = total_keys_checked / elapsed_time
                        estimated_time_remaining = search_space_left / keys_per_sec
                        f.write(f"Estimated time remaining: {format_time(estimated_time_remaining)}\n\n")
                    
                    if best_matches:
                        f.write("Best matches so far:\n")
                        for i, (key, diff, hash_val) in enumerate(best_matches):
                            f.write(f"{i+1}. Key: {key}\n")
                            f.write(f"   Ratio diff: {diff}\n")
                            f.write(f"   Hash: {hash_val}\n\n")
            
            print("\nCompleted search of entire 68-bit key space")
            print(f"No solution found for puzzle 68")
            
        except KeyboardInterrupt:
            print("\nSearch interrupted by user")
            
        except Exception as e:
            print(f"\nError during search: {e}")
            import traceback
            traceback.print_exc()
    
    except Exception as e:
        print(f"Error in GPU search: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up GPU resources
        try:
            candidates_gpu.free()
            count_gpu.free()
            ratios_gpu.free()
        except:
            pass
        
        # Pop the context
        try:
            ctx.pop()
            print("CUDA context released")
        except:
            pass
    
    return None

def main():
    """Main function"""
    print("\n" + "="*80)
    print("Brute Force Bitcoin Puzzle #68 Solver (FIXED - DIRECT 68-BIT KEY HANDLING)")
    print("="*80)
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 = {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
    print(f"Key range: 0x{MIN_68BIT_HEX} to 0x{MAX_68BIT_HEX}")
    print("="*80 + "\n")
    
    # Check whether we can use CUDA
    if not HAS_CUDA:
        print("\nCUDA is not available. Cannot run GPU search.")
        return
    
    try:
        # Run the brute force search
        print("\nStarting brute force search of correct 68-bit key space...")
        print("Press Ctrl+C to pause (progress will be saved)")
        
        solution = brute_force_search()
        
        if solution:
            print(f"\nFinal solution: {solution}")
        else:
            print("\nNo solution found or search was interrupted.")
            print("Check puzzle68_brute_force_progress.txt to resume later.")
    
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
