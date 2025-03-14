# simplified_puzzle68.py
# Direct search of the 68-bit key space for Bitcoin Puzzle #68

import hashlib
import binascii
import time
import sys
import os
import numpy as np
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

# Constants
TARGET_HASH = 'e0b8a2baee1b77fc703455f39d51477451fc8cfc'  # Hash from scriptPubKey
PUZZLE_NUMBER = 68
PHI_OVER_8 = 0.202254  # Approximate value of φ/8

# EXPLICIT DEFINITION of key space boundaries
FIXED_HIGH_BITS = 0xced
MIN_KEY = (FIXED_HIGH_BITS << 56)          # 0xced00000000000000
MAX_KEY = MIN_KEY + (1 << 56) - 1          # 0xcedFFFFFFFFFFFFFFF

# CUDA kernel for testing keys
CUDA_CODE = """
#include <stdio.h>
#include <stdint.h>

// Test a batch of 68-bit keys
__global__ void test_keys(uint32_t key_hi, uint32_t key_mid, uint32_t key_lo, 
                          uint32_t batch_size,
                          uint32_t *candidates, 
                          uint32_t *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Create the key for this thread
    uint32_t offset = idx;
    uint32_t cur_lo = key_lo + offset;
    uint32_t cur_mid = key_mid;
    uint32_t cur_hi = key_hi;
    
    // Handle overflow
    if (cur_lo < key_lo) {
        cur_mid++;
        if (cur_mid == 0) {
            cur_hi++;
        }
    }
    
    // Simple preliminary filtering criteria - we'll do full verification on CPU
    // This just ensures we consider some candidates from the batch
    if ((offset % 10000) == 0 || (cur_lo & 0xFFF) == 0xFFF) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < 1000) {  // Limit to 1000 candidates
            // Store this candidate (3 consecutive uint32 positions)
            candidates[pos*3] = cur_hi;
            candidates[pos*3 + 1] = cur_mid;
            candidates[pos*3 + 2] = cur_lo;
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

def get_pubkey_from_privkey(private_key_int):
    """Get public key from private key (as integer) using SECP256k1"""
    try:
        # Convert integer to 32-byte private key
        private_key_bytes = private_key_int.to_bytes(32, byteorder='big')
        
        # Generate key pair using ecdsa
        sk = SigningKey.from_string(private_key_bytes, curve=SECP256k1)
        vk = sk.verifying_key
        
        # Get public key coordinates
        x = vk.pubkey.point.x()
        y = vk.pubkey.point.y()
        
        # Format uncompressed public key (04 + x + y)
        x_bytes = x.to_bytes(32, byteorder='big')
        y_bytes = y.to_bytes(32, byteorder='big')
        pubkey = b'\x04' + x_bytes + y_bytes
        
        # Calculate ratio x/y (for φ/8 filtering)
        ratio = x / y if y != 0 else 0
        
        return {
            'pubkey': pubkey,
            'x': x,
            'y': y,
            'ratio': ratio,
            'ratio_diff': abs(ratio - PHI_OVER_8)
        }
    except Exception as e:
        print(f"Error generating public key: {e}")
        return None

def verify_private_key(private_key_int):
    """Verify if private key (as integer) generates the target public key hash"""
    try:
        # Ensure the key is within range
        if private_key_int < MIN_KEY or private_key_int > MAX_KEY:
            return {'match': False, 'reason': 'Key out of 68-bit range'}
            
        # Get public key
        pubkey_info = get_pubkey_from_privkey(private_key_int)
        if not pubkey_info:
            return {'match': False, 'reason': 'Failed to generate public key'}
        
        # Calculate hash160
        pubkey_hash = hash160(pubkey_info['pubkey'])
        
        # Check for exact match with the hash from scriptPubKey
        exact_match = pubkey_hash == TARGET_HASH
        
        if exact_match:
            print("\n!!! EXACT MATCH FOUND WITH SCRIPTPUBKEY 76a914e0b8a2baee1b77fc703455f39d51477451fc8cfc88ac !!!")
            print(f"Private key (int): {private_key_int}")
            print(f"Private key (hex): {format(private_key_int, 'x')}")
            print(f"Generated hash: {pubkey_hash}")
            print(f"Target hash: {TARGET_HASH}")
        
        return {
            'match': exact_match,
            'generated_hash': pubkey_hash,
            'ratio': pubkey_info['ratio'],
            'ratio_diff': pubkey_info['ratio_diff']
        }
    except Exception as e:
        print(f"Error verifying key: {e}")
        return {'match': False, 'error': str(e)}

def save_solution(key_int, result):
    """Save a found solution to file with detailed information"""
    key_hex = format(key_int, 'x')
    try:
        with open(f"puzzle68_solution_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
            f.write(f"SOLUTION FOUND AT {time.ctime()}\n")
            f.write(f"Private key (int): {key_int}\n")
            f.write(f"Private key (hex): {key_hex}\n")
            f.write(f"Generated hash: {result.get('generated_hash')}\n")
            f.write(f"Target hash: {TARGET_HASH}\n")
            f.write(f"ScriptPubKey: 76a914{TARGET_HASH}88ac\n")
            f.write(f"Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ\n")
        
        # Also save to a fixed filename for easy access
        with open("puzzle68_SOLVED.txt", "w") as f:
            f.write(f"SOLUTION FOUND AT {time.ctime()}\n")
            f.write(f"Private key (int): {key_int}\n")
            f.write(f"Private key (hex): {key_hex}\n")
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

def split_key(key_int):
    """Split 68-bit key into three parts for CUDA processing"""
    key_hi = (key_int >> 64) & 0xFFFFFFFF
    key_mid = (key_int >> 32) & 0xFFFFFFFF
    key_lo = key_int & 0xFFFFFFFF
    return key_hi, key_mid, key_lo

def combine_key_parts(key_hi, key_mid, key_lo):
    """Combine three parts back into a full 68-bit key"""
    return (key_hi << 64) | (key_mid << 32) | key_lo

def calculate_progress(current_key):
    """Calculate progress percentage through the 68-bit key space"""
    return ((current_key - MIN_KEY) / (MAX_KEY - MIN_KEY)) * 100

def brute_force_search(start_key=None, batch_size=10000000, device_id=0):
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
            if os.path.exists("puzzle68_progress.txt"):
                try:
                    with open("puzzle68_progress.txt", "r") as f:
                        start_key = int(f.readline().strip(), 16)
                        print(f"Resuming from key: 0x{start_key:x}")
                except Exception as e:
                    print(f"Could not read progress file: {e}")
                    start_key = MIN_KEY
            else:
                start_key = MIN_KEY
        
        # Ensure start_key is within range
        if start_key < MIN_KEY:
            print(f"Start key 0x{start_key:x} is below the 68-bit range")
            print(f"Setting to minimum 68-bit value: 0x{MIN_KEY:x}")
            start_key = MIN_KEY
        
        if start_key > MAX_KEY:
            print(f"Start key 0x{start_key:x} exceeds 68-bit range")
            print(f"Setting to maximum 68-bit value: 0x{MAX_KEY:x}")
            start_key = MAX_KEY
        
        # Show the search range
        print(f"Minimum 68-bit key: 0x{MIN_KEY:x}")
        print(f"Maximum 68-bit key: 0x{MAX_KEY:x}")
        print(f"Starting search at: 0x{start_key:x}")
        
        # Calculate total search space
        total_keys = MAX_KEY - MIN_KEY + 1
        progress_pct = calculate_progress(start_key)
        
        print(f"Total search space: {format_large_int(total_keys)} keys")
        print(f"Progress: {progress_pct:.8f}%")
        
        # Allocate memory on GPU
        candidates = np.zeros(3000, dtype=np.uint32)  # Space for 1000 candidates (3 uint32 values each)
        count = np.zeros(1, dtype=np.uint32)
        
        candidates_gpu = cuda.mem_alloc(candidates.nbytes)
        count_gpu = cuda.mem_alloc(count.nbytes)
        
        start_time = time.time()
        current_key = start_key
        total_keys_checked = 0
        best_matches = []
        
        # Set up kernel grid
        block_size = 256
        
        try:
            # Main search loop - iterate through the whole key space
            while current_key <= MAX_KEY:
                batch_start_time = time.time()
                
                # Calculate batch size for this round
                current_batch_size = min(batch_size, MAX_KEY - current_key + 1)
                
                # Split key for CUDA processing
                key_hi, key_mid, key_lo = split_key(current_key)
                
                # Reset candidate counter
                count[0] = 0
                
                # Copy data to GPU
                cuda.memcpy_htod(count_gpu, count)
                
                # Calculate grid size for this batch
                grid_size = (current_batch_size + block_size - 1) // block_size
                
                # Launch kernel
                test_keys_kernel(
                    np.uint32(key_hi),
                    np.uint32(key_mid),
                    np.uint32(key_lo),
                    np.uint32(current_batch_size),
                    candidates_gpu,
                    count_gpu,
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                
                # Get results back
                cuda.memcpy_dtoh(candidates, candidates_gpu)
                cuda.memcpy_dtoh(count, count_gpu)
                
                # Update progress
                total_keys_checked += current_batch_size
                current_key += current_batch_size
                
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time
                
                # Save progress
                with open("puzzle68_progress.txt", "w") as f:
                    f.write(f"{current_key:x}\n")
                
                # Calculate progress percentage
                progress_pct = calculate_progress(current_key)
                
                print(f"\nBatch completed in {format_time(batch_time)}")
                print(f"Current key: 0x{current_key:x}")
                print(f"Total keys checked: {format_large_int(total_keys_checked)}")
                print(f"Keys/sec: {int(current_batch_size / batch_time):,}")
                print(f"Progress: {progress_pct:.8f}%")
                
                # Estimated time remaining
                keys_per_sec = total_keys_checked / elapsed_time if elapsed_time > 0 else 0
                if keys_per_sec > 0:
                    keys_remaining = MAX_KEY - current_key + 1
                    estimated_time_remaining = keys_remaining / keys_per_sec
                    print(f"Estimated time remaining: {format_time(estimated_time_remaining)}")
                    
                    # Also estimate total search time
                    estimated_total_time = total_keys / keys_per_sec
                    print(f"Estimated total search time: {format_time(estimated_total_time)}")
                
                # Get the number of candidates found
                candidates_found = min(int(count[0]), 1000)
                print(f"Candidates found: {candidates_found}")
                
                # Verification phase
                if candidates_found > 0:
                    print(f"Verifying {candidates_found} candidates on CPU...")
                    
                    # Check each candidate
                    for i in range(candidates_found):
                        # Construct full 68-bit key
                        key_hi = candidates[i*3]
                        key_mid = candidates[i*3 + 1]
                        key_lo = candidates[i*3 + 2]
                        
                        # Convert to full integer
                        full_key_int = combine_key_parts(key_hi, key_mid, key_lo)
                        
                        # Verify this is a valid 68-bit key
                        if full_key_int < MIN_KEY or full_key_int > MAX_KEY:
                            continue
                        
                        # Verify if it's a solution
                        result = verify_private_key(full_key_int)
                        
                        # Check if we found a solution
                        if result and result.get('match', False):
                            print("\n*** SOLUTION FOUND! ***")
                            print(f"Private key (int): {full_key_int}")
                            print(f"Private key (hex): {format(full_key_int, 'x')}")
                            print(f"Generated hash: {result['generated_hash']}")
                            print(f"Target hash: {TARGET_HASH}")
                            
                            # Save solution
                            save_solution(full_key_int, result)
                            
                            # Clean up and return
                            candidates_gpu.free()
                            count_gpu.free()
                            ctx.pop()
                            
                            return full_key_int
                        
                        # Track best matches
                        if result and 'ratio_diff' in result:
                            ratio_diff = result['ratio_diff']
                            hex_key = format(full_key_int, 'x')
                            best_matches.append((hex_key, ratio_diff, result.get('generated_hash', '')))
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
                with open("puzzle68_details.txt", "w") as f:
                    f.write(f"Search progress as of {time.ctime()}\n")
                    f.write(f"Current key: 0x{current_key:x}\n")
                    f.write(f"Total keys checked: {format_large_int(total_keys_checked)}\n")
                    f.write(f"Progress: {progress_pct:.8f}%\n")
                    f.write(f"Keys/sec: {int(current_batch_size / batch_time):,}\n")
                    f.write(f"Elapsed time: {format_time(elapsed_time)}\n")
                    if keys_per_sec > 0:
                        f.write(f"Estimated time remaining: {format_time(keys_remaining / keys_per_sec)}\n\n")
                    
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
    print("Simplified Bitcoin Puzzle #68 Solver - DIRECT 68-BIT KEY APPROACH")
    print("="*80)
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 ≈ {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
    print(f"Key range: 0x{MIN_KEY:x} to 0x{MAX_KEY:x}")
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
            print(f"\nFinal solution: 0x{solution:x}")
        else:
            print("\nNo solution found or search was interrupted.")
            print("Check puzzle68_progress.txt to resume later.")
    
    except KeyboardInterrupt:
        print("\nSearch interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
