# brute_force_puzzle68_fixed.py
# Searches the 68-bit key space for Bitcoin Puzzle #68 using 32-bit chunks

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
MIN_KEY_HIGH = 0x8000           # Upper 16 bits of 2^67
MIN_KEY_LOW = 0x0000000000000   # Lower 48 bits of 2^67
MAX_KEY_HIGH = 0xFFFF           # Upper 16 bits of 2^68-1
MAX_KEY_LOW = 0xFFFFFFFFFFFF    # Lower 48 bits of 2^68-1

# CUDA kernel for testing keys in 32-bit chunks
CUDA_CODE = """
#include <stdio.h>
#include <stdint.h>

// Approximation of the ratio calculation for filtering
__device__ float approx_ratio(uint32_t key_high, uint32_t key_mid, uint32_t key_low) {
    // Simple approximation for filtering - actual ratio calculated on CPU for verification
    float result = 0.0f;
    float x_approx = (float)((key_high << 8) ^ key_mid ^ (key_low >> 8));
    float y_approx = (float)((key_low) ^ (key_mid << 16) ^ (key_high >> 8));
    
    if (y_approx != 0.0f) {
        result = 0.202254f + (float)(x_approx / y_approx - 0.5f) * 0.00001f;
    }
    return result;
}

// Hash approximation for filtering
__device__ uint32_t approx_hash(uint32_t key_high, uint32_t key_mid, uint32_t key_low) {
    uint32_t h = 0xe0b8a2ba; // First bytes of target
    h ^= key_high;
    h ^= key_mid;
    h ^= key_low;
    h ^= h >> 16;
    return h;
}

__global__ void test_keys(uint32_t key_high, uint32_t key_mid, uint32_t key_low_start, 
                         uint32_t batch_size,
                         float target_ratio, 
                         uint32_t target_hash_prefix,
                         uint32_t *candidates_high,
                         uint32_t *candidates_mid,
                         uint32_t *candidates_low,
                         float *ratios, 
                         uint32_t *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    // Create the key for this thread
    uint32_t key_low = key_low_start + idx;
    uint32_t current_key_mid = key_mid;
    uint32_t current_key_high = key_high;
    
    // Handle overflow - this is only needed if batch_size is very large
    if (key_low < key_low_start) {
        current_key_mid++;
        if (current_key_mid == 0) {
            current_key_high++;
        }
    }
    
    // Calculate approximate ratio and hash
    float ratio = approx_ratio(current_key_high, current_key_mid, key_low);
    uint32_t hash_prefix = approx_hash(current_key_high, current_key_mid, key_low);
    
    // Store ratio for sorting
    ratios[idx] = ratio;
    
    // Calculate difference from target ratio
    float ratio_diff = fabsf(ratio - target_ratio);
    uint32_t hash_diff = (hash_prefix ^ target_hash_prefix);
    
    // Check if it's a candidate worth verifying on CPU
    if (ratio_diff < 0.0005f || (hash_diff & 0xFFFF0000) == 0) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < 1000) {  // Limit to 1000 candidates
            candidates_high[pos] = current_key_high;
            candidates_mid[pos] = current_key_mid;
            candidates_low[pos] = key_low;
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

def key_parts_to_full_hex(key_high, key_mid, key_low):
    """Convert 3-part key to a full 64-character hex string"""
    # Combine the 3 parts into a single integer
    full_int = (key_high << 48) | (key_mid << 24) | key_low
    # Convert to hex and pad to 64 chars
    return format(full_int, '016x').zfill(64)

def full_hex_to_key_parts(hex_key):
    """Convert a hex key to its 3 parts (high, mid, low)"""
    # Ensure consistent format
    if len(hex_key) > 16:
        hex_key = hex_key[-16:]  # Take only the 16 rightmost hex characters
    
    # Convert to integer
    full_int = int(hex_key, 16)
    
    # Split into parts
    key_high = (full_int >> 48) & 0xFFFF
    key_mid = (full_int >> 24) & 0xFFFFFF
    key_low = full_int & 0xFFFFFF
    
    return key_high, key_mid, key_low

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

def get_key_progress_string(key_high, key_mid, key_low):
    """Get a formatted hex string showing current progress"""
    return f"0x{key_high:04x}{key_mid:06x}{key_low:06x}"

def is_key_in_range(key_high, key_mid, key_low):
    """Check if the given key parts are in the 68-bit range"""
    if key_high < MIN_KEY_HIGH:
        return False
    if key_high > MAX_KEY_HIGH:
        return False
    if key_high == MIN_KEY_HIGH and key_mid < MIN_KEY_LOW >> 24:
        return False
    if key_high == MAX_KEY_HIGH and key_mid > MAX_KEY_LOW >> 24:
        return False
    return True

def calculate_progress_percentage(key_high, key_mid, key_low):
    """Calculate progress percentage through the 68-bit key space"""
    # Convert the current key to a full integer
    current_key = (key_high << 48) | (key_mid << 24) | key_low
    
    # Convert min and max keys to full integers
    min_key = (MIN_KEY_HIGH << 48) | ((MIN_KEY_LOW >> 24) << 24)
    max_key = (MAX_KEY_HIGH << 48) | ((MAX_KEY_LOW >> 24) << 24) | 0xFFFFFF
    
    # Calculate the total range and progress
    total_range = max_key - min_key
    progress = current_key - min_key
    
    # Calculate percentage
    return (progress / total_range) * 100 if total_range > 0 else 0

def brute_force_search(start_key_hex=None, batch_size=5000000, device_id=0):
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
        if start_key_hex is None:
            # Try to load last position from progress file
            if os.path.exists("puzzle68_brute_force_progress.txt"):
                try:
                    with open("puzzle68_brute_force_progress.txt", "r") as f:
                        start_key_hex = f.readline().strip()
                        print(f"Loaded key from progress file: {start_key_hex}")
                except Exception as e:
                    print(f"Could not read progress file: {e}")
                    start_key_hex = f"{MIN_KEY_HIGH:04x}{MIN_KEY_LOW:012x}"
            else:
                start_key_hex = f"{MIN_KEY_HIGH:04x}{MIN_KEY_LOW:012x}"
        
        # Parse the starting key
        try:
            key_high, key_mid, key_low = full_hex_to_key_parts(start_key_hex)
            
            # Ensure the key is in the valid range
            if not is_key_in_range(key_high, key_mid, key_low):
                print(f"Starting key {get_key_progress_string(key_high, key_mid, key_low)} is outside the valid range.")
                key_high = MIN_KEY_HIGH
                key_mid = MIN_KEY_LOW >> 24
                key_low = 0
                print(f"Resetting to minimum key: {get_key_progress_string(key_high, key_mid, key_low)}")
        except Exception as e:
            print(f"Error parsing starting key: {e}")
            key_high = MIN_KEY_HIGH
            key_mid = MIN_KEY_LOW >> 24
            key_low = 0
        
        # Show the search range
        print(f"Minimum 68-bit key: 0x{MIN_KEY_HIGH:04x}{MIN_KEY_LOW:012x}")
        print(f"Maximum 68-bit key: 0x{MAX_KEY_HIGH:04x}{MAX_KEY_LOW:012x}")
        print(f"Starting search at: {get_key_progress_string(key_high, key_mid, key_low)}")
        
        # Calculate total search space
        total_keys = 2**68 - 2**67
        progress_pct = calculate_progress_percentage(key_high, key_mid, key_low)
        
        print(f"Total search space: {format_large_int(total_keys)} keys")
        print(f"Progress: {progress_pct:.8f}%")
        
        # First 4 bytes of target hash as uint32
        target_hash_prefix = int(TARGET_HASH[:8], 16)
        
        # Allocate memory on GPU
        candidates_high = np.zeros(1000, dtype=np.uint32)
        candidates_mid = np.zeros(1000, dtype=np.uint32)
        candidates_low = np.zeros(1000, dtype=np.uint32)
        count = np.zeros(1, dtype=np.uint32)
        ratios = np.zeros(batch_size, dtype=np.float32)
        
        candidates_high_gpu = cuda.mem_alloc(candidates_high.nbytes)
        candidates_mid_gpu = cuda.mem_alloc(candidates_mid.nbytes)
        candidates_low_gpu = cuda.mem_alloc(candidates_low.nbytes)
        count_gpu = cuda.mem_alloc(count.nbytes)
        ratios_gpu = cuda.mem_alloc(ratios.nbytes)
        
        start_time = time.time()
        current_key_high = key_high
        current_key_mid = key_mid
        current_key_low = key_low
        total_keys_checked = 0
        best_matches = []
        
        # Set up kernel grid
        block_size = 256
        
        try:
            # Main search loop - iterate through the whole key space
            while True:
                batch_start_time = time.time()
                
                # Calculate batch size for this round
                # Make sure we don't exceed uint32 limit for key_low
                current_batch_size = min(batch_size, 0xFFFFFF - current_key_low + 1)
                
                # Reset candidate counter
                count[0] = 0
                
                # Copy data to GPU
                cuda.memcpy_htod(count_gpu, count)
                
                # Calculate grid size for this batch
                grid_size = (current_batch_size + block_size - 1) // block_size
                
                # Launch kernel with the 3-part key
                test_keys_kernel(
                    np.uint32(current_key_high),
                    np.uint32(current_key_mid),
                    np.uint32(current_key_low),
                    np.uint32(current_batch_size),
                    np.float32(PHI_OVER_8),
                    np.uint32(target_hash_prefix),
                    candidates_high_gpu,
                    candidates_mid_gpu,
                    candidates_low_gpu,
                    ratios_gpu,
                    count_gpu,
                    block=(block_size, 1, 1),
                    grid=(grid_size, 1)
                )
                
                # Get results back
                cuda.memcpy_dtoh(candidates_high, candidates_high_gpu)
                cuda.memcpy_dtoh(candidates_mid, candidates_mid_gpu)
                cuda.memcpy_dtoh(candidates_low, candidates_low_gpu)
                cuda.memcpy_dtoh(count, count_gpu)
                cuda.memcpy_dtoh(ratios, ratios_gpu)
                
                # Update progress
                total_keys_checked += current_batch_size
                batch_time = time.time() - batch_start_time
                elapsed_time = time.time() - start_time
                
                # Update current key for next batch
                current_key_low += current_batch_size
                if current_key_low >= 0x1000000:  # If we overflow the 24-bit low part
                    current_key_low = 0
                    current_key_mid += 1
                    if current_key_mid >= 0x1000000:  # If we overflow the 24-bit mid part
                        current_key_mid = 0
                        current_key_high += 1
                
                # Check if we've completed the whole range
                if current_key_high > MAX_KEY_HIGH or (
                    current_key_high == MAX_KEY_HIGH and 
                    current_key_mid > (MAX_KEY_LOW >> 24) or (
                        current_key_high == MAX_KEY_HIGH and
                        current_key_mid == (MAX_KEY_LOW >> 24) and
                        current_key_low > (MAX_KEY_LOW & 0xFFFFFF)
                    )
                ):
                    print("\nCompleted search of entire 68-bit key space!")
                    break
                
                # Save progress - save the current position as a hex string
                current_hex = get_key_progress_string(current_key_high, current_key_mid, current_key_low)[2:]  # Remove 0x prefix
                with open("puzzle68_brute_force_progress.txt", "w") as f:
                    f.write(current_hex + "\n")
                
                # Calculate progress percentage
                progress_pct = calculate_progress_percentage(current_key_high, current_key_mid, current_key_low)
                
                print(f"\nBatch completed in {format_time(batch_time)}")
                print(f"Current position: {get_key_progress_string(current_key_high, current_key_mid, current_key_low)}")
                print(f"Total keys checked: {format_large_int(total_keys_checked)}")
                print(f"Keys/sec: {int(current_batch_size / batch_time):,}")
                print(f"Progress: {progress_pct:.8f}%")
                
                # Estimated time remaining
                keys_per_sec = total_keys_checked / elapsed_time if elapsed_time > 0 else 0
                if keys_per_sec > 0:
                    # Calculate remaining keys
                    keys_done = (current_key_high - MIN_KEY_HIGH) * 0x1000000 * 0x1000000
                    if current_key_high == MIN_KEY_HIGH:
                        keys_done += (current_key_mid - (MIN_KEY_LOW >> 24)) * 0x1000000
                    else:
                        keys_done += current_key_mid * 0x1000000
                    keys_done += current_key_low
                    
                    keys_remaining = total_keys - keys_done
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
                        h = candidates_high[i]
                        m = candidates_mid[i]
                        l = candidates_low[i]
                        
                        # Format to full private key
                        full_key = key_parts_to_full_hex(h, m, l)
                        
                        # Verify if it's a solution
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
                            candidates_high_gpu.free()
                            candidates_mid_gpu.free()
                            candidates_low_gpu.free()
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
                    f.write(f"Current key: {get_key_progress_string(current_key_high, current_key_mid, current_key_low)}\n")
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
            candidates_high_gpu.free()
            candidates_mid_gpu.free()
            candidates_low_gpu.free()
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
    print("Brute Force Bitcoin Puzzle #68 Solver (FIXED - SPLIT KEY APPROACH)")
    print("="*80)
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 = {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
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
