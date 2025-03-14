# brute_force_puzzle68.py
# Systematically searches the entire 68-bit key space for Bitcoin Puzzle #68

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

# CUDA kernel for testing keys
CUDA_CODE = """
#include <stdio.h>
#include <stdint.h>

// Approximation of the ratio calculation for filtering
__device__ float approx_ratio(uint64_t key_high, uint64_t key_low) {
    float result = 0.0f;
    float x_approx = (float)(key_high ^ (key_low >> 12));
    float y_approx = (float)(key_low ^ (key_high << 8));
    
    if (y_approx != 0.0f) {
        result = 0.202254f + (float)((x_approx * 723467.0f + y_approx * 13498587.0f) / 
                                   (1e12 + key_low + key_high) - 0.5f) * 0.00001f;
    }
    return result;
}

// Hash approximation for filtering
__device__ uint32_t approx_hash(uint64_t key_high, uint64_t key_low) {
    uint32_t h = 0xe0b8a2ba; // First bytes of target
    h ^= key_high >> 32;
    h ^= key_high;
    h ^= key_low >> 32;
    h ^= key_low;
    h ^= h >> 16;
    return h;
}

__global__ void test_keys(uint64_t start_high, uint64_t start_low, 
                         uint32_t *offsets, int offset_count,
                         float target_ratio, uint32_t target_hash_prefix,
                         uint32_t *candidates, float *ratios, uint32_t *count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= offset_count) return;
    
    // Get this thread's offset
    uint32_t offset = offsets[idx];
    
    // Create full key by adding offset
    uint64_t key_low = start_low + offset;
    uint64_t key_high = start_high;
    if (key_low < start_low) {  // Handle overflow
        key_high++;
    }
    
    // Calculate approximate ratio and hash
    float ratio = approx_ratio(key_high, key_low);
    uint32_t hash_prefix = approx_hash(key_high, key_low);
    
    // Store ratio for sorting
    ratios[idx] = ratio;
    
    // Calculate difference from target ratio
    float ratio_diff = fabsf(ratio - target_ratio);
    uint32_t hash_diff = (hash_prefix ^ target_hash_prefix);
    
    // Check if it's a candidate worth verifying on CPU
    if (ratio_diff < 0.0005f || (hash_diff & 0xFFFF0000) == 0) {
        uint32_t pos = atomicAdd(count, 1);
        if (pos < 1000) {  // Limit to 1000 candidates
            candidates[pos] = offset;
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
        
        # Calculate key space boundaries
        max_68bit = (1 << 68) - 1  # Maximum 68-bit value
        if start_key is None:
            # Try to load last position from progress file
            if os.path.exists("puzzle68_brute_force_progress.txt"):
                try:
                    with open("puzzle68_brute_force_progress.txt", "r") as f:
                        start_key = int(f.readline().strip(), 16)
                        print(f"Resuming from key: {format_private_key(start_key)}")
                except:
                    print("Could not read progress file, starting from 0")
                    start_key = 0
            else:
                start_key = 0
        
        if start_key > max_68bit:
            print(f"Start key exceeds 68-bit range. Setting to maximum 68-bit value.")
            start_key = max_68bit
        
        print(f"Starting search at key: {format_private_key(start_key)}")
        print(f"Maximum 68-bit key: {format_private_key(max_68bit)}")
        
        # Calculate total search space and progress
        total_space = max_68bit + 1  # Total number of 68-bit keys
        search_space_left = max_68bit - start_key + 1
        progress_pct = (1 - (search_space_left / total_space)) * 100
        print(f"Total search space: {format_large_int(total_space)} keys")
        print(f"Search space remaining: {format_large_int(search_space_left)} keys")
        print(f"Progress: {progress_pct:.8f}%")
        
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
        
        start_time = time.time()
        current_key = start_key
        total_keys_checked = 0
        best_matches = []
        
        try:
            # Process batches until we reach the end of 68-bit range
            while current_key <= max_68bit:
                batch_start_time = time.time()
                
                # Calculate actual batch size (might be smaller near the end)
                current_batch_size = min(batch_size, max_68bit - current_key + 1)
                
                # Split the current key into high and low parts
                current_key_high = current_key >> 32
                current_key_low = current_key & 0xFFFFFFFF
                
                # Prepare offsets for this batch
                offsets = np.arange(min(current_batch_size, batch_size), dtype=np.uint32)
                
                # Reset candidate counter
                count[0] = 0
                
                # Copy data to GPU
                cuda.memcpy_htod(offsets_gpu, offsets)
                cuda.memcpy_htod(count_gpu, count)
                
                # Set up kernel grid
                block_size = 256
                grid_size = (current_batch_size + block_size - 1) // block_size
                
                # Launch kernel
                test_keys_kernel(
                    np.uint64(current_key_high),
                    np.uint64(current_key_low),
                    offsets_gpu,
                    np.int32(current_batch_size),
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
                
                # Save progress
                with open("puzzle68_brute_force_progress.txt", "w") as f:
                    f.write(format_private_key(current_key) + "\n")
                
                # Calculate progress
                search_space_left = max_68bit - current_key + 1
                progress_pct = (1 - (search_space_left / total_space)) * 100
                
                print(f"\nBatch completed in {format_time(batch_time)}")
                print(f"Current key: {format_private_key(current_key)}")
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
                    
                    # Sort candidates by ratio proximity to PHI/8
                    best_ratio_indices = np.argsort(np.abs(ratios[:current_batch_size] - PHI_OVER_8))[:min(200, current_batch_size)]
                    
                    # Combine explicit candidates with best ratio candidates
                    all_offsets = set()
                    for i in range(candidates_found):
                        all_offsets.add(int(candidates[i]))
                    
                    for idx in best_ratio_indices:
                        if idx < current_batch_size:  # Make sure index is in bounds
                            all_offsets.add(int(offsets[idx]))
                    
                    # Check each candidate - adjust base key for this batch
                    batch_base_key = current_key - current_batch_size
                    for offset in all_offsets:
                        if offset < current_batch_size:  # Make sure offset is valid
                            full_key_int = batch_base_key + offset
                            
                            # Ensure it's exactly 68 bits
                            if count_bits(full_key_int) != 68:
                                continue
                            
                            # Format and verify
                            full_key = format_private_key(full_key_int)
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
                                offsets_gpu.free()
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
                    f.write(f"Current key: {format_private_key(current_key)}\n")
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
            offsets_gpu.free()
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
    print("Brute Force Bitcoin Puzzle #68 Solver")
    print(f"Target hash: {TARGET_HASH}")
    print(f"Target scriptPubKey: 76a914{TARGET_HASH}88ac")
    print(f"Target Bitcoin address: 1MVDYgVaSN6iKKEsbzRUAYFrYJadLYZvvZ")
    print(f"Target ratio: φ/8 = {PHI_OVER_8}")
    print(f"Searching for private keys with EXACTLY 68 bits")
    
    # Check whether we can use CUDA
    if not HAS_CUDA:
        print("\nCUDA is not available. Cannot run GPU search.")
        return
    
    try:
        # Run the brute force search
        print("\nStarting brute force search of 68-bit key space...")
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
