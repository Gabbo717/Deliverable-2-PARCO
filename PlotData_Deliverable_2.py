import matplotlib.pyplot as plt

def read_floats_in_chunks(filename, chunk_size=200):  
    """
    Reads the file and returns a list of (checkSym_avg, matTranspose_avg) for each chunk of 200 lines.
    - checkSym corresponds to odd-indexed lines in that chunk.
    - matTranspose corresponds to even-indexed lines in that chunk.
    """
    with open(filename, 'r') as f:
        lines = f.read().strip().splitlines()
    
    float_values = [float(line.strip()) for line in lines]
    
    num_chunks = len(float_values) // chunk_size
    
    results = [] 
    
    idx = 0
    for _ in range(num_chunks):
        chunk = float_values[idx : idx + chunk_size]
        idx += chunk_size
        
        odd_values  = [chunk[i] for i in range(0, chunk_size, 2)]  
        even_values = [chunk[i] for i in range(1, chunk_size, 2)]  
        
        # Compute averages
        avg_odds  = sum(odd_values) / len(odd_values)
        avg_evens = sum(even_values) / len(even_values)
        
        results.append((avg_odds, avg_evens))
    
    return results

def main():
    filename = input("Enter the path of the input text file: ").strip()
    mode_str = input("Enter the mode (1 = Sequential, 2 = OMP, 3 = MPI): ").strip()
    mode = int(mode_str)
    
    results = read_floats_in_chunks(filename, chunk_size=200)
    num_blocks = len(results)
    
    matrix_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    if mode == 1:
        if num_blocks != 9:
            print("Warning: For Sequential mode, expected 9 data blocks. Found:", num_blocks)
        
        checkSym_vals  = [r[0] for r in results]  
        transpose_vals = [r[1] for r in results]  
        
        plt.figure(figsize=(8, 6))
        plt.plot(matrix_sizes, checkSym_vals, marker='o', label='checkSym Sequential')
        plt.title("checkSym Sequential")
        plt.xlabel("Matrix Size (n)")
        plt.ylabel("Elaboration Time (s)")
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
        plt.figure(figsize=(8, 6))
        plt.plot(matrix_sizes, transpose_vals, marker='o', label='matTranspose Sequential')
        plt.title("matTranspose Sequential")
        plt.xlabel("Matrix Size (n)")
        plt.ylabel("Elaboration Time (s)")
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    elif mode == 2 or mode == 3:
        if num_blocks != 54:
            print("Warning: For OMP/MPI mode, expected 54 data blocks. Found:", num_blocks)
        
        if mode == 2:
            title_prefix_odd  = "checkSym OMP"
            title_prefix_even = "matTranspose OMP"
            line_label_prefix = "# Threads : "
        else:
            title_prefix_odd  = "checkSym MPI"
            title_prefix_even = "matTranspose MPI"
            line_label_prefix = "# Processes : "
        
        thread_counts = [1, 2, 4, 8, 16, 32]  
        
        checkSym_data  = [[] for _ in range(len(thread_counts))]
        matTrans_data  = [[] for _ in range(len(thread_counts))]
        
        for i in range(num_blocks):
            group_idx = i // 9  
            
            avg_checkSym, avg_matTranspose = results[i]
            
            checkSym_data[group_idx].append(avg_checkSym)
            matTrans_data[group_idx].append(avg_matTranspose)
        
        plt.figure(figsize=(8, 6))
        for i, tc in enumerate(thread_counts):
            plt.plot(
                matrix_sizes,
                checkSym_data[i],
                marker='o',
                label=f"{line_label_prefix}{tc}"
            )
        plt.title(title_prefix_odd)
        plt.xlabel("Matrix Size (n)")
        plt.ylabel("Elaboration Time (s)")
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
        
        plt.figure(figsize=(8, 6))
        for i, tc in enumerate(thread_counts):
            plt.plot(
                matrix_sizes,
                matTrans_data[i],
                marker='o',
                label=f"{line_label_prefix}{tc}"
            )
        plt.title(title_prefix_even)
        plt.xlabel("Matrix Size (n)")
        plt.ylabel("Elaboration Time (s)")
        plt.grid(True)
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
        plt.show()
    
    else:
        print("Invalid mode. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()
