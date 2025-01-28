import cupy as cp
import numpy as np
import scipy.io
import time

# ---------------------------------------------------------------------
# Load all .mat files (still on CPU via NumPy), then convert to CuPy
# ---------------------------------------------------------------------
LLA = scipy.io.loadmat('LLA.mat')
LLA_h  = cp.asarray(LLA["h"])   # shape (1, 2000) or similar
LLA_lat= cp.asarray(LLA["lat"])
LLA_lon= cp.asarray(LLA["lon"])
LLA_t  = cp.asarray(LLA["t"])
LLA_Vn = cp.asarray(LLA["Vn"])

Xplane_Pos    = scipy.io.loadmat('Xplane_Pos.mat')
Xplane_Pos_Lat_ref  = cp.asarray(Xplane_Pos["Lat_ref"])
Xplane_Pos_Long_ref = cp.asarray(Xplane_Pos["Long_ref"])

AccForces = scipy.io.loadmat('AccForces.mat')
AccForces_An = cp.asarray(AccForces["An"])

DEM_ROI_sub        = scipy.io.loadmat('DEM_ROI_sub.mat')
DEM_ROI_sub_DEM_Z2 = cp.asarray(DEM_ROI_sub["DEM_Z2"])

Selected_DEM       = scipy.io.loadmat('Selected_DEM.mat')
Selected_DEM_in_Python = Selected_DEM["Selected_DEM"] 
# Note: "Selected_DEM_in_Python" might be a cell array; 
# you can leave it as NumPy objects if you only index by row/col occasionally.

DataV4 = scipy.io.loadmat('DataV4.mat')
DataV4_DataV4 = cp.asarray(DataV4["DataV4"])

GroundData = scipy.io.loadmat('GroundData.mat')
GroundData_grounddata = cp.asarray(GroundData["grounddata"])

# ---------------------------------------------------------------------
# Initialize constants (as floats so they work easily with CuPy)
# ---------------------------------------------------------------------
Rm = cp.float64(6335.439e3)
g  = cp.float64(9.8)
Rp = cp.float64(6399.594e3)
ws = cp.float64(7.3e-5)
Re = cp.float64(6371.0072e3)
B  = cp.float64(0.1)
KK = cp.float64(4)
W  = cp.int32(10)

# ...
# Fill in the rest of your constant definitions

# ---------------------------------------------------------------------
# Example arrays on GPU
# ---------------------------------------------------------------------
VE = LLA_Vn[0, :]
VN = LLA_Vn[1, :]
VU = LLA_Vn[2, :]
VD = -VU
Lambda = LLA_lon
phi    = LLA_lat

Lambda_dot = VE / ((Rp + LLA_h) * cp.cos(phi))
phi_dot    = VN / (Rm + LLA_h)

# etc.

# ---------------------------------------------------------------------
# Kalman Filter init (on GPU)
# ---------------------------------------------------------------------
delta_x = cp.zeros((15, 1), dtype=cp.float64)
I = cp.eye(15, dtype=cp.float64)

Q = (2e-3)*cp.eye(15, dtype=cp.float64)
Q[0, 0] = 1e-3
Q[1, 1] = 12.0

P = 0.5*cp.eye(15, dtype=cp.float64)

a = cp.eye(3, dtype=cp.float64)
b = cp.zeros((3, 12), dtype=cp.float64)
Ck = cp.hstack((a, b))

Rk = cp.diag(cp.array([50, 50, 200], dtype=cp.float64))

# INS Position
Lins = LLA_lat
lins = LLA_lon
hins = LLA_h

# ---------------------------------------------------------------------
# Prepare big arrays for storing results (on GPU)
# ---------------------------------------------------------------------
Ns = 201
L_Tercom = cp.zeros((1, Ns), dtype=cp.float64)
l_Tercom = cp.zeros((1, Ns), dtype=cp.float64)
h_Tercom = cp.zeros((1, Ns), dtype=cp.float64)
Delta_x  = cp.zeros((3, Ns), dtype=cp.float64)
LAT      = cp.zeros((1, Ns), dtype=cp.float64)
LON      = cp.zeros((1, Ns), dtype=cp.float64)
HGT      = cp.zeros((1, Ns), dtype=cp.float64)

fE = AccForces_An[0, :]
fN = AccForces_An[1, :]
fD = -AccForces_An[2, :]

start_time = 600
end_time   = 800

# DEM for the MAD step
hDB = DEM_ROI_sub_DEM_Z2
R, C = hDB.shape

# Extend hDB to accommodate the strip length
# For the loop, we might do this once outside:
SL = 10
placeholder = cp.max(hDB) + 100
pad_array   = cp.full((SL-1, C), placeholder, dtype=cp.float64)
hDB         = cp.concatenate((hDB, pad_array), axis=0)

# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
start_execution_time = time.time()

PHI_k_1 = None  # Will store from previous iteration

for idx, i in enumerate(range(start_time, end_time + 10, 10)):
    print("Iteration i:", i)
    
    # ------------------------------------------------
    # 1. Build A (15x15) matrix (on GPU)
    # ------------------------------------------------
    # Example: be sure to wrap all values in cp.float64(...)
    A1 = -phi[0, i-1]/(Rm + LLA_h[0, i-1])  # etc.
    # Repeat for all A2, A3, ...
    # Convert them to cp.float64 to avoid type issues:
    A1 = cp.float64(A1)
    
    # ... define all A2..A33 similarly ...
    # This is a small matrix, so building it in Python is fine.

    A = cp.array([
       [0,   0,   A1,  ..., 0],
       [...],
       ...
       [0,   0,    0,  ..., 0]
    ], dtype=cp.float64)
    
    F = A
    
    # ------------------------------------------------
    # 2. Model Discretization
    # ------------------------------------------------
    if i == start_time:
        dt = LLA_t[0, 0]
    else:
        # watch indexing carefully
        dt = LLA_t[i-1, 0] - LLA_t[i-2, 0]
    
    PHI = I + F * dt
    
    if PHI_k_1 is None:
        PHI_K_1 = PHI
    else:
        PHI_K_1 = PHI_k_1

    # ------------------------------------------------
    # 3. Kalman Filter: Prediction
    # ------------------------------------------------
    delta_x = PHI_K_1 @ delta_x
    P = PHI_K_1 @ P @ PHI_K_1.T + Q
    
    PHI_k_1 = PHI  # store for next iteration

    # ------------------------------------------------
    # 4. MAD Algorithm
    # ------------------------------------------------
    h_RADAR = GroundData_grounddata[i-1, 0]
    
    # Grab next 10 consecutive values from grounddata
    h_RADAR_SL = GroundData_grounddata[i-1 : i-1+SL, 0]
    
    # hMeas, mean, etc.
    # Notice we subtract: hins[0, i-1] - h_RADAR_SL
    # Make sure all are CuPy arrays
    hMeas     = hins[0, i-1] - h_RADAR_SL
    hMeas_bar = cp.mean(hMeas)
    
    # We could do the “parallel” approach by writing a custom CuPy kernel or by vectorizing.
    # For demonstration, let’s illustrate a naive approach:
    
    # shape of hDB is (R + SL-1, C)
    # We compute an MAD array of shape (R-1, C-1) or (R, C)... 
    # The direct approach was nested loops. That’s not ideal for GPU, 
    # but it will still run on GPU. 
    
    # Pre-allocate
    MAD = cp.zeros((R-1, C-1), dtype=cp.float64)
    
    for m in range(1, C):
        for n in range(1, R):
            # index = n + k, but we used 'k' = range(SL); 
            # you can adapt your logic as in the original code
            # example:
            index = n + cp.arange(SL)
            
            # Compute local stats
            # hDB_ind is a single value or a small subarray
            # but here you had hDB[index-1, m-1]
            hDB_slice = hDB[index-1, m-1]
            hDB_mean  = cp.mean(hDB_slice)
            
            T = (hMeas - hMeas_bar) - (hDB_slice - hDB_mean)
            Tabs = cp.abs(T)
            
            # store the mean absolute deviation
            MAD[n-1, m-1] = cp.sum(Tabs) / SL
    
    # find minimum
    Height_min = cp.amin(MAD)
    # find location
    loc = cp.where(MAD == Height_min)
    r_val = loc[0][0]
    c_val = loc[1][0]
    
    # Retrieve lat/long from your cell array Selected_DEM_in_Python
    # If "Selected_DEM_in_Python" is a Python object array, you might do:
    Pos = Selected_DEM_in_Python[r_val, c_val]  # still CPU-based?
    # If it’s a nested array from MATLAB (cell), you might have to handle the conversion carefully.
    
    # Suppose Pos is something like [[Longitude, Latitude]]:
    # or you said Pos[0,1] is lat, Pos[0,0] is long
    # This part is tricky because it’s not all CuPy arrays. 
    # Typically, you’d keep these lookups on CPU. 
    # For demonstration:
    Lat  = Pos[0, 1]
    Long = Pos[0, 0]
    
    L_Tercom[0, i - start_time] = Lat
    l_Tercom[0, i - start_time] = Long
    h_Tercom[0, i - start_time] = hDB[r_val, c_val]
    
    # ------------------------------------------------
    # 5. Kalman Filter: Correction
    # ------------------------------------------------
    zk = cp.array([
        [Lins[0, i-1] - L_Tercom[0, i - start_time]],
        [lins[0, i-1] - l_Tercom[0, i - start_time]],
        [hins[0, i-1] - h_Tercom[0, i - start_time]]
    ], dtype=cp.float64)
    
    Sk = Rk + (Ck @ P @ Ck.T)
    
    K = P @ Ck.T @ cp.linalg.inv(Sk)
    
    Ik = zk - (Ck @ delta_x)
    
    delta_xhat = K @ Ik
    delta_x = delta_xhat
    
    P_hat = (cp.eye(15) - (K @ Ck)) @ P
    P = P_hat
    
    # This is the standard transformation:
    Delta_x[:, i - start_time] = (Ck @ delta_x)[:, 0]
    
    X_c = cp.array([
        [Lins[0, i-1] - Delta_x[0, i - start_time]],
        [lins[0, i-1] - Delta_x[1, i - start_time]],
        [hins[0, i-1] - Delta_x[2, i - start_time]]
    ], dtype=cp.float64)
    
    LAT[0, i - start_time] = X_c[0, 0]
    LON[0, i - start_time] = X_c[1, 0]
    HGT[0, i - start_time] = X_c[2, 0]

end_execution_time = time.time() - start_execution_time
print(f"--- {end_execution_time} seconds on GPU ---")

print("Alhumdulillah all is well (on GPU).")

# If you need final results on CPU as NumPy arrays:
# LAT_cpu = cp.asnumpy(LAT)
# LON_cpu = cp.asnumpy(LON)
# HGT_cpu = cp.asnumpy(HGT)
